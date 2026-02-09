import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from types import SimpleNamespace

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        From: https://github.com/meta-llama/llama/blob/main/llama/model.py
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @classmethod
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    def forward(self, x):
        output = self.norm(x.float()).type_as(x)
        return output * self.weight

NORM = RMSNorm

#######################################################
#######################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import math
import numpy as np
from typing import Tuple

@torch.compiler.disable
@torch.no_grad()
def init_nonrope_bands(module, L_max: int, rho_std: float = 0.03):
    H, K = module.H, module.K
    device = module.alpha_base.device
    dtype  = module.alpha_base.dtype  #（fp32/fp16/bf16）

    # 1) band: log-uniform in [2π/L_max, π/2]
    omega_min = (2.0 * math.pi) / float(L_max)
    omega_max = (0.5 * math.pi)

    log_omin = math.log(omega_min)
    log_omax = math.log(omega_max)

    log_omega = torch.linspace(log_omin, log_omax, K, device=device, dtype=dtype)  # (K,)
    # alpha_base/beta_base
    alpha_base = log_omega.reshape(1, K)
    beta_base  = log_omega.reshape(1, K)

    module.alpha_base.copy_(alpha_base)
    module.beta_base.copy_(beta_base)

    module.delta_alpha.zero_()
    module.delta_beta.zero_()
    module.rho.normal_(mean=0.0, std=rho_std)

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def _pairwise_blocks(x: torch.Tensor) -> torch.Tensor:
    # *prefix, D = x.shape
    # assert D % 2 == 0, "D must be even"
    return x.reshape(*list(x.size()[:-1]), -1, 2)

def _apply_J2_block(u: torch.Tensor) -> torch.Tensor:
    x, y = u.unbind(dim=-1)
    return torch.stack((y, -x), dim=-1)

def apply_J(x: torch.Tensor) -> torch.Tensor:
    xb = _pairwise_blocks(x)
    yb = _apply_J2_block(xb)
    return yb.reshape_as(x)  # yb.view_as(x)

def _init_base_omegas(head_dim: int, base: int = 10_000) -> torch.Tensor:
    assert head_dim % 2 == 0
    # (K,)
    return 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

def _expand_to_HK(t: torch.Tensor, H, K) -> torch.Tensor:
    H = int(H)
    K = int(K)
    if t.size(0) == 1 and t.size(1) == 1:
        return t.expand(H, K)
    if t.size(0) == 1 and t.size(1) == K:
        return t.expand(H, -1)
    if t.size(0) == H and t.size(1) == 1:
        return t.expand(-1, K)
    return t

def apply_symplectic_pos_emb(
    x: torch.Tensor,
    alpha: torch.Tensor,  # (H,K)
    beta: torch.Tensor,   # (H,K)
    rho: torch.Tensor,    # (H,K)
    transpose: bool = False,
    rho_clip: float = 0.99,
    eps: float = 1e-6,
    clock=None,
) -> torch.Tensor:
    B, L, H, D = list(x.detach().size())
    K = D // 2

    alpha = alpha.reshape(H, K)
    beta  = beta.reshape(H, K)
    rho = rho.reshape(H, K)
    # rho   = torch.clamp(rho, -rho_clip, rho_clip)
    clipping = rho_clip * torch.ones_like(rho)
    rho = torch.maximum(torch.minimum(rho, clipping), -clipping)

    a = torch.exp(alpha)
    b = torch.exp(beta)
    c = rho * torch.sqrt(a * b)

    detK  = a * b - c * c
    omega = torch.sqrt(detK + eps)

    if clock is None:
        t = torch.arange(L, device=x.device).reshape(1, L, 1, 1)
    else:
        t = clock.to(device=x.device)
    if transpose:
        t = -t

    omega_f = omega.to(device=x.device).reshape(1, 1, H, K)

    theta  = t * omega_f
    cos_t  = theta.cos()
    sin_om = theta.sin() / (omega_f + eps)

    x1, x2 = x.view(B, L, H, K, 2).unbind(-1)

    y1 = cos_t * x1 + sin_om * (c * x1 + b * x2)
    y2 = cos_t * x2 + sin_om * (-a * x1 - c * x2)

    return torch.stack((y1, y2), dim=-1).view(B, L, H, D)

# ---------------- module ----------------

class SymplecticPE(nn.Module):
    """
    Symplectic Positional Embedding (SyPE) with flexible parameter sharing.

    share_mode:
      - "global"         -> delta/rho shapes (1,1)       -> 3 params
      - "per_head"       -> delta/rho shapes (H,1)       -> 3*H
      - "per_block"      -> delta/rho shapes (1,K)       -> 3*K
      - "per_head_block" -> delta/rho shapes (H,K)       -> 3*H*K

    NOTE:
      RoPE base spectrum (alpha_base/beta_base) is ALWAYS (1,K) so you never lose the per-block RoPE frequencies.
      Deltas are added on top and broadcast to (H,K) according to share_mode.
    """
    def __init__(
        self,
        n_head: int,
        head_dim: int,
        base: int = 10_000,
        learnable: bool = True,
        share_mode: str = "per_head_block",
        fix_rope_base: bool = True,
        apply_J_for_keys_by_default: bool = True,
        # non-RoPE base init (optional)
        nonrope_init: bool = False,
        nonrope_log_mean: float = -3.0,
        nonrope_log_std: float = 0.02,
        nonrope_rho_std: float = 0.02,
        learnable_clock: bool = True,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        assert share_mode in {"global", "per_head", "per_block", "per_head_block"}
        self.H = n_head
        self.D = head_dim
        self.K = head_dim // 2
        self.share_mode = share_mode
        self.apply_J_for_keys_by_default = apply_J_for_keys_by_default
        self.fix_rope_base = fix_rope_base

        # RoPE base spectrum: (1,K) always
        if not nonrope_init:
            omega = _init_base_omegas(head_dim=head_dim, base=base).reshape(1, self.K)  # (1,K)
            alpha_base = torch.log(omega + 1e-30)   # (1,K)
            beta_base  = torch.log(omega + 1e-30)   # (1,K)
        else:
            # non-RoPE base: per-block random logs (still (1,K), shared across heads)
            alpha_base = torch.empty(1, self.K).normal_(mean=nonrope_log_mean, std=nonrope_log_std)
            beta_base  = torch.empty(1, self.K).normal_(mean=nonrope_log_mean, std=nonrope_log_std)

        # register base
        self.alpha_base = nn.Parameter(alpha_base, requires_grad=(not fix_rope_base and learnable))
        self.beta_base  = nn.Parameter(beta_base,  requires_grad=(not fix_rope_base and learnable))

        # deltas/rho shapes by share_mode
        if share_mode == "global":
            shape = (1, 1)
        elif share_mode == "per_head":
            shape = (self.H, 1)
        elif share_mode == "per_block":
            shape = (1, self.K)
        else:  # "per_head_block"
            shape = (self.H, self.K)

        self.delta_alpha = nn.Parameter(torch.zeros(shape), requires_grad=learnable)
        self.delta_beta  = nn.Parameter(torch.zeros(shape), requires_grad=learnable)
        if nonrope_init:
            self.rho = nn.Parameter(torch.empty(shape).normal_(mean=0.0, std=nonrope_rho_std), requires_grad=learnable)
        else:
            self.rho = nn.Parameter(torch.zeros(shape), requires_grad=learnable)

        self.learnable_clock = learnable_clock
        if learnable_clock:
            self.clock = nn.Linear(self.H * self.D, self.H * self.K, bias=False)

    def _expanded_params_HK(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # alpha_base/beta_base: (1,K) -> (H,K)
        alpha_base = self.alpha_base.expand(self.H, self.K)
        beta_base  = self.beta_base .expand(self.H, self.K)

        # deltas/rho: (pH,pK) with pH in {1,H}, pK in {1,K} -> (H,K)
        delta_alpha = _expand_to_HK(self.delta_alpha, self.H, self.K)
        delta_beta  = _expand_to_HK(self.delta_beta,  self.H, self.K)
        rho         = _expand_to_HK(self.rho,         self.H, self.K)

        alpha = alpha_base + delta_alpha
        beta  = beta_base  + delta_beta
        return alpha, beta, rho

    def forward(self, x: torch.Tensor, *, is_key: bool = False, transpose: bool = False, clock=None) -> torch.Tensor:
        alpha, beta, rho = self._expanded_params_HK()
        y = apply_symplectic_pos_emb(x, alpha, beta, rho, transpose=transpose, clock=clock)
        if is_key and self.apply_J_for_keys_by_default:
            y = apply_J(y)
        return y

    def rpe(self, q: torch.Tensor, k: torch.Tensor, clock_x=None):
        if self.learnable_clock:
            assert clock_x is not None, "When learnable_clock=True, pass clock_x of shape (B, L, H*D)."
            clock = self.clock(clock_x).reshape(q.size(0), q.size(1), self.H, -1)  # (B, L, H, K)
            clock = F.softplus(clock)
            clock = clock.cumsum(dim=1)
        else:
            clock = None
        q_pos = self.forward(q, is_key=False, transpose=False, clock=clock)
        k_pos = self.forward(k, is_key=True,  transpose=False, clock=clock)
        return q_pos, k_pos

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = False
 
        # --- replaced RotaryPositionalEmbedding with SymplecticPE ---
        self.rope = SymplecticPE(
            n_head=self.n_head,
            head_dim=config.n_embd // self.n_head,
            base=10_000,

            learnable=True,
            share_mode=getattr(config, "share_mode", "per_head_block"),
            fix_rope_base=True,
            apply_J_for_keys_by_default=True,
            nonrope_init=getattr(config, "nonrope_init", False),
            learnable_clock=True,
        )
        # ------------------------------------------------------------
 
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )
 
    def forward(self, x, *args, **kwargs):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # shapes: (B, T, H, D)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)

        # Symplectic position embedding (identical usage to your second code)
        q, k = self.rope.rpe(q, k, clock_x=x)   # q,k: (B, T, H, D)
        q, k = q.transpose(1, 2), k.transpose(1, 2)  # (B, H, T, D)

        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, H, T, D)
 
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.is_causal,
        )
 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias) # 4 * config.n_embd
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias) 
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layer = config.n_layer
        self.n_layer = n_layer
        self.predictor = config.predictor
        self.attn = nn.ModuleList([CausalSelfAttention(config) for i in range(n_layer)])
        self.pns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.lns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.mlps = nn.ModuleList([MLP(config) for i in range(n_layer)])
        self.n_head = config.n_head
        self.d = config.n_embd // self.n_head

    def forward(self, x):
        for attn, pn, ln, mlp in zip(self.attn, self.pns, self.lns, self.mlps):
            x = x + attn(pn(x))
            x = x + mlp(ln(x))
        return x

def random_ratio_channel_dropout_vectorized(x, training=True):
    if not training:
        return x

    B, C, _ = x.shape
    device = x.device
    r = torch.rand(B, device=device)
    scores = torch.rand(B, C, device=device)
    threshold = r.unsqueeze(1)
    # mask: True = keep
    mask = (scores > threshold).float()
    # (B, C, 1)
    mask = mask.unsqueeze(-1)
    keep_ratio = mask.mean(dim=1, keepdim=True).clamp(min=1e-6)
    mask = mask / keep_ratio
    return x * mask

def nearest_power_of_two_to_sqrt(x: float) -> float:
    if x <= 0:
        raise ValueError("x must be positive")

    sqrt_x = math.sqrt(x)
    n = round(math.log2(sqrt_x))
    n = max(1, n)
    return 2 ** n

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.time_emb_dim = configs.time_emb_dim
        self.enc_in = configs.enc_in + configs.time_emb_dim
        self.number_of_targets = configs.number_of_targets if configs.number_of_targets else configs.enc_in
        n_series = self.number_of_targets
        self.n_series = n_series
        self.endog_dim_factor = configs.endog_dim_factor
        self.exog_dim_factor = configs.exog_dim_factor
        self.d_input = self.exog_dim_factor if not configs.d_model else configs.d_model//2       # int(np.sqrt(self.enc_in)) * 4
        self.seq_emb_dim = self.endog_dim_factor if not configs.d_model else configs.d_model//2

        # Use ARX Tokenization
        self.dropout = nn.Dropout(configs.dropout)
        self.n_head = self.d_input // 16 if not configs.n_heads else configs.n_heads
        self.d = self.d_input // self.n_head
        self.local_attention = getattr(configs, "local_attention", False)   # Disabled
        self.local_attention_window = getattr(configs, "local_attention_window", 8)

        
        self.channel_token_mixer_seq = nn.Linear(self.enc_in, self.d_input, bias=False)
        self.value_token_mixer = nn.Parameter(0.02*torch.randn(self.number_of_targets, 1, self.seq_emb_dim))

        hidden_dim = self.d_input + self.seq_emb_dim
        print(f'Current Dimension: {hidden_dim}')
        transformer_config = SimpleNamespace(n_layer=configs.e_layers, n_embd=hidden_dim, n_head=self.n_head, dropout=configs.dropout, 
                                             bias=False, max_len=self.seq_len+self.pred_len, predictor=configs.predictor, decay=True, enc_in=self.number_of_targets,
                                             local_attention=self.local_attention, local_attention_window=self.local_attention_window)
        self.temporal_processor = nn.Sequential(
                                                #NORM(hidden_dim),
                                                Transformer(transformer_config),
                                                #NORM(hidden_dim)
                                                )

        self.pos_emb = nn.Embedding(self.seq_len+self.pred_len, hidden_dim)
        self.channel_emb_2 = nn.Parameter(torch.zeros(1, self.number_of_targets, 1, hidden_dim))
        self.out_fill = nn.Linear(self.seq_len, self.pred_len, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * configs.e_layers))

        self.random_channel_dropout = configs.random_channel_dropout

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.00)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # B C L

        # RevIN: preprocessing
        mean = x[:, :, [-1]] # x[:, :, -self.pred_len:].mean(dim=2, keepdim=True) # # B C 1
        std = torch.ones(x.size(0), x.size(1), 1, device=x.device) # x.std(dim=2, keepdim=True) + 1e-8 # torch.ones_like(x.std(dim=2, keepdim=True)) #                    # B C 1
        x = (x - mean) / std                                       # B C L

        out_fill = self.out_fill(x)  # x @ self.out_fill #
        # out_fill = torch.einsum('bcl,clp->bcp', x, self.out_fill)
        x = torch.cat([x, out_fill], dim=2)  # B C L+pred_len
        if x_mark_enc is not None and x_mark_dec is not None and self.time_emb_dim:
            mark = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:]], dim=1).permute(0, 2, 1) # B C L
            mark = (mark - x_mark_enc.permute(0, 2, 1)[:, :, [-1]]) # / (x_mark_enc.permute(0, 2, 1).std(dim=-1, keepdim=True) + 1e-8)  # ???
            x = torch.cat([mark, x], dim=1)  # B C L

        ######
        # For series c
        # x_1: n_1 * v_{c,in}, v \in R^d_{in}
        # channel_token_1: [n_1, ..., n_C] 
        # channel_token_1: for each of the series, decide whether to drop it
        # channel_token_1 \times M, M \in R^{C \times d_{channel}}
        # [p_c+, p_1+, channel_token_1, x_1]: d_{channel} + d_{in}
        # [p_c+, p_1+, channel_token_2, x_2]: d_{channel} + d_{in}
        # ...
        # [p_c+, p_1+, channel_token_n, x_n]: d_{channel} + d_{in}
        # Transformer
        # [...,                         y_n] last d_{in} dimension
        # y_n \dot v_{c,out} -> \hat{x}_n
        #####

        # !!!
        x_dropout = random_ratio_channel_dropout_vectorized(x, self.training and self.random_channel_dropout)
        channel_token = self.channel_token_mixer_seq(x_dropout.permute(0, 2, 1))   # B L C -> B L d
        channel_token = channel_token.unsqueeze(1).expand(-1, self.number_of_targets, -1, -1)
        x = x[:, -self.number_of_targets:].unsqueeze(-1) # B C L 1
        x = torch.einsum('bcli,cid->bcld', x, self.value_token_mixer)   # B C L d
        x = torch.cat([channel_token, x], dim=-1)    # B C L 2d

        B, C, N, _ = x.shape
        pos = torch.arange(0, N, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos)
        x = x + self.channel_emb_2 + pos_emb

        # x = x.reshape(B * C, N, -1)                   # B*C L 2d    
        # x = self.temporal_processor(x)              # B*C L 2d
        # x = x[:, -self.pred_len:].reshape(B, C, self.pred_len, -1)   # B C pred_len 2d
        B, C, N, D = x.shape
        chunk_size = 128

        x = x.reshape(B * C, N, D)

        outputs = []
        for i in range(0, C, chunk_size):
            x_chunk = x[i*B : min((i+chunk_size)*B, B*C)]  # (B*chunk, N, D)
            
            out_chunk = self.temporal_processor(x_chunk)   # (B*chunk, N, D)
            outputs.append(out_chunk)

        x = torch.cat(outputs, dim=0)  # (B*C, N, D)

        x = x[:, -self.pred_len:].reshape(B, C, self.pred_len, D)
        y = torch.einsum('bcld,cdi->bcli', x[:, :, :, -self.seq_emb_dim:], self.value_token_mixer.transpose(-2, -1)).squeeze(-1)  # B C L_P
        x = y

        # RevIN: inverse processing
        x = (x*std[:, -self.number_of_targets:] + mean[:, -self.number_of_targets:]).permute(0, 2, 1)                  # B s Ct

        if self.training:
            return x, torch.tensor(0, device=x.device)
        else:
            return x
 
