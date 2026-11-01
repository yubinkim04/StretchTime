# StretchTime: Adaptive Time Series Forecasting via Symplectic Attention

**ICML 2026** · [Paper (PMLR)](https://proceedings.mlr.press/v306/kim26bk.html) · [arXiv](https://arxiv.org/abs/2602.08983) · [OpenReview](https://openreview.net/forum?id=6NER0ZYcop)

Yubin Kim, Viresh Pati, Jevon Twitty, Vinh Pham, Shihao Yang, Jiecheng Lu

This repository hosts the official code for our ICML 2026 paper "StretchTime: Adaptive Time Series Forecasting via Symplectic Attention".

The main model can be found in `models/StretchTime.py`

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2026stretchtime,
  title     = {StretchTime: Adaptive Time Series Forecasting via Symplectic Attention},
  author    = {Kim, Yubin and Pati, Viresh and Twitty, Jevon and Pham, Vinh and Yang, Shihao and Lu, Jiecheng},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  pages     = {58878--58898},
  publisher = {PMLR},
  year      = {2026},
  url       = {https://proceedings.mlr.press/v306/kim26bk.html}
}
```

## 1. Install Required Packages

Install PyTorch first, then install the additional dependencies with:

```bash
pip install -r requirements.txt
```

## 2. Download Datasets

You can download the datasets used in the paper from the link provided by the iTransformer or Time-Series-Library githubs. Place the downloaded files in `dataset/`.

## 3. Run the Training Scripts

To train the StretchTime model, run:

```bash
bash stretchtime.sh
```

If you don't have a Weights & Biases account, run in offline mode instead:

```bash
WANDB_MODE=offline bash stretchtime.sh
```

## 4. Overall Architecture

![Overall Architecture](figs/STime.png)

## 5. Monitoring

Metrics are logged to Weights & Biases.

## Acknowledgements

This codebase builds on the following open-source repositories. We thank the authors for releasing their code. Files adapted from these projects remain under their original licenses.

| Project | License |
|---|---|
| [Time-Series-Library](https://github.com/thuml/Time-Series-Library) | MIT |
| [PatchTST](https://github.com/yuqinie98/PatchTST) | Apache-2.0 |
| [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | Apache-2.0 |
| [iTransformer](https://github.com/thuml/iTransformer) | MIT |
| [Autoformer](https://github.com/thuml/Autoformer) | MIT |
| [Informer](https://github.com/zhouhaoyi/Informer2020) | Apache-2.0 |
| [TimeMixer](https://github.com/kwuking/TimeMixer) | Apache-2.0 |
| [DUET](https://github.com/decisionintelligence/DUET) | MIT |
| [FITS](https://github.com/VEWOXIC/FITS) | Apache-2.0 |
| [OLinear](https://github.com/jackyue1994/OLinear) | — |
| [RevIN](https://github.com/ts-kim/RevIN) | MIT |
| [Time-LLM](https://github.com/KimMeen/Time-LLM) | Apache-2.0 |
| [labml.ai annotated implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) | MIT |
| [LLaMA](https://github.com/meta-llama/llama) | Llama 2 Community License |

## Contact

For questions, please open an issue or email yubinkim@mit.edu.
