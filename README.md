![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

# Adapting PLMs to the candidate pool: exploring transductive learning for antibody developability under distribution shift

This repository contains the code, curated benchmarks and evaluation protocol
for our paper on adapting protein language models (PLMs) to antibody property
prediction.

We compare zero-shot likelihoods, a linear probe (a prototype classifier on
frozen embeddings), a small feature transformation (FT), LoRA, and Gini-TIM, a
transductive objective that uses the unlabeled candidate pool and no test
labels. All methods are evaluated under identical features, splits and metrics.

---

## Requirements

| Requirement | Details |
|-------------|---------|
| **OS** | Linux or macOS (Windows untested) |
| **Python** | ≥ 3.10, < 3.12 |
| **GPU / CUDA** | CUDA ≥ 11.8 strongly recommended; CPU-only runs are possible but very slow |
| **PyTorch** | 2.6.0 (pinned) — must match your CUDA version |
| **Package manager** | [Poetry](https://python-poetry.org/) ≥ 1.2 |

Key Python dependencies (all managed by Poetry):

| Package | Version |
|---------|---------|
| `torch` | 2.6.0 |
| `fair-esm` | ^2.0.0 |
| `transformers` | ^4.26.1 |
| `peft` | ≥0.10.0 |
| `pytorch-lightning` | ^1.9.5 |
| `scikit-learn` | ^1.0.0 |
| `pandas` | ^2.2.0 |

---

## Installation

```bash
git clone https://github.com/AstraZeneca/PLM-Developability.git
cd PLM-Developability
pip install poetry
poetry install
```

For GPU environments, verify that your PyTorch build matches your CUDA version after install.

---

## Supported PLMs

| Key in config | Model |
|---------------|-------|
| `esm.t6-8m` … `esm.t48-15b` | ESM-2 family (8 M – 15 B parameters) |
| `saprot-seq-af2` / `saprot-seq-pdb` | SaProt (sequence-only mode) |
| `onehot` | One-hot amino-acid encoding |

SaProt requires local model weights; set `local_saprot_dir_af2` / `local_saprot_dir_pdb` in the config.

---

## Downloading model weights

### ESM-2
No action required. Weights are downloaded automatically via `fair-esm` on first run and cached in `~/.cache/torch/hub/`.

### SaProt 650M
Download manually from HuggingFace and place under `models/`:

```bash
# AF2 version
git clone https://huggingface.co/westlake-repl/SaProt_650M_AF2 models/SaProt_650M_AF2

# PDB version
git clone https://huggingface.co/westlake-repl/SaProt_650M_PDB models/SaProt_650M_PDB
```

Then set in your config:
```yaml
local_saprot_dir_af2: "models/SaProt_650M_AF2"
local_saprot_dir_pdb: "models/SaProt_650M_PDB"
```

> **Note:** `git clone` from HuggingFace requires [Git LFS](https://git-lfs.com) to download the actual weights (`git lfs install` before cloning).

---

## Benchmark modes

The `mode` key in each YAML config selects the pipeline:

| `mode` | Description |
|--------|-------------|
| `tim` | Frozen PLM embeddings → TIM (transductive inference) |
| `tim` + `update_feat: true` | FT-TIM: learnable feature transform + TIM |
| `lora_ce` | LoRA fine-tune PLM + cross-entropy classifier head |
| `lora_tim` | LoRA fine-tune PLM → re-embed → TIM |

CE and FT-CE are the `tim` mode with `loss_weights: [1,0,0]` (entropy terms zeroed out).

**Zero-shot** (no training, no labels at training time):

```
benchmark_flab_zero_shot.py
```

Scores sequences by WT-relative masked pseudo-log-likelihood (LLR) and finds an optimal threshold on the training split for classification metrics.

---

## Quick start

```bash
cd notebooks

# TIM on frozen ESM embeddings
poetry run python benchmark_all.py --config configs/flab/config-gini-tim-random.yaml

# FT-TIM
poetry run python benchmark_all.py --config configs/flab/config-ft-gini-tim-random.yaml

# LoRA-CE
poetry run python benchmark_all.py --config configs/lora-flab/config-flab-lora-ce-random.yaml

# LoRA-TIM
poetry run python benchmark_all.py --config configs/lora-flab/config-flab-lora-gini-tim-random.yaml

# Zero-shot LLR
poetry run python benchmark_flab_zero_shot.py --config configs/config-zero-shot.yaml
```

---

## Running the full benchmark suite

### Flab dataset

```bash
cd notebooks
bash run-flab.sh
```

Runs CE, FT-CE, TIM, FT-TIM across Random / 1H→3H / 3H→1H splits.

Override output and config directories:

```bash
LOG_DIR=/path/to/logs CONFIGS_DIR=/path/to/configs/flab bash run-flab.sh
```

### Polyreactivity (Harvey et. al. 2022) dataset

```bash
cd notebooks
bash run-poly.sh
```

Runs LoRA-CE, FT-CE, CE across splits A and B.

Override:

```bash
LOG_DIR=/path/to/logs CONFIGS_DIR=/path/to/configs bash run-poly.sh
```

---

## Config reference

Every YAML config must specify:

| Key | Description |
|-----|-------------|
| `mode` | Pipeline: `tim`, `lora_ce`, or `lora_tim` |
| `data_dir` | Path to dataset directory |
| `train_file` | CSV filename for training split |
| `test_file` | CSV filename for test split |
| `val_file` | CSV filename for validation split (optional) |
| `target_column` | Column name for binary labels |
| `log_dir` | Output directory for logs, plots, and CSVs |
| `plms` | List of PLM keys to benchmark |
| `seeds` | List of random seeds for multi-seed aggregation |

For zero-shot configs, replace `target_column` with `target_column_binary` and optionally `target_column_scores` (continuous values for Spearman correlation). Add `wt_seq` with the full wild-type VH-VL sequence.

---

## Output

Each run writes:

- `{log_dir}/results_table/results_*.csv` — per-model metrics (train / val / test splits)
- `{log_dir}/plots/` — t-SNE visualisations and PR curves
- `{data_dir}/encodings/` — cached PLM embeddings (`.pt` files, reused on subsequent runs)

Multi-seed runs additionally produce `*_agg.csv` files with `mean ± std` across seeds.

---

## Citation

If you use this code, please cite:

```
@misc{PLM-Developability,
  author = {Ziko Masud, Mehdi Boroumand},
  title  = {Adapting protein language models to the candidate pool:
Transductive learning for antibody developability under
distribution shift},
  year   = {2026},
  url    = {https://github.com/AstraZeneca/PLM-Developability}
}
```
