# Dual-Attention Multimodal Framework for Molecular Property Prediction

This README reflects the current state of the code in this branch.

## Overview

The model uses two branches:

- Graph branch: `DAGT` in `models/dagt.py`
- SMILES branch: 3072-d embeddings (prefer loading from precomputed `.npy` files)
- Fusion: `CrossAttention` in `models/cross_attention.py`
- Output heads: classification, multilabel classification, or regression

The main training entry point is `run_training.py` (not `training/train.py` directly).

## Installation

```bash
pip install -r requirements.txt
```

If `rdkit-pypi` fails on your machine, install RDKit via conda:

```bash
conda install -c conda-forge rdkit
```

## Data and Embeddings

The code currently supports these datasets (defined in `data/dataset.py`):

- Classification: `HIV`, `BACE`, `BBBP`
- Regression: `ESOL`, `FreeSolv`, `Lipophilicity`
- Multilabel: `Tox21`, `SIDER`, `ClinTox`

Place CSV files in `dataset/` with names like:

- `dataset/HIV.csv`
- `dataset/BACE.csv`
- ...

Each CSV must contain a `smiles` column.
Label columns are fixed per dataset in `DATASET_CONFIG`.

Precomputed SMILES embeddings (can be downloaded full in [here](https://drive.google.com/drive/folders/16nPisYpVO0pshPvIibRBJgoMooovuS-V?usp=drive_link)) (optional but recommended) should be stored as:

- `embeddings/<DATASET>_embeddings.npy`

Example: `embeddings/BACE_embeddings.npy`.

If no `.npy` embeddings are found, the code uses `LLMEncoder`:

- If `openai_api_key` is set and `use_local_fallback=false`: calls OpenAI embeddings API.
- Default (`use_local_fallback=true`): random fallback embeddings (fine for smoke tests, not for reporting real model quality).

## Training

### Recommended command

```bash
python run_training.py --config configs/default_config.json
```

### Example with overrides

```bash
python run_training.py --dataset BACE --split_type scaffold --epochs 30 --batch_size 32 --num_seeds 3
```

### Main CLI arguments (`run_training.py`)

- `--config`: JSON config path (default `configs/default_config.json`)
- `--dataset`: dataset name
- `--task_type`: `classification|multilabel|regression` (if mismatched, code auto-syncs to dataset task)
- `--batch_size`
- `--epochs`
- `--lr`
- `--embeddings_dir`
- `--split_type`: `random|scaffold`
- `--num_seeds`: number of runs with consecutive seeds (for mean/std reporting)

### GPU note

Default is `require_gpu=true`. If CUDA is unavailable, set `require_gpu=false` in config.

## Inference

```bash
python inference/predict.py --checkpoint checkpoints/best_model_BACE_seed42.pt --smiles "CCO" "c1ccccc1"
```

Arguments:

- `--checkpoint` (required)
- `--smiles` (one or more SMILES strings)
- `--config` (optional; needed only if checkpoint does not include `config`)

## Evaluation

The repo includes `evaluate.py` for evaluating checkpoints on train/val/test splits and for scanning a checkpoint directory.


## Project Structure

```text
.
|-- configs/
|   `-- default_config.json
|-- data/
|   `-- dataset.py
|-- dataset/
|   |-- BACE.csv
|   |-- BBBP.csv
|   |-- ClinTox.csv
|   |-- ESOL.csv
|   |-- FreeSolv.csv
|   |-- HIV.csv
|   |-- Lipophilicity.csv
|   |-- SIDER.csv
|   `-- Tox21.csv
|-- embeddings/
|   `-- *_embeddings.npy
|-- inference/
|   `-- predict.py
|-- models/
|   |-- cross_attention.py
|   |-- dagt.py
|   `-- llm_encoder.py
|-- training/
|   |-- losses.py
|   `-- train.py
|-- evaluate.py
|-- requirements.txt
|-- run_training.py
`-- README.md
```
