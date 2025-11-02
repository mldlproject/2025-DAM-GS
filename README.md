# Dual-Attention Multimodal Framework for Molecular Property Prediction

This repository implements a Dual-Attention Multimodal Framework for molecular property prediction, integrating both structural (graph) and semantic (SMILES) representations of molecules. The method combines a Dual-Attention Graph Transformer (DAGT) encoder with OpenAI's text-embedding-3-large model through a Cross-Attention module.

## Architecture
The proposed framework consists of:
- **DAGT Encoder (fg)**: Maps molecular graphs to 512-dimensional representations
- **Text-Embedding-3-Large Encoder (fs)**: Maps SMILES strings to 3072-dimensional representations  
- **Cross-Attention Module**: Fuses graph and SMILES representations into a joint representation
- **Predictor Head**: Makes final property predictions

## Features

- **Dual-Attention Graph Transformer (DAGT)**: 
  - Bond-level message passing with attention
  - Atom-level attention for global structure understanding
- **Cross-Modal Fusion**: Cross-attention mechanism to combine graph and SMILES embeddings
- **Contrastive Learning**: Aligns graph and SMILES embeddings in shared representation space
- **Multi-Task Support**: Classification, regression, and multi-label prediction

## Installation

1. Clone this repository:
```bash
git clone https://github.com/mldlproject/2025-DAM-GS.git
cd 2025-DAM-GS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: RDKit installation may require conda:
```bash
conda install -c conda-forge rdkit
```

## Dataset

The repository supports the following molecular property prediction datasets:
- **Classification**: HIV, BBBP, BACE
- **Regression**: ESOL, FreeSolv, Lipophilicity
- **Multi-label**: Tox21, SIDER, ClinTox

Datasets should be placed in the `dataset/` directory with CSV format containing SMILES strings and labels.

## Usage

### Training

Train a model on a dataset:

```bash
python training/train.py
```

```python
from training.train import Trainer

config = {
    'dataset_name': 'HIV',
    'data_dir': 'dataset',
    'task_type': 'classification',
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'lambda_contrastive': 0.1,
    'use_local_fallback': True,  # Use local model instead of OpenAI API
    # 'openai_api_key': 'your-api-key',  # Optional: for OpenAI API
}

trainer = Trainer(config)
trainer.train()
```

## Project Structure

```
.
├── models/
│   ├── dagt.py              # Dual-Attention Graph Transformer
│   ├── cross_attention.py   # Cross-attention fusion module
│   └── llm_encoder.py       # LLM integration
├── data/
│   └── dataset.py            # Data loading and preprocessing
├── training/
│   ├── train.py             # Training script
│   └── losses.py             # Loss functions
├── inference/
│   └── predict.py            # Inference script
├── configs/
│   └── default_config.json  # Default configuration
├── dataset/                  # Dataset directory
├── checkpoints/              # Saved models
└── requirements.txt          # Dependencies
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI for text-embedding-3-large model
- PyTorch Geometric for graph neural network utilities
- RDKit for molecular processing
