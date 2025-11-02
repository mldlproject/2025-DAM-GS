"""
Main script to run training with configuration file or command-line arguments.
"""

import json
import argparse
import sys
from training.train import Trainer


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Train dual-branch molecular property prediction model')
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                        help='Path to configuration JSON file')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'dataset_name': 'HIV',
            'data_dir': 'dataset',
            'task_type': 'classification',
            'batch_size': 32,
            'num_epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'lambda_contrastive': 0.1,
            'temperature': 0.07,
            'graph_dim': 512,
            'smiles_dim': 3072,
            'hidden_dim': 512,
            'num_layers': 3,
            'num_heads': 8,
            'checkpoint_dir': 'checkpoints',
            'use_local_fallback': True,
            'openai_api_key': None,
            'num_workers': 0,
            'seed': 42
        }
    
    # Override with command-line arguments
    if args.dataset:
        config['dataset_name'] = args.dataset
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    
    print("=" * 50)
    print("Training Configuration:")
    print("=" * 50)
    for key, value in config.items():
        if key != 'openai_api_key' or value is not None:
            print(f"  {key}: {value}")
    print("=" * 50)
    
    # Initialize and run trainer
    try:
        trainer = Trainer(config)
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

