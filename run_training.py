"""
Main script to run training with configuration file or command-line arguments.
"""

import json
import argparse
import sys
import numpy as np
from data.dataset import DATASET_CONFIG
from training.train import Trainer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def _report_aggregate(all_metrics: list, dataset_name: str):
    print(f"\n{'='*55}")
    print(f"Aggregate results ({len(all_metrics)} seeds) — {dataset_name}")
    print(f"{'='*55}")
    keys = [k for k in all_metrics[0] if k != 'loss']
    for key in keys:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            print(f"  {key}: {np.mean(vals):.4f} ({np.std(vals):.4f})")
    print(f"{'='*55}")


def _sync_task_type(config: dict, explicit_task_type=None):
    dataset_name = config.get('dataset_name')
    if dataset_name not in DATASET_CONFIG:
        valid = ", ".join(sorted(DATASET_CONFIG.keys()))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Valid: {valid}")

    expected_task_type = str(DATASET_CONFIG[dataset_name]["task_type"])
    current_task_type = explicit_task_type if explicit_task_type is not None else config.get('task_type')

    if current_task_type is not None and current_task_type != expected_task_type:
        print(
            f"[Config] task_type '{current_task_type}' khong khop dataset '{dataset_name}'. "
            f"Tu dong doi thanh '{expected_task_type}'."
        )

    config['task_type'] = expected_task_type


def main():
    parser = argparse.ArgumentParser(description='Train dual-branch molecular property prediction model')
    parser.add_argument('--config', type=str, default='configs/default_config.json')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--task_type', type=str, default=None,
                        choices=['classification', 'multilabel', 'regression'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--embeddings_dir', type=str, default=None)
    parser.add_argument('--split_type', type=str, default=None, choices=['random', 'scaffold'])
    parser.add_argument('--num_seeds', type=int, default=1,
                        help='Number of independent runs with consecutive seeds')

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'dataset_name': 'HIV',
            'data_dir': 'dataset',
            'embeddings_dir': 'embeddings',
            'task_type': 'classification',
            'split_type': 'random',
            'batch_size': 32,
            'num_epochs': 50,
            'warmup_epochs': 5,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'lambda_contrastive': 0.1,
            'temperature': 0.07,
            'graph_dim': 512,
            'smiles_dim': 3072,
            'projection_dim': 512,
            'hidden_dim': 512,
            'num_layers': 3,
            'num_heads': 8,
            'pool_type': 'attention',
            'checkpoint_dir': 'checkpoints',
            'require_gpu': True,
            'device': 'cuda',
            'use_local_fallback': True,
            'openai_api_key': None,
            'num_workers': 0,
            'seed': 42,
        }

    if args.dataset:
        config['dataset_name'] = args.dataset
    if args.task_type:
        config['task_type'] = args.task_type
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.embeddings_dir:
        config['embeddings_dir'] = args.embeddings_dir
    if args.split_type:
        config['split_type'] = args.split_type

    _sync_task_type(config, explicit_task_type=args.task_type)

    print("=" * 55)
    print("Training Configuration:")
    print("=" * 55)
    for key, value in config.items():
        if key != 'openai_api_key' or value is not None:
            print(f"  {key}: {value}")
    print("=" * 55)

    base_seed = config.get('seed', 42)
    num_seeds = max(1, args.num_seeds)
    all_test_metrics = []

    try:
        for run in range(num_seeds):
            run_config = {**config, 'seed': base_seed + run}
            if num_seeds > 1:
                print(f"\n{'='*55}")
                print(f"Run {run + 1}/{num_seeds}  seed={run_config['seed']}")
                print(f"{'='*55}")

            trainer = Trainer(run_config)
            trainer.train()
            test_m = trainer.evaluate_test()
            all_test_metrics.append(test_m)
            print(f"\n[Run {run + 1}] Test: {test_m}")

        if num_seeds > 1:
            _report_aggregate(all_test_metrics, config['dataset_name'])

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
