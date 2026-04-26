"""
Training script for dual-branch molecular property prediction model.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cross_attention import DualBranchModel
from models.llm_encoder import LLMEncoder, FrozenLLMEncoder
from data.dataset import DATASET_CONFIG, collate_fn, get_dataset, scaffold_split, random_split
from training.losses import ContrastiveLoss, ClassificationLoss, RegressionLoss, MultiLabelLoss


class Trainer:
    """
    Trainer class for dual-branch molecular property prediction.
    """

    def __init__(self, config):
        self.config = config
        require_gpu = config.get('require_gpu', True)
        requested_device = config.get('device', 'cuda')

        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is required for training but CUDA is not available.\n"
                "Please install a CUDA-enabled PyTorch build and run on a machine with NVIDIA GPU.\n"
                "If you intentionally want CPU fallback, set require_gpu=false in config."
            )

        if requested_device.startswith('cuda') and torch.cuda.is_available():
            self.device = torch.device(requested_device)
            gpu_idx = self.device.index if self.device.index is not None else 0
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            total_mem_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)
            print(f"Using device: {self.device} ({gpu_name}, {total_mem_gb:.1f} GB)")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: {self.device}")

        dataset_name = config['dataset_name']
        if dataset_name not in DATASET_CONFIG:
            valid = ", ".join(sorted(DATASET_CONFIG.keys()))
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Valid: {valid}")

        expected_task_type = str(DATASET_CONFIG[dataset_name]['task_type'])
        configured_task_type = config.get('task_type')
        if configured_task_type is not None and configured_task_type != expected_task_type:
            print(
                f"[Config] task_type '{configured_task_type}' khong khop dataset '{dataset_name}'. "
                f"Tu dong doi thanh '{expected_task_type}'."
            )
        self.config['task_type'] = expected_task_type

        # Load dataset
        self.dataset = get_dataset(
            dataset_name,
            data_dir=config['data_dir'],
            embeddings_dir=config.get('embeddings_dir', 'embeddings'),
            task_type=expected_task_type,
        )

        # Split dataset
        seed = config.get('seed', 42)
        split_type = config.get('split_type', 'random')

        if split_type == 'scaffold':
            train_idx, val_idx, test_idx = scaffold_split(self.dataset, seed=seed)
        else:
            train_idx, val_idx, test_idx = random_split(self.dataset, seed=seed)

        self.train_indices = train_idx

        self.train_loader = DataLoader(
            Subset(self.dataset, train_idx),
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 0)
        )
        self.val_loader = DataLoader(
            Subset(self.dataset, val_idx),
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 0)
        )
        self.test_loader = DataLoader(
            Subset(self.dataset, test_idx),
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 0)
        )

        # Infer feature dimensions
        sample_graph = self.dataset[0]['graph']
        atom_dim = sample_graph.x.size(1)
        bond_dim = (
            sample_graph.edge_attr.size(1)
            if sample_graph.edge_attr.numel() > 0
            else config.get('bond_dim', 10)
        )
        self.config['atom_dim'] = int(atom_dim)
        self.config['bond_dim'] = int(bond_dim)

        # Determine number of classes
        if self.dataset.task_type == 'regression':
            num_classes = 1
        elif self.dataset.task_type == 'multilabel':
            num_classes = len(self.dataset.label_cols)
        else:
            num_classes = len(torch.unique(torch.tensor(self.dataset.valid_labels)))
        
        # Initialize LLM encoder
        self.frozen_llm = None
        if self.dataset.precomputed_embeddings is None:
            llm = LLMEncoder(
                api_key=config.get('openai_api_key'),
                use_local_fallback=config.get('use_local_fallback', True)
            )
            self.frozen_llm = FrozenLLMEncoder(llm)

        # Initialize model
        self.model = DualBranchModel(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            graph_dim=config.get('graph_dim', 512),
            smiles_dim=config.get('smiles_dim', 3072),
            hidden_dim=config.get('hidden_dim', 512),
            num_classes=num_classes,
            num_layers=config.get('num_layers', 3),
            num_heads=config.get('num_heads', 8),
            pool_type=config.get('pool_type', 'attention'),
        ).to(self.device)

        # Loss functions
        self.contrastive_loss = ContrastiveLoss(
            graph_dim=config.get('graph_dim', 512),
            smiles_dim=config.get('smiles_dim', 3072),
            projection_dim=config.get('projection_dim', config.get('graph_dim', 512)),
            temperature=config.get('temperature', 0.07)
        ).to(self.device)
        self.lambda_contrastive = config.get('lambda_contrastive', 0.1)

        if self.dataset.task_type == 'regression':
            self.task_loss = RegressionLoss()
        elif self.dataset.task_type == 'multilabel':
            self.task_loss = MultiLabelLoss()
        else:
            self.task_loss = ClassificationLoss()

        # Optimizer
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.contrastive_loss.parameters()),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # Scheduler: linear warmup -> cosine annealing
        num_epochs = config['num_epochs']
        warmup_epochs = config.get('warmup_epochs', 5)
        lr = config.get('learning_rate', 1e-4)

        if 0 < warmup_epochs < num_epochs:
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[
                    LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                              total_iters=warmup_epochs),
                    CosineAnnealingLR(self.optimizer,
                                      T_max=max(1, num_epochs - warmup_epochs),
                                      eta_min=lr * 0.01),
                ],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=lr * 0.01
            )

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self):
        self.model.train()
        total_loss = total_task = total_contra = 0.0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            graphs = batch['graph'].to(self.device)
            labels = batch['label'].to(self.device)
            smiles_embeddings = self._get_embeddings(batch)

            self.optimizer.zero_grad()
            logits, h_g, _ = self.model(graphs, smiles_embeddings)

            if self.dataset.task_type == 'regression':
                if logits.dim() > 1:
                    logits = logits.squeeze(1)
                task_loss = self.task_loss(logits, labels.float())
            else:
                task_loss = self.task_loss(logits, labels)

            contra_loss = self.contrastive_loss(h_g, smiles_embeddings)
            loss = task_loss + self.lambda_contrastive * contra_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_task += task_loss.item()
            total_contra += contra_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", task=f"{task_loss.item():.4f}", refresh=False)

        n = len(self.train_loader)
        return {
            'total_loss': total_loss / n,
            'task_loss': total_task / n,
            'contrastive_loss': total_contra / n,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _get_embeddings(self, batch) -> torch.Tensor:
        emb = batch.get('smiles_embedding')
        if emb is not None:
            return emb.to(self.device)
        with torch.no_grad():
            return self.frozen_llm(batch['smiles']).to(self.device)

    def _evaluate_loader(self, loader, desc='Eval') -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                graphs = batch['graph'].to(self.device)
                labels = batch['label'].to(self.device)
                smiles_embeddings = self._get_embeddings(batch)

                logits, _, _ = self.model(graphs, smiles_embeddings)

                if self.dataset.task_type == 'regression':
                    if logits.dim() > 1:
                        logits = logits.squeeze(1)
                    loss = self.task_loss(logits, labels.float())
                    preds = logits.cpu().numpy()
                    label_vals = labels.float().cpu().numpy()
                else:
                    loss = self.task_loss(logits, labels)
                    if self.dataset.task_type == 'multilabel':
                        preds = torch.sigmoid(logits).cpu().numpy()
                    else:
                        preds = torch.softmax(logits, dim=1).cpu().numpy()
                    label_vals = labels.cpu().numpy()

                total_loss += loss.item()
                all_preds.append(preds)
                all_labels.append(label_vals)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = {'loss': total_loss / len(loader)}

        if self.dataset.task_type == 'regression':
            metrics['rmse'] = float(np.sqrt(mean_squared_error(all_labels, all_preds)))
            metrics['mae'] = float(mean_absolute_error(all_labels, all_preds))
        elif self.dataset.task_type == 'multilabel':
            try:
                metrics['roc_auc'] = roc_auc_score(all_labels, all_preds, average='macro')
            except Exception:
                metrics['roc_auc'] = 0.0
        else:
            if len(np.unique(all_labels)) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(all_labels, all_preds[:, 1])
                except Exception:
                    metrics['roc_auc'] = 0.0

        return metrics

    def validate(self) -> dict:
        return self._evaluate_loader(self.val_loader, desc='Validation')

    def evaluate_test(self) -> dict:
        seed = self.config.get('seed', 42)
        ckpt_path = os.path.join(
            self.config.get('checkpoint_dir', 'checkpoints'),
            f"best_model_{self.config['dataset_name']}_seed{seed}.pt",
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded best checkpoint: {ckpt_path}")
        return self._evaluate_loader(self.test_loader, desc='Test')

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        dataset_name = self.config['dataset_name']
        split_type = self.config.get('split_type', 'random')
        seed = self.config.get('seed', 42)
        print(f"Training {dataset_name} | split={split_type} | seed={seed}")
        print(f"  train={len(self.train_loader.dataset)}  "
              f"val={len(self.val_loader.dataset)}  "
              f"test={len(self.test_loader.dataset)}")

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            train_m = self.train_epoch()
            self.train_losses.append(train_m)

            val_m = self.validate()
            self.val_losses.append(val_m)

            print(f"  Train loss: {train_m['total_loss']:.4f} "
                  f"(task={train_m['task_loss']:.4f}, contra={train_m['contrastive_loss']:.4f})")
            print(f"  Val   loss: {val_m['loss']:.4f}", end='')
            for k, v in val_m.items():
                if k != 'loss':
                    print(f"  {k}={v:.4f}", end='')
            print()

            self.scheduler.step()

            if val_m['loss'] < self.best_val_loss:
                self.best_val_loss = val_m['loss']
                self.save_model(f"best_model_{dataset_name}_seed{seed}.pt")
                print(f"  Saved best (val_loss={self.best_val_loss:.4f})")

        self.save_training_history()

    def save_model(self, filename: str):
        os.makedirs(self.config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
        path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }, path)

    def save_training_history(self):
        seed = self.config.get('seed', 42)
        os.makedirs(self.config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
        path = os.path.join(
            self.config.get('checkpoint_dir', 'checkpoints'),
            f"history_{self.config['dataset_name']}_seed{seed}.json",
        )
        with open(path, 'w') as f:
            json.dump({'train_losses': self.train_losses, 'val_losses': self.val_losses}, f, indent=2)


if __name__ == '__main__':
    config = {
        'dataset_name': 'BACE',
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

    trainer = Trainer(config)
    trainer.train()
    test_metrics = trainer.evaluate_test()
    print(f"\nTest: {test_metrics}")
