"""
Training script for dual-branch molecular property prediction model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error

from models.cross_attention import DualBranchModel
from models.llm_encoder import LLMEncoder, FrozenLLMEncoder
from data.dataset import MolecularDataset, collate_fn, get_dataset
from training.losses import ContrastiveLoss, ClassificationLoss, RegressionLoss, MultiLabelLoss


class Trainer:
    """
    Trainer class for dual-branch molecular property prediction.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load dataset
        self.dataset = get_dataset(
            config['dataset_name'],
            data_dir=config['data_dir'],
            task_type=config.get('task_type', 'classification')
        )
        
        # Split dataset
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config.get('seed', 42))
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 0)
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 0)
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 0)
        )
        
        # Determine number of classes
        if self.dataset.task_type == 'regression':
            num_classes = 1
        elif self.dataset.task_type == 'multilabel':
            num_classes = len(self.dataset.label_cols)
        else:
            num_classes = len(torch.unique(torch.tensor(self.dataset.valid_labels)))
        
        # Initialize LLM encoder
        self.llm_encoder = LLMEncoder(
            api_key=config.get('openai_api_key'),
            use_local_fallback=config.get('use_local_fallback', True)
        )
        self.frozen_llm = FrozenLLMEncoder(self.llm_encoder)
        
        # Initialize model
        self.model = DualBranchModel(
            atom_dim=config.get('atom_dim', 41),
            bond_dim=config.get('bond_dim', 10),
            graph_dim=config.get('graph_dim', 512),
            smiles_dim=config.get('smiles_dim', 3072),
            hidden_dim=config.get('hidden_dim', 512),
            num_classes=num_classes,
            num_layers=config.get('num_layers', 3),
            num_heads=config.get('num_heads', 8)
        ).to(self.device)
        
        # Loss functions
        self.contrastive_loss = ContrastiveLoss(temperature=config.get('temperature', 0.07))
        self.lambda_contrastive = config.get('lambda_contrastive', 0.1)
        
        # Projection layer for SMILES embeddings to graph dimension (for contrastive loss)
        self.smiles_projection = nn.Linear(
            config.get('smiles_dim', 3072),
            config.get('graph_dim', 512)
        ).to(self.device)
        
        if self.dataset.task_type == 'regression':
            self.task_loss = RegressionLoss()
        elif self.dataset.task_type == 'multilabel':
            self.task_loss = MultiLabelLoss()
        else:
            self.task_loss = ClassificationLoss()
        
        # Optimizer (include projection layer parameters)
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.smiles_projection.parameters()),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_task_loss = 0
        total_contrastive_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            graphs = batch['graph'].to(self.device)
            smiles_list = batch['smiles']
            labels = batch['label'].to(self.device)
            
            # Encode SMILES with frozen LLM
            with torch.no_grad():
                smiles_embeddings = self.frozen_llm(smiles_list).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, h_g, h_joint = self.model(graphs, smiles_embeddings)
            
            # Compute losses
            # Task loss
            if self.dataset.task_type == 'regression':
                if logits.dim() > 1:
                    logits = logits.squeeze(1)
                task_loss = self.task_loss(logits, labels.float())
            else:
                task_loss = self.task_loss(logits, labels)
            
            # Contrastive loss
            # Project h_s to match h_g dimensions for contrastive loss
            h_s_proj = self.smiles_projection(smiles_embeddings)
            contrastive_loss = self.contrastive_loss(h_g, h_s_proj)
            
            # Total loss
            total_loss_batch = task_loss + self.lambda_contrastive * contrastive_loss
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_task_loss += task_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            
            pbar.set_postfix({
                'loss': total_loss_batch.item(),
                'task': task_loss.item(),
                'contrastive': contrastive_loss.item()
            })
        
        return {
            'total_loss': total_loss / len(self.train_loader),
            'task_loss': total_task_loss / len(self.train_loader),
            'contrastive_loss': total_contrastive_loss / len(self.train_loader)
        }
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                graphs = batch['graph'].to(self.device)
                smiles_list = batch['smiles']
                labels = batch['label'].to(self.device)
                
                # Encode SMILES
                smiles_embeddings = self.frozen_llm(smiles_list).to(self.device)
                
                # Forward pass
                logits, h_g, h_joint = self.model(graphs, smiles_embeddings)
                
                # Compute loss
                if self.dataset.task_type == 'regression':
                    if logits.dim() > 1:
                        logits = logits.squeeze(1)
                    loss = self.task_loss(logits, labels.float())
                    predictions = logits.cpu().numpy()
                else:
                    loss = self.task_loss(logits, labels)
                    if self.dataset.task_type == 'multilabel':
                        predictions = torch.sigmoid(logits).cpu().numpy()
                    else:
                        predictions = torch.softmax(logits, dim=1).cpu().numpy()
                
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_labels.append(labels.cpu().numpy())
        
        # Compute metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        metrics = {'loss': total_loss / len(self.val_loader)}
        
        if self.dataset.task_type == 'regression':
            metrics['mse'] = mean_squared_error(all_labels, all_predictions)
            metrics['mae'] = mean_absolute_error(all_labels, all_predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        elif self.dataset.task_type == 'multilabel':
            # Multi-label metrics
            all_predictions_binary = (all_predictions > 0.5).astype(int)
            metrics['accuracy'] = accuracy_score(all_labels, all_predictions_binary)
            try:
                metrics['roc_auc'] = roc_auc_score(all_labels, all_predictions, average='macro')
            except:
                metrics['roc_auc'] = 0.0
        else:
            # Classification metrics
            pred_classes = np.argmax(all_predictions, axis=1)
            metrics['accuracy'] = accuracy_score(all_labels, pred_classes)
            if len(np.unique(all_labels)) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(all_labels, all_predictions[:, 1])
                except:
                    metrics['roc_auc'] = 0.0
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.config['dataset_name']}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics)
            
            print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Task: {train_metrics['task_loss']:.4f}, "
                  f"Contrastive: {train_metrics['contrastive_loss']:.4f})")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            for key, value in val_metrics.items():
                if key != 'loss':
                    print(f"  {key}: {value:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_model(f"best_model_{self.config['dataset_name']}.pt")
                print(f"Saved best model (val loss: {self.best_val_loss:.4f})")
        
        # Save training history
        self.save_training_history()
    
    def save_model(self, filename):
        """Save model checkpoint."""
        os.makedirs(self.config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
        path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        os.makedirs(self.config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
        path = os.path.join(
            self.config.get('checkpoint_dir', 'checkpoints'),
            f"history_{self.config['dataset_name']}.json"
        )
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)


if __name__ == '__main__':
    # Example configuration
    config = {
        'dataset_name': 'HIV',  # Change to desired dataset
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
        'openai_api_key': None,  # Set if using OpenAI API
        'num_workers': 0,
        'seed': 42
    }
    
    trainer = Trainer(config)
    trainer.train()

