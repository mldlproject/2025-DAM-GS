"""
Inference script for making predictions on new molecules.
"""

import torch
from models.cross_attention import DualBranchModel
from models.llm_encoder import LLMEncoder, FrozenLLMEncoder
from data.dataset import smiles_to_graph
import json
import os


class Predictor:
    """
    Predictor class for making predictions on new molecules.
    """
    
    def __init__(self, checkpoint_path, config_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if config_path is None and 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            if config_path:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                raise ValueError("Config not found in checkpoint and no config_path provided")
        
        # Determine number of classes
        num_classes = self.config.get('num_classes', 2)
        
        # Initialize LLM encoder
        self.llm_encoder = LLMEncoder(
            api_key=self.config.get('openai_api_key'),
            use_local_fallback=self.config.get('use_local_fallback', True)
        )
        self.frozen_llm = FrozenLLMEncoder(self.llm_encoder)
        
        # Initialize model
        self.model = DualBranchModel(
            atom_dim=self.config.get('atom_dim', 41),
            bond_dim=self.config.get('bond_dim', 10),
            graph_dim=self.config.get('graph_dim', 512),
            smiles_dim=self.config.get('smiles_dim', 3072),
            hidden_dim=self.config.get('hidden_dim', 512),
            num_classes=num_classes,
            num_layers=self.config.get('num_layers', 3),
            num_heads=self.config.get('num_heads', 8)
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.task_type = self.config.get('task_type', 'classification')
    
    def predict(self, smiles_list):
        """
        Make predictions for a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings or single SMILES string
        Returns:
            predictions: Predictions for each molecule
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Convert SMILES to graphs
        graphs = []
        valid_smiles = []
        for smiles in smiles_list:
            graph = smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
                valid_smiles.append(smiles)
        
        if len(graphs) == 0:
            return None
        
        # Batch graphs
        from torch_geometric.data import Batch
        batch_graph = Batch.from_data_list(graphs)
        batch_graph = batch_graph.to(self.device)
        
        # Encode SMILES
        smiles_embeddings = self.frozen_llm(valid_smiles).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits, h_g, h_joint = self.model(batch_graph, smiles_embeddings)
            
            if self.task_type == 'regression':
                if logits.dim() > 1:
                    predictions = logits.squeeze(1).cpu().numpy()
                else:
                    predictions = logits.cpu().numpy()
            elif self.task_type == 'multilabel':
                predictions = torch.sigmoid(logits).cpu().numpy()
            else:
                predictions = torch.softmax(logits, dim=1).cpu().numpy()
        
        return predictions, valid_smiles
    
    def predict_proba(self, smiles_list):
        """Get probability predictions."""
        predictions, valid_smiles = self.predict(smiles_list)
        return predictions, valid_smiles


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions on new molecules')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--smiles', type=str, nargs='+', required=True,
                        help='SMILES strings to predict')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional if in checkpoint)')
    
    args = parser.parse_args()
    
    predictor = Predictor(args.checkpoint, args.config)
    predictions, valid_smiles = predictor.predict(args.smiles)
    
    print("\n=== Predictions ===")
    for smiles, pred in zip(valid_smiles, predictions):
        print(f"{smiles}: {pred}")

