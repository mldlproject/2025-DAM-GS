"""
Cross-Attention module for fusing graph and SMILES representations.
"""

import torch
import torch.nn as nn
import math
from models.dagt import DAGT



class CrossAttention(nn.Module):
    """
    Cross-Attention module to fuse heterogeneous features from graph and SMILES modalities.
    Uses h_g as query and h_s as key and value.
    """
    
    def __init__(self, graph_dim=512, smiles_dim=3072, hidden_dim=512, num_heads=8):
        super(CrossAttention, self).__init__()
        self.graph_dim = graph_dim
        self.smiles_dim = smiles_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        
        # Projection matrices for each head
        self.W_Q = nn.ModuleList([
            nn.Linear(graph_dim, self.d_k) for _ in range(num_heads)
        ])
        self.W_K = nn.ModuleList([
            nn.Linear(smiles_dim, self.d_k) for _ in range(num_heads)
        ])
        self.W_V = nn.ModuleList([
            nn.Linear(smiles_dim, self.d_k) for _ in range(num_heads)
        ])
        
        # Output projection
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP for joint representation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, h_g, h_s):
        """
        Args:
            h_g: Graph representation [batch_size, graph_dim]
            h_s: SMILES representation [batch_size, smiles_dim]
        Returns:
            h_joint: Joint representation [batch_size, hidden_dim]
        """
        batch_size = h_g.size(0)
        
        # Multi-head cross-attention
        attn_outputs = []
        for i in range(self.num_heads):
            Q = self.W_Q[i](h_g)  # [batch_size, d_k]
            K = self.W_K[i](h_s)  # [batch_size, d_k]
            V = self.W_V[i](h_s)  # [batch_size, d_k]
            
            # Scaled dot-product attention
            scores = torch.matmul(Q.unsqueeze(1), K.unsqueeze(1).transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V.unsqueeze(1))
            attn_outputs.append(attn_output.squeeze(1))
        
        # Concatenate all heads
        h_concat = torch.cat(attn_outputs, dim=1)  # [batch_size, hidden_dim]
        
        # Apply output projection and residual connection
        h_attn = self.W_O(h_concat)
        h_attn = self.norm1(h_attn + h_g)  # Residual connection with h_g
        
        # MLP for final joint representation
        h_joint = self.mlp(h_attn)
        h_joint = self.norm2(h_joint + h_attn)  # Residual connection
        
        return h_joint


class DualBranchModel(nn.Module):
    """
    Complete dual-branch model combining DAGT and LLM with Cross-Attention.
    """
    
    def __init__(self, atom_dim=41, bond_dim=10, graph_dim=512, smiles_dim=3072, 
                 hidden_dim=512, num_classes=2, num_layers=3, num_heads=8):
        super(DualBranchModel, self).__init__()
        
        # DAGT encoder
        self.dagt = DAGT(atom_dim, bond_dim, graph_dim, num_layers)
        
        # Cross-Attention module
        self.cross_attention = CrossAttention(graph_dim, smiles_dim, hidden_dim, num_heads)
        
        # Predictor head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, graph_data, smiles_embeddings):
        """
        Args:
            graph_data: PyTorch Geometric Batch object
            smiles_embeddings: SMILES embeddings from LLM [batch_size, smiles_dim]
        Returns:
            logits: Prediction logits [batch_size, num_classes]
            h_g: Graph representation [batch_size, graph_dim]
            h_joint: Joint representation [batch_size, hidden_dim]
        """
        # Get graph representation
        h_g = self.dagt(graph_data)  # [batch_size, graph_dim]
        
        # Cross-attention fusion
        h_joint = self.cross_attention(h_g, smiles_embeddings)  # [batch_size, hidden_dim]
        
        # Prediction
        logits = self.predictor(h_joint)  # [batch_size, num_classes]
        
        return logits, h_g, h_joint

