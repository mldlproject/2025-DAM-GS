"""
Dual-Attention Graph Transformer (DAGT) for molecular graph representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import math


class BondAttentionBlock(nn.Module):
    """Bond-level attention block (τ_bond)."""
    
    def __init__(self, hidden_dim, num_heads=8):
        super(BondAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input features [num_edges, hidden_dim] or [1, num_edges, hidden_dim]
        Returns:
            output: Attention output + residual
        """
        was_2d = x.dim() == 2
        if was_2d:
            x = x.unsqueeze(0)
        
        x_norm = self.norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        output = attn_out + x  # Residual connection
        return output.squeeze(0) if was_2d else output


class BondMessagePassing(nn.Module):
    """Message passing phase for bond-level attention."""
    
    def __init__(self, bond_dim, hidden_dim, num_layers):
        super(BondMessagePassing, self).__init__()
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bond embedding layers (Equation 1)
        self.bond_embedding = nn.Sequential(
            nn.Linear(bond_dim, hidden_dim),
            nn.GELU()
        )
        
        # Bond attention blocks for each layer
        self.bond_attention_blocks = nn.ModuleList([
            BondAttentionBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Weight matrices for message passing
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        
        # Linear layer for updating bond features (Equation 4)
        self.bond_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for _ in range(num_layers)
        ])
        
    def forward(self, edge_index, bond_features):
        """
        Args:
            edge_index: Edge connectivity matrix [2, num_edges]
            bond_features: Bond feature matrix [num_edges, bond_dim]
        Returns:
            h_ij: Bond embeddings [num_edges, hidden_dim]
        """
        num_nodes = edge_index.max().item() + 1
        num_edges = edge_index.size(1)
        
        # Initial bond embeddings (Equation 1): h_ij^(0) = σ(W_h * e_ij + b_h)
        h_ij = self.bond_embedding(bond_features)  # [num_edges, hidden_dim]
        h_ij = self.W_h(h_ij)
        
        # Message passing iterations
        for t in range(self.num_layers):
            # Compute aggregated bond features r_ij(t) (Equation 2)
            r_ij = self.compute_bond_aggregation(edge_index, h_ij, num_nodes)
            
            # Bond-level attention (Equation 3): t_ij(t) = τ_bond(r_ij(t)) + r_ij(t)
            t_ij = self.bond_attention_blocks[t](r_ij) + r_ij
            
            # Update bond features (Equation 4): h_ij(t) = σ(W_t * t_ij(t) + b_t)
            h_ij = self.bond_update[t](t_ij)
        
        return h_ij
    
    def compute_bond_aggregation(self, edge_index, h_ij, num_nodes):
        """
        Compute bond aggregation r_ij(t) = Σ_{u∈N(i)\j} [h_ui^(t-1) - h_ji^(t-1)]
        Equation 2 from paper.
        Optimized version using tensor operations.
        """
        num_edges = edge_index.size(1)
        r_ij = torch.zeros(num_edges, h_ij.size(1), device=h_ij.device)
        
        # Build node-to-edges mapping for efficiency
        node_to_edges = {}
        for e_idx in range(num_edges):
            target_node = edge_index[1, e_idx].item()
            if target_node not in node_to_edges:
                node_to_edges[target_node] = []
            node_to_edges[target_node].append(e_idx)
        
        # For each edge (i,j), aggregate from neighboring bonds
        for e_idx in range(num_edges):
            j = edge_index[0, e_idx].item()  # source node
            i = edge_index[1, e_idx].item()  # target node
            
            # Get all edges incident to node i
            if i in node_to_edges:
                edges_to_i = node_to_edges[i]
                
                # Compute aggregation: Σ [h_ui^(t-1) - h_ji^(t-1)]
                for u_edge_idx in edges_to_i:
                    u = edge_index[0, u_edge_idx].item()
                    if u != j:  # Exclude j (u ∈ N(i)\j)
                        r_ij[e_idx] += h_ij[u_edge_idx] - h_ij[e_idx]
        
        return r_ij


class AtomAttentionBlock(nn.Module):
    """Atom-level attention block (τ_atom)."""
    
    def __init__(self, hidden_dim, num_heads=8):
        super(AtomAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input features [num_nodes, hidden_dim] or [1, num_nodes, hidden_dim]
        Returns:
            output: Attention output + residual
        """
        was_2d = x.dim() == 2
        if was_2d:
            x = x.unsqueeze(0)
        
        x_norm = self.norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        output = attn_out + x  # Residual connection
        return output.squeeze(0) if was_2d else output


class AtomAttention(nn.Module):
    """Atom-level attention block."""
    
    def __init__(self, atom_dim, hidden_dim, num_heads=8):
        super(AtomAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Atom embedding (Equation 6): a_i = σ(W_m * m_i + b_m)
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Projection for concatenated features
        self.feature_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Atom-level attention block (Equation 8)
        self.atom_attention = AtomAttentionBlock(hidden_dim, num_heads)
        
    def forward(self, atom_features, bond_aggregated):
        """
        Args:
            atom_features: [num_nodes, atom_dim]
            bond_aggregated: [num_nodes, hidden_dim] (h_i^(T) from Equation 5)
        """
        # Atom linear embeddings (Equation 6)
        a_i = self.atom_embedding(atom_features)  # [num_nodes, hidden_dim]
        
        # Concatenate atom embeddings with bond-aggregated features (Equation 7)
        # x_i = [a_i || h_i^(T)]
        x_i = torch.cat([a_i, bond_aggregated], dim=1)  # [num_nodes, 2*hidden_dim]
        x_i = self.feature_projection(x_i)  # [num_nodes, hidden_dim]
        
        # Apply atom-level attention (Equation 8): h_i = τ_atom(x_i) + x_i
        h_i = self.atom_attention(x_i) + x_i  # Residual connection
        
        return h_i


class DAGT(nn.Module):
    """
    Dual-Attention Graph Transformer for molecular representation learning.
    """
    
    def __init__(self, atom_dim=41, bond_dim=10, hidden_dim=512, num_layers=3):
        super(DAGT, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Bond-level message passing
        self.bond_mp = BondMessagePassing(bond_dim, hidden_dim, num_layers)
        
        # Atom-level attention
        self.atom_attention = AtomAttention(atom_dim, hidden_dim)
        
        # Graph-level pooling
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Batch object with:
                - x: atom features [num_nodes, atom_dim]
                - edge_index: edge connectivity [2, num_edges]
                - edge_attr: bond features [num_edges, bond_dim]
                - batch: batch assignment [num_nodes]
        Returns:
            h_g: graph representation [batch_size, hidden_dim]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Step 1: Bond-level message passing
        h_ij = self.bond_mp(edge_index, edge_attr)  # [num_edges, hidden_dim]
        
        # Step 2: Aggregate bond embeddings per atom
        num_nodes = x.size(0)
        h_i_bond = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        h_i_bond.scatter_add_(0, edge_index[1].unsqueeze(1).expand(-1, self.hidden_dim), h_ij)
        
        # Step 3: Atom-level attention
        h_i = self.atom_attention(x, h_i_bond)  # [num_nodes, hidden_dim]
        
        # Step 4: Graph-level pooling
        # Use mean pooling over atoms in each graph
        batch_size = batch.max().item() + 1
        graph_embeddings = []
        for i in range(batch_size):
            mask = (batch == i)
            graph_emb = h_i[mask].mean(dim=0)  # Mean pooling
            graph_embeddings.append(graph_emb)
        
        h_g = torch.stack(graph_embeddings)  # [batch_size, hidden_dim]
        h_g = self.graph_pooling(h_g)  # [batch_size, hidden_dim]
        
        return h_g

