"""
Integration with OpenAI's text-embedding-3-large model for SMILES encoding.
"""

import torch
import torch.nn as nn
from openai import OpenAI
import numpy as np
from typing import List, Union


class LLMEncoder:
    """
    Wrapper for OpenAI's text-embedding-3-large model.
    Note: This requires an OpenAI API key. For local testing, we provide a fallback.
    """
    
    def __init__(self, api_key=None, use_local_fallback=True):
        self.api_key = api_key
        self.use_local_fallback = use_local_fallback
        self.embedding_dim = 3072
       
        self.client = OpenAI(api_key=api_key)
    
    
    def encode(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Encode SMILES strings to embeddings.
        
        Args:
            smiles_list: List of SMILES strings
        Returns:
            embeddings: Tensor of shape [batch_size, 3072]
        """
        if self.api_key and not self.use_local_fallback:
            # Use OpenAI API
            embeddings = []
            for smiles in smiles_list:
                response = self.client.embeddings.create(
                    model="text-embedding-3-large",
                    input=smiles
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            return torch.tensor(embeddings, dtype=torch.float32)
        
        elif hasattr(self, 'local_mode') and self.local_mode:
            # Use local transformer model
            embeddings = self.model.encode(smiles_list, convert_to_tensor=True)
            embeddings = self.projection(embeddings.to(torch.float32))
            return embeddings
        
        else:
            # Fallback: random embeddings (for testing without API)
            batch_size = len(smiles_list)
            return torch.randn(batch_size, self.embedding_dim, dtype=torch.float32)


class FrozenLLMEncoder(nn.Module):
    """
    PyTorch module wrapper for frozen LLM encoder.
    """
    
    def __init__(self, llm_encoder: LLMEncoder):
        super(FrozenLLMEncoder, self).__init__()
        self.llm_encoder = llm_encoder
        # Freeze the encoder
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Args:
            smiles_list: List of SMILES strings
        Returns:
            embeddings: Tensor of shape [batch_size, 3072]
        """
        with torch.no_grad():
            embeddings = self.llm_encoder.encode(smiles_list)
        return embeddings

