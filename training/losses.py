"""
Loss functions for dual-branch molecular property prediction.
Includes contrastive loss and classification loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to align graph and SMILES embeddings.
    Based on InfoNCE loss.
    """
    
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, h_g, h_s):
        """
        Args:
            h_g: Graph embeddings [batch_size, dim]
            h_s: SMILES embeddings [batch_size, dim]
        Returns:
            loss: Contrastive loss value
        """
        batch_size = h_g.size(0)
        
        # Normalize embeddings
        h_g_norm = F.normalize(h_g, p=2, dim=1)
        h_s_norm = F.normalize(h_s, p=2, dim=1)
        
        # Compute cosine similarities
        # Positive pairs: diagonal elements (h_g[i] and h_s[i])
        # Negative pairs: off-diagonal elements
        similarity_matrix = torch.matmul(h_g_norm, h_s_norm.t()) / self.temperature
        
        # Create labels: diagonal is positive (1), others are negative (0)
        labels = torch.arange(batch_size, device=h_g.device)
        
        # InfoNCE loss for h_g -> h_s direction
        loss_g_to_s = F.cross_entropy(similarity_matrix, labels)
        
        # InfoNCE loss for h_s -> h_g direction
        loss_s_to_g = F.cross_entropy(similarity_matrix.t(), labels)
        
        # Average both directions
        loss = (loss_g_to_s + loss_s_to_g) / 2
        
        return loss


class ClassificationLoss(nn.Module):
    """
    Classification loss (cross-entropy) for molecular property prediction.
    """
    
    def __init__(self):
        super(ClassificationLoss, self).__init__()
    
    def forward(self, logits, labels):
        """
        Args:
            logits: Prediction logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
        Returns:
            loss: Cross-entropy loss
        """
        return F.cross_entropy(logits, labels)


class RegressionLoss(nn.Module):
    """
    Regression loss (MSE) for molecular property prediction.
    """
    
    def __init__(self):
        super(RegressionLoss, self).__init__()
    
    def forward(self, predictions, labels):
        """
        Args:
            predictions: Predictions [batch_size] or [batch_size, 1]
            labels: Ground truth values [batch_size]
        Returns:
            loss: MSE loss
        """
        if predictions.dim() > 1 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)
        return F.mse_loss(predictions, labels)


class MultiLabelLoss(nn.Module):
    """
    Multi-label classification loss (BCE with logits).
    """
    
    def __init__(self):
        super(MultiLabelLoss, self).__init__()
    
    def forward(self, logits, labels):
        """
        Args:
            logits: Prediction logits [batch_size, num_labels]
            labels: Ground truth labels [batch_size, num_labels]
        Returns:
            loss: BCE loss
        """
        return F.binary_cross_entropy_with_logits(logits, labels)

