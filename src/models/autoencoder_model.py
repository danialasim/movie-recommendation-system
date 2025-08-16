"""
Stacked AutoEncoder (SAE) model for movie recommendation system.

This module implements a deep autoencoder for collaborative filtering,
learning user representations from rating patterns.

Author: Movie Recommendation System Team
Date: August 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.base_model import BaseRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StackedAutoEncoder(BaseRecommender):
    """
    Stacked AutoEncoder for collaborative filtering recommendation.
    
    Architecture:
    Input Layer (n_movies) → Hidden Layer 1 → Hidden Layer 2 (bottleneck) → 
    Hidden Layer 3 → Output Layer (n_movies)
    
    The bottleneck layer learns compact user representations.
    """
    
    def __init__(self, n_movies: int, hidden_dims: List[int] = [20, 10], 
                 dropout_rate: float = 0.2, activation: str = 'sigmoid', 
                 config: Dict[str, Any] = None):
        """
        Initialize the Stacked AutoEncoder.
        
        Args:
            n_movies: Number of movies (input/output dimension)
            hidden_dims: List of hidden layer dimensions [encoder layers]
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('sigmoid', 'relu', 'tanh')
            config: Configuration dictionary
        """
        
        if config is None:
            config = {}
        
        # Update config with model parameters
        config.update({
            'n_movies': n_movies,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
            'activation': activation,
            'model_type': 'stacked_autoencoder'
        })
        
        super(StackedAutoEncoder, self).__init__('SAE', config)
        
        self.n_movies = n_movies
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_mask = True  # Use mask for loss computation on rated items only
        
        # Build the model
        self.build_model()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"SAE initialized: {n_movies} → {' → '.join(map(str, hidden_dims))} → {n_movies}")
    
    def _get_activation(self):
        """Get activation function."""
        if self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def build_model(self) -> None:
        """Build the autoencoder architecture."""
        
        # Encoder layers
        encoder_layers = []
        input_dim = self.n_movies
        
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(self.dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Remove last dropout for bottleneck
        if encoder_layers:
            encoder_layers = encoder_layers[:-1]
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (symmetric to encoder)
        decoder_layers = []
        reversed_dims = list(reversed(self.hidden_dims[:-1])) + [self.n_movies]
        input_dim = self.hidden_dims[-1]  # Bottleneck dimension
        
        for i, hidden_dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Use sigmoid for output layer to keep ratings in [0,1] range
            if i == len(reversed_dims) - 1:  # Output layer
                decoder_layers.append(nn.Sigmoid())
            else:
                decoder_layers.extend([
                    self._get_activation(),
                    nn.Dropout(self.dropout_rate)
                ])
            
            input_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        logger.info(f"SAE architecture built with {self.count_parameters()} parameters")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            # Xavier initialization for better convergence
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, n_movies)
            
        Returns:
            torch.Tensor: Reconstructed ratings of shape (batch_size, n_movies)
        """
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoded (compressed) representation of user preferences.
        
        Args:
            x: Input tensor of shape (batch_size, n_movies)
            
        Returns:
            torch.Tensor: Encoded representation (bottleneck layer)
        """
        with torch.no_grad():
            x = x.to(self.device)
            encoded = self.encoder(x)
            return encoded.cpu()
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute reconstruction loss for autoencoder.
        
        Args:
            predictions: Predicted ratings
            targets: True ratings
            mask: Mask for rated items (optional)
            
        Returns:
            torch.Tensor: MSE loss (only on rated items if mask provided)
        """
        if mask is not None:
            # Only compute loss on rated items
            loss = F.mse_loss(predictions * mask, targets * mask, reduction='sum')
            # Normalize by number of rated items
            num_rated = mask.sum()
            if num_rated > 0:
                loss = loss / num_rated
            else:
                loss = torch.tensor(0.0, device=predictions.device)
        else:
            loss = F.mse_loss(predictions, targets)
        
        return loss
    
    def train_step(self, user_ratings: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Single training step for one user.
        
        Args:
            user_ratings: User rating vector
            optimizer: Optimizer
            
        Returns:
            float: Loss value
        """
        self.train()
        
        # Create mask for rated items
        mask = (user_ratings > 0).float()
        
        # Forward pass
        optimizer.zero_grad()
        predictions = self.forward(user_ratings.unsqueeze(0))
        
        # Compute loss only on rated items
        loss = self.compute_loss(predictions.squeeze(), user_ratings, mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def predict_for_user(self, user_ratings: torch.Tensor, 
                        return_all: bool = False) -> torch.Tensor:
        """
        Predict ratings for a single user.
        
        Args:
            user_ratings: User's rating vector
            return_all: Whether to return all predictions or only for unrated items
            
        Returns:
            torch.Tensor: Predicted ratings
        """
        self.eval()
        with torch.no_grad():
            user_ratings = user_ratings.to(self.device)
            predictions = self.forward(user_ratings.unsqueeze(0)).squeeze()
            
            if not return_all:
                # Only return predictions for unrated items
                unrated_mask = user_ratings == 0
                predictions = predictions * unrated_mask.float()
            
            return predictions.cpu()
    
    def get_user_similarity(self, user_ratings1: torch.Tensor, 
                           user_ratings2: torch.Tensor) -> float:
        """
        Compute similarity between two users using their encoded representations.
        
        Args:
            user_ratings1: First user's rating vector
            user_ratings2: Second user's rating vector
            
        Returns:
            float: Cosine similarity between encoded representations
        """
        # Get encoded representations
        encoded1 = self.encode(user_ratings1.unsqueeze(0)).squeeze()
        encoded2 = self.encode(user_ratings2.unsqueeze(0)).squeeze()
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(encoded1, encoded2, dim=0)
        return similarity.item()
    
    def get_top_similar_users(self, target_user_ratings: torch.Tensor, 
                             all_user_ratings: torch.Tensor, k: int = 10) -> List[Tuple[int, float]]:
        """
        Find most similar users to target user.
        
        Args:
            target_user_ratings: Target user's rating vector
            all_user_ratings: All users' rating matrix
            k: Number of similar users to return
            
        Returns:
            List[Tuple[int, float]]: List of (user_idx, similarity_score)
        """
        target_encoded = self.encode(target_user_ratings.unsqueeze(0)).squeeze()
        all_encoded = self.encode(all_user_ratings)
        
        # Compute similarities
        similarities = F.cosine_similarity(target_encoded.unsqueeze(0), all_encoded, dim=1)
        
        # Get top-k similar users
        _, top_indices = torch.topk(similarities, k + 1)  # +1 to exclude self
        
        # Exclude the target user itself (if present)
        similar_users = []
        for idx in top_indices:
            if not torch.equal(all_user_ratings[idx], target_user_ratings):
                similar_users.append((idx.item(), similarities[idx].item()))
                if len(similar_users) == k:
                    break
        
        return similar_users
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def analyze_reconstruction_quality(self, user_ratings: torch.Tensor) -> Dict[str, float]:
        """
        Analyze reconstruction quality for a user.
        
        Args:
            user_ratings: User's rating vector
            
        Returns:
            Dict[str, float]: Reconstruction quality metrics
        """
        predictions = self.predict_for_user(user_ratings, return_all=True)
        mask = user_ratings > 0
        
        if mask.sum() == 0:
            return {'rmse': float('inf'), 'mae': float('inf'), 'num_ratings': 0}
        
        # Denormalize if needed (assuming ratings were normalized to [0,1])
        true_ratings = user_ratings[mask] * 4 + 1  # Convert [0,1] back to [1,5]
        pred_ratings = predictions[mask] * 4 + 1
        
        # Compute metrics
        rmse = torch.sqrt(F.mse_loss(pred_ratings, true_ratings)).item()
        mae = F.l1_loss(pred_ratings, true_ratings).item()
        
        return {
            'rmse': rmse,
            'mae': mae,
            'num_ratings': mask.sum().item(),
            'mean_error': (pred_ratings - true_ratings).mean().item()
        }
    
    def save_embeddings(self, user_ratings_matrix: torch.Tensor, 
                       save_path: str) -> None:
        """
        Save user embeddings (encoded representations) to file.
        
        Args:
            user_ratings_matrix: Complete user-item matrix
            save_path: Path to save embeddings
        """
        embeddings = self.encode(user_ratings_matrix)
        torch.save({
            'embeddings': embeddings,
            'model_config': self.config,
            'embedding_dim': self.hidden_dims[-1]
        }, save_path)
        
        logger.info(f"User embeddings saved to {save_path}")
        logger.info(f"Embedding shape: {embeddings.shape}")


def create_sae_data_loader(user_item_matrix: torch.Tensor, batch_size: int = 128, 
                          shuffle: bool = True, num_workers: int = 0) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for SAE training.
    
    Args:
        user_item_matrix: User-item rating matrix
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for training
    """
    dataset = torch.utils.data.TensorDataset(user_item_matrix)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return data_loader


def normalize_ratings_for_sae(ratings_matrix: torch.Tensor, 
                             method: str = 'global') -> torch.Tensor:
    """
    Normalize ratings for SAE training.
    
    Args:
        ratings_matrix: Raw rating matrix (typically 1-5 scale)
        method: Normalization method ('global', 'user', 'none')
        
    Returns:
        torch.Tensor: Normalized rating matrix (0-1 scale)
    """
    if method == 'global':
        # Global normalization: (rating - min) / (max - min)
        # Assumes ratings are 1-5, maps to 0-1
        normalized = (ratings_matrix - 1) / 4
        # Keep unrated items as 0
        normalized = normalized * (ratings_matrix > 0).float()
        
    elif method == 'user':
        # Per-user normalization
        normalized = torch.zeros_like(ratings_matrix)
        for user_idx in range(ratings_matrix.shape[0]):
            user_ratings = ratings_matrix[user_idx]
            rated_mask = user_ratings > 0
            
            if rated_mask.sum() > 0:
                min_rating = user_ratings[rated_mask].min()
                max_rating = user_ratings[rated_mask].max()
                
                if max_rating > min_rating:
                    norm_ratings = (user_ratings - min_rating) / (max_rating - min_rating)
                    normalized[user_idx] = norm_ratings * rated_mask.float()
                else:
                    # If user has same rating for all items, set to 0.5
                    normalized[user_idx] = 0.5 * rated_mask.float()
    
    elif method == 'none':
        normalized = ratings_matrix.clone()
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized
