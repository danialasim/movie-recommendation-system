"""
Restricted Boltzmann Machine (RBM) model for movie recommendation system.

This module implements an RBM for collaborative filtering using binary
user preferences (like/dislike) with Contrastive Divergence training.

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


class RestrictedBoltzmannMachine(BaseRecommender):
    """
    Restricted Boltzmann Machine for collaborative filtering recommendation.
    
    RBM is a generative model that learns probability distributions over
    binary user preferences (like/dislike) using energy-based modeling.
    
    Architecture:
    Hidden Layer (n_hidden units) ↔ Visible Layer (n_movies units)
    
    No connections within hidden or visible layers.
    """
    
    def __init__(self, n_movies: int, n_hidden: int = 100, 
                 learning_rate: float = 0.01, cd_k: int = 10,
                 l2_penalty: float = 0.0, config: Dict[str, Any] = None):
        """
        Initialize the RBM.
        
        Args:
            n_movies: Number of movies (visible units)
            n_hidden: Number of hidden units
            learning_rate: Learning rate for CD training
            cd_k: Number of Contrastive Divergence steps
            l2_penalty: L2 regularization penalty
            config: Configuration dictionary
        """
        
        if config is None:
            config = {}
        
        # Update config with model parameters
        config.update({
            'n_movies': n_movies,
            'n_hidden': n_hidden,
            'learning_rate': learning_rate,
            'cd_k': cd_k,
            'l2_penalty': l2_penalty,
            'model_type': 'restricted_boltzmann_machine'
        })
        
        super(RestrictedBoltzmannMachine, self).__init__('RBM', config)
        
        self.n_movies = n_movies
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.cd_k = cd_k
        self.l2_penalty = l2_penalty
        self.use_mask = False  # RBM doesn't use mask-based loss
        
        # Build the model
        self.build_model()
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"RBM initialized: {n_movies} visible ↔ {n_hidden} hidden units")
    
    def build_model(self) -> None:
        """Build the RBM architecture."""
        
        # Weight matrix connecting visible and hidden units
        # Shape: (n_hidden, n_movies)
        self.W = nn.Parameter(torch.randn(self.n_hidden, self.n_movies) * 0.1)
        
        # Bias for visible units (movies)
        self.b_visible = nn.Parameter(torch.zeros(self.n_movies))
        
        # Bias for hidden units (latent factors)
        self.b_hidden = nn.Parameter(torch.zeros(self.n_hidden))
        
        logger.info(f"RBM architecture built with {self.count_parameters()} parameters")
    
    def _init_weights(self):
        """Initialize RBM weights using small random values."""
        nn.init.normal_(self.W, mean=0, std=0.1)
        nn.init.zeros_(self.b_visible)
        nn.init.zeros_(self.b_hidden)
    
    def sample_hidden(self, visible: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given visible units.
        
        Args:
            visible: Visible unit states (batch_size, n_movies)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (probabilities, samples)
        """
        # Compute P(h_j = 1 | v) = sigmoid(sum_i W_ji * v_i + b_j)
        activation = F.linear(visible, self.W, self.b_hidden)
        prob_hidden = torch.sigmoid(activation)
        
        # Sample binary hidden states
        hidden_sample = torch.bernoulli(prob_hidden)
        
        return prob_hidden, hidden_sample
    
    def sample_visible(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units.
        
        Args:
            hidden: Hidden unit states (batch_size, n_hidden)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (probabilities, samples)
        """
        # Compute P(v_i = 1 | h) = sigmoid(sum_j W_ji * h_j + b_i)
        activation = F.linear(hidden, self.W.t(), self.b_visible)
        prob_visible = torch.sigmoid(activation)
        
        # Sample binary visible states
        visible_sample = torch.bernoulli(prob_visible)
        
        return prob_visible, visible_sample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: reconstruct visible units through hidden layer.
        
        Args:
            x: Input visible units (batch_size, n_movies)
            
        Returns:
            torch.Tensor: Reconstructed visible units
        """
        # Sample hidden given visible
        prob_hidden, _ = self.sample_hidden(x)
        
        # Reconstruct visible given hidden probabilities
        prob_visible, _ = self.sample_visible(prob_hidden)
        
        return prob_visible
    
    def contrastive_divergence(self, visible_data: torch.Tensor, 
                              k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Contrastive Divergence (CD-k) training step.
        
        Args:
            visible_data: Input data (batch_size, n_movies)
            k: Number of CD steps (uses self.cd_k if None)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (reconstructed_visible, positive_phase, negative_phase)
        """
        if k is None:
            k = self.cd_k
        
        batch_size = visible_data.size(0)
        
        # Positive phase: compute statistics from data
        prob_hidden_pos, hidden_sample = self.sample_hidden(visible_data)
        
        # Positive statistics
        positive_visible_activity = visible_data.mean(0)
        positive_hidden_activity = prob_hidden_pos.mean(0)
        positive_associations = torch.mm(prob_hidden_pos.t(), visible_data) / batch_size
        
        # Negative phase: run Gibbs sampling for k steps
        visible_sample = visible_data
        
        for step in range(k):
            # Sample hidden given current visible
            prob_hidden_neg, hidden_sample = self.sample_hidden(visible_sample)
            
            # Sample visible given hidden
            prob_visible_neg, visible_sample = self.sample_visible(hidden_sample)
            
            # For last step, use probabilities instead of samples
            if step == k - 1:
                final_visible = prob_visible_neg
                final_hidden = prob_hidden_neg
        
        # Negative statistics
        negative_visible_activity = final_visible.mean(0)
        negative_hidden_activity = final_hidden.mean(0)
        negative_associations = torch.mm(final_hidden.t(), final_visible) / batch_size
        
        # Return reconstructed data and phase statistics
        return (final_visible, 
                (positive_associations, positive_visible_activity, positive_hidden_activity),
                (negative_associations, negative_visible_activity, negative_hidden_activity))
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute RBM reconstruction loss (binary cross-entropy).
        
        Args:
            predictions: Predicted probabilities
            targets: Target binary values
            mask: Not used for RBM
            
        Returns:
            torch.Tensor: Binary cross-entropy loss
        """
        # Use binary cross-entropy for binary data
        # Handle the three states: like (1), dislike (0), unknown (ignore)
        
        # Create mask for known preferences (exclude unknown/unrated)
        known_mask = (targets != 0.5) & (targets >= 0)  # Exclude unknown ratings
        
        if known_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # Convert targets to binary: like=1, dislike=0
        binary_targets = (targets > 0.5).float()
        
        # Compute loss only on known preferences
        loss = F.binary_cross_entropy(predictions[known_mask], 
                                    binary_targets[known_mask],
                                    reduction='mean')
        
        return loss
    
    def train_step(self, visible_data: torch.Tensor, 
                  optimizer: torch.optim.Optimizer) -> float:
        """
        Single training step using Contrastive Divergence.
        
        Args:
            visible_data: Batch of visible data
            optimizer: Optimizer (not used, RBM uses custom updates)
            
        Returns:
            float: Reconstruction error
        """
        self.train()
        
        # Perform contrastive divergence
        reconstructed, pos_stats, neg_stats = self.contrastive_divergence(visible_data)
        
        # Unpack statistics
        pos_assoc, pos_vis_act, pos_hid_act = pos_stats
        neg_assoc, neg_vis_act, neg_hid_act = neg_stats
        
        # Update weights and biases (custom RBM update rule)
        with torch.no_grad():
            # Weight update: positive phase - negative phase + L2 regularization
            dW = self.learning_rate * (pos_assoc - neg_assoc)
            if self.l2_penalty > 0:
                dW -= self.learning_rate * self.l2_penalty * self.W.data
            self.W.data += dW
            
            # Visible bias update
            db_visible = self.learning_rate * (pos_vis_act - neg_vis_act)
            self.b_visible.data += db_visible
            
            # Hidden bias update
            db_hidden = self.learning_rate * (pos_hid_act - neg_hid_act)
            self.b_hidden.data += db_hidden
        
        # Compute reconstruction error
        error = F.binary_cross_entropy(reconstructed, visible_data, reduction='mean')
        
        return error.item()
    
    def free_energy(self, visible: torch.Tensor) -> torch.Tensor:
        """
        Compute free energy of visible configuration.
        
        Lower free energy indicates higher probability.
        
        Args:
            visible: Visible unit configuration
            
        Returns:
            torch.Tensor: Free energy values
        """
        # F(v) = -sum_i b_i * v_i - sum_j log(1 + exp(sum_i W_ji * v_i + b_j))
        
        visible_term = torch.mv(visible, self.b_visible)
        
        hidden_activation = F.linear(visible, self.W, self.b_hidden)
        hidden_term = torch.sum(torch.log(1 + torch.exp(hidden_activation)), dim=1)
        
        free_energy = -visible_term - hidden_term
        
        return free_energy
    
    def predict_ratings(self, user_data: torch.Tensor, 
                       num_samples: int = 100) -> torch.Tensor:
        """
        Predict ratings using Gibbs sampling.
        
        Args:
            user_data: User preference data
            num_samples: Number of Gibbs samples
            
        Returns:
            torch.Tensor: Predicted probabilities
        """
        self.eval()
        
        with torch.no_grad():
            user_data = user_data.to(self.device)
            
            # Initialize with input data
            visible = user_data.clone()
            
            # Run Gibbs sampling
            for _ in range(num_samples):
                prob_hidden, hidden = self.sample_hidden(visible)
                prob_visible, visible = self.sample_visible(hidden)
            
            # Return final probabilities
            prob_hidden_final, _ = self.sample_hidden(visible)
            prob_visible_final, _ = self.sample_visible(prob_hidden_final)
            
            return prob_visible_final.cpu()
    
    def generate_samples(self, num_samples: int = 10, 
                        gibbs_steps: int = 1000) -> torch.Tensor:
        """
        Generate samples from the learned distribution.
        
        Args:
            num_samples: Number of samples to generate
            gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            torch.Tensor: Generated samples
        """
        self.eval()
        
        with torch.no_grad():
            # Initialize with random binary states
            visible = torch.randint(0, 2, (num_samples, self.n_movies), 
                                  dtype=torch.float32, device=self.device)
            
            # Run Gibbs sampling
            for _ in range(gibbs_steps):
                prob_hidden, hidden = self.sample_hidden(visible)
                prob_visible, visible = self.sample_visible(hidden)
            
            return visible.cpu()
    
    def get_hidden_representations(self, visible_data: torch.Tensor) -> torch.Tensor:
        """
        Get hidden layer representations for visible data.
        
        Args:
            visible_data: Input visible data
            
        Returns:
            torch.Tensor: Hidden layer activations
        """
        self.eval()
        with torch.no_grad():
            visible_data = visible_data.to(self.device)
            prob_hidden, _ = self.sample_hidden(visible_data)
            return prob_hidden.cpu()
    
    def compute_similarity(self, user1_data: torch.Tensor, 
                          user2_data: torch.Tensor) -> float:
        """
        Compute similarity between two users using hidden representations.
        
        Args:
            user1_data: First user's preference data
            user2_data: Second user's preference data
            
        Returns:
            float: Cosine similarity
        """
        hidden1 = self.get_hidden_representations(user1_data.unsqueeze(0)).squeeze()
        hidden2 = self.get_hidden_representations(user2_data.unsqueeze(0)).squeeze()
        
        similarity = F.cosine_similarity(hidden1, hidden2, dim=0)
        return similarity.item()
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_weights_info(self) -> Dict[str, Any]:
        """Get information about learned weights."""
        return {
            'weight_matrix_shape': self.W.shape,
            'weight_mean': self.W.data.mean().item(),
            'weight_std': self.W.data.std().item(),
            'visible_bias_mean': self.b_visible.data.mean().item(),
            'hidden_bias_mean': self.b_hidden.data.mean().item(),
            'weight_norm': torch.norm(self.W.data).item()
        }


def convert_ratings_to_binary(ratings_matrix: torch.Tensor, 
                             threshold: float = 3.0,
                             like_value: float = 1.0,
                             dislike_value: float = 0.0,
                             unknown_value: float = 0.5) -> torch.Tensor:
    """
    Convert rating matrix to binary format for RBM.
    
    Args:
        ratings_matrix: Original rating matrix (typically 1-5 scale)
        threshold: Rating threshold for like/dislike
        like_value: Value for liked items
        dislike_value: Value for disliked items
        unknown_value: Value for unknown/unrated items
        
    Returns:
        torch.Tensor: Binary rating matrix
    """
    binary_matrix = torch.full_like(ratings_matrix, unknown_value)
    
    # Items rated >= threshold are liked
    liked_mask = ratings_matrix >= threshold
    binary_matrix[liked_mask] = like_value
    
    # Items rated < threshold (but > 0) are disliked
    disliked_mask = (ratings_matrix > 0) & (ratings_matrix < threshold)
    binary_matrix[disliked_mask] = dislike_value
    
    # Unrated items remain as unknown_value
    
    return binary_matrix


def create_rbm_data_loader(binary_matrix: torch.Tensor, batch_size: int = 100,
                          shuffle: bool = True, num_workers: int = 0) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for RBM training.
    
    Args:
        binary_matrix: Binary user-item matrix
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for training
    """
    # Filter out users with no preferences (all unknown)
    valid_users = []
    for i in range(binary_matrix.shape[0]):
        if (binary_matrix[i] != 0.5).any():  # Has at least one known preference
            valid_users.append(binary_matrix[i])
    
    if not valid_users:
        raise ValueError("No users with known preferences found")
    
    valid_matrix = torch.stack(valid_users)
    dataset = torch.utils.data.TensorDataset(valid_matrix)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    
    return data_loader


def evaluate_rbm_reconstruction(rbm: RestrictedBoltzmannMachine,
                               test_data: torch.Tensor,
                               num_samples: int = 10) -> Dict[str, float]:
    """
    Evaluate RBM reconstruction quality.
    
    Args:
        rbm: Trained RBM model
        test_data: Test data for evaluation
        num_samples: Number of reconstruction samples
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    rbm.eval()
    total_error = 0.0
    total_free_energy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(test_data), 100):  # Process in batches
            batch = test_data[i:i+100]
            
            # Compute reconstruction error
            reconstructed = rbm.forward(batch)
            error = F.binary_cross_entropy(reconstructed, batch, reduction='mean')
            total_error += error.item()
            
            # Compute free energy
            free_energy = rbm.free_energy(batch).mean()
            total_free_energy += free_energy.item()
            
            num_batches += 1
    
    return {
        'reconstruction_error': total_error / num_batches,
        'average_free_energy': total_free_energy / num_batches,
        'num_samples': len(test_data)
    }
