"""
Base Model module for movie recommendation system.

This module provides an abstract base class for all recommendation models,
with MLflow integration and common functionality.

Author: Movie Recommendation System Team
Date: August 2025
"""

import os
import sys
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import numpy as np
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRecommender(ABC, nn.Module):
    """
    Abstract base class for recommendation models with MLflow integration.
    
    Provides common functionality for model training, evaluation, and tracking.
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the base recommender.
        
        Args:
            model_name: Name of the model (e.g., 'SAE', 'RBM')
            config: Configuration dictionary with model parameters
        """
        super(BaseRecommender, self).__init__()
        
        self.model_name = model_name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlflow_run = None
        
        # Model metadata
        self.n_users = config.get('n_users', 0)
        self.n_movies = config.get('n_movies', 0)
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"Initialized {model_name} on device: {self.device}")
    
    @abstractmethod
    def build_model(self) -> None:
        """Build the neural network architecture."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    mask: torch.Tensor = None) -> torch.Tensor:
        """Compute model-specific loss."""
        pass
    
    def train_epoch(self, data_loader: torch.utils.data.DataLoader, 
                   optimizer: torch.optim.Optimizer, epoch: int) -> float:
        """
        Train the model for one epoch.
        
        Args:
            data_loader: DataLoader with training data
            optimizer: Optimizer for training
            epoch: Current epoch number
            
        Returns:
            float: Average training loss for the epoch
        """
        self.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(data_loader):
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 2:
                    # Supervised learning case (inputs, targets)
                    inputs, targets = batch_data
                else:
                    # Autoencoder case with TensorDataset wrapping
                    inputs = batch_data[0]
                    targets = inputs  # For autoencoders
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                inputs = batch_data.to(self.device)
                targets = inputs  # For autoencoders
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.forward(inputs)
            
            # Compute loss with mask for rated items only
            mask = targets > 0 if hasattr(self, 'use_mask') and self.use_mask else None
            loss = self.compute_loss(predictions, targets, mask)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            data_loader: DataLoader with validation data
            
        Returns:
            float: Average validation loss
        """
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) == 2:
                        # Supervised learning case (inputs, targets)
                        inputs, targets = batch_data
                    else:
                        # Autoencoder case with TensorDataset wrapping
                        inputs = batch_data[0]
                        targets = inputs  # For autoencoders
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    inputs = batch_data.to(self.device)
                    targets = inputs
                
                predictions = self.forward(inputs)
                mask = targets > 0 if hasattr(self, 'use_mask') and self.use_mask else None
                loss = self.compute_loss(predictions, targets, mask)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def predict_ratings(self, user_data: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for given user data.
        
        Args:
            user_data: User rating vector or batch of user vectors
            
        Returns:
            torch.Tensor: Predicted ratings
        """
        self.eval()
        with torch.no_grad():
            user_data = user_data.to(self.device)
            predictions = self.forward(user_data)
            return predictions.cpu()
    
    def recommend_items(self, user_idx: int, user_item_matrix: torch.Tensor, 
                       k: int = 10, exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Recommend top-k items for a user.
        
        Args:
            user_idx: Index of the user
            user_item_matrix: Complete user-item matrix
            k: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List[Tuple[int, float]]: List of (movie_idx, predicted_rating) tuples
        """
        if user_idx >= user_item_matrix.shape[0]:
            raise ValueError(f"User index {user_idx} out of bounds")
        
        user_ratings = user_item_matrix[user_idx:user_idx+1]  # Keep batch dimension
        predicted_ratings = self.predict_ratings(user_ratings).squeeze()
        
        if exclude_rated:
            # Set predictions for already rated items to negative infinity
            rated_mask = user_item_matrix[user_idx] > 0
            predicted_ratings[rated_mask] = float('-inf')
        
        # Get top-k items
        _, top_indices = torch.topk(predicted_ratings, k)
        recommendations = [(idx.item(), predicted_ratings[idx].item()) 
                         for idx in top_indices]
        
        return recommendations
    
    def save_model(self, save_path: str, save_state_dict_only: bool = False) -> None:
        """
        Save the model to disk.
        
        Args:
            save_path: Path to save the model
            save_state_dict_only: Whether to save only state dict or full model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_state_dict_only:
            torch.save(self.state_dict(), save_path)
        else:
            # Save complete model info
            model_info = {
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'model_name': self.model_name,
                'training_history': self.training_history,
                'n_users': self.n_users,
                'n_movies': self.n_movies
            }
            torch.save(model_info, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str, load_state_dict_only: bool = False) -> None:
        """
        Load the model from disk.
        
        Args:
            load_path: Path to load the model from
            load_state_dict_only: Whether loading only state dict or full model
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        if load_state_dict_only:
            self.load_state_dict(torch.load(load_path, map_location=self.device))
        else:
            model_info = torch.load(load_path, map_location=self.device)
            self.load_state_dict(model_info['model_state_dict'])
            self.config.update(model_info.get('config', {}))
            self.training_history = model_info.get('training_history', {'train_loss': [], 'val_loss': []})
            self.n_users = model_info.get('n_users', self.n_users)
            self.n_movies = model_info.get('n_movies', self.n_movies)
        
        self.to(self.device)
        logger.info(f"Model loaded from {load_path}")
    
    def start_mlflow_run(self, experiment_name: str = "movie_recommendation") -> None:
        """Start MLflow run for experiment tracking."""
        try:
            # Try to set experiment, create if doesn't exist
            try:
                mlflow.set_experiment(experiment_name)
            except:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
            
            self.mlflow_run = mlflow.start_run(run_name=f"{self.model_name}_training")
            
            # Log model configuration
            mlflow.log_params({f"config_{k}": v for k, v in self.config.items()})
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_param("device", str(self.device))
            
            logger.info(f"Started MLflow run: {self.mlflow_run.info.run_id}")
            
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            self.mlflow_run = None
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        if self.mlflow_run:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def log_model(self, artifact_path: str = "model") -> None:
        """Log model to MLflow."""
        if self.mlflow_run:
            try:
                mlflow.pytorch.log_model(self, artifact_path)
                logger.info(f"Model logged to MLflow at {artifact_path}")
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")
    
    def end_mlflow_run(self) -> None:
        """End MLflow run."""
        if self.mlflow_run:
            try:
                mlflow.end_run()
                logger.info("MLflow run ended")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
            finally:
                self.mlflow_run = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'n_users': self.n_users,
            'n_movies': self.n_movies,
            'config': self.config
        }
    
    @staticmethod
    def save_model_with_hyperparameters(model_path: str, hyperparams: Dict[str, Any], 
                                       model_instance: 'BaseRecommender',
                                       metrics: Dict[str, Any] = None) -> None:
        """
        Save model with its best hyperparameters and configuration.
        
        Args:
            model_path: Path to save the model
            hyperparams: Best hyperparameters dictionary
            model_instance: The model instance to save
            metrics: Final evaluation metrics
        """
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_instance.save_model(model_path)
        
        # Save hyperparameters
        hyperparams_path = model_dir / 'best_hyperparameters.json'
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        # Save complete configuration
        config_data = {
            'model_type': model_instance.model_name,
            'hyperparameters': hyperparams,
            'model_info': model_instance.get_model_info(),
            'metrics': metrics or {},
            'save_timestamp': torch.save.__module__,  # Use available timestamp
            'model_file': Path(model_path).name
        }
        
        config_path = model_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Model, hyperparameters, and config saved to {model_dir}")
    
    @staticmethod
    def load_model_with_hyperparameters(model_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load hyperparameters and configuration from saved model directory.
        
        Args:
            model_dir: Directory containing saved model files
            
        Returns:
            Tuple of (hyperparameters, full_config)
        """
        model_dir = Path(model_dir)
        
        # Load hyperparameters
        hyperparams_path = model_dir / 'best_hyperparameters.json'
        hyperparams = {}
        if hyperparams_path.exists():
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
        
        # Load full configuration
        config_path = model_dir / 'model_config.json'
        full_config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                full_config = json.load(f)
        
        return hyperparams, full_config
