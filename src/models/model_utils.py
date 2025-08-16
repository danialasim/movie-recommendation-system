"""
Model Utilities for Movie Recommendation System
Common utilities for model training, data handling, and evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import pickle
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MovieDataset(Dataset):
    """Custom dataset for movie ratings"""
    
    def __init__(self, ratings_matrix: torch.Tensor, 
                 transform=None, target_transform=None):
        """
        Initialize dataset
        
        Args:
            ratings_matrix: User-item rating matrix (n_users x n_movies)
            transform: Transform to apply to input
            target_transform: Transform to apply to target
        """
        self.ratings_matrix = ratings_matrix
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.ratings_matrix)
    
    def __getitem__(self, idx):
        """Get user rating vector"""
        user_ratings = self.ratings_matrix[idx]
        
        if self.transform:
            user_ratings = self.transform(user_ratings)
        
        # For autoencoders, input and target are the same
        target = user_ratings.clone()
        if self.target_transform:
            target = self.target_transform(target)
            
        return user_ratings, target


class DataProcessor:
    """Data processing utilities for recommendation models"""
    
    @staticmethod
    def load_preprocessed_data(data_path: str) -> Dict[str, torch.Tensor]:
        """
        Load preprocessed data from files
        
        Args:
            data_path: Path to preprocessed data directory
            
        Returns:
            Dictionary containing tensors
        """
        data_path = Path(data_path)
        
        data = {}
        
        # Load main datasets
        files_to_load = [
            'train_set.pt',
            'test_set.pt', 
            'user_item_matrix.pt'
        ]
        
        for filename in files_to_load:
            file_path = data_path / filename
            if file_path.exists():
                key = filename.replace('.pt', '')
                data[key] = torch.load(file_path)
                logger.info(f"Loaded {key}: {data[key].shape}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        # Load metadata if exists
        metadata_path = data_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data['metadata'] = json.load(f)
            logger.info("Loaded metadata")
        
        return data
    
    @staticmethod
    def create_train_val_split(ratings_matrix: torch.Tensor, 
                              val_ratio: float = 0.2,
                              random_seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create train/validation split
        
        Args:
            ratings_matrix: User-item rating matrix
            val_ratio: Fraction of users for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_matrix, val_matrix)
        """
        torch.manual_seed(random_seed)
        
        n_users = ratings_matrix.shape[0]
        n_val_users = int(n_users * val_ratio)
        
        # Random permutation of user indices
        perm = torch.randperm(n_users)
        
        val_indices = perm[:n_val_users]
        train_indices = perm[n_val_users:]
        
        train_matrix = ratings_matrix[train_indices]
        val_matrix = ratings_matrix[val_indices]
        
        logger.info(f"Train users: {len(train_indices)}, Val users: {len(val_indices)}")
        
        return train_matrix, val_matrix
    
    @staticmethod
    def create_data_loaders(train_data: torch.Tensor,
                           val_data: Optional[torch.Tensor] = None,
                           batch_size: int = 256,
                           shuffle: bool = True,
                           num_workers: int = 0) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create PyTorch data loaders
        
        Args:
            train_data: Training data tensor
            val_data: Validation data tensor (optional)
            batch_size: Batch size
            shuffle: Whether to shuffle training data
            num_workers: Number of data loading workers
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create datasets
        train_dataset = MovieDataset(train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = None
        if val_data is not None:
            val_dataset = MovieDataset(val_data)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        
        return train_loader, val_loader
    
    @staticmethod
    def normalize_ratings(ratings_matrix: torch.Tensor, 
                         method: str = 'user_mean') -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Normalize ratings matrix
        
        Args:
            ratings_matrix: Raw ratings matrix
            method: Normalization method ('user_mean', 'global_mean', 'z_score')
            
        Returns:
            Tuple of (normalized_matrix, normalization_stats)
        """
        normalized_matrix = ratings_matrix.clone().float()
        stats = {}
        
        if method == 'user_mean':
            # Subtract user mean from each user's ratings
            user_means = []
            for i in range(ratings_matrix.shape[0]):
                user_ratings = ratings_matrix[i]
                rated_mask = user_ratings > 0
                if rated_mask.sum() > 0:
                    user_mean = user_ratings[rated_mask].float().mean()
                    normalized_matrix[i, rated_mask] = user_ratings[rated_mask].float() - user_mean
                    user_means.append(user_mean.item())
                else:
                    user_means.append(0.0)
            
            stats['user_means'] = torch.tensor(user_means)
            
        elif method == 'global_mean':
            # Subtract global mean from all ratings
            all_ratings = ratings_matrix[ratings_matrix > 0].float()
            global_mean = all_ratings.mean()
            
            mask = ratings_matrix > 0
            normalized_matrix[mask] = ratings_matrix[mask].float() - global_mean
            
            stats['global_mean'] = global_mean.item()
            
        elif method == 'z_score':
            # Z-score normalization
            all_ratings = ratings_matrix[ratings_matrix > 0].float()
            global_mean = all_ratings.mean()
            global_std = all_ratings.std()
            
            mask = ratings_matrix > 0
            normalized_matrix[mask] = (ratings_matrix[mask].float() - global_mean) / global_std
            
            stats['global_mean'] = global_mean.item()
            stats['global_std'] = global_std.item()
        
        return normalized_matrix, stats
    
    @staticmethod
    def denormalize_ratings(normalized_ratings: torch.Tensor,
                           stats: Dict[str, Any],
                           method: str = 'user_mean',
                           user_idx: Optional[int] = None) -> torch.Tensor:
        """
        Denormalize ratings back to original scale
        
        Args:
            normalized_ratings: Normalized ratings
            stats: Normalization statistics
            method: Normalization method used
            user_idx: User index (for user_mean method)
            
        Returns:
            Denormalized ratings
        """
        denormalized_ratings = normalized_ratings.clone()
        
        if method == 'user_mean' and user_idx is not None:
            user_mean = stats['user_means'][user_idx]
            denormalized_ratings += user_mean
            
        elif method == 'global_mean':
            global_mean = stats['global_mean']
            denormalized_ratings += global_mean
            
        elif method == 'z_score':
            global_mean = stats['global_mean']
            global_std = stats['global_std']
            denormalized_ratings = denormalized_ratings * global_std + global_mean
        
        return denormalized_ratings


class ModelFactory:
    """Factory class for creating recommendation models"""
    
    @staticmethod
    def create_model(model_type: str, n_movies: int, config: Dict[str, Any]):
        """
        Create recommendation model
        
        Args:
            model_type: Type of model ('sae' or 'rbm')
            n_movies: Number of movies
            config: Model configuration
            
        Returns:
            Initialized model
        """
        if model_type.lower() == 'sae' or model_type.lower() == 'autoencoder':
            from .autoencoder_model import StackedAutoEncoder
            return StackedAutoEncoder(n_movies, config)
            
        elif model_type.lower() == 'rbm':
            from .rbm_model import RestrictedBoltzmannMachine
            n_hidden = config.get('n_hidden', 300)
            return RestrictedBoltzmannMachine(n_movies, n_hidden, config)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_config(model_type: str, config_variant: str = 'default') -> Dict[str, Any]:
        """
        Get model configuration
        
        Args:
            model_type: Type of model ('sae' or 'rbm')
            config_variant: Configuration variant ('default', 'large', 'fast', etc.)
            
        Returns:
            Model configuration dictionary
        """
        if model_type.lower() == 'sae' or model_type.lower() == 'autoencoder':
            from .autoencoder_model import SAEConfig
            
            if config_variant == 'large':
                return SAEConfig.get_large_config()
            elif config_variant == 'deep':
                return SAEConfig.get_deep_config()
            else:
                return SAEConfig.get_default_config()
                
        elif model_type.lower() == 'rbm':
            from .rbm_model import RBMConfig
            
            if config_variant == 'large':
                return RBMConfig.get_large_config()
            elif config_variant == 'fast':
                return RBMConfig.get_fast_config()
            else:
                return RBMConfig.get_default_config()
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class ModelCheckpoint:
    """Model checkpointing utilities"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch: int, 
                       loss: float, is_best: bool = False) -> str:
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'model_config': model.config,
            'model_name': model.model_name
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"{model.model_name.lower()}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / f"{model.model_name.lower()}_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, model, optimizer, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss']
        }


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Model being trained
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                
            return False
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                # Restore best weights if requested
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best weights")
                
                return True
            
            return False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name()
        info['memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
        info['memory_reserved'] = torch.cuda.memory_reserved() / 1024**2  # MB
    
    return info