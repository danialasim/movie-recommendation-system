"""
Hyperparameter optimization using Optuna for movie recommendation models.

This module implements automated hyperparameter tuning for both SAE and RBM models
using Optuna optimization framework with MLflow tracking.

Author: Movie Recommendation System Team
Date: August 2025
"""

import optuna
import torch
import mlflow
import logging
from typing import Dict, Any, Tuple
from pathlib import Path
import sys
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.autoencoder_model import StackedAutoEncoder, create_sae_data_loader
from models.rbm_model import RestrictedBoltzmannMachine, convert_ratings_to_binary
from training.evaluate import ModelEvaluator
import torch.optim as optim

# Setup proper logging
from utils.logging_utils import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__, 'training')


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for movie recommendation models using Optuna.
    """
    
    def __init__(self, train_matrix: torch.Tensor, val_matrix: torch.Tensor, 
                 test_matrix: torch.Tensor, device: torch.device = None):
        """
        Initialize the optimizer.
        
        Args:
            train_matrix: Training data matrix
            val_matrix: Validation data matrix  
            test_matrix: Test data matrix
            device: Computing device
        """
        self.train_matrix = train_matrix
        self.val_matrix = val_matrix
        self.test_matrix = test_matrix
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ModelEvaluator()
        
        logger.info(f"Optimizer initialized on device: {self.device}")
        logger.info(f"Data shapes - Train: {train_matrix.shape}, Val: {val_matrix.shape}, Test: {test_matrix.shape}")
    
    def optimize_sae(self, n_trials: int = 50, timeout: int = 3600) -> Dict[str, Any]:
        """
        Optimize SAE hyperparameters using Optuna.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Best hyperparameters found
        """
        logger.info(f"Starting SAE hyperparameter optimization with {n_trials} trials")
        
        def objective(trial):
            """Optuna objective function for SAE."""
            
            # Define hyperparameter search space based on MLOps guide recommendations
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-1),
                'hidden_dim1': trial.suggest_int('hidden_dim1', 15, 50),  # Increased from [10,50]
                'hidden_dim2': trial.suggest_int('hidden_dim2', 8, 25),   # Increased from [5,20]
                'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'activation': trial.suggest_categorical('activation', ['sigmoid', 'relu', 'tanh'])
            }
            
            try:
                # Create model with suggested parameters
                n_movies = self.train_matrix.shape[1]
                model = StackedAutoEncoder(
                    n_movies=n_movies,
                    hidden_dims=[params['hidden_dim1'], params['hidden_dim2']],
                    dropout_rate=params['dropout_rate'],
                    activation=params['activation']
                ).to(self.device)
                
                # Create data loaders
                train_loader = create_sae_data_loader(
                    self.train_matrix, 
                    batch_size=params['batch_size'],
                    shuffle=True
                )
                
                val_loader = create_sae_data_loader(
                    self.val_matrix,
                    batch_size=params['batch_size'],
                    shuffle=False
                )
                
                # Setup optimizer
                optimizer = optim.RMSprop(
                    model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=params['weight_decay']
                )
                
                # Training loop (reduced epochs for optimization speed)
                num_epochs = 50  # Reduced for faster optimization
                best_val_loss = float('inf')
                patience = 10
                patience_counter = 0
                
                model.train()
                for epoch in range(num_epochs):
                    # Training epoch
                    epoch_loss = 0.0
                    num_batches = 0
                    
                    for batch_data in train_loader:
                        if isinstance(batch_data, (list, tuple)):
                            batch_data = batch_data[0]
                        
                        batch_data = batch_data.to(self.device)
                        
                        optimizer.zero_grad()
                        reconstructed = model(batch_data)
                        loss = model.compute_loss(reconstructed, batch_data)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                    
                    train_loss = epoch_loss / num_batches
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for batch_data in val_loader:
                            if isinstance(batch_data, (list, tuple)):
                                batch_data = batch_data[0]
                            
                            batch_data = batch_data.to(self.device)
                            reconstructed = model(batch_data)
                            loss = model.compute_loss(reconstructed, batch_data)
                            val_loss += loss.item()
                            val_batches += 1
                    
                    val_loss = val_loss / val_batches
                    model.train()
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
                
                # Report intermediate value for pruning
                trial.report(best_val_loss, epoch)
                
                # Prune unpromising trials
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return best_val_loss
                
            except Exception as e:
                logger.warning(f"Trial failed with error: {e}")
                return float('inf')
        
        # Create study and optimize
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        best_params['hidden_dims'] = [best_params.pop('hidden_dim1'), best_params.pop('hidden_dim2')]
        
        logger.info(f"SAE optimization completed. Best validation loss: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def optimize_rbm(self, n_trials: int = 30, timeout: int = 2400) -> Dict[str, Any]:
        """
        Optimize RBM hyperparameters using Optuna.
        
        Args:
            n_trials: Number of optimization trials  
            timeout: Timeout in seconds
            
        Returns:
            Best hyperparameters found
        """
        logger.info(f"Starting RBM hyperparameter optimization with {n_trials} trials")
        
        # Convert to binary for RBM
        train_binary = convert_ratings_to_binary(self.train_matrix, threshold=3.0)
        val_binary = convert_ratings_to_binary(self.val_matrix, threshold=3.0)
        
        def objective(trial):
            """Optuna objective function for RBM."""
            
            # Define hyperparameter search space based on MLOps guide
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                'n_hidden': trial.suggest_int('n_hidden', 80, 250),  # Expanded range
                'cd_k': trial.suggest_int('cd_k', 5, 15),  # Focused on better range
                'batch_size': trial.suggest_categorical('batch_size', [50, 100, 150, 200]),
                'l2_penalty': trial.suggest_loguniform('l2_penalty', 1e-6, 1e-2)
            }
            
            try:
                # Create model
                n_movies = train_binary.shape[1]
                model = RestrictedBoltzmannMachine(
                    n_movies=n_movies,
                    n_hidden=params['n_hidden'],
                    learning_rate=params['learning_rate'],
                    cd_k=params['cd_k'],
                    l2_penalty=params['l2_penalty']
                ).to(self.device)
                
                # Training loop (reduced epochs for optimization)
                num_epochs = 30  # Reduced for faster optimization
                best_val_loss = float('inf')
                
                # Create batches manually for RBM
                batch_size = params['batch_size']
                n_batches = len(train_binary) // batch_size
                
                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    
                    # Shuffle data
                    indices = torch.randperm(len(train_binary))
                    
                    for i in range(n_batches):
                        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                        batch_data = train_binary[batch_indices].to(self.device)
                        
                        # RBM training step
                        reconstructed, pos_stats, neg_stats = model.contrastive_divergence(batch_data)
                        
                        # Compute reconstruction loss for optimization
                        loss = torch.mean((batch_data - reconstructed) ** 2)
                        epoch_loss += loss.item()
                    
                    train_loss = epoch_loss / n_batches
                    
                    # Validation
                    val_loss = 0.0
                    val_batches = len(val_binary) // batch_size
                    
                    with torch.no_grad():
                        for i in range(val_batches):
                            start_idx = i * batch_size
                            end_idx = min((i + 1) * batch_size, len(val_binary))
                            batch_data = val_binary[start_idx:end_idx].to(self.device)
                            
                            # Compute reconstruction error
                            _, h = model.sample_hidden(batch_data)
                            _, v_recon = model.sample_visible(h)
                            loss = torch.mean((batch_data - v_recon) ** 2)
                            val_loss += loss.item()
                    
                    val_loss = val_loss / val_batches
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                
                # Report for pruning
                trial.report(best_val_loss, epoch)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return best_val_loss
                
            except Exception as e:
                logger.warning(f"RBM trial failed with error: {e}")
                return float('inf')
        
        # Create study and optimize
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        
        logger.info(f"RBM optimization completed. Best validation loss: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params


def run_hyperparameter_optimization(train_matrix: torch.Tensor, 
                                   val_matrix: torch.Tensor,
                                   test_matrix: torch.Tensor,
                                   optimize_sae: bool = True,
                                   optimize_rbm: bool = True,
                                   sae_trials: int = 50,
                                   rbm_trials: int = 30) -> Dict[str, Dict[str, Any]]:
    """
    Run hyperparameter optimization for both models.
    
    Args:
        train_matrix: Training data
        val_matrix: Validation data
        test_matrix: Test data
        optimize_sae: Whether to optimize SAE
        optimize_rbm: Whether to optimize RBM
        sae_trials: Number of SAE trials
        rbm_trials: Number of RBM trials
        
    Returns:
        Dictionary with best parameters for each model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = HyperparameterOptimizer(train_matrix, val_matrix, test_matrix, device)
    
    results = {}
    
    if optimize_sae:
        logger.info("=" * 50)
        logger.info("OPTIMIZING SAE HYPERPARAMETERS")
        logger.info("=" * 50)
        results['sae'] = optimizer.optimize_sae(n_trials=sae_trials)
    
    if optimize_rbm:
        logger.info("=" * 50)
        logger.info("OPTIMIZING RBM HYPERPARAMETERS")
        logger.info("=" * 50)
        results['rbm'] = optimizer.optimize_rbm(n_trials=rbm_trials)
    
    return results


if __name__ == "__main__":
    # Example usage
    logger.info("Starting hyperparameter optimization...")
    
    # This would be called from the main training script
    # with actual data matrices
    pass
