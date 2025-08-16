"""
Main training pipeline for movie recommendation system.

This module handles training of both SAE and RBM models with comprehensive
MLflow tracking, hyperparameter optimization, and evaluation.

Author: Movie Recommendation System Team
Date: August 2025
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow imports
import mlflow
import mlflow.pytorch
import dagshub

# Optuna for hyperparameter optimization
import optuna
from optuna.integration.mlflow import MLflowCallback

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_preprocessing import MovieLensPreprocessor
from data.data_utils import DataValidator
from models.autoencoder_model import (StackedAutoEncoder, create_sae_data_loader, 
                                     normalize_ratings_for_sae)
from models.rbm_model import (RestrictedBoltzmannMachine, create_rbm_data_loader,
                             convert_ratings_to_binary, evaluate_rbm_reconstruction)
from training.evaluate import ModelEvaluator
from training.hyperparameter_optimization import run_hyperparameter_optimization

# Setup proper logging
from utils.logging_utils import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__, 'training')

# Suppress warnings
warnings.filterwarnings('ignore')


class MovieRecommendationTrainer:
    """
    Main trainer class for movie recommendation models.
    
    Handles both SAE and RBM training with MLflow tracking and optimization.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize paths
        self.data_path = Path(self.config.get('data_path', 'data'))
        self.model_path = Path(self.config.get('model_path', 'models'))
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data preprocessor
        self.preprocessor = MovieLensPreprocessor(
            raw_data_path=str(self.data_path / 'raw'),
            processed_data_path=str(self.data_path / 'preprocessed')
        )
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator()
        
        # Setup MLflow
        self._setup_mlflow()
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Model path: {self.model_path}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        
        default_config = {
            'data_path': 'data',
            'model_path': 'models',
            'dataset': 'ml-100k',  # or 'ml-1m'
            'test_size': 0.2,
            'random_seed': 42,
            
            # SAE config - Optimized based on MLOps guide recommendations
            'sae': {
                'hidden_dims': [32, 16],  # Increased capacity for better learning
                'dropout_rate': 0.3,      # Increased regularization
                'activation': 'relu',     # Better gradient flow than sigmoid
                'learning_rate': 0.005,   # Slightly lower for stable training
                'weight_decay': 0.001,    # Reduced from 0.5 for better generalization
                'batch_size': 256,        # Larger batches for stable gradients
                'num_epochs': 300,        # More epochs for convergence
                'patience': 30            # More patience for complex model
            },
            
            # RBM config - Optimized based on MLOps guide recommendations  
            'rbm': {
                'n_hidden': 150,          # Increased hidden units for better capacity
                'learning_rate': 0.005,   # Lower learning rate for stable CD
                'cd_k': 10,               # Keep at 10 as recommended
                'batch_size': 128,        # Increased batch size
                'num_epochs': 250,        # More epochs for RBM convergence
                'rating_threshold': 3.0,
                'l2_penalty': 1e-4        # Added L2 regularization
            },
            
            # Training config
            'train_both_models': True,
            'use_hyperparameter_optimization': True,  # Enable optimization by default
            'n_trials_sae': 30,                      # Reasonable number for SAE
            'n_trials_rbm': 20,                      # Reasonable number for RBM
            'optimization_timeout': 7200,            # 2 hours timeout
            'early_stopping': True,
            'save_best_model': True,
            
            # MLflow config
            'experiment_name': 'movie_recommendation_experiment',
            'mlflow_tracking_uri': None,
            'dagshub_repo': None
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            # Setup DagsHub if configured
            if self.config.get('dagshub_repo'):
                dagshub.init(repo=self.config['dagshub_repo'], mlflow=True)
            
            # Set tracking URI
            if self.config.get('mlflow_tracking_uri'):
                mlflow.set_tracking_uri(self.config['mlflow_tracking_uri'])
            
            # Set experiment
            experiment_name = self.config.get('experiment_name', 'movie_recommendation_experiment')
            try:
                mlflow.set_experiment(experiment_name)
            except:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
            
            logger.info("MLflow setup completed")
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
    
    def load_and_preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and preprocess data for training.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (train_matrix_sae, test_matrix_sae, train_matrix_rbm, test_matrix_rbm)
        """
        logger.info("Loading and preprocessing data...")
        
        # Set random seed for reproducibility
        torch.manual_seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))
        
        # Load dataset
        dataset = self.config.get('dataset', 'ml-100k')
        
        if dataset == 'ml-100k':
            ratings_df = self.preprocessor.load_ml_100k()
        elif dataset == 'ml-1m':
            ratings_df = self.preprocessor.load_ml_1m()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        logger.info(f"Loaded {dataset} dataset: {len(ratings_df)} ratings")
        
        # Validate data
        DataValidator.validate_ratings_df(ratings_df)
        
        # Create user-movie mappings
        self.preprocessor.create_user_movie_mappings(ratings_df)
        
        # Create user-item matrix
        user_item_matrix = self.preprocessor.create_user_item_matrix(ratings_df)
        logger.info(f"Created user-item matrix: {user_item_matrix.shape}")
        
        # Create train/test split
        train_matrix, test_matrix = self.preprocessor.create_train_test_split(
            user_item_matrix, test_ratio=self.config.get('test_size', 0.2)
        )
        
        logger.info(f"Train matrix sparsity: {(train_matrix == 0).float().mean():.3f}")
        logger.info(f"Test matrix sparsity: {(test_matrix == 0).float().mean():.3f}")
        
        # Prepare data for SAE (normalized ratings)
        train_matrix_sae = normalize_ratings_for_sae(train_matrix, method='global')
        test_matrix_sae = normalize_ratings_for_sae(test_matrix, method='global')
        
        # Prepare data for RBM (binary ratings)
        threshold = self.config['rbm'].get('rating_threshold', 3.0)
        train_matrix_rbm = convert_ratings_to_binary(train_matrix, threshold=threshold)
        test_matrix_rbm = convert_ratings_to_binary(test_matrix, threshold=threshold)
        
        # Save preprocessed data
        self._save_preprocessed_data(train_matrix_sae, test_matrix_sae, 
                                   train_matrix_rbm, test_matrix_rbm)
        
        logger.info("Data preprocessing completed")
        
        return train_matrix_sae, test_matrix_sae, train_matrix_rbm, test_matrix_rbm
    
    def _save_preprocessed_data(self, train_sae: torch.Tensor, test_sae: torch.Tensor,
                               train_rbm: torch.Tensor, test_rbm: torch.Tensor):
        """Save preprocessed data to disk."""
        save_dir = self.data_path / 'preprocessed'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(train_sae, save_dir / 'train_matrix_sae.pt')
        torch.save(test_sae, save_dir / 'test_matrix_sae.pt')
        torch.save(train_rbm, save_dir / 'train_matrix_rbm.pt')
        torch.save(test_rbm, save_dir / 'test_matrix_rbm.pt')
        
        # Save metadata
        metadata = {
            'dataset': self.config.get('dataset'),
            'n_users': int(train_sae.shape[0]),
            'n_movies': int(train_sae.shape[1]),
            'train_sparsity_sae': float((train_sae == 0).float().mean().item()),
            'test_sparsity_sae': float((test_sae == 0).float().mean().item()),
            'user_to_idx': {int(k): int(v) for k, v in self.preprocessor.user_to_idx.items()},
            'movie_to_idx': {int(k): int(v) for k, v in self.preprocessor.movie_to_idx.items()}
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Preprocessed data saved to {save_dir}")
    
    def train_sae(self, train_matrix: torch.Tensor, test_matrix: torch.Tensor,
                  hyperparams: Optional[Dict[str, Any]] = None) -> StackedAutoEncoder:
        """
        Train Stacked AutoEncoder model.
        
        Args:
            train_matrix: Training data
            test_matrix: Test data
            hyperparams: Hyperparameters (uses config if None)
            
        Returns:
            StackedAutoEncoder: Trained model
        """
        logger.info("Training Stacked AutoEncoder...")
        
        # Use provided hyperparams or config defaults
        if hyperparams is None:
            hyperparams = self.config['sae'].copy()
        
        # Initialize model
        n_movies = train_matrix.shape[1]
        model = StackedAutoEncoder(
            n_movies=n_movies,
            hidden_dims=hyperparams.get('hidden_dims', [20, 10]),
            dropout_rate=hyperparams.get('dropout_rate', 0.2),
            activation=hyperparams.get('activation', 'sigmoid'),
            config=hyperparams
        )
        
        model.start_mlflow_run(experiment_name=self.config.get('experiment_name'))
        
        # Create data loaders
        train_loader = create_sae_data_loader(
            train_matrix, 
            batch_size=hyperparams.get('batch_size', 128),
            shuffle=True
        )
        
        test_loader = create_sae_data_loader(
            test_matrix,
            batch_size=hyperparams.get('batch_size', 128),
            shuffle=False
        )
        
        # Setup optimizer
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=hyperparams.get('learning_rate', 0.01),
            weight_decay=hyperparams.get('weight_decay', 0.5)
        )
        
        # Training loop
        num_epochs = hyperparams.get('num_epochs', 200)
        patience = hyperparams.get('patience', 20)
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            train_loss = model.train_epoch(train_loader, optimizer, epoch)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = model.validate(test_loader)
            val_losses.append(val_loss)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"SAE Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}")
            
            # MLflow logging
            model.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch
            }, step=epoch)
            
            # Early stopping
            if self.config.get('early_stopping', True):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model with hyperparameters
                    if self.config.get('save_best_model', True):
                        best_model_path = self.model_path / 'autoencoder' / 'best_model.pt'
                        best_model_path.parent.mkdir(parents=True, exist_ok=True)
                        model.save_model(str(best_model_path))
                        
                        # Save best hyperparameters at checkpoint
                        if hyperparams:
                            best_hyperparams_path = self.model_path / 'autoencoder' / 'best_hyperparameters_checkpoint.json'
                            checkpoint_config = {
                                'epoch': epoch,
                                'best_val_loss': best_val_loss,
                                'hyperparameters': hyperparams,
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                            with open(best_hyperparams_path, 'w') as f:
                                json.dump(checkpoint_config, f, indent=2)
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Final evaluation
        final_metrics = self.evaluator.evaluate_sae(model, test_matrix, train_matrix)
        model.log_metrics(final_metrics)
        
        # Save final model with hyperparameters
        final_model_path = self.model_path / 'autoencoder' / 'final_model.pt'
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(final_model_path))
        
        # Save hyperparameters alongside the model
        if hyperparams:
            hyperparams_path = self.model_path / 'autoencoder' / 'best_hyperparameters.json'
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparams, f, indent=2)
            logger.info(f"Saved best hyperparameters to {hyperparams_path}")
        
        # Save complete model configuration
        model_config = {
            'model_type': 'SAE',
            'architecture': {
                'input_dim': hyperparams.get('input_dim', test_matrix.shape[1]) if hyperparams else test_matrix.shape[1],
                'hidden_dims': hyperparams.get('hidden_dims', [128, 64]) if hyperparams else [128, 64],
                'activation': hyperparams.get('activation', 'relu') if hyperparams else 'relu'
            },
            'training_params': hyperparams if hyperparams else {},
            'final_metrics': final_metrics,
            'model_path': str(final_model_path.name),
            'training_completed': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_path = self.model_path / 'autoencoder' / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"Saved model configuration to {config_path}")
        
        # Log model to MLflow
        model.log_model("sae_model")
        
        # Plot and save training curves
        self._plot_training_curves(train_losses, val_losses, 'SAE', model)
        
        model.end_mlflow_run()
        
        logger.info("SAE training completed")
        logger.info(f"Final metrics: {final_metrics}")
        
        return model
    
    def train_rbm(self, train_matrix: torch.Tensor, test_matrix: torch.Tensor,
                  hyperparams: Optional[Dict[str, Any]] = None) -> RestrictedBoltzmannMachine:
        """
        Train Restricted Boltzmann Machine model.
        
        Args:
            train_matrix: Training data (binary)
            test_matrix: Test data (binary)
            hyperparams: Hyperparameters (uses config if None)
            
        Returns:
            RestrictedBoltzmannMachine: Trained model
        """
        logger.info("Training Restricted Boltzmann Machine...")
        
        # Use provided hyperparams or config defaults
        if hyperparams is None:
            hyperparams = self.config['rbm'].copy()
        
        # Initialize model
        n_movies = train_matrix.shape[1]
        model = RestrictedBoltzmannMachine(
            n_movies=n_movies,
            n_hidden=hyperparams.get('n_hidden', 100),
            learning_rate=hyperparams.get('learning_rate', 0.01),
            cd_k=hyperparams.get('cd_k', 10),
            l2_penalty=hyperparams.get('l2_penalty', 0.0),
            config=hyperparams
        )
        
        model.start_mlflow_run(experiment_name=self.config.get('experiment_name'))
        
        # Create data loaders
        train_loader = create_rbm_data_loader(
            train_matrix,
            batch_size=hyperparams.get('batch_size', 100),
            shuffle=True
        )
        
        test_loader = create_rbm_data_loader(
            test_matrix,
            batch_size=hyperparams.get('batch_size', 100),
            shuffle=False
        )
        
        # Training loop (RBM uses custom training)
        num_epochs = hyperparams.get('num_epochs', 200)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data in train_loader:
                batch_data = batch_data[0].to(model.device)  # Unpack from DataLoader
                loss = model.train_step(batch_data, None)  # RBM doesn't use optimizer
                epoch_loss += loss
                num_batches += 1
            
            avg_train_loss = epoch_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)
            
            # Validation
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_data = batch_data[0].to(model.device)
                    reconstructed = model.forward(batch_data)
                    loss = model.compute_loss(reconstructed, batch_data)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"RBM Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                          f"Val Loss = {avg_val_loss:.4f}")
            
            # MLflow logging
            model.log_metrics({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch
            }, step=epoch)
        
        # Final evaluation
        final_metrics = evaluate_rbm_reconstruction(model, test_matrix)
        model.log_metrics(final_metrics)
        
        # Save model with hyperparameters
        final_model_path = self.model_path / 'rbm' / 'final_model.pt'
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(final_model_path))
        
        # Save hyperparameters alongside the model
        if hyperparams:
            hyperparams_path = self.model_path / 'rbm' / 'best_hyperparameters.json'
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparams, f, indent=2)
            logger.info(f"Saved best hyperparameters to {hyperparams_path}")
        
        # Save complete model configuration
        model_config = {
            'model_type': 'RBM',
            'architecture': {
                'visible_units': hyperparams.get('visible_units', test_matrix.shape[1]) if hyperparams else test_matrix.shape[1],
                'hidden_units': hyperparams.get('n_hidden', 128) if hyperparams else 128,
                'cd_k': hyperparams.get('cd_k', 1) if hyperparams else 1
            },
            'training_params': hyperparams if hyperparams else {},
            'final_metrics': final_metrics,
            'model_path': str(final_model_path.name),
            'training_completed': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_path = self.model_path / 'rbm' / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"Saved model configuration to {config_path}")
        
        # Log model to MLflow
        model.log_model("rbm_model")
        
        # Plot training curves
        self._plot_training_curves(train_losses, val_losses, 'RBM', model)
        
        model.end_mlflow_run()
        
        logger.info("RBM training completed")
        logger.info(f"Final metrics: {final_metrics}")
        
        return model
    
    def _plot_training_curves(self, train_losses: List[float], val_losses: List[float],
                             model_name: str, model):
        """Plot and save training curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Training Curves')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = self.model_path / model_name.lower() / 'training_curves.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow if available
        if model.mlflow_run:
            try:
                mlflow.log_artifact(str(plot_path))
            except:
                pass
    
    def optimize_hyperparameters(self, model_type: str, train_matrix: torch.Tensor,
                                test_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Optimize hyperparameters using our advanced Optuna implementation.
        
        Args:
            model_type: 'sae' or 'rbm'
            train_matrix: Training data
            test_matrix: Test data
            
        Returns:
            Dict[str, Any]: Best hyperparameters
        """
        logger.info(f"Optimizing hyperparameters for {model_type.upper()} using advanced Optuna...")
        
        # Create validation split from training data
        val_size = int(0.2 * len(train_matrix))
        train_size = len(train_matrix) - val_size
        
        # Split data
        train_subset = train_matrix[:train_size]
        val_subset = train_matrix[train_size:]
        
        # Run optimization using our dedicated optimizer
        if model_type == 'sae':
            n_trials = self.config.get('n_trials_sae', 30)
            results = run_hyperparameter_optimization(
                train_subset, val_subset, test_matrix,
                optimize_sae=True, optimize_rbm=False,
                sae_trials=n_trials
            )
            best_params = results['sae']
            
        elif model_type == 'rbm':
            n_trials = self.config.get('n_trials_rbm', 20)
            results = run_hyperparameter_optimization(
                train_subset, val_subset, test_matrix,
                optimize_sae=False, optimize_rbm=True,
                rbm_trials=n_trials
            )
            best_params = results['rbm']
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Best {model_type.upper()} hyperparameters: {best_params}")
        
        return best_params
    
    def _sae_objective(self, trial, train_matrix: torch.Tensor, 
                      test_matrix: torch.Tensor) -> float:
        """Optuna objective for SAE hyperparameter optimization."""
        
        # Define hyperparameter search space
        hyperparams = {
            'hidden_dims': trial.suggest_categorical('hidden_dims', 
                                                    [[20, 10], [50, 20], [100, 50], [30, 15]]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-1),
            'dropout_rate': trial.suggest_uniform('dropout_rate', 0.0, 0.5),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
            'activation': trial.suggest_categorical('activation', ['sigmoid', 'relu']),
            'num_epochs': 50,  # Reduced for optimization
            'patience': 10
        }
        
        # Train model with these hyperparameters
        try:
            model = self.train_sae(train_matrix, test_matrix, hyperparams)
            
            # Return validation loss (to minimize)
            val_loss = model.validate(create_sae_data_loader(test_matrix, shuffle=False))
            
            return val_loss
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')
    
    def _rbm_objective(self, trial, train_matrix: torch.Tensor,
                      test_matrix: torch.Tensor) -> float:
        """Optuna objective for RBM hyperparameter optimization."""
        
        # Define hyperparameter search space
        hyperparams = {
            'n_hidden': trial.suggest_int('n_hidden', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'cd_k': trial.suggest_int('cd_k', 1, 20),
            'batch_size': trial.suggest_categorical('batch_size', [50, 100, 200]),
            'num_epochs': 50  # Reduced for optimization
        }
        
        # Train model with these hyperparameters
        try:
            model = self.train_rbm(train_matrix, test_matrix, hyperparams)
            
            # Return reconstruction error (to minimize)
            metrics = evaluate_rbm_reconstruction(model, test_matrix)
            
            return metrics['reconstruction_error']
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')
    
    def run_full_training_pipeline(self):
        """Run the complete training pipeline for both models."""
        logger.info("Starting full training pipeline...")
        
        # Load and preprocess data
        train_sae, test_sae, train_rbm, test_rbm = self.load_and_preprocess_data()
        
        results = {}
        
        if self.config.get('train_both_models', True):
            
            # Train SAE
            logger.info("=" * 50)
            logger.info("TRAINING STACKED AUTOENCODER")
            logger.info("=" * 50)
            
            if self.config.get('use_hyperparameter_optimization', False):
                best_sae_params = self.optimize_hyperparameters('sae', train_sae, test_sae)
                sae_model = self.train_sae(train_sae, test_sae, best_sae_params)
                results['sae_best_params'] = best_sae_params
            else:
                sae_model = self.train_sae(train_sae, test_sae)
            
            results['sae_model'] = sae_model
            
            # Train RBM
            logger.info("=" * 50)
            logger.info("TRAINING RESTRICTED BOLTZMANN MACHINE")
            logger.info("=" * 50)
            
            if self.config.get('use_hyperparameter_optimization', False):
                best_rbm_params = self.optimize_hyperparameters('rbm', train_rbm, test_rbm)
                rbm_model = self.train_rbm(train_rbm, test_rbm, best_rbm_params)
                results['rbm_best_params'] = best_rbm_params
            else:
                rbm_model = self.train_rbm(train_rbm, test_rbm)
            
            results['rbm_model'] = rbm_model
            
            # Compare models
            logger.info("=" * 50)
            logger.info("MODEL COMPARISON")
            logger.info("=" * 50)
            
            self._compare_models(sae_model, rbm_model, test_sae, test_rbm)
        
        logger.info("Full training pipeline completed!")
        return results
    
    def _compare_models(self, sae_model: StackedAutoEncoder, 
                       rbm_model: RestrictedBoltzmannMachine,
                       test_sae: torch.Tensor, test_rbm: torch.Tensor):
        """Compare trained models and log results."""
        
        logger.info("Comparing model performance...")
        
        # SAE evaluation
        sae_metrics = self.evaluator.evaluate_sae(sae_model, test_sae, None)
        
        # RBM evaluation
        rbm_metrics = evaluate_rbm_reconstruction(rbm_model, test_rbm)
        
        # Model info
        sae_info = sae_model.get_model_info()
        rbm_info = rbm_model.get_weights_info()
        
        # Create comparison report
        comparison = {
            'SAE': {
                'metrics': sae_metrics,
                'parameters': sae_info['total_parameters'],
                'model_size_mb': sae_model.get_model_size_mb()
            },
            'RBM': {
                'metrics': rbm_metrics,
                'parameters': rbm_model.count_parameters(),
                'model_size_mb': rbm_info['weight_norm']
            }
        }
        
        # Save comparison
        comparison_path = self.model_path / 'model_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info("Model comparison:")
        logger.info(f"SAE - RMSE: {sae_metrics.get('rmse', 'N/A'):.4f}, "
                   f"Parameters: {sae_info['total_parameters']}")
        logger.info(f"RBM - Reconstruction Error: {rbm_metrics.get('reconstruction_error', 'N/A'):.4f}, "
                   f"Parameters: {rbm_model.count_parameters()}")


def main():
    """Main function to run training."""
    
    # Parse command line arguments (if any)
    import argparse
    parser = argparse.ArgumentParser(description='Train movie recommendation models')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['ml-100k', 'ml-1m'], 
                       default='ml-100k', help='Dataset to use')
    parser.add_argument('--optimize', action='store_true', 
                       help='Use hyperparameter optimization')
    parser.add_argument('--model', type=str, choices=['sae', 'rbm', 'both'], 
                       default='both', help='Which model(s) to train')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MovieRecommendationTrainer(config_path=args.config)
    
    # Update config based on arguments
    if args.dataset:
        trainer.config['dataset'] = args.dataset
    if args.optimize:
        trainer.config['use_hyperparameter_optimization'] = True
    if args.model != 'both':
        trainer.config['train_both_models'] = False
    
    # Run training
    if args.model == 'both' or trainer.config.get('train_both_models', True):
        results = trainer.run_full_training_pipeline()
    else:
        # Load data
        train_sae, test_sae, train_rbm, test_rbm = trainer.load_and_preprocess_data()
        
        if args.model == 'sae':
            if args.optimize:
                best_params = trainer.optimize_hyperparameters('sae', train_sae, test_sae)
                model = trainer.train_sae(train_sae, test_sae, best_params)
            else:
                model = trainer.train_sae(train_sae, test_sae)
        
        elif args.model == 'rbm':
            if args.optimize:
                best_params = trainer.optimize_hyperparameters('rbm', train_rbm, test_rbm)
                model = trainer.train_rbm(train_rbm, test_rbm, best_params)
            else:
                model = trainer.train_rbm(train_rbm, test_rbm)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
