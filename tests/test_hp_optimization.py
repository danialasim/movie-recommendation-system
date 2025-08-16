#!/usr/bin/env python3
"""
Test script for hyperparameter optimization functionality.

This script tests the Optuna-based hyperparameter optimization with a small
number of trials for quick verification.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from training.train import MovieRecommendationTrainer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hyperparameter_optimization():
    """Test hyperparameter optimization with a small number of trials."""
    
    logger.info("Testing hyperparameter optimization...")
    
    # Create trainer with optimization enabled
    trainer = MovieRecommendationTrainer()
    
    # Enable hyperparameter optimization with fewer trials for testing
    trainer.config['use_hyperparameter_optimization'] = True
    trainer.config['n_trials_sae'] = 3  # Small number for quick test
    trainer.config['n_trials_rbm'] = 2  # Small number for quick test
    
    # Load and preprocess data
    train_sae, test_sae, train_rbm, test_rbm = trainer.load_and_preprocess_data()
    
    # Test SAE optimization only (faster)
    logger.info("Testing SAE hyperparameter optimization...")
    try:
        best_sae_params = trainer.optimize_hyperparameters('sae', train_sae, test_sae)
        logger.info(f"✅ SAE optimization completed: {best_sae_params}")
    except Exception as e:
        logger.error(f"❌ SAE optimization failed: {e}")
        raise
    
    logger.info("Hyperparameter optimization test completed successfully!")
    
    return best_sae_params

if __name__ == "__main__":
    try:
        results = test_hyperparameter_optimization()
        logger.info("✅ Hyperparameter optimization test passed!")
        logger.info(f"Best parameters found: {results}")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
