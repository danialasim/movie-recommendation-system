#!/usr/bin/env python3
"""
Test script for optimized training with improved hyperparameters.

This script tests the training pipeline with the new optimized hyperparameters
without running full hyperparameter optimization (for faster testing).
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

def test_optimized_training():
    """Test training with optimized hyperparameters."""
    
    logger.info("Testing optimized training pipeline...")
    
    # Create trainer with optimized config (without hyperparameter search)
    trainer = MovieRecommendationTrainer()
    
    # Temporarily disable hyperparameter optimization for testing
    trainer.config['use_hyperparameter_optimization'] = False
    
    # Run the training pipeline
    results = trainer.run_full_training_pipeline()
    
    logger.info("Optimized training test completed successfully!")
    
    # Print results summary
    if 'sae_model' in results:
        sae_model = results['sae_model']
        logger.info(f"SAE Model trained with {sae_model.config['hidden_dims']} hidden dims")
    
    if 'rbm_model' in results:
        rbm_model = results['rbm_model']
        logger.info(f"RBM Model trained with {rbm_model.n_hidden} hidden units")
    
    return results

if __name__ == "__main__":
    try:
        results = test_optimized_training()
        logger.info("✅ All tests passed! Optimized training pipeline is working.")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
