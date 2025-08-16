#!/usr/bin/env python3
"""
Full training script with hyperparameter optimization enabled.

This script runs the complete training pipeline with Optuna-based hyperparameter
optimization for both SAE and RBM models to achieve the best performance.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from training.train import MovieRecommendationTrainer

# Setup proper logging
from src.utils.logging_utils import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__, 'training')

def run_full_optimized_training():
    """Run complete training with hyperparameter optimization."""
    
    logger.info("🚀 Starting FULL training with hyperparameter optimization...")
    
    # Create trainer
    trainer = MovieRecommendationTrainer()
    
    # Enable hyperparameter optimization with reasonable trial counts
    trainer.config['use_hyperparameter_optimization'] = True
    trainer.config['n_trials_sae'] = 20  # Good balance of exploration vs time
    trainer.config['n_trials_rbm'] = 15  # RBM is slower, so fewer trials
    trainer.config['optimization_timeout'] = 10800  # 3 hours max
    
    # Enhanced training settings
    trainer.config['sae']['num_epochs'] = 400  # More epochs for better convergence
    trainer.config['rbm']['num_epochs'] = 300  # More epochs for RBM
    trainer.config['sae']['patience'] = 40     # More patience for complex optimization
    
    logger.info(f"Configuration:")
    logger.info(f"  - SAE trials: {trainer.config['n_trials_sae']}")
    logger.info(f"  - RBM trials: {trainer.config['n_trials_rbm']}")
    logger.info(f"  - SAE epochs: {trainer.config['sae']['num_epochs']}")
    logger.info(f"  - RBM epochs: {trainer.config['rbm']['num_epochs']}")
    logger.info(f"  - Timeout: {trainer.config['optimization_timeout']/3600:.1f} hours")
    
    # Run the full training pipeline
    results = trainer.run_full_training_pipeline()
    
    logger.info("🎉 FULL OPTIMIZED TRAINING COMPLETED!")
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*80)
    
    if 'sae_model' in results:
        sae_model = results['sae_model']
        print(f"\n📊 STACKED AUTOENCODER (SAE):")
        print(f"  Architecture: {sae_model.config['hidden_dims']}")
        print(f"  Parameters: {sae_model.count_parameters():,}")
        print(f"  Activation: {sae_model.activation}")
        
        if 'sae_best_params' in results:
            best_params = results['sae_best_params']
            print(f"  🎯 OPTIMIZED HYPERPARAMETERS:")
            print(f"    Learning Rate: {best_params.get('learning_rate', 'N/A'):.6f}")
            print(f"    Weight Decay: {best_params.get('weight_decay', 'N/A'):.6f}")
            print(f"    Dropout: {best_params.get('dropout_rate', 'N/A'):.3f}")
            print(f"    Batch Size: {best_params.get('batch_size', 'N/A')}")
            print(f"    Hidden Dims: {best_params.get('hidden_dims', 'N/A')}")
    
    if 'rbm_model' in results:
        rbm_model = results['rbm_model']
        print(f"\n🔥 RESTRICTED BOLTZMANN MACHINE (RBM):")
        print(f"  Architecture: {rbm_model.n_movies} visible ↔ {rbm_model.n_hidden} hidden")
        print(f"  Parameters: {rbm_model.count_parameters():,}")
        print(f"  CD Steps: {rbm_model.cd_k}")
        
        if 'rbm_best_params' in results:
            best_params = results['rbm_best_params']
            print(f"  🎯 OPTIMIZED HYPERPARAMETERS:")
            print(f"    Learning Rate: {best_params.get('learning_rate', 'N/A'):.6f}")
            print(f"    Hidden Units: {best_params.get('n_hidden', 'N/A')}")
            print(f"    CD K: {best_params.get('cd_k', 'N/A')}")
            print(f"    L2 Penalty: {best_params.get('l2_penalty', 'N/A'):.6f}")
            print(f"    Batch Size: {best_params.get('batch_size', 'N/A')}")
    
    print(f"\n🎊 Training completed! Check MLflow UI for detailed metrics and visualizations.")
    print("="*80)
    
    return results

if __name__ == "__main__":
    try:
        results = run_full_optimized_training()
        logger.info("✅ SUCCESS: Full optimized training completed!")
    except Exception as e:
        logger.error(f"❌ FAILED: Training failed with error: {e}")
        raise
