#!/usr/bin/env python3
"""
Simple training script to get started with movie recommendation models.

This script provides an easy way to train both SAE and RBM models
on the MovieLens dataset with minimal setup.

Usage:
    python start_training.py
"""

import os
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

def main():
    """Main function to start training."""
    
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - TRAINING")
    print("=" * 60)
    print()
    
    # Check if data exists
    data_path = project_root / 'data' / 'raw'
    if not data_path.exists():
        print("⚠️  Data directory not found!")
        print(f"Please ensure MovieLens data is in: {data_path}")
        print()
        print("Expected structure:")
        print("data/")
        print("├── raw/")
        print("│   ├── ml-100k/")
        print("│   │   ├── u1.base")
        print("│   │   ├── u1.test")
        print("│   │   └── u.item")
        print("│   └── ml-1m/")
        print("│       ├── movies.dat")
        print("│       ├── ratings.dat")
        print("│       └── users.dat")
        return
    
    print("🚀 Starting training pipeline...")
    print()
    
    try:
        # Import the training module
        from training.train import MovieRecommendationTrainer
        
        # Initialize trainer with config
        config_path = project_root / 'config' / 'training_config.json'
        trainer = MovieRecommendationTrainer(config_path=str(config_path) if config_path.exists() else None)
        
        print("✅ Trainer initialized successfully")
        print(f"📊 Using dataset: {trainer.config.get('dataset', 'ml-100k')}")
        print(f"🎯 Device: {trainer.device}")
        print()
        
        # Run training
        print("🔄 Starting model training...")
        results = trainer.run_full_training_pipeline()
        
        print()
        print("=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Show results summary
        if 'sae_model' in results:
            sae_info = results['sae_model'].get_model_info()
            print(f"📈 SAE Model: {sae_info['total_parameters']} parameters")
        
        if 'rbm_model' in results:
            rbm_params = results['rbm_model'].count_parameters()
            print(f"📈 RBM Model: {rbm_params} parameters")
        
        print()
        print("📁 Models saved in: models/")
        print("📊 Check MLflow UI for detailed metrics and comparisons")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install torch pandas numpy scikit-learn mlflow optuna matplotlib seaborn")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Check the logs above for detailed error information")


if __name__ == "__main__":
    main()
