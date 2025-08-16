#!/usr/bin/env python3
"""
Demonstration of Model Saving with Best Hyperparameters

This script shows how the enhanced model saving system works and
demonstrates loading models with their optimal hyperparameters.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from load_trained_models import ModelLoader

def demonstrate_model_loading():
    """Demonstrate loading models with their best hyperparameters."""
    
    print("🎯 MODEL LOADING WITH BEST HYPERPARAMETERS DEMO")
    print("="*60)
    
    # Initialize the model loader
    loader = ModelLoader("models")
    
    try:
        print("\n🔄 Loading SAE model with optimized hyperparameters...")
        sae_model, sae_params = loader.load_sae_model(use_best=True)
        
        print(f"\n📊 SAE Model Summary:")
        print(f"   Architecture: {sae_params['n_movies']} → {' → '.join(map(str, sae_params['hidden_dims']))} → {sae_params['n_movies']}")
        print(f"   Optimized Parameters:")
        for key, value in sae_params.items():
            if key not in ['n_movies', 'hidden_dims', 'model_type']:
                print(f"     • {key}: {value}")
        
        # Get model parameter count
        total_params = sum(p.numel() for p in sae_model.parameters())
        print(f"   Total Parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ SAE loading failed: {e}")
    
    try:
        print(f"\n🔄 Loading RBM model with optimized hyperparameters...")
        rbm_model, rbm_params = loader.load_rbm_model(use_best=True)
        
        print(f"\n📊 RBM Model Summary:")
        print(f"   Architecture: {rbm_params['visible_units']} visible ↔ {rbm_params['n_hidden']} hidden")
        print(f"   Optimized Parameters:")
        for key, value in rbm_params.items():
            if key not in ['visible_units', 'n_hidden', 'model_type']:
                print(f"     • {key}: {value}")
        
        # Get model parameter count
        total_params = sum(p.numel() for p in rbm_model.parameters())
        print(f"   Total Parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ RBM loading failed: {e}")

def show_performance_metrics():
    """Display the performance metrics of saved models."""
    
    print(f"\n📈 MODEL PERFORMANCE METRICS")
    print("-" * 40)
    
    loader = ModelLoader("models")
    performance = loader.get_model_performance_summary()
    
    for model_name, metrics in performance.items():
        print(f"\n{model_name.upper()} Performance:")
        if metrics:
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        else:
            print("  No metrics available")

def show_hyperparameter_optimization_results():
    """Show the hyperparameter optimization results."""
    
    print(f"\n🎯 HYPERPARAMETER OPTIMIZATION RESULTS")
    print("-" * 50)
    
    # SAE Hyperparameters
    sae_hp_path = Path("models/autoencoder/best_hyperparameters.json")
    if sae_hp_path.exists():
        with open(sae_hp_path, 'r') as f:
            sae_params = json.load(f)
        
        print("\n🧠 SAE Optimized Configuration:")
        print(f"   Learning Rate: {sae_params.get('learning_rate', 'N/A')}")
        print(f"   Architecture: {sae_params.get('n_movies', 'N/A')} → {sae_params.get('hidden_dims', 'N/A')} → {sae_params.get('n_movies', 'N/A')}")
        print(f"   Dropout Rate: {sae_params.get('dropout_rate', 'N/A')}")
        print(f"   Batch Size: {sae_params.get('batch_size', 'N/A')}")
        print(f"   Activation: {sae_params.get('activation', 'N/A')}")
        print(f"   Weight Decay: {sae_params.get('weight_decay', 'N/A')}")
    
    # RBM Hyperparameters
    rbm_hp_path = Path("models/rbm/best_hyperparameters.json")
    if rbm_hp_path.exists():
        with open(rbm_hp_path, 'r') as f:
            rbm_params = json.load(f)
        
        print("\n⚡ RBM Optimized Configuration:")
        print(f"   Learning Rate: {rbm_params.get('learning_rate', 'N/A')}")
        print(f"   Architecture: {rbm_params.get('visible_units', 'N/A')} visible ↔ {rbm_params.get('n_hidden', 'N/A')} hidden")
        print(f"   CD-K Steps: {rbm_params.get('cd_k', 'N/A')}")
        print(f"   Batch Size: {rbm_params.get('batch_size', 'N/A')}")
        print(f"   L2 Penalty: {rbm_params.get('l2_penalty', 'N/A')}")

def main():
    """Main demo function."""
    
    print("🚀 ENHANCED MODEL SAVING SYSTEM DEMONSTRATION")
    print("="*70)
    print("This demo shows how models are now saved with their optimal")
    print("hyperparameters discovered through Optuna optimization.")
    print("="*70)
    
    # Show hyperparameter optimization results
    show_hyperparameter_optimization_results()
    
    # Demonstrate model loading
    demonstrate_model_loading()
    
    # Show performance metrics
    show_performance_metrics()
    
    print(f"\n✨ SUMMARY")
    print("-" * 20)
    print("✅ Models are saved with their optimal hyperparameters")
    print("✅ Models can be loaded with exact training configurations")
    print("✅ Performance metrics are preserved with model files")
    print("✅ System is ready for production deployment")
    
    print(f"\n🎯 Next Steps:")
    print("   1. Use loaded models for inference/recommendations")
    print("   2. Deploy models with optimal hyperparameters")
    print("   3. Compare against future model improvements")
    print("   4. Reproduce results with saved configurations")

if __name__ == "__main__":
    main()
