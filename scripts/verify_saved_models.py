#!/usr/bin/env python3
"""
Script to verify that models are saved with their best hyperparameters.

This script checks the model directories and displays the saved configurations.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

def display_model_info(model_dir: Path, model_name: str) -> None:
    """Display saved model information."""
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} MODEL INFORMATION")
    print(f"{'='*60}")
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    print(f"📁 Model Directory: {model_dir}")
    
    # Check for model files
    model_files = []
    for pattern in ['*.pt', '*.pth', '*.pkl']:
        model_files.extend(model_dir.glob(pattern))
    
    if model_files:
        print(f"🤖 Model Files:")
        for model_file in model_files:
            print(f"   - {model_file.name}")
    else:
        print("❌ No model files found")
    
    # Check for hyperparameters
    hyperparams_path = model_dir / 'best_hyperparameters.json'
    if hyperparams_path.exists():
        print(f"⚙️  Best Hyperparameters: ✅")
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
        
        print("   Parameters:")
        for key, value in hyperparams.items():
            print(f"   - {key}: {value}")
    else:
        print(f"⚙️  Best Hyperparameters: ❌ Not found")
    
    # Check for model configuration
    config_path = model_dir / 'model_config.json'
    if config_path.exists():
        print(f"📋 Model Configuration: ✅")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("   Configuration Summary:")
        if 'model_type' in config:
            print(f"   - Model Type: {config['model_type']}")
        if 'metrics' in config and config['metrics']:
            print(f"   - Performance Metrics:")
            for metric, value in config['metrics'].items():
                if isinstance(value, float):
                    print(f"     • {metric}: {value:.4f}")
                else:
                    print(f"     • {metric}: {value}")
        if 'training_completed' in config:
            print(f"   - Training Completed: {config['training_completed']}")
    else:
        print(f"📋 Model Configuration: ❌ Not found")
    
    # Check for checkpoint files
    checkpoint_files = list(model_dir.glob('*checkpoint*'))
    if checkpoint_files:
        print(f"💾 Checkpoint Files: ✅")
        for checkpoint in checkpoint_files:
            print(f"   - {checkpoint.name}")
    else:
        print(f"💾 Checkpoint Files: ❌ Not found")

def main():
    """Main function to verify saved models."""
    print("🔍 VERIFYING SAVED MODELS WITH HYPERPARAMETERS")
    print("="*60)
    
    # Base model directory
    models_dir = Path("models")
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    # Check SAE models
    sae_dir = models_dir / "autoencoder"
    display_model_info(sae_dir, "Stacked AutoEncoder (SAE)")
    
    # Check RBM models
    rbm_dir = models_dir / "rbm"
    display_model_info(rbm_dir, "Restricted Boltzmann Machine (RBM)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    sae_complete = (sae_dir / 'best_hyperparameters.json').exists()
    rbm_complete = (rbm_dir / 'best_hyperparameters.json').exists()
    
    print(f"SAE Model with Hyperparameters: {'✅ Complete' if sae_complete else '❌ Incomplete'}")
    print(f"RBM Model with Hyperparameters: {'✅ Complete' if rbm_complete else '❌ Incomplete'}")
    
    if sae_complete and rbm_complete:
        print("\n🎉 All models are properly saved with their best hyperparameters!")
    else:
        print("\n⚠️  Some models are missing hyperparameter configurations.")
        print("   Run training again to generate complete model saves.")

if __name__ == "__main__":
    main()
