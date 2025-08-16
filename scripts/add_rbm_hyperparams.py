#!/usr/bin/env python3
"""
Script to retroactively add hyperparameters to existing RBM model.

Since the RBM was trained before we added hyperparameter saving,
this script creates the missing configuration files.
"""

import json
import time
from pathlib import Path

def add_rbm_hyperparameters():
    """Add hyperparameters to existing RBM model."""
    
    rbm_dir = Path("models/rbm")
    
    if not rbm_dir.exists():
        print("❌ RBM model directory not found")
        return
    
    # Best hyperparameters from the training output we saw
    best_hyperparams = {
        "learning_rate": 0.016811361311235073,
        "n_hidden": 89,
        "cd_k": 9,
        "batch_size": 50,
        "l2_penalty": 0.00429255823866641,
        "visible_units": 1682,
        "model_type": "restricted_boltzmann_machine"
    }
    
    # Save hyperparameters
    hyperparams_path = rbm_dir / 'best_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(best_hyperparams, f, indent=2)
    
    # Create model configuration
    model_config = {
        "model_type": "RBM",
        "architecture": {
            "visible_units": 1682,
            "hidden_units": 89,
            "cd_k": 9
        },
        "training_params": best_hyperparams,
        "final_metrics": {
            "reconstruction_error": 0.6942779839038848,
            "average_free_energy": -57.616796875,
            "num_samples": 943
        },
        "model_path": "final_model.pt",
        "training_completed": "2025-08-16 05:51:48"
    }
    
    config_path = rbm_dir / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print("✅ RBM hyperparameters and configuration added successfully!")
    print(f"📁 Saved to: {rbm_dir}")
    print(f"📄 Files created:")
    print(f"   - best_hyperparameters.json")
    print(f"   - model_config.json")

if __name__ == "__main__":
    print("🔧 ADDING HYPERPARAMETERS TO EXISTING RBM MODEL")
    print("="*60)
    add_rbm_hyperparameters()
