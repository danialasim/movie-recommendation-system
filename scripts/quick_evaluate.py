#!/usr/bin/env python3
"""
Quick Model Evaluation Script

Simple script to quickly evaluate your trained models and show results.
"""

import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from load_trained_models import ModelLoader
from data.data_preprocessing import MovieLensPreprocessor
from models.autoencoder_model import normalize_ratings_for_sae
from models.rbm_model import convert_ratings_to_binary, evaluate_rbm_reconstruction
from training.evaluate import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_evaluation():
    """Run quick evaluation of both models."""
    
    print("🚀 QUICK MODEL EVALUATION")
    print("="*50)
    
    # Initialize loader
    loader = ModelLoader("models")
    evaluator = ModelEvaluator()
    
    # Load preprocessed data
    print("\n📊 Loading test data...")
    try:
        # Try different possible file names
        preprocessed_dir = Path("data/preprocessed")
        
        # Check for SAE format files
        if (preprocessed_dir / "test_matrix_sae.pt").exists():
            test_matrix_sae = torch.load(preprocessed_dir / "test_matrix_sae.pt")
            train_matrix_sae = torch.load(preprocessed_dir / "train_matrix_sae.pt")
            test_matrix_rbm = torch.load(preprocessed_dir / "test_matrix_rbm.pt")
            train_matrix_rbm = torch.load(preprocessed_dir / "train_matrix_rbm.pt")
            print(f"✅ Data loaded - SAE Test: {test_matrix_sae.shape}, RBM Test: {test_matrix_rbm.shape}")
        elif (preprocessed_dir / "test_matrix.pt").exists():
            test_matrix = torch.load(preprocessed_dir / "test_matrix.pt")
            train_matrix = torch.load(preprocessed_dir / "train_matrix.pt")
            print(f"✅ Data loaded - Test: {test_matrix.shape}, Train: {train_matrix.shape}")
            # Use same matrices for both models
            test_matrix_sae = test_matrix
            train_matrix_sae = train_matrix
            test_matrix_rbm = test_matrix
            train_matrix_rbm = train_matrix
        else:
            raise FileNotFoundError("No preprocessed matrices found")
            
    except FileNotFoundError:
        print("❌ Preprocessed data not found. Please run training first.")
        return
    
    # Evaluate SAE
    print("\n🧠 Evaluating SAE Model...")
    try:
        sae_model, sae_params = loader.load_sae_model(use_best=True)
        sae_model.eval()
        
        # Prepare data for SAE (data is already normalized)
        test_sae = test_matrix_sae  # Already normalized
        
        # Get predictions (already normalized)
        with torch.no_grad():
            predictions = sae_model(test_sae)
        
        # Calculate metrics on normalized scale
        test_mask = (test_sae > 0)
        
        if test_mask.sum() > 0:
            true_ratings = test_sae[test_mask]
            pred_ratings = predictions[test_mask]
            
            rmse = torch.sqrt(torch.mean((true_ratings - pred_ratings) ** 2)).item()
            mae = torch.mean(torch.abs(true_ratings - pred_ratings)).item()
            
            # R² score
            ss_res = torch.sum((true_ratings - pred_ratings) ** 2)
            ss_tot = torch.sum((true_ratings - torch.mean(true_ratings)) ** 2)
            r2_score = (1 - ss_res / ss_tot).item()
            
            # Convert back to original scale for interpretability
            rmse_original = rmse * 4  # Since we normalized by dividing by 4
            mae_original = mae * 4
            
            print(f"✅ SAE Results:")
            print(f"   📊 Architecture: {sae_params['n_movies']} → {' → '.join(map(str, sae_params['hidden_dims']))} → {sae_params['n_movies']}")
            print(f"   📏 Parameters: {sum(p.numel() for p in sae_model.parameters()):,}")
            print(f"   🎯 RMSE (normalized): {rmse:.4f}")
            print(f"   🎯 RMSE (original scale): {rmse_original:.4f}")
            print(f"   📐 MAE (normalized): {mae:.4f}")
            print(f"   📐 MAE (original scale): {mae_original:.4f}")
            print(f"   📈 R² Score: {r2_score:.4f}")
            print(f"   🔢 Test Ratings: {test_mask.sum().item():,}")
            
    except Exception as e:
        print(f"❌ SAE evaluation failed: {e}")
    
    # Evaluate RBM
    print("\n⚡ Evaluating RBM Model...")
    try:
        rbm_model, rbm_params = loader.load_rbm_model(use_best=True)
        rbm_model.eval()
        
        # Prepare data for RBM
        test_rbm = convert_ratings_to_binary(test_matrix_rbm)
        
        # Evaluate reconstruction
        metrics = evaluate_rbm_reconstruction(rbm_model, test_rbm)
        
        # Additional metrics using forward pass
        with torch.no_grad():
            # Use forward pass for reconstruction
            reconstructed = rbm_model(test_rbm)
            binary_accuracy = (reconstructed.round() == test_rbm).float().mean().item()
        
        print(f"✅ RBM Results:")
        print(f"   📊 Architecture: {rbm_params['visible_units']} visible ↔ {rbm_params['n_hidden']} hidden")
        print(f"   📏 Parameters: {sum(p.numel() for p in rbm_model.parameters()):,}")
        print(f"   🎯 Reconstruction Error: {metrics['reconstruction_error']:.4f}")
        print(f"   📐 Binary Accuracy: {binary_accuracy:.4f}")
        print(f"   🔢 Test Samples: {metrics['num_samples']:,}")
        
    except Exception as e:
        print(f"❌ RBM evaluation failed: {e}")
    
    print("\n✨ Quick evaluation completed!")

if __name__ == "__main__":
    quick_evaluation()
