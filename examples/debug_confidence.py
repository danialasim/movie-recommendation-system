#!/usr/bin/env python3
"""
Debug script to check model predictions and confidence scores.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from load_trained_models import ModelLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_predictions():
    """Test raw model predictions to understand confidence issues."""
    
    try:
        # Load models
        model_loader = ModelLoader("models")
        
        print("🔬 Testing Model Predictions and Confidence Calculations")
        print("=" * 60)
        
        # Load SAE model
        try:
            sae_model, sae_params = model_loader.load_sae_model(use_best=True)
            sae_model.eval()
            print("✅ SAE model loaded successfully")
            
            # Test SAE predictions
            print("\n📊 SAE Model Analysis:")
            
            # Create test user vectors
            test_cases = [
                ("User likes movies 1, 2, 3", [1, 2, 3]),
                ("User likes movies 100, 200, 300", [100, 200, 300]),
                ("User likes movies 1, 50, 100, 150", [1, 50, 100, 150])
            ]
            
            for case_name, liked_movies in test_cases:
                print(f"\n🎬 {case_name}:")
                
                # Create user vector
                user_vector = torch.zeros(1682)
                for movie_id in liked_movies:
                    if 1 <= movie_id <= 1682:
                        user_vector[movie_id - 1] = 0.8
                
                # Get baseline (empty user)
                baseline_vector = torch.zeros(1682)
                
                with torch.no_grad():
                    # User predictions
                    user_input = user_vector.unsqueeze(0)
                    user_predictions = sae_model(user_input).squeeze(0)
                    
                    # Baseline predictions
                    baseline_input = baseline_vector.unsqueeze(0)
                    baseline_predictions = sae_model(baseline_input).squeeze(0)
                
                # Calculate preference scores
                preference_scores = user_predictions - baseline_predictions
                
                # Analyze the predictions
                print(f"   Raw prediction range: {user_predictions.min().item():.4f} to {user_predictions.max().item():.4f}")
                print(f"   Baseline range: {baseline_predictions.min().item():.4f} to {baseline_predictions.max().item():.4f}")
                print(f"   Preference score range: {preference_scores.min().item():.4f} to {preference_scores.max().item():.4f}")
                
                # Get top 5 recommendations
                sorted_indices = torch.argsort(preference_scores, descending=True)
                
                print(f"   Top 5 recommendations:")
                for i, idx in enumerate(sorted_indices[:5]):
                    movie_id = int(idx.item()) + 1
                    if movie_id not in liked_movies:
                        preference_score = float(preference_scores[idx].item())
                        raw_prediction = float(user_predictions[idx].item())
                        
                        # Our current confidence calculation
                        base_confidence = min(0.95, max(0.1, preference_score * 2))
                        prediction_confidence = min(0.9, raw_prediction)
                        position_penalty = 0.05 * i
                        final_confidence = (base_confidence * 0.4 + prediction_confidence * 0.4 + (1 - position_penalty) * 0.2)
                        final_confidence = max(0.1, min(0.95, final_confidence))
                        
                        print(f"     Movie {movie_id}: pred={raw_prediction:.4f}, pref={preference_score:.4f}, conf={final_confidence:.3f}")
                        
                        if i >= 2:  # Show first 3 valid recommendations
                            break
            
        except Exception as e:
            print(f"❌ SAE model test failed: {e}")
        
        # Load RBM model
        try:
            rbm_model, rbm_params = model_loader.load_rbm_model(use_best=True)
            rbm_model.eval()
            print("\n✅ RBM model loaded successfully")
            
            # Test RBM predictions
            print("\n📊 RBM Model Analysis:")
            
            for case_name, liked_movies in test_cases:
                print(f"\n🎬 {case_name}:")
                
                # Create user vector (binary for RBM)
                user_vector = torch.zeros(1682)
                for movie_id in liked_movies:
                    if 1 <= movie_id <= 1682:
                        user_vector[movie_id - 1] = 1.0
                
                # Get baseline (empty user)
                baseline_vector = torch.zeros(1682)
                
                with torch.no_grad():
                    # User predictions
                    user_input = user_vector.unsqueeze(0)
                    user_probabilities = rbm_model(user_input).squeeze(0)
                    
                    # Baseline predictions
                    baseline_input = baseline_vector.unsqueeze(0)
                    baseline_probabilities = rbm_model(baseline_input).squeeze(0)
                
                # Calculate preference scores
                preference_scores = user_probabilities - baseline_probabilities
                
                # Analyze the predictions
                print(f"   Raw probability range: {user_probabilities.min().item():.4f} to {user_probabilities.max().item():.4f}")
                print(f"   Baseline range: {baseline_probabilities.min().item():.4f} to {baseline_probabilities.max().item():.4f}")
                print(f"   Preference score range: {preference_scores.min().item():.4f} to {preference_scores.max().item():.4f}")
                
                # Get top 5 recommendations
                sorted_indices = torch.argsort(preference_scores, descending=True)
                
                print(f"   Top 5 recommendations:")
                for i, idx in enumerate(sorted_indices[:5]):
                    movie_id = int(idx.item()) + 1
                    if movie_id not in liked_movies:
                        preference_score = float(preference_scores[idx].item())
                        raw_probability = float(user_probabilities[idx].item())
                        
                        # Our current RBM confidence calculation
                        base_confidence = min(0.9, max(0.15, preference_score * 3))
                        probability_confidence = min(0.85, raw_probability * 1.2)
                        position_penalty = 0.04 * i
                        final_confidence = (base_confidence * 0.5 + probability_confidence * 0.3 + (1 - position_penalty) * 0.2)
                        final_confidence = max(0.15, min(0.9, final_confidence))
                        
                        print(f"     Movie {movie_id}: prob={raw_probability:.4f}, pref={preference_score:.4f}, conf={final_confidence:.3f}")
                        
                        if i >= 2:  # Show first 3 valid recommendations
                            break
            
        except Exception as e:
            print(f"❌ RBM model test failed: {e}")
        
        print("\n🔍 Confidence Analysis:")
        print("- If raw predictions are very low (< 0.1), that's expected for sparse data")
        print("- If preference scores are negative or very small, confidence will be low")
        print("- The models might be outputting realistic but low confidence predictions")
        print("- This could indicate the models are being conservative, which is good!")
        
        # Test with a known high-confidence scenario
        print("\n🧪 Testing High-Confidence Scenario:")
        print("Creating a user who likes many popular movies...")
        
        # Create a user who likes the first 20 movies (likely popular)
        high_conf_user = torch.zeros(1682)
        for i in range(20):
            high_conf_user[i] = 0.8  # SAE input
        
        if 'sae_model' in locals():
            with torch.no_grad():
                predictions = sae_model(high_conf_user.unsqueeze(0)).squeeze(0)
                baseline = sae_model(torch.zeros(1682).unsqueeze(0)).squeeze(0)
                pref_scores = predictions - baseline
                
                # Get best recommendation
                best_idx = torch.argmax(pref_scores)
                if best_idx.item() >= 20:  # Not a liked movie
                    best_pref = float(pref_scores[best_idx].item())
                    best_pred = float(predictions[best_idx].item())
                    
                    # Calculate confidence
                    base_conf = min(0.95, max(0.1, best_pref * 2))
                    pred_conf = min(0.9, best_pred)
                    final_conf = (base_conf * 0.4 + pred_conf * 0.4 + 0.2)
                    final_conf = max(0.1, min(0.95, final_conf))
                    
                    print(f"SAE Best recommendation confidence: {final_conf:.3f}")
                    print(f"  - Raw prediction: {best_pred:.4f}")
                    print(f"  - Preference score: {best_pref:.4f}")
        
    except Exception as e:
        print(f"❌ Error in confidence testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_predictions()
