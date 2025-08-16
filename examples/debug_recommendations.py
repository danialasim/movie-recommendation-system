#!/usr/bin/env python3
"""
Debug script to test recommendation logic with different inputs.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_recommendation_logic():
    """Test if different inputs give different outputs."""
    print("🔍 Testing recommendation logic...")
    
    try:
        from load_trained_models import ModelLoader
        
        # Load models
        model_loader = ModelLoader("models")
        sae_model, sae_params = model_loader.load_sae_model(use_best=True)
        rbm_model, rbm_params = model_loader.load_rbm_model(use_best=True)
        
        print("✅ Models loaded successfully")
        
        # Test with different user profiles
        test_cases = [
            ("User 1 - Action movies", [1, 2, 3]),  # First few movies
            ("User 2 - Different movies", [10, 20, 30]),  # Different movies
            ("User 3 - High rated movies", [100, 200, 300]),  # Different set
        ]
        
        for user_name, liked_movies in test_cases:
            print(f"\n🎭 Testing {user_name} with movies {liked_movies}")
            
            # Create user vector
            user_vector = torch.zeros(1682)
            for movie_id in liked_movies:
                if 1 <= movie_id <= 1682:
                    user_vector[movie_id - 1] = 0.8
            
            # Test SAE
            with torch.no_grad():
                user_input = user_vector.unsqueeze(0)
                sae_output = sae_model(user_input).squeeze(0)
                
                # Get top 5 predictions (excluding input movies)
                sae_sorted = torch.argsort(sae_output, descending=True)[:20]
                sae_recs = []
                for idx in sae_sorted:
                    movie_id = idx.item() + 1
                    if movie_id not in liked_movies:
                        sae_recs.append((movie_id, sae_output[idx].item()))
                        if len(sae_recs) >= 5:
                            break
                
                print(f"  SAE top 5: {[(mid, f'{score:.3f}') for mid, score in sae_recs]}")
            
            # Test RBM
            user_vector_binary = torch.zeros(1682)
            for movie_id in liked_movies:
                if 1 <= movie_id <= 1682:
                    user_vector_binary[movie_id - 1] = 1.0
            
            with torch.no_grad():
                user_input_binary = user_vector_binary.unsqueeze(0)
                rbm_output = rbm_model(user_input_binary).squeeze(0)
                
                # Get top 5 predictions (excluding input movies)
                rbm_sorted = torch.argsort(rbm_output, descending=True)[:20]
                rbm_recs = []
                for idx in rbm_sorted:
                    movie_id = idx.item() + 1
                    if movie_id not in liked_movies:
                        rbm_recs.append((movie_id, rbm_output[idx].item()))
                        if len(rbm_recs) >= 5:
                            break
                
                print(f"  RBM top 5: {[(mid, f'{score:.3f}') for mid, score in rbm_recs]}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recommendation_logic()
