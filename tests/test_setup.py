#!/usr/bin/env python3
"""
Test script to verify that the training pipeline is set up correctly.

This script runs basic tests on the models and data preprocessing
to ensure everything is working before full training.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("🧪 Testing data preprocessing...")
    
    try:
        from data.data_preprocessing import MovieLensPreprocessor
        from data.data_utils import DataValidator
        
        # Create small test data
        test_data = {
            'user_id': [1, 1, 2, 2, 3, 3],
            'movie_id': [1, 2, 1, 3, 2, 3],
            'rating': [5, 3, 4, 2, 5, 4]
        }
        
        import pandas as pd
        test_df = pd.DataFrame(test_data)
        
        # Validate data
        DataValidator.validate_ratings_df(test_df)
        
        # Test preprocessor
        preprocessor = MovieLensPreprocessor()
        preprocessor.create_user_movie_mappings(test_df)
        matrix = preprocessor.create_user_item_matrix(test_df)
        
        print(f"✅ Data preprocessing test passed - Matrix shape: {matrix.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        return False

def test_sae_model():
    """Test SAE model functionality."""
    print("🧪 Testing SAE model...")
    
    try:
        from models.autoencoder_model import StackedAutoEncoder, normalize_ratings_for_sae
        
        # Create test data
        n_users, n_movies = 10, 20
        test_matrix = torch.randint(0, 6, (n_users, n_movies)).float()
        test_matrix[test_matrix == 0] = 0  # Some unrated items
        
        # Normalize data
        normalized_matrix = normalize_ratings_for_sae(test_matrix)
        
        # Create model
        model = StackedAutoEncoder(n_movies=n_movies, hidden_dims=[10, 5])
        
        # Test forward pass
        output = model(normalized_matrix)
        assert output.shape == normalized_matrix.shape
        
        # Test encoding
        encoded = model.encode(normalized_matrix)
        assert encoded.shape == (n_users, 5)  # Bottleneck dimension
        
        print(f"✅ SAE model test passed - Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ SAE model test failed: {e}")
        return False

def test_rbm_model():
    """Test RBM model functionality."""
    print("🧪 Testing RBM model...")
    
    try:
        from models.rbm_model import RestrictedBoltzmannMachine, convert_ratings_to_binary
        
        # Create test data
        n_users, n_movies = 10, 20
        test_matrix = torch.randint(1, 6, (n_users, n_movies)).float()
        
        # Convert to binary
        binary_matrix = convert_ratings_to_binary(test_matrix, threshold=3.0)
        
        # Create model
        model = RestrictedBoltzmannMachine(n_movies=n_movies, n_hidden=15)
        
        # Test forward pass
        output = model(binary_matrix)
        assert output.shape == binary_matrix.shape
        
        # Test hidden sampling
        prob_hidden, hidden_sample = model.sample_hidden(binary_matrix)
        assert prob_hidden.shape == (n_users, 15)
        assert hidden_sample.shape == (n_users, 15)
        
        print(f"✅ RBM model test passed - Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ RBM model test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation functionality."""
    print("🧪 Testing evaluation...")
    
    try:
        from training.evaluate import ModelEvaluator
        from models.autoencoder_model import StackedAutoEncoder
        
        # Create test data
        n_users, n_movies = 50, 100
        test_matrix = torch.rand(n_users, n_movies)
        test_matrix[test_matrix < 0.7] = 0  # Sparse matrix
        
        # Create simple model
        model = StackedAutoEncoder(n_movies=n_movies, hidden_dims=[20, 10])
        
        # Test evaluation
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_sae(model, test_matrix)
        
        assert 'rmse' in metrics or 'num_test_ratings' in metrics
        
        print(f"✅ Evaluation test passed - Metrics: {list(metrics.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - TESTS")
    print("=" * 60)
    print()
    
    tests = [
        ("Data Preprocessing", test_data_preprocessing),
        ("SAE Model", test_sae_model),
        ("RBM Model", test_rbm_model),
        ("Evaluation", test_evaluation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to start training.")
        print()
        print("Next steps:")
        print("1. Download MovieLens data to data/raw/")
        print("2. Run: python start_training.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed:")
        print("pip install torch pandas numpy scikit-learn mlflow optuna matplotlib seaborn")

if __name__ == "__main__":
    main()
