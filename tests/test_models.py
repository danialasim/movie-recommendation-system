#!/usr/bin/env python3
"""
Simple test script to verify model loading works.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_model_loading():
    """Test if models can be loaded successfully."""
    print("🧪 Testing model loading...")
    
    try:
        # Test importing model loader
        print("📦 Importing ModelLoader...")
        from load_trained_models import ModelLoader
        print("✅ ModelLoader imported successfully")
        
        # Initialize model loader
        print("🔧 Initializing ModelLoader...")
        model_loader = ModelLoader("models")
        print("✅ ModelLoader initialized")
        
        # Test SAE model loading
        print("🤖 Testing SAE model loading...")
        try:
            sae_model, sae_params = model_loader.load_sae_model(use_best=True)
            print("✅ SAE model loaded successfully")
            print(f"📊 SAE params: {sae_params}")
        except Exception as e:
            print(f"❌ SAE model loading failed: {e}")
        
        # Test RBM model loading
        print("🎲 Testing RBM model loading...")
        try:
            rbm_model, rbm_params = model_loader.load_rbm_model(use_best=True)
            print("✅ RBM model loaded successfully")
            print(f"📊 RBM params: {rbm_params}")
        except Exception as e:
            print(f"❌ RBM model loading failed: {e}")
    
    except Exception as e:
        print(f"💥 Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()
