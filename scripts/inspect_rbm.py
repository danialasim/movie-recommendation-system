#!/usr/bin/env python3
"""
Script to inspect the saved RBM model and determine the correct architecture.
"""

import torch
from pathlib import Path

def inspect_rbm_model():
    """Inspect the saved RBM model to determine its architecture."""
    model_path = Path("models/rbm/best_model.pt")
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"🔍 Inspecting RBM model: {model_path}")
    
    try:
        # Load the state dict
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("📊 Checkpoint structure:")
        print(f"  Type: {type(checkpoint)}")
        print(f"  Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("  Found 'state_dict' key")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("  Found 'model_state_dict' key")
            else:
                state_dict = checkpoint
                print("  Using checkpoint as state_dict")
        else:
            state_dict = checkpoint
        
        print(f"\n📊 Model state dict keys and shapes:")
        for key, tensor in state_dict.items():
            if hasattr(tensor, 'shape'):
                print(f"  {key}: {tensor.shape}")
            else:
                print(f"  {key}: {type(tensor)} (no shape attribute)")
        
        # Extract architecture info from weight shapes
        if 'W' in state_dict and hasattr(state_dict['W'], 'shape'):
            W_shape = state_dict['W'].shape
            n_hidden = W_shape[0]
            n_visible = W_shape[1]
            print(f"\n🏗️  Inferred architecture:")
            print(f"  Visible units (movies): {n_visible}")
            print(f"  Hidden units: {n_hidden}")
        
        if 'b_visible' in state_dict and hasattr(state_dict['b_visible'], 'shape'):
            b_visible_shape = state_dict['b_visible'].shape
            print(f"  Visible bias shape: {b_visible_shape}")
        
        if 'b_hidden' in state_dict and hasattr(state_dict['b_hidden'], 'shape'):
            b_hidden_shape = state_dict['b_hidden'].shape
            print(f"  Hidden bias shape: {b_hidden_shape}")
            
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_rbm_model()
