#!/usr/bin/env python3
"""
Test the improved recommendation logic with the Flask app functions.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

def test_improved_recommendations():
    """Test the improved recommendation functions."""
    print("🧪 Testing improved recommendation logic...")
    
    try:
        # Import the Flask app components
        from src.api.flask_app import MovieRecommendationService, get_sae_recommendations, get_rbm_recommendations
        
        # Initialize the recommendation service
        print("🔧 Initializing recommendation service...")
        global recommendation_service
        recommendation_service = MovieRecommendationService()
        
        # Test cases with different movie preferences
        test_cases = [
            ("Action fan", [1, 2, 3]),  # Toy Story, GoldenEye, Four Weddings
            ("Drama lover", [50, 100, 150]),  # Different genre preferences
            ("Comedy enthusiast", [7, 8, 9]),  # Twelve Monkeys, Babe, Dead Man Walking
        ]
        
        for user_type, liked_movies in test_cases:
            print(f"\n🎭 Testing {user_type} with movies {liked_movies}")
            
            # Test SAE recommendations
            sae_recs = get_sae_recommendations(liked_movies, 5)
            print(f"  SAE recommendations:")
            for i, rec in enumerate(sae_recs[:5], 1):
                print(f"    {i}. {rec['title']} (Rating: {rec['rating']}, Confidence: {rec['confidence']})")
            
            # Test RBM recommendations
            rbm_recs = get_rbm_recommendations(liked_movies, 5)
            print(f"  RBM recommendations:")
            for i, rec in enumerate(rbm_recs[:5], 1):
                print(f"    {i}. {rec['title']} (Rating: {rec['rating']}, Confidence: {rec['confidence']})")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_recommendations()
