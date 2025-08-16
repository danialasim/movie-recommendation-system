#!/usr/bin/env python3
"""
Quick test for rating values in Flask app.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_rating_values():
    """Test that rating values are being calculated correctly."""
    
    try:
        from src.api.flask_app import create_app
        
        print("🔍 Testing Rating Values")
        print("=" * 30)
        
        app = create_app()
        
        with app.test_client() as client:
            # Test SAE recommendations
            response = client.post('/recommend', json={
                'movies': [1, 2, 3],
                'model': 'sae',
                'num_recommendations': 3
            })
            
            if response.status_code == 200:
                data = response.get_json()
                recommendations = data.get('recommendations', [])
                
                print("✅ SAE Recommendations:")
                for i, rec in enumerate(recommendations):
                    movie_id = rec.get('movie_id', 'Unknown')
                    title = rec.get('title', 'Unknown')[:30]
                    rating = rec.get('rating', 'undefined')
                    confidence = rec.get('confidence', 'undefined')
                    print(f"  {i+1}. {title}")
                    print(f"     Movie ID: {movie_id}")
                    print(f"     Rating: {rating}/5")
                    print(f"     Confidence: {confidence}")
                    print(f"     Rating type: {type(rating)}")
                    print()
            else:
                print(f"❌ SAE test failed with status {response.status_code}")
                
            # Test RBM recommendations
            response = client.post('/recommend', json={
                'movies': [1, 2, 3],
                'model': 'rbm',
                'num_recommendations': 3
            })
            
            if response.status_code == 200:
                data = response.get_json()
                recommendations = data.get('recommendations', [])
                
                print("✅ RBM Recommendations:")
                for i, rec in enumerate(recommendations):
                    movie_id = rec.get('movie_id', 'Unknown')
                    title = rec.get('title', 'Unknown')[:30]
                    rating = rec.get('rating', 'undefined')
                    confidence = rec.get('confidence', 'undefined')
                    print(f"  {i+1}. {title}")
                    print(f"     Movie ID: {movie_id}")
                    print(f"     Rating: {rating}/5")
                    print(f"     Confidence: {confidence}")
                    print(f"     Rating type: {type(rating)}")
                    print()
            else:
                print(f"❌ RBM test failed with status {response.status_code}")
                
    except Exception as e:
        print(f"❌ Error testing ratings: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rating_values()
