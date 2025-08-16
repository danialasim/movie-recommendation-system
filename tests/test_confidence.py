#!/usr/bin/env python3
"""
Test the improved confidence calculations in Flask app.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_improved_confidence():
    """Test the Flask app with improved confidence calculations."""
    
    try:
        from src.api.flask_app import create_app
        
        print("🧪 Testing Improved Confidence Calculations")
        print("=" * 50)
        
        app = create_app()
        
        with app.test_client() as client:
            # Test different scenarios
            test_scenarios = [
                {
                    "name": "Light user (3 movies)",
                    "movies": [1, 2, 3],
                    "model": "sae"
                },
                {
                    "name": "Moderate user (7 movies)", 
                    "movies": [1, 10, 50, 100, 150, 200, 250],
                    "model": "sae"
                },
                {
                    "name": "Light user with RBM",
                    "movies": [1, 2, 3],
                    "model": "rbm"
                },
                {
                    "name": "Moderate user with RBM",
                    "movies": [1, 10, 50, 100, 150, 200, 250],
                    "model": "rbm"
                }
            ]
            
            for scenario in test_scenarios:
                print(f"\n🎬 {scenario['name']}:")
                
                response = client.post('/recommend', json={
                    'movies': scenario['movies'],
                    'model': scenario['model'],
                    'num_recommendations': 5
                })
                
                if response.status_code == 200:
                    data = response.get_json()
                    recommendations = data.get('recommendations', [])
                    
                    print(f"   Status: ✅ {len(recommendations)} recommendations")
                    
                    if recommendations:
                        confidences = [rec.get('confidence', 0) for rec in recommendations]
                        avg_confidence = sum(confidences) / len(confidences)
                        min_confidence = min(confidences)
                        max_confidence = max(confidences)
                        
                        print(f"   Confidence range: {min_confidence:.3f} - {max_confidence:.3f}")
                        print(f"   Average confidence: {avg_confidence:.3f}")
                        
                        # Show top 3 recommendations
                        for i, rec in enumerate(recommendations[:3]):
                            title = rec.get('title', 'Unknown')[:25]
                            rating = rec.get('rating', 0)
                            confidence = rec.get('confidence', 0)
                            print(f"     {i+1}. {title} | Rating: {rating:.2f} | Confidence: {confidence:.3f}")
                else:
                    print(f"   Status: ❌ Error {response.status_code}")
                    
        print("\n📊 Summary:")
        print("- Confidence scores should now be higher and more realistic")
        print("- SAE should show confidence range roughly 0.4-0.9")
        print("- RBM should show confidence range roughly 0.3-0.8")
        print("- More liked movies should generally lead to higher confidence")
        
    except Exception as e:
        print(f"❌ Error testing confidence: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_confidence()
