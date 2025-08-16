#!/usr/bin/env python3
"""
Test the exact API response format to debug frontend issues.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_api_response_format():
    """Test the exact format of API responses."""
    
    try:
        from src.api.flask_app import create_app
        
        print("🔍 Testing API Response Format")
        print("=" * 40)
        
        app = create_app()
        
        with app.test_client() as client:
            # Test with the exact same request format as the frontend
            test_requests = [
                {
                    "name": "Frontend format (movies array)",
                    "data": {
                        'movies': [1, 2, 3],
                        'model': 'sae',
                        'num_recommendations': 3
                    }
                },
                {
                    "name": "Alternative format (liked_movies)",
                    "data": {
                        'liked_movies': [1, 2, 3],
                        'model_type': 'sae',
                        'num_recommendations': 3
                    }
                }
            ]
            
            for test in test_requests:
                print(f"\n🧪 {test['name']}:")
                print(f"Request: {json.dumps(test['data'], indent=2)}")
                
                response = client.post('/recommend', 
                                     json=test['data'],
                                     content_type='application/json')
                
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.get_json()
                    print(f"Response Keys: {list(data.keys())}")
                    
                    if 'recommendations' in data:
                        recs = data['recommendations']
                        print(f"Number of recommendations: {len(recs)}")
                        
                        if recs:
                            first_rec = recs[0]
                            print(f"First recommendation keys: {list(first_rec.keys())}")
                            print(f"First recommendation:")
                            print(json.dumps(first_rec, indent=2))
                            
                            # Check specific fields
                            rating = first_rec.get('rating')
                            confidence = first_rec.get('confidence')
                            print(f"Rating value: {rating} (type: {type(rating)})")
                            print(f"Confidence value: {confidence} (type: {type(confidence)})")
                    else:
                        print("No recommendations in response")
                        print(f"Full response: {json.dumps(data, indent=2)}")
                else:
                    print(f"Error response: {response.get_data(as_text=True)}")
                
                print("-" * 40)
                
    except Exception as e:
        print(f"❌ Error testing API format: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_response_format()
