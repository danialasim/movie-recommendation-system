#!/usr/bin/env python3
"""
Test FastAPI endpoints with sample requests.
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🏥 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Health check failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_model_info():
    """Test the model info endpoint."""
    print("\n🧠 Testing Model Info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model Info Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Model info failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_movie_search():
    """Test the movie search endpoint."""
    print("\n🔍 Testing Movie Search...")
    try:
        search_data = {
            "query": "Toy Story",
            "limit": 5
        }
        
        response = requests.post(
            f"{BASE_URL}/search",
            json=search_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Request: {json.dumps(search_data, indent=2)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Movie Search Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Movie search failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_recommendations():
    """Test the recommendations endpoint."""
    print("\n🎬 Testing Movie Recommendations...")
    
    test_cases = [
        {
            "name": "SAE Model Test",
            "data": {
                "movies": [1, 2, 3],
                "model": "sae",
                "num_recommendations": 5
            }
        },
        {
            "name": "RBM Model Test",
            "data": {
                "movies": [1, 2, 3],
                "model": "rbm",
                "num_recommendations": 5
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}:")
        try:
            response = requests.post(
                f"{BASE_URL}/recommend",
                json=test_case['data'],
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Request: {json.dumps(test_case['data'], indent=2)}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Recommendations Response:")
                print(f"Model Used: {data.get('model_used')}")
                print(f"Number of Recommendations: {len(data.get('recommendations', []))}")
                
                # Show first few recommendations
                for i, rec in enumerate(data.get('recommendations', [])[:3]):
                    print(f"  {i+1}. {rec.get('title')} - Rating: {rec.get('rating')}/5, Confidence: {rec.get('confidence')*100:.1f}%")
                    
            else:
                print(f"❌ Recommendations failed: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_popular_movies():
    """Test the popular movies endpoint."""
    print("\n⭐ Testing Popular Movies...")
    try:
        response = requests.get(f"{BASE_URL}/popular?limit=5", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Popular Movies Response:")
            for i, movie in enumerate(data[:5]):
                print(f"  {i+1}. {movie.get('title')} (ID: {movie.get('movie_id')})")
        else:
            print(f"❌ Popular movies failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_server_running():
    """Check if the FastAPI server is running."""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main function to test all FastAPI endpoints."""
    print("🧪 FastAPI Movie Recommendation System Tests")
    print("=" * 50)
    
    # Check if server is running
    if not check_server_running():
        print("❌ FastAPI server is not running!")
        print("🚀 Please start the server first with: python run_fastapi.py")
        return
    
    print("✅ FastAPI server is running")
    print("🔗 Testing endpoints at:", BASE_URL)
    
    # Run all tests
    test_health_check()
    test_model_info()
    test_movie_search()
    test_recommendations()
    test_popular_movies()
    
    print("\n" + "=" * 50)
    print("🎉 FastAPI testing completed!")
    print("📚 Visit http://localhost:8000/docs for interactive API documentation")

if __name__ == "__main__":
    main()
