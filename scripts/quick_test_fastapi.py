#!/usr/bin/env python3
"""
Quick FastAPI test without requests dependency.
"""

import urllib.request
import urllib.parse
import json

def test_fastapi_health():
    """Test FastAPI health endpoint using urllib."""
    try:
        print("🧪 Testing FastAPI Health Check...")
        
        # Test health endpoint
        with urllib.request.urlopen('http://127.0.0.1:8001/health') as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print("✅ Health Check Response:")
                print(json.dumps(data, indent=2))
            else:
                print(f"❌ Health check failed with status: {response.status}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure FastAPI server is running on port 8001")

def test_recommendations():
    """Test recommendations endpoint."""
    try:
        print("\n🎬 Testing Recommendations...")
        
        # Prepare request data
        data = {
            "movies": [1, 2, 3],
            "model": "sae",
            "num_recommendations": 5
        }
        
        # Prepare request
        req_data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(
            'http://127.0.0.1:8001/recommend',
            data=req_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Send request
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                result = json.loads(response.read().decode())
                print("✅ Recommendations Response:")
                print(f"Model Used: {result.get('model_used')}")
                print(f"Number of Recommendations: {len(result.get('recommendations', []))}")
                
                # Show first few recommendations
                for i, rec in enumerate(result.get('recommendations', [])[:3]):
                    print(f"  {i+1}. {rec.get('title')} - Rating: {rec.get('rating')}/5, Confidence: {rec.get('confidence')*100:.1f}%")
            else:
                print(f"❌ Recommendations failed with status: {response.status}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 FastAPI Quick Test")
    print("=" * 30)
    
    test_fastapi_health()
    test_recommendations()
    
    print("\n🎉 FastAPI testing completed!")
    print("📚 Visit http://127.0.0.1:8001/docs for interactive API documentation")
