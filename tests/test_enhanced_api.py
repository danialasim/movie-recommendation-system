#!/usr/bin/env python3
"""
Test script for enhanced FastAPI with Prometheus metrics.
"""

import subprocess
import sys
import time
import urllib.request
import urllib.parse
import json

def install_dependencies():
    """Install required dependencies for monitoring."""
    print("📦 Installing monitoring dependencies...")
    
    dependencies = [
        'prometheus-client',
        'psutil'
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    return True

def test_enhanced_api():
    """Test the enhanced FastAPI with metrics."""
    print("\n🧪 Testing Enhanced FastAPI...")
    
    base_url = "http://127.0.0.1:8001"
    
    # Test health endpoint
    try:
        print("🏥 Testing health endpoint...")
        with urllib.request.urlopen(f'{base_url}/health') as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print("✅ Health Check Response:")
                print(f"  Status: {data.get('status')}")
                print(f"  Models: {data.get('models_loaded')}")
                print(f"  Metrics Enabled: {data.get('metrics_enabled')}")
                print(f"  Uptime: {data.get('uptime_seconds')}s")
            else:
                print(f"❌ Health check failed: {response.status}")
                return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test metrics endpoint
    try:
        print("\n📊 Testing metrics endpoint...")
        with urllib.request.urlopen(f'{base_url}/metrics') as response:
            if response.status == 200:
                metrics_data = response.read().decode()
                print("✅ Metrics endpoint working")
                print(f"  Metrics data length: {len(metrics_data)} characters")
                
                # Count number of metrics
                metric_lines = [line for line in metrics_data.split('\n') if line and not line.startswith('#')]
                print(f"  Number of metric entries: {len(metric_lines)}")
            else:
                print(f"❌ Metrics failed: {response.status}")
    except Exception as e:
        print(f"⚠️  Metrics endpoint error: {e}")
    
    # Test recommendations with metrics
    try:
        print("\n🎬 Testing recommendations with metrics...")
        data = {
            "movies": [1, 2, 3],
            "model": "sae",
            "num_recommendations": 5
        }
        
        req_data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(
            f'{base_url}/recommend',
            data=req_data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                result = json.loads(response.read().decode())
                print("✅ Enhanced Recommendations Response:")
                print(f"  Model Used: {result.get('model_used')}")
                print(f"  Success: {result.get('success')}")
                print(f"  Recommendations: {len(result.get('recommendations', []))}")
                
                # Show metrics
                metrics = result.get('metrics', {})
                if metrics:
                    print("  📊 Performance Metrics:")
                    for key, value in metrics.items():
                        print(f"    {key}: {value}")
                
                # Show first recommendation
                recs = result.get('recommendations', [])
                if recs:
                    first_rec = recs[0]
                    print(f"  🎯 First Recommendation: {first_rec.get('title')}")
                    print(f"    Rating: {first_rec.get('rating')}/5")
                    print(f"    Confidence: {first_rec.get('confidence')*100:.1f}%")
            else:
                print(f"❌ Recommendations failed: {response.status}")
                return False
    except Exception as e:
        print(f"❌ Recommendations error: {e}")
        return False
    
    return True

def generate_test_traffic():
    """Generate some test traffic for metrics."""
    print("\n🚗 Generating test traffic for metrics...")
    
    base_url = "http://127.0.0.1:8001"
    
    test_requests = [
        {"movies": [1, 2, 3], "model": "sae", "num_recommendations": 5},
        {"movies": [4, 5, 6], "model": "rbm", "num_recommendations": 3},
        {"movies": [7, 8, 9], "model": "sae", "num_recommendations": 10},
        {"movies": [10, 11, 12], "model": "rbm", "num_recommendations": 7}
    ]
    
    for i, request_data in enumerate(test_requests):
        try:
            print(f"  Request {i+1}/4: {request_data['model']} model...")
            
            req_data = json.dumps(request_data).encode('utf-8')
            req = urllib.request.Request(
                f'{base_url}/recommend',
                data=req_data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print(f"    ✅ Success")
                else:
                    print(f"    ❌ Failed: {response.status}")
            
            time.sleep(0.5)  # Small delay between requests
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    print("✅ Test traffic generation completed")

def main():
    """Main function."""
    print("🎬 Enhanced FastAPI with Prometheus Metrics - Test Suite")
    print("=" * 60)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    print("\n" + "=" * 60)
    
    # Test the API
    if not test_enhanced_api():
        print("❌ API tests failed")
        return
    
    print("\n" + "=" * 60)
    
    # Generate test traffic
    generate_test_traffic()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\n📊 Next Steps:")
    print("1. Start the monitoring stack: ./start_monitoring.sh")
    print("2. Open Grafana: http://localhost:3000 (admin/admin123)")
    print("3. View Prometheus: http://localhost:9090")
    print("4. Check metrics: http://127.0.0.1:8001/metrics")
    print("\n📈 The API is now generating metrics that can be visualized in Grafana!")

if __name__ == "__main__":
    main()
