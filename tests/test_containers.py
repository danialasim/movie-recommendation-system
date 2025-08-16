#!/usr/bin/env python3
"""
Quick test script for the containerized Movie Recommendation System
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8001"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"

def test_api_health() -> bool:
    """Test if the API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ API Health Check: PASSED")
            return True
        else:
            print(f"❌ API Health Check: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ API Health Check: FAILED (Error: {e})")
        return False

def test_api_recommendations() -> bool:
    """Test the recommendations endpoint"""
    try:
        # Test data
        test_payload = {
            "user_id": 1,
            "num_recommendations": 5,
            "model_type": "autoencoder"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/recommend", 
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "recommendations" in data and len(data["recommendations"]) > 0:
                print("✅ Recommendations Endpoint: PASSED")
                print(f"   └─ Received {len(data['recommendations'])} recommendations")
                return True
            else:
                print("❌ Recommendations Endpoint: FAILED (No recommendations returned)")
                return False
        else:
            print(f"❌ Recommendations Endpoint: FAILED (Status: {response.status_code})")
            print(f"   └─ Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Recommendations Endpoint: FAILED (Error: {e})")
        return False

def test_prometheus_metrics() -> bool:
    """Test if Prometheus is collecting metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        if response.status_code == 200:
            metrics_text = response.text
            if "http_requests_total" in metrics_text:
                print("✅ Prometheus Metrics: PASSED")
                return True
            else:
                print("❌ Prometheus Metrics: FAILED (No metrics found)")
                return False
        else:
            print(f"❌ Prometheus Metrics: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Prometheus Metrics: FAILED (Error: {e})")
        return False

def test_prometheus_server() -> bool:
    """Test if Prometheus server is accessible"""
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/label/__name__/values", timeout=10)
        if response.status_code == 200:
            print("✅ Prometheus Server: PASSED")
            return True
        else:
            print(f"❌ Prometheus Server: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Prometheus Server: FAILED (Error: {e})")
        return False

def test_grafana_server() -> bool:
    """Test if Grafana server is accessible"""
    try:
        response = requests.get(f"{GRAFANA_URL}/api/health", timeout=10)
        if response.status_code == 200:
            print("✅ Grafana Server: PASSED")
            return True
        else:
            print(f"❌ Grafana Server: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Grafana Server: FAILED (Error: {e})")
        return False

def load_test() -> bool:
    """Perform a basic load test"""
    print("\n🔄 Running load test (10 requests)...")
    success_count = 0
    total_requests = 10
    
    for i in range(total_requests):
        try:
            test_payload = {
                "user_id": i % 5 + 1,  # Rotate between users 1-5
                "num_recommendations": 3,
                "model_type": "autoencoder"
            }
            
            response = requests.post(
                f"{API_BASE_URL}/recommend", 
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                success_count += 1
                print(f"   Request {i+1}: ✅")
            else:
                print(f"   Request {i+1}: ❌ (Status: {response.status_code})")
                
            time.sleep(0.1)  # Small delay between requests
            
        except Exception as e:
            print(f"   Request {i+1}: ❌ (Error: {e})")
    
    success_rate = (success_count / total_requests) * 100
    print(f"\n📊 Load Test Results: {success_count}/{total_requests} successful ({success_rate:.1f}%)")
    
    return success_rate >= 80  # Consider 80% success rate as passing

def main():
    """Main test function"""
    print("🎬 Movie Recommendation System - Container Test Suite")
    print("=" * 60)
    
    tests = [
        ("API Health Check", test_api_health),
        ("API Recommendations", test_api_recommendations),
        ("Prometheus Metrics", test_prometheus_metrics),
        ("Prometheus Server", test_prometheus_server),
        ("Grafana Server", test_grafana_server),
    ]
    
    print("\n🧪 Running integration tests...\n")
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed_tests += 1
        print()
    
    # Run load test
    if load_test():
        passed_tests += 1
        total_tests += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your Movie Recommendation System is ready!")
        print("\n🔗 Quick Links:")
        print(f"   • API Documentation: {API_BASE_URL}/docs")
        print(f"   • Prometheus: {PROMETHEUS_URL}")
        print(f"   • Grafana: {GRAFANA_URL} (admin/admin123)")
        return True
    else:
        print("❌ Some tests failed. Please check the logs and configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
