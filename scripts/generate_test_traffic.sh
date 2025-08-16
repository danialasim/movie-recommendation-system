#!/bin/bash

echo "Generating test traffic for movie recommendation API..."

# Test different endpoints multiple times
for i in {1..10}; do
    echo "Request $i: Health check"
    curl -s http://127.0.0.1:8001/health > /dev/null
    
    echo "Request $i: Get recommendations"
    curl -s -X POST "http://127.0.0.1:8001/recommend" \
         -H "Content-Type: application/json" \
         -d '{"user_id": '$((RANDOM % 943 + 1))', "num_recommendations": 5}' > /dev/null
    
    # Random delay between requests
    sleep $((RANDOM % 3 + 1))
done

echo "Generating some load tests..."
for i in {1..20}; do
    echo "Load test $i"
    curl -s -X POST "http://127.0.0.1:8001/recommend" \
         -H "Content-Type: application/json" \
         -d '{"user_id": '$((RANDOM % 943 + 1))', "num_recommendations": 10, "model_type": "sae"}' > /dev/null &
done

wait

echo "Test traffic generation complete!"
echo "Check Prometheus at: http://localhost:9090"
echo "Check Grafana at: http://localhost:3000 (admin/admin123)"
