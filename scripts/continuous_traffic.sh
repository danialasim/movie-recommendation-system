#!/bin/bash

echo "🚀 Generating continuous test traffic for monitoring..."

# Run in background to generate steady traffic
while true; do
    # Health checks
    curl -s http://127.0.0.1:8001/health > /dev/null
    
    # Random recommendation requests
    user_id=$((RANDOM % 943 + 1))
    curl -s -X POST "http://127.0.0.1:8001/recommend" \
         -H "Content-Type: application/json" \
         -d "{\"user_id\": $user_id, \"num_recommendations\": 5}" > /dev/null
    
    # Random delay between 1-3 seconds
    sleep $((RANDOM % 3 + 1))
done
