"""
MLflow setup script for DagsHub integration.
Run this script to configure MLflow tracking.
"""

import mlflow
import dagshub
import os

def setup_mlflow():
    """Setup MLflow with DagsHub integration."""
    
    # Initialize DagsHub
    dagshub.init(
        repo_owner="danialasim",  # Replace with your username
        repo_name="movie-rec-system",  # Replace with your repo name
        mlflow=True
    )
    
    # Set tracking URI
    tracking_uri = "https://dagshub.com/your-username/movie-rec-system.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"MLflow configured with tracking URI: {tracking_uri}")
    print("Please update the config.yaml file with your actual DagsHub details.")

if __name__ == "__main__":
    setup_mlflow()
