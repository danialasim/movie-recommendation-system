#!/usr/bin/env python3
"""
Movie Recommendation System - Project Structure Setup Script

This script creates the complete folder structure for the MLOps movie recommendation system
as defined in the movie_rec_project_guide.md and mlops_movie_rec_guide.md.

Usage:
    python setup_project_structure.py

Author: Generated for Movie Recommendation MLOps Project
Date: August 2025
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

def create_directory_structure() -> Dict[str, List[str]]:
    """
    Define the complete directory structure for the movie recommendation system.
    
    Returns:
        Dict containing directory paths and their required files
    """
    
    structure = {
        # Data directories
        "data/raw/ml-1m": [
            ".gitkeep",  # Keep empty directories in git
        ],
        "data/raw/ml-100k": [
            ".gitkeep",
        ],
        "data/preprocessed": [
            ".gitkeep",
        ],
        "data/external": [
            ".gitkeep",
        ],
        
        # Source code directories
        "src/data": [
            "__init__.py",
            "data_ingestion.py",
            "data_preprocessing.py", 
            "data_utils.py",
        ],
        "src/models": [
            "__init__.py",
            "base_model.py",
            "autoencoder_model.py",
            "rbm_model.py",
            "ensemble_model.py",
            "content_based.py",
            "explanation.py",
            "model_utils.py",
        ],
        "src/training": [
            "__init__.py",
            "train.py",
            "evaluate.py",
            "hyperparameter_tuning.py",
            "model_validation.py",
        ],
        "src/api": [
            "__init__.py",
            "flask_app.py",
            "fastapi_app.py",
            "prediction_service.py",
        ],
        "src/monitoring": [
            "__init__.py",
            "metrics.py",
            "model_monitor.py",
            "logging_config.py",
            "business_metrics.py",
            "ab_testing.py",
            "grafana_dashboards.py",
        ],
        "src/utils": [
            "__init__.py",
            "config.py",
            "torch_utils.py",
            "helpers.py",
            "model_optimization.py",
            "caching.py",
        ],
        
        # Model artifacts
        "models/autoencoder": [
            ".gitkeep",
        ],
        "models/rbm": [
            ".gitkeep", 
        ],
        "models/ensemble": [
            ".gitkeep",
        ],
        
        # Logs
        "logs": [
            ".gitkeep",
        ],
        
        # Tests
        "tests/test_models": [
            "__init__.py",
            "test_autoencoder.py",
            "test_rbm.py",
            "test_ensemble.py",
        ],
        "tests/test_data": [
            "__init__.py",
            "test_preprocessing.py",
            "test_ingestion.py",
        ],
        "tests/test_api": [
            "__init__.py",
            "test_flask_app.py",
            "test_fastapi_app.py",
        ],
        "tests/test_training": [
            "__init__.py",
            "test_evaluation.py",
            "test_hyperparameter_tuning.py",
        ],
        "tests/integration": [
            "__init__.py",
            "test_end_to_end.py",
            "test_api_integration.py",
        ],
        
        # Notebooks
        "notebooks": [
            "eda.ipynb",
            "model_comparison.ipynb", 
            "hyperparameter_analysis.ipynb",
            "data_exploration.ipynb",
        ],
        
        # Docker configuration
        "docker": [
            "Dockerfile.app",
            "Dockerfile.training",
            "docker-compose.yml",
            "docker-compose.monitoring.yml",
            "requirements.txt",
            "entrypoint-training.sh",
            "entrypoint-app.sh",
        ],
        
        # Monitoring configuration
        "monitoring/prometheus": [
            "prometheus.yml",
            "alert_rules.yml",
        ],
        "monitoring/grafana/dashboards": [
            "model_performance.json",
            "system_metrics.json",
            "business_metrics.json",
        ],
        "monitoring/grafana/provisioning/dashboards": [
            "dashboard.yml",
        ],
        "monitoring/grafana/provisioning/datasources": [
            "prometheus.yml",
        ],
        
        # Web templates
        "templates": [
            "base.html",
            "index.html", 
            "recommendations.html",
            "model_comparison.html",
            "profile.html",
            "analytics.html",
        ],
        
        # Static assets
        "static/css": [
            "custom.css",
            "dashboard.css",
        ],
        "static/js": [
            "app.js",
            "charts.js",
            "recommendations.js",
        ],
        "static/images": [
            ".gitkeep",
        ],
        
        # GitHub workflows
        ".github/workflows": [
            "ci.yml",
            "cd.yml", 
            "model-training.yml",
            "data-pipeline.yml",
        ],
        
        # SQL scripts
        "sql": [
            "init.sql",
            "create_tables.sql",
        ],
        
        # Configuration
        "config": [
            "config.yaml",
            "logging.yaml",
            "prometheus.yml",
        ],
        
        # Scripts
        "scripts": [
            "download_data.sh",
            "setup_environment.sh",
            "run_training.sh",
            "deploy.sh",
        ],
    }
    
    return structure

def create_file_with_content(file_path: Path, file_type: str) -> None:
    """
    Create a file with appropriate initial content based on file type.
    
    Args:
        file_path: Path to the file to create
        file_type: Type of file (extension or special name)
    """
    
    content = ""
    
    if file_type == "__init__.py":
        module_name = file_path.parent.name
        content = f'"""{module_name.title()} module for movie recommendation system."""\n'
        
    elif file_type == ".gitkeep":
        content = "# This file keeps the directory in git\n"
        
    elif file_type.endswith(".py"):
        filename = file_path.stem
        content = f'''"""
{filename.replace("_", " ").title()} module for movie recommendation system.

This module provides functionality for {filename.replace("_", " ")}.
"""

# TODO: Implement {filename} functionality
pass
'''
        
    elif file_type.endswith(".yml") or file_type.endswith(".yaml"):
        content = f"# {file_path.name} configuration\n# TODO: Add configuration settings\n"
        
    elif file_type.endswith(".html"):
        template_name = file_path.stem
        content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{template_name.title()} - Movie Recommender</title>
</head>
<body>
    <!-- TODO: Implement {template_name} template -->
    <h1>{template_name.title()}</h1>
</body>
</html>
'''
        
    elif file_type.endswith(".css"):
        content = f'''/* {file_path.name} - Movie Recommendation System Styles */

/* TODO: Add CSS styles for {file_path.stem} */
'''
        
    elif file_type.endswith(".js"):
        content = f'''// {file_path.name} - Movie Recommendation System JavaScript

// TODO: Implement {file_path.stem} functionality
'''
        
    elif file_type.endswith(".sh"):
        content = f'''#!/bin/bash
# {file_path.name} - Movie Recommendation System Script

# TODO: Implement {file_path.stem} script functionality
'''
        
    elif file_type.endswith(".sql"):
        content = f'''-- {file_path.name} - Movie Recommendation System Database Script

-- TODO: Implement {file_path.stem} SQL functionality
'''
        
    elif file_type.endswith(".json"):
        content = '''{\n    "TODO": "Add JSON configuration"\n}\n'''
        
    elif file_type.endswith(".ipynb"):
        content = '''{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    # Write content to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_project_structure(base_path: str = ".") -> None:
    """
    Create the complete project structure.
    
    Args:
        base_path: Base directory to create the structure in
    """
    
    base_dir = Path(base_path)
    structure = create_directory_structure()
    
    print("🎬 Creating Movie Recommendation System Project Structure...")
    print("=" * 60)
    
    created_dirs = 0
    created_files = 0
    
    # Create directories and files
    for dir_path, files in structure.items():
        full_dir_path = base_dir / dir_path
        
        # Create directory
        full_dir_path.mkdir(parents=True, exist_ok=True)
        created_dirs += 1
        print(f"📁 Created directory: {dir_path}")
        
        # Create files in directory
        for file_name in files:
            file_path = full_dir_path / file_name
            
            # Only create file if it doesn't exist
            if not file_path.exists():
                create_file_with_content(file_path, file_name)
                created_files += 1
                print(f"   📄 Created file: {dir_path}/{file_name}")
            else:
                print(f"   ⚠️  File exists: {dir_path}/{file_name}")
    
    # Create root-level configuration files
    root_files = {
        "requirements.txt": '''# Core ML and Deep Learning
torch==1.13.1
torchvision==0.14.1
torchaudio==0.13.1
numpy==1.21.0
pandas==1.5.0
scikit-learn==1.1.0

# MLOps and Experiment Tracking
mlflow==2.0.1
dagshub==0.3.1
dvc==2.30.0
optuna==3.0.0
tensorboard==2.10.0

# API and Web Framework
flask==2.2.0
fastapi==0.88.0
uvicorn==0.20.0
jinja2==3.1.2
pydantic==1.10.0

# Monitoring and Metrics
prometheus-client==0.15.0
grafana-api==1.0.3
redis==4.3.4

# Data Processing
scipy==1.9.0
matplotlib==3.6.0
seaborn==0.12.0

# Development and Testing
pytest==7.2.0
pytest-cov==4.0.0
black==22.10.0
flake8==6.0.0
jupyter==1.0.0

# Deployment
docker==6.0.0
gunicorn==20.1.0
psycopg2-binary==2.9.5
''',
        
        "config.yaml": '''# Movie Recommendation System Configuration

# Dataset configuration
data:
  raw_path: "data/raw"
  processed_path: "data/preprocessed"
  ml_1m_path: "data/raw/ml-1m"
  ml_100k_path: "data/raw/ml-100k"
  
# Model configurations
models:
  autoencoder:
    architecture:
      input_dim: 1682
      hidden_layers: [20, 10]
      output_dim: 1682
    training:
      learning_rate: 0.01
      epochs: 200
      optimizer: "RMSprop"
      weight_decay: 0.5
      
  rbm:
    architecture:
      n_visible: 1682
      n_hidden: 300
    training:
      learning_rate: 0.01
      epochs: 200
      cd_k: 20
      
# MLflow configuration
mlflow:
  experiment_name: "movie-recommendations"
  tracking_uri: "https://dagshub.com/username/movie-rec-system.mlflow"
  
# API configuration
api:
  flask:
    host: "0.0.0.0"
    port: 5000
  fastapi:
    host: "0.0.0.0"
    port: 8000
    
# Monitoring
monitoring:
  prometheus:
    port: 9090
  grafana:
    port: 3000
''',
        
        "dvc.yaml": '''stages:
  data_preprocessing:
    cmd: python src/data/data_preprocessing.py
    deps:
    - data/raw/
    - src/data/data_preprocessing.py
    outs:
    - data/preprocessed/
    
  train_sae:
    cmd: python src/training/train.py --model sae
    deps:
    - data/preprocessed/
    - src/training/train.py
    - src/models/autoencoder_model.py
    outs:
    - models/autoencoder/
    metrics:
    - metrics/sae_metrics.json
    
  train_rbm:
    cmd: python src/training/train.py --model rbm
    deps:
    - data/preprocessed/
    - src/training/train.py
    - src/models/rbm_model.py
    outs:
    - models/rbm/
    metrics:
    - metrics/rbm_metrics.json
    
  evaluate:
    cmd: python src/training/evaluate.py
    deps:
    - models/autoencoder/
    - models/rbm/
    - src/training/evaluate.py
    metrics:
    - metrics/evaluation_results.json
''',
        
        "setup.py": '''from setuptools import setup, find_packages

setup(
    name="movie-recommendation-system",
    version="1.0.0",
    description="MLOps Movie Recommendation System with Deep Learning",
    author="Movie Rec Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.1.0",
        "mlflow>=2.0.0",
        "fastapi>=0.88.0",
        "flask>=2.2.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "monitoring": ["prometheus-client", "grafana-api"],
    },
)
''',
        
        "README.md": '''# Movie Recommendation System - MLOps Project

A comprehensive movie recommendation system built with deep learning models, featuring complete MLOps pipeline with monitoring, deployment, and continuous integration.

## Features

- **Deep Learning Models**: Stacked AutoEncoder (SAE) and Restricted Boltzmann Machine (RBM)
- **MLOps Pipeline**: Complete CI/CD with DVC, MLflow, and automated deployment
- **Monitoring**: Real-time performance monitoring with Prometheus and Grafana
- **APIs**: Both Flask web interface and FastAPI for production use
- **Containerization**: Docker support for easy deployment

## Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv movie-rec-env
   source movie-rec-env/bin/activate
   pip install -r requirements.txt
   ```

2. **Download Data**:
   ```bash
   bash scripts/download_data.sh
   ```

3. **Train Models**:
   ```bash
   python src/training/train.py --model sae
   python src/training/train.py --model rbm
   ```

4. **Run API**:
   ```bash
   python src/api/fastapi_app.py
   ```

## Project Structure

See `movie_rec_project_guide.md` and `mlops_movie_rec_guide.md` for detailed documentation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
''',
        
        ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
movie-rec-env/

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt

# MLflow
mlruns/
mlartifacts/

# DVC
/data/raw/
/data/preprocessed/
.dvc

# Logs
logs/
*.log

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Model artifacts (large files)
models/*.pt
models/*.pth
models/*.joblib

# Monitoring data
prometheus_data/
grafana_data/
''',
        
        ".dvcignore": '''# Add patterns of files dvc should ignore, which could improve
# the performance. Learn more at
# https://dvc.org/doc/user-guide/dvcignore

*.log
.git/
__pycache__/
.pytest_cache/
''',
        
        "mlflow_setup.py": '''"""
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
        repo_owner="your-username",  # Replace with your username
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
''',
    }
    
    print("\n📋 Creating root configuration files...")
    for filename, content in root_files.items():
        file_path = base_dir / filename
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files += 1
            print(f"📄 Created: {filename}")
        else:
            print(f"⚠️  Exists: {filename}")
    
    print("\n" + "=" * 60)
    print(f"✅ Project structure created successfully!")
    print(f"📁 Directories created: {created_dirs}")
    print(f"📄 Files created: {created_files}")
    print("\n🎯 Next steps:")
    print("1. Update config.yaml with your DagsHub credentials")
    print("2. Run: bash scripts/download_data.sh")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Initialize DVC: dvc init")
    print("5. Start building your recommendation system!")

def main():
    """Main function to run the project setup."""
    
    print("🎬 Movie Recommendation System - Project Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not any(f.name in ['movie-recommendation-system', 'movie_rec_project_guide.md'] 
               for f in current_dir.iterdir()):
        print("⚠️  Warning: You might not be in the project root directory.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    try:
        create_project_structure()
    except Exception as e:
        print(f"❌ Error creating project structure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
