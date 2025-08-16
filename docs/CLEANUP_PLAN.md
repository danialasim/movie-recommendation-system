# Movie Recommendation System - Project Cleanup and Organization

## Files to Remove (Irrelevant/Temporary)
- .DS_Store files
- __pycache__ directories  
- *.pyc files
- movies-env/ (virtual environment - should not be in Git)
- mlruns/ (MLflow runs - should be in .gitignore)
- logs/ (log files - should be in .gitignore) 
- .env (environment file - should be in .gitignore)

## Files to Organize

### Root Level Files (to be moved or consolidated)
- test_*.py files → tests/ directory
- debug_*.py files → scripts/ or remove if not needed
- demo_*.py files → examples/ directory
- quick_*.py files → scripts/ directory
- Duplicate guide files → docs/ directory

### Scripts to Keep in Root
- setup.py (Python package setup)
- start_training.py (main training script)
- load_trained_models.py (model loading utility)

### Configuration Files
- config.yaml (main config)
- requirements.txt (dependencies)
- dvc.yaml (DVC pipeline)
- docker-compose.*.yml (Docker configs)

### Documentation
- README.md
- *.md files → docs/ directory

### New Structure
```
movie-recommendation-system/
├── README.md
├── requirements.txt
├── setup.py
├── config.yaml
├── dvc.yaml
├── .gitignore
├── .dockerignore
├── docker-compose.quick.yml
├── docker-manager.sh
├── src/
│   ├── api/
│   ├── models/
│   ├── training/
│   ├── utils/
│   ├── config/
│   └── monitoring/
├── docker/
├── monitoring/
├── tests/
├── scripts/
├── docs/
├── examples/
├── data/
├── models/
├── static/
├── templates/
└── notebooks/
```
