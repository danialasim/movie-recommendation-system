# Movie Recommendation System - Complete MLOps Project Guide

## Project Overview

You want to build a comprehensive movie recommendation system with:
- Two different ML models for recommendations
- Complete MLOps pipeline with experiment tracking
- Data versioning and preprocessing
- Model deployment with APIs
- Monitoring and CI/CD automation
- Containerization and cloud deployment

## Project Architecture

```
movie-recommendation-system/
├── data/
│   ├── raw/                    # Original datasets (ml-1m, ml-100k)
│   │   ├── ml-1m/
│   │   │   ├── movies.dat
│   │   │   ├── users.dat
│   │   │   └── ratings.dat
│   │   └── ml-100k/
│   │       ├── u1.base         # Training set
│   │       └── u1.test         # Test set
│   ├── preprocessed/           # Cleaned and processed data
│   │   ├── user_item_matrix.pt # PyTorch tensor format
│   │   ├── train_set.pt
│   │   ├── test_set.pt
│   │   └── metadata.json       # Dataset statistics
│   └── external/               # External data sources
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py   # MovieLens data loading
│   │   ├── data_preprocessing.py # Matrix creation and normalization
│   │   └── data_utils.py       # Conversion utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py       # Abstract base model with MLflow integration
│   │   ├── autoencoder_model.py # SAE implementation with PyTorch
│   │   ├── rbm_model.py        # RBM implementation
│   │   └── model_utils.py      # Common model utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py           # Main training pipeline for both models
│   │   ├── evaluate.py        # RMSE, MAE, precision@k evaluation
│   │   └── hyperparameter_tuning.py # Optuna/MLflow integration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── flask_app.py       # Web interface with movie search
│   │   ├── fastapi_app.py     # REST API endpoints
│   │   └── prediction_service.py # Model inference logic
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Custom recommendation metrics
│   │   ├── logging_config.py  # Structured logging
│   │   └── model_monitor.py   # Model drift detection
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration management
│       ├── torch_utils.py     # PyTorch utilities
│       └── helpers.py         # General utilities
├── models/                     # Saved PyTorch model artifacts
│   ├── autoencoder/
│   │   ├── best_model.pt
│   │   └── model_metadata.json
│   └── rbm/
│       ├── best_model.pt
│       └── model_metadata.json
├── logs/                       # Structured application logs
├── tests/                      # Comprehensive test suite
│   ├── test_models/
│   ├── test_data/
│   └── test_api/
├── notebooks/                  # Analysis and experimentation
│   ├── eda.ipynb             # Exploratory data analysis
│   ├── model_comparison.ipynb # Model performance analysis
│   └── hyperparameter_analysis.ipynb
├── docker/
│   ├── Dockerfile.app         # Application container
│   ├── Dockerfile.training    # Training container
│   ├── docker-compose.yml     # Multi-service setup
│   └── requirements.txt
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alert_rules.yml
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── model_performance.json
│   │   │   └── system_metrics.json
│   │   └── provisioning/
│   └── docker-compose.monitoring.yml
├── templates/                  # HTML templates for Flask
│   ├── index.html
│   ├── recommendations.html
│   └── model_comparison.html
├── static/                     # Frontend assets
│   ├── css/
│   ├── js/
│   └── images/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       ├── model-training.yml  # Automated retraining
│       └── data-pipeline.yml
├── dvc.yaml                    # DVC pipeline with PyTorch models
├── dvc.lock
├── .dvcignore
├── mlflow_setup.py
├── requirements.txt            # PyTorch, MLflow, DagsHub dependencies
├── setup.py
├── README.md
└── config.yaml                 # Enhanced configuration for deep learning
```

## Step-by-Step Implementation Plan

### Phase 1: Project Setup and Data Management

#### Step 1: Enhanced Environment Setup
```bash
# Create virtual environment
python -m venv movie-rec-env
source movie-rec-env/bin/activate  # Linux/Mac

# Install PyTorch and deep learning dependencies
pip install torch torchvision torchaudio
pip install mlflow dagshub dvc pandas numpy scikit-learn 
pip install flask fastapi uvicorn docker prometheus-client grafana-api
pip install optuna tensorboard jupyter matplotlib seaborn
pip install pytest pytest-cov black flake8
```

#### Step 2: Initialize Version Control and DVC
```bash
# Initialize Git
git init
git remote add origin <your-github-repo>

# Initialize DVC with DagsHub
dvc init
dvc remote add -d origin <dagshub-repo-url>
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <token>
```

#### Step 3: Enhanced Data Ingestion and Preprocessing
Create `src/data/data_ingestion.py` to handle MovieLens datasets:
- Load ml-1m (movies.dat, users.dat, ratings.dat) and ml-100k (u1.base, u1.test)
- Validate data integrity and handle encoding issues
- Create user-item interaction matrices

Implement `src/data/data_preprocessing.py` for:
- **Matrix Conversion**: Transform rating data into user-item matrices
- **Data Normalization**: Handle missing ratings and create sparse tensors
- **Binary Transformation**: Convert ratings to binary (for RBM: >=3 = liked)
- **Train/Test Consistency**: Ensure same user/movie indices across sets
- **PyTorch Tensors**: Save preprocessed data as `.pt` files for efficient loading

Create `src/data/data_utils.py` with conversion utilities:
```python
def convert_to_matrix(data, nb_users, nb_movies):
    """Convert rating data to user-item matrix"""
    
def prepare_autoencoder_data(ratings_matrix):
    """Prepare continuous ratings for SAE"""
    
def prepare_rbm_data(ratings_matrix, threshold=3):
    """Convert to binary ratings for RBM"""
```

Save processed data to `data/preprocessed/` and track with DVC

### Phase 2: Model Development

#### Step 4: Create Enhanced Base Model Architecture
Implement `src/models/base_model.py` with PyTorch integration:
```python
class BaseRecommender(ABC):
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.mlflow_run = None
    
    @abstractmethod
    def build_model(self):
        """Build the neural network architecture"""
        
    @abstractmethod
    def train(self, train_data, val_data=None):
        """Train with MLflow logging"""
        
    @abstractmethod
    def predict(self, user_data):
        """Generate recommendations"""
        
    @abstractmethod
    def evaluate(self, test_data):
        """Compute evaluation metrics"""
        
    def save_model(self, path):
        """Save PyTorch model state"""
        
    def load_model(self, path):
        """Load trained model"""
        
    def log_metrics(self, metrics, step=None):
        """Log to MLflow"""
```

#### Step 5: Implement Two Deep Learning Recommendation Models

**Model 1: Stacked AutoEncoder (`src/models/autoencoder_model.py`)**
- Neural network with encoder-decoder architecture
- Hidden layers: Input → 20 → 10 → 20 → Output
- Uses sigmoid activation and MSE loss
- Handles sparse rating matrices effectively
- Good for learning non-linear user-item interactions

**Model 2: Restricted Boltzmann Machine (`src/models/rbm_model.py`)**
- Generative model with visible and hidden units
- Binary rating transformation (liked/not liked)
- Contrastive Divergence training (CD-k)
- Excellent for handling binary preferences
- Probabilistic approach to recommendation

#### Step 6: Enhanced Training Pipeline
Create `src/training/train.py` with advanced features:
- **Multi-Model Training**: Train both SAE and RBM simultaneously
- **Hyperparameter Optimization**: Use Optuna for automated tuning
- **Advanced MLflow Logging**: 
  - Model architecture diagrams
  - Training curves and loss plots
  - Hyperparameter importance plots
  - Model comparison metrics
- **GPU Support**: Automatic CUDA detection and utilization
- **Checkpointing**: Save best models during training
- **Early Stopping**: Prevent overfitting with validation monitoring

Example training configuration:
```python
# Hyperparameter spaces for optimization
sae_space = {
    'hidden_layers': trial.suggest_categorical('hidden_layers', [(20, 10), (50, 20), (100, 50)]),
    'learning_rate': trial.suggest_loguniform('lr', 1e-4, 1e-1),
    'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-1),
    'dropout': trial.suggest_uniform('dropout', 0.0, 0.5)
}

rbm_space = {
    'n_hidden': trial.suggest_int('n_hidden', 100, 500),
    'learning_rate': trial.suggest_loguniform('lr', 1e-4, 1e-1),
    'cd_k': trial.suggest_int('cd_k', 1, 20),
    'batch_size': trial.suggest_categorical('batch_size', [50, 100, 200])
}
```

### Phase 3: MLflow and Experiment Tracking

#### Step 7: MLflow Setup with DagsHub
```python
# mlflow_setup.py
import mlflow
import dagshub

dagshub.init(repo_owner="username", repo_name="movie-rec-system", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/username/movie-rec-system.mlflow")
```

#### Step 8: Experiment Tracking
- Log hyperparameters for both models
- Track training/validation metrics (RMSE, MAE, Precision@K, Recall@K)
- Save model artifacts and preprocessing pipelines
- Compare model performance in DagsHub MLflow UI

### Phase 4: Model Evaluation and Selection

#### Step 9: Advanced Model Evaluation (`src/training/evaluate.py`)
Implement comprehensive recommendation system metrics:

**Traditional Metrics:**
- RMSE (Root Mean Square Error) for rating prediction accuracy
- MAE (Mean Absolute Error) for average prediction error
- R² Score for explained variance

**Ranking Metrics:**
- Precision@K: Fraction of recommended items that are relevant
- Recall@K: Fraction of relevant items that are recommended  
- NDCG@K: Normalized Discounted Cumulative Gain for ranking quality
- MAP@K: Mean Average Precision across all users

**Coverage and Diversity Metrics:**
- Catalog Coverage: Percentage of items that can be recommended
- Intra-list Diversity: Average dissimilarity within recommendation lists
- Novelty: Average popularity of recommended items (lower = more novel)

**Business Metrics:**
- Click-through Rate simulation
- Conversion rate estimation
- User satisfaction scores

**Cold Start Analysis:**
- Performance on new users with few ratings
- Performance on new movies with few ratings

Create evaluation reports and log all metrics to MLflow with visualizations

### Phase 5: Logging and Monitoring

#### Step 10: Implement Comprehensive Logging
Create `src/monitoring/logging_config.py`:
- Configure Python logging with different levels
- Create separate loggers for data, training, and API
- Log to both files and console
- Structured logging with JSON format

### Phase 6: API Development

#### Step 11: Enhanced Flask Web Interface (`src/api/flask_app.py`)
Create a sophisticated web application:

**Frontend Features:**
- **Movie Search**: Real-time search with autocomplete
- **User Profile**: Rate movies and build preference profile  
- **Recommendations Display**: Side-by-side comparison of SAE vs RBM recommendations
- **Recommendation Explanations**: Show why movies were recommended
- **Interactive Rating**: Ajax-based rating system
- **Model Performance**: Live metrics dashboard
- **A/B Testing UI**: Compare model recommendations for same user

**Backend Features:**
- **Session Management**: Track user interactions
- **Real-time Inference**: Fast model prediction with caching
- **Recommendation Caching**: Redis integration for performance
- **Analytics Tracking**: Log user interactions for model improvement
- **Model Switching**: Toggle between SAE and RBM recommendations

**Templates Structure:**
```
templates/
├── base.html           # Base template with navigation
├── index.html          # Home page with movie search
├── profile.html        # User rating interface  
├── recommendations.html # Recommendation results
├── model_compare.html  # Side-by-side model comparison
└── analytics.html      # Model performance dashboard
```

#### Step 12: Advanced FastAPI Service (`src/api/fastapi_app.py`)
Build a production-ready API with advanced features:

**Core Endpoints:**
```python
@app.post("/recommend/sae")
async def get_sae_recommendations(user_ratings: UserRatings)

@app.post("/recommend/rbm") 
async def get_rbm_recommendations(user_ratings: UserRatings)

@app.post("/recommend/hybrid")
async def get_hybrid_recommendations(user_ratings: UserRatings, weights: ModelWeights)

@app.get("/models/compare")
async def compare_models(user_id: int)

@app.post("/feedback")
async def log_user_feedback(feedback: UserFeedback)
```

**Advanced Features:**
- **Hybrid Recommendations**: Combine SAE and RBM predictions
- **Batch Predictions**: Handle multiple users efficiently
- **Explanation API**: Return recommendation reasoning
- **Model Health Checks**: Monitor model performance
- **Rate Limiting**: Prevent API abuse
- **Authentication**: JWT token-based auth
- **Caching**: Redis for frequently requested predictions
- **Monitoring**: Prometheus metrics integration
- **A/B Testing**: Route users to different models for testing

**Performance Optimizations:**
- **Async Processing**: Non-blocking I/O operations
- **Model Caching**: Keep models in memory
- **Connection Pooling**: Efficient database connections
- **Response Compression**: Gzip compression for large responses

### Phase 7: Containerization

#### Step 13: Docker Implementation
Create `docker/Dockerfile`:
- Multi-stage build for optimization
- Include all dependencies and models
- Configure for production deployment

Create `docker/docker-compose.yml`:
- Application container
- Database (if needed)
- Redis for caching
- Network configuration

### Phase 8: Monitoring and Observability

#### Step 14: Advanced Prometheus Metrics (`src/monitoring/metrics.py`)
Create comprehensive monitoring for deep learning models:

**Model Performance Metrics:**
```python
# Recommendation accuracy metrics
recommendation_accuracy = Gauge('recommendation_accuracy', 'Model accuracy score')
model_rmse = Gauge('model_rmse', 'Root Mean Square Error', ['model_type'])
model_precision_at_k = Gauge('precision_at_k', 'Precision at K', ['model_type', 'k'])

# Training metrics
training_loss = Gauge('training_loss', 'Current training loss', ['model_type', 'epoch'])
validation_loss = Gauge('validation_loss', 'Validation loss', ['model_type', 'epoch'])
model_convergence_time = Histogram('model_training_duration_seconds', 'Training time', ['model_type'])

# Inference performance
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction response time', ['model_type'])
predictions_total = Counter('predictions_total', 'Total predictions made', ['model_type', 'endpoint'])
model_memory_usage = Gauge('model_memory_usage_bytes', 'Memory usage by model', ['model_type'])

# Data drift detection
feature_drift = Gauge('feature_drift_score', 'Data drift detection score')
prediction_drift = Gauge('prediction_drift_score', 'Prediction drift score')

# User interaction metrics
user_satisfaction = Gauge('user_satisfaction_score', 'Average user satisfaction')
recommendation_click_rate = Gauge('recommendation_click_rate', 'Click-through rate')
api_error_rate = Counter('api_errors_total', 'API errors', ['endpoint', 'error_type'])
```

**GPU/Hardware Monitoring:**
- CUDA memory usage tracking
- GPU utilization metrics  
- Model inference throughput
- Batch processing efficiency

#### Step 15: Advanced Grafana Dashboards
Create specialized dashboards for deep learning recommendation systems:

**Model Performance Dashboard (`model_performance.json`):**
- Real-time RMSE, MAE, and accuracy trends
- Side-by-side SAE vs RBM performance comparison
- Training convergence curves with loss plots
- Hyperparameter impact visualization
- Model inference latency percentiles
- GPU utilization and memory usage

**User Behavior Analytics Dashboard:**
- Recommendation click-through rates
- User rating distribution over time
- Popular movie trends and cold start performance
- A/B testing results visualization
- User satisfaction scores
- Session duration and engagement metrics

**System Health Dashboard:**
- API response times and error rates
- Database query performance
- Cache hit/miss rates
- Model memory consumption
- Alert status and notifications
- Data pipeline health

**Business Intelligence Dashboard:**
- Revenue impact predictions
- User retention correlations
- Recommendation diversity metrics
- Coverage and long-tail performance
- Seasonal trends in recommendations
- Model ROI analysis

**Alert Configurations:**
- Model accuracy drop alerts
- High API error rate warnings
- GPU memory exhaustion alerts
- Data drift detection notifications

### Phase 9: Testing

#### Step 16: Comprehensive Testing
- Unit tests for all modules
- Integration tests for API endpoints
- Model performance tests
- Data validation tests

### Phase 10: CI/CD Pipeline

#### Step 17: GitHub Actions Workflows

**Continuous Integration (`.github/workflows/ci.yml`)**
- Run tests on every push
- Code quality checks (linting, formatting)
- Security vulnerability scanning

**Continuous Deployment (`.github/workflows/cd.yml`)**
- Automated deployment on main branch
- Docker image building and pushing
- Model retraining triggers

**Data Pipeline (`.github/workflows/data-pipeline.yml`)**
- Automated data preprocessing
- Model retraining on new data
- DVC pipeline execution

## Key Features and Benefits

### 1. **MLOps Pipeline**
- Automated model training and deployment
- Experiment tracking and model versioning
- Data pipeline automation with DVC

### 2. **Model Management**
- Two complementary recommendation approaches
- A/B testing framework for model comparison
- Automated model selection based on performance

### 3. **Data Management**
- Version-controlled datasets with DVC
- Automated data preprocessing pipeline
- Data validation and quality checks

### 4. **Monitoring and Observability**
- Real-time performance monitoring with Prometheus
- Visual dashboards with Grafana
- Comprehensive logging throughout the pipeline

### 5. **Deployment Options**
- Web interface for end-users (Flask)
- API endpoints for integration (FastAPI)
- Containerized deployment with Docker

### 6. **Automation**
- End-to-end CI/CD pipeline
- Automated testing and quality assurance
- Triggered retraining on data changes

## Configuration Management

The enhanced `config.yaml` file will contain:
```yaml
# Dataset configuration
data:
  raw_path: "data/raw"
  processed_path: "data/preprocessed" 
  ml_1m_path: "data/raw/ml-1m"
  ml_100k_path: "data/raw/ml-100k"
  
# Model configurations based on your working implementations
models:
  autoencoder:
    architecture:
      input_dim: 1682  # Number of movies in ml-100k
      hidden_layers: [20, 10]  # Your tested architecture
      output_dim: 1682
    training:
      learning_rate: 0.01
      epochs: 200
      optimizer: "RMSprop"
      weight_decay: 0.5
      criterion: "MSELoss"
      device: "cuda"  # Auto-detect GPU
    
  rbm:
    architecture:
      n_visible: 1682  # Number of movies
      n_hidden: 300    # Your tested hidden units
    training:
      learning_rate: 0.01
      epochs: 200
      batch_size: 100
      cd_k: 20  # Contrastive Divergence steps
      binary_threshold: 3  # Rating >= 3 = liked
      
# MLflow configuration for DagsHub
mlflow:
  experiment_name: "deep-movie-recommendations"
  tracking_uri: "https://dagshub.com/username/movie-rec-system.mlflow"
  run_name_prefix: "deep_learning"
  log_models: true
  log_artifacts: true
  
# API configuration  
api:
  flask:
    host: "0.0.0.0"
    port: 5000
    debug: false
  fastapi:
    host: "0.0.0.0" 
    port: 8000
    workers: 4
    
# Monitoring setup
monitoring:
  prometheus:
    port: 9090
    scrape_interval: "15s"
  grafana:
    port: 3000
    admin_password: "secure_password"
  logging:
    level: "INFO"
    format: "json"
    
# Training configuration
training:
  validation_split: 0.2
  early_stopping:
    patience: 10
    min_delta: 0.001
  checkpointing:
    save_best: true
    save_frequency: 10
  hyperparameter_tuning:
    n_trials: 50
    optimization_direction: "minimize"
    
# Evaluation metrics
evaluation:
  metrics: ["rmse", "mae", "precision_at_k", "recall_at_k", "ndcg_at_k"]
  k_values: [5, 10, 20]
  test_users: "all"  # or specific number for faster evaluation
```

## Execution Workflow

1. **Data Processing**: Raw data → Preprocessed data (tracked with DVC)
2. **Model Training**: Train both models with MLflow tracking
3. **Model Evaluation**: Compare performance and select best model
4. **API Deployment**: Deploy selected model via Flask/FastAPI
5. **Monitoring**: Track performance with Prometheus/Grafana
6. **Automation**: CI/CD pipeline handles updates and redeployment

This comprehensive setup will give you a production-ready movie recommendation system with full MLOps capabilities, monitoring, and automated deployment pipeline.