# 🎬 Movie Recommendation System - Training Guide

## 🚀 Quick Start Training

You now have a complete implementation of both **Stacked AutoEncoder (SAE)** and **Restricted Boltzmann Machine (RBM)** models for movie recommendations. Here's how to start training:

### 1. 📋 Prerequisites

Make sure you have Python 3.8+ and the required dependencies:

```bash
# Install dependencies
pip install -r requirements.txt

# Or install key packages manually:
pip install torch pandas numpy scikit-learn mlflow optuna matplotlib seaborn
```

### 2. 📁 Download MovieLens Data

Download and extract MovieLens datasets to the `data/raw/` directory:

```bash
# Create data directories
mkdir -p data/raw

# Download ml-100k (smaller dataset, good for testing)
cd data/raw
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip

# Download ml-1m (larger dataset, better for final models)
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

Your data structure should look like:
```
data/
├── raw/
│   ├── ml-100k/
│   │   ├── u1.base
│   │   ├── u1.test
│   │   └── u.item
│   └── ml-1m/
│       ├── movies.dat
│       ├── ratings.dat
│       └── users.dat
```

### 3. 🧪 Test Your Setup

Run the test script to verify everything is working:

```bash
python test_setup.py
```

This will test:
- ✅ Data preprocessing functionality
- ✅ SAE model architecture
- ✅ RBM model architecture  
- ✅ Evaluation metrics

### 4. 🏃‍♂️ Start Training

#### Option A: Simple Training (Recommended for first time)
```bash
python start_training.py
```

This will:
- Use default hyperparameters
- Train both SAE and RBM models
- Use ml-100k dataset (smaller, faster)
- Save models to `models/` directory
- Track experiments with MLflow

#### Option B: Advanced Training with CLI
```bash
# Train both models with hyperparameter optimization
python src/training/train.py --dataset ml-1m --optimize

# Train only SAE
python src/training/train.py --model sae --dataset ml-100k

# Train only RBM
python src/training/train.py --model rbm --dataset ml-1m

# Use custom config
python src/training/train.py --config config/training_config.json
```

### 5. 📊 Monitor Training

#### MLflow UI (Local)
```bash
# Start MLflow UI
mlflow ui

# Open browser to: http://localhost:5000
```

#### DagsHub (Optional - for remote tracking)
1. Create account at [dagshub.com](https://dagshub.com)
2. Create a new repository
3. Update `config/training_config.json`:
```json
{
  "dagshub_repo": "username/movie-rec-system",
  "mlflow_tracking_uri": "https://dagshub.com/username/movie-rec-system.mlflow"
}
```

## 🎯 What Each Model Does

### 🧠 Stacked AutoEncoder (SAE)
- **Input**: User rating vectors (normalized 0-1)
- **Architecture**: Input → 20 → 10 → 20 → Output
- **Learning**: Reconstructs user preferences, learns compressed representations
- **Output**: Continuous rating predictions (1-5 scale)
- **Best for**: Precise rating prediction, user similarity analysis

### ⚡ Restricted Boltzmann Machine (RBM)
- **Input**: Binary user preferences (like/dislike)
- **Architecture**: 100 hidden ↔ visible units (no hidden-hidden connections)
- **Learning**: Contrastive Divergence, models probability distributions
- **Output**: Binary preference probabilities
- **Best for**: Binary recommendation decisions, generative modeling

## 📈 Expected Training Results

### SAE Training
```
Epoch 0: Train Loss = 0.8245, Val Loss = 0.7932
Epoch 10: Train Loss = 0.3421, Val Loss = 0.3654
Epoch 50: Train Loss = 0.1876, Val Loss = 0.2103
...
Final RMSE: ~0.85-1.2 (good), 0.6-0.9 (excellent)
```

### RBM Training
```
Epoch 0: Train Loss = 0.6831, Val Loss = 0.6745
Epoch 10: Train Loss = 0.4932, Val Loss = 0.5123
Epoch 50: Train Loss = 0.3654, Val Loss = 0.3821
...
Final Reconstruction Error: ~0.3-0.5 (good)
```

## 🔧 Configuration Options

Edit `config/training_config.json` to customize training:

```json
{
  "dataset": "ml-100k",  // or "ml-1m"
  "sae": {
    "hidden_dims": [20, 10],  // Architecture
    "learning_rate": 0.01,
    "num_epochs": 200,
    "batch_size": 128
  },
  "rbm": {
    "n_hidden": 100,
    "learning_rate": 0.01,
    "cd_k": 10,  // Contrastive Divergence steps
    "num_epochs": 200
  },
  "use_hyperparameter_optimization": false,  // Set true for Optuna optimization
  "early_stopping": true
}
```

## 🎊 After Training

### Model Files Created
```
models/
├── autoencoder/
│   ├── best_model.pt          # Best SAE model
│   ├── final_model.pt         # Final SAE model
│   └── training_curves.png    # Loss curves
├── rbm/
│   ├── final_model.pt         # Trained RBM model
│   └── training_curves.png    # Loss curves
└── model_comparison.json      # Performance comparison
```

### Check Results
```python
# Load and use trained models
import torch
from src.models.autoencoder_model import StackedAutoEncoder

# Load SAE model
model = StackedAutoEncoder(n_movies=1682)  # ml-100k has 1682 movies
model.load_model('models/autoencoder/best_model.pt')

# Get recommendations for user
user_ratings = torch.zeros(1682)  # User's rating vector
user_ratings[0] = 0.8  # Liked movie 0
user_ratings[10] = 0.6  # Somewhat liked movie 10

recommendations = model.recommend_items(0, user_ratings.unsqueeze(0), k=10)
print("Top 10 recommendations:", recommendations)
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config
   "batch_size": 64  # Instead of 128
   ```

2. **Import Errors**
   ```bash
   # Make sure you're in the project root
   cd /path/to/movie-recommendation-system
   python start_training.py
   ```

3. **Data Not Found**
   ```bash
   # Check data directory structure
   ls -la data/raw/ml-100k/
   # Should show: u1.base, u1.test, u.item, etc.
   ```

4. **MLflow Connection Issues**
   ```bash
   # Start local MLflow server
   mlflow server --host 0.0.0.0 --port 5000
   ```

### Performance Tips

1. **For Faster Training** (Development):
   - Use `ml-100k` dataset
   - Reduce `num_epochs` to 50-100
   - Set `use_hyperparameter_optimization: false`

2. **For Best Results** (Production):
   - Use `ml-1m` dataset
   - Enable hyperparameter optimization
   - Increase epochs to 200-500

3. **GPU Acceleration**:
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## 📚 Understanding the Code

### Key Files
- `src/models/base_model.py` - Abstract base class with MLflow integration
- `src/models/autoencoder_model.py` - SAE implementation
- `src/models/rbm_model.py` - RBM implementation
- `src/training/train.py` - Main training pipeline
- `src/training/evaluate.py` - Comprehensive evaluation metrics
- `src/data/data_preprocessing.py` - Data preprocessing pipeline

### Model Architecture Details

#### SAE Architecture
```python
# Input: [batch_size, n_movies]
encoder: Linear(n_movies, 20) → Sigmoid → Linear(20, 10) → Sigmoid
decoder: Linear(10, 20) → Sigmoid → Linear(20, n_movies) → Sigmoid
# Output: [batch_size, n_movies] (reconstructed ratings)
```

#### RBM Architecture  
```python
# Visible: [batch_size, n_movies] (binary preferences)
# Hidden: [batch_size, n_hidden] (latent factors)
# Weights: [n_hidden, n_movies] (fully connected)
# Training: Contrastive Divergence (CD-k)
```

## 🎯 Next Steps

After successful training, you can:

1. **Explore Results**: Check MLflow UI for detailed metrics
2. **Deploy Models**: Use the trained models in the API (see `src/api/`)
3. **Experiment**: Try different architectures, hyperparameters
4. **Add Features**: Implement additional models or evaluation metrics

Happy training! 🚀
