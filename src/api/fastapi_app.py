#!/usr/bin/env python3
"""
FastAPI Movie Recommendation System
High-performance API with automatic documentation and type validation.
"""

import sys
import json
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# FastAPI and Pydantic imports
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "api"))

# Import our modules
try:
    # Try importing from the project root
    sys.path.append(str(project_root))
    import load_trained_models
    
    # Create a compatibility function
    def load_all_models():
        """Compatibility function to load all models."""
        loader = load_trained_models.ModelLoader()
        models_dict = loader.load_both_models(use_best=True)
        
        # Convert to the expected format
        models = {}
        model_info = {}
        
        for model_name, (model_instance, hyperparams) in models_dict.items():
            models[model_name] = model_instance
            model_info[model_name] = {
                "hyperparameters": hyperparams,
                "parameters": getattr(model_instance, 'get_parameter_count', lambda: 0)(),
                "architecture": str(model_instance)
            }
        
        return models, model_info
    
    print("✅ Successfully imported load_trained_models and created compatibility layer")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔍 Project root:", project_root)
    print("🔍 Current working directory:", Path.cwd())
    print("🔍 Files in project root:")
    import os
    print([f for f in os.listdir(project_root) if f.endswith('.py')])
    sys.exit(1)

# Setup simple logging
import logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class MovieRecommendationRequest(BaseModel):
    """Request model for movie recommendations."""
    movies: List[int] = Field(..., description="List of movie IDs that the user likes", min_items=1, max_items=50)
    model: str = Field("sae", description="Model to use for recommendations", regex="^(sae|rbm)$")
    num_recommendations: int = Field(10, description="Number of recommendations to return", ge=1, le=50)

    @validator('movies')
    def validate_movies(cls, v):
        if not all(isinstance(movie_id, int) and movie_id > 0 for movie_id in v):
            raise ValueError('All movie IDs must be positive integers')
        return v

class MovieSearchRequest(BaseModel):
    """Request model for movie search."""
    query: str = Field(..., description="Search query for movies", min_length=1, max_length=100)
    limit: int = Field(10, description="Maximum number of results to return", ge=1, le=50)

class MovieItem(BaseModel):
    """Response model for a single movie."""
    movie_id: int = Field(..., description="Unique movie identifier")
    title: str = Field(..., description="Movie title")
    rating: Optional[float] = Field(None, description="Predicted rating (1-5 scale)")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")

class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    success: bool = Field(..., description="Whether the request was successful")
    recommendations: List[MovieItem] = Field(..., description="List of recommended movies")
    model_used: str = Field(..., description="Model that was used for recommendations")
    input_movies: List[int] = Field(..., description="Original input movie IDs")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the recommendations were generated")

class SearchResponse(BaseModel):
    """Response model for movie search."""
    success: bool = Field(..., description="Whether the search was successful")
    movies: List[MovieItem] = Field(..., description="List of matching movies")
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total number of results found")

class ModelInfo(BaseModel):
    """Response model for model information."""
    available: bool = Field(..., description="Whether the model is available")
    architecture: Optional[str] = Field(None, description="Model architecture description")
    parameters: Optional[int] = Field(None, description="Number of model parameters")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")

class ModelInfoResponse(BaseModel):
    """Response model for all model information."""
    sae_model: ModelInfo = Field(..., description="SAE model information")
    rbm_model: ModelInfo = Field(..., description="RBM model information")
    total_movies: int = Field(..., description="Total number of movies in the dataset")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")

# Global service instance
recommendation_service = None

class FastAPIRecommendationService:
    """FastAPI Movie Recommendation Service with enhanced features."""
    
    def __init__(self):
        """Initialize the recommendation service."""
        logger.info("Initializing FastAPI Movie Recommendation Service...")
        
        self.models = {}
        self.movies_df = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.model_info = {}
        
        # Load models and data
        self._load_models()
        self._load_movie_data()
        
        logger.info("FastAPI Movie Recommendation Service initialized successfully")
    
    def _load_models(self):
        """Load trained models."""
        try:
            loaded_models, model_info = load_all_models()
            self.models = loaded_models
            self.model_info = model_info
            
            for model_name, model in self.models.items():
                if model is not None:
                    logger.info(f"{model_name.upper()} model loaded successfully")
                else:
                    logger.warning(f"{model_name.upper()} model failed to load")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _load_movie_data(self):
        """Load movie metadata."""
        try:
            # Load movies data
            movies_path = project_root / "data" / "preprocessed" / "ml-100k" / "movies.csv"
            
            if not movies_path.exists():
                logger.warning(f"Movies file not found at {movies_path}")
                return
            
            import pandas as pd
            self.movies_df = pd.read_csv(movies_path)
            
            # Create mapping dictionaries
            if 'movie_id' in self.movies_df.columns:
                self.movie_id_to_idx = {row['movie_id']: idx for idx, row in self.movies_df.iterrows()}
                self.idx_to_movie_id = {idx: row['movie_id'] for idx, row in self.movies_df.iterrows()}
            
            logger.info(f"Loaded {len(self.movies_df)} movies")
            
        except Exception as e:
            logger.error(f"Error loading movie data: {e}")
            # Continue without movies data
            self.movies_df = None
    
    def get_movie_title(self, movie_id: int) -> str:
        """Get movie title by ID."""
        if self.movies_df is not None:
            movie_row = self.movies_df[self.movies_df['movie_id'] == movie_id]
            if not movie_row.empty:
                return movie_row.iloc[0]['title']
        return f"Movie {movie_id}"
    
    def search_movies(self, query: str, limit: int = 10) -> List[MovieItem]:
        """Search for movies by title."""
        if self.movies_df is None:
            return []
        
        # Case-insensitive search
        mask = self.movies_df['title'].str.contains(query, case=False, na=False)
        results = self.movies_df[mask].head(limit)
        
        return [
            MovieItem(
                movie_id=int(row['movie_id']),
                title=row['title']
            )
            for _, row in results.iterrows()
        ]
    
    def get_recommendations(self, liked_movies: List[int], model_type: str = "sae", 
                          num_recommendations: int = 10) -> List[MovieItem]:
        """Get movie recommendations."""
        
        # Validate model availability
        if model_type not in self.models or self.models[model_type] is None:
            raise ValueError(f"Model {model_type} is not available")
        
        model = self.models[model_type]
        
        try:
            import torch
            import numpy as np
            
            # Create user vector
            n_movies = 1682  # MovieLens 100k dataset size
            user_vector = torch.zeros(n_movies)
            
            # Set liked movies to positive rating
            for movie_id in liked_movies:
                if movie_id <= n_movies:
                    user_vector[movie_id - 1] = 4.0  # High rating
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                if model_type == "sae":
                    predictions = model(user_vector.unsqueeze(0)).squeeze()
                else:  # RBM
                    # For RBM, we need to use the reconstruct method
                    predictions = model.reconstruct(user_vector.unsqueeze(0)).squeeze()
            
            # Convert to numpy for easier processing
            predictions = predictions.numpy()
            
            # Mask out already liked movies
            for movie_id in liked_movies:
                if movie_id <= len(predictions):
                    predictions[movie_id - 1] = 0
            
            # Get top recommendations
            top_indices = np.argsort(predictions)[-num_recommendations:][::-1]
            
            recommendations = []
            for idx in top_indices:
                movie_id = idx + 1  # Convert back to 1-indexed
                raw_score = predictions[idx]
                
                # Calculate rating (1-5 scale)
                rating = max(1.0, min(5.0, float(raw_score * 5.0)))
                if rating <= 0 or np.isnan(rating):
                    rating = 3.0
                
                # Calculate confidence with enhanced scoring
                base_confidence = min(0.99, max(0.01, abs(float(raw_score))))
                
                # Enhanced confidence calculation
                if base_confidence > 0.3:
                    confidence = 0.75 + (base_confidence - 0.3) * 0.5  # 75-100%
                else:
                    confidence = 0.5 + base_confidence * 0.8  # 50-75%
                
                confidence = min(0.99, max(0.50, confidence))
                
                recommendations.append(MovieItem(
                    movie_id=movie_id,
                    title=self.get_movie_title(movie_id),
                    rating=round(rating, 2),
                    confidence=round(confidence, 3)
                ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting {model_type} recommendations: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "sae_model": {
                "available": self.models.get("sae") is not None,
                "architecture": self.model_info.get("sae", {}).get("architecture"),
                "parameters": self.model_info.get("sae", {}).get("parameters"),
                "hyperparameters": self.model_info.get("sae", {}).get("hyperparameters")
            },
            "rbm_model": {
                "available": self.models.get("rbm") is not None,
                "architecture": self.model_info.get("rbm", {}).get("architecture"),
                "parameters": self.model_info.get("rbm", {}).get("parameters"),
                "hyperparameters": self.model_info.get("rbm", {}).get("hyperparameters")
            },
            "total_movies": len(self.movies_df) if self.movies_df is not None else 0
        }

# Create FastAPI app
app = FastAPI(
    title="MovieRec AI - FastAPI",
    description="High-performance movie recommendation API using SAE and RBM models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation service on startup."""
    global recommendation_service
    try:
        recommendation_service = FastAPIRecommendationService()
        logger.info("FastAPI Movie Recommendation Service started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health status of the API."""
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return HealthResponse(
        status="healthy",
        models_loaded={
            "sae": recommendation_service.models.get("sae") is not None,
            "rbm": recommendation_service.models.get("rbm") is not None
        }
    )

# Movie search endpoint
@app.post("/search", response_model=SearchResponse, tags=["Movies"])
async def search_movies(request: MovieSearchRequest):
    """Search for movies by title."""
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        movies = recommendation_service.search_movies(request.query, request.limit)
        
        return SearchResponse(
            success=True,
            movies=movies,
            query=request.query,
            total_results=len(movies)
        )
        
    except Exception as e:
        logger.error(f"Error searching movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Recommendations endpoint
@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: MovieRecommendationRequest):
    """Get movie recommendations based on liked movies."""
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Getting recommendations for {len(request.movies)} liked movies using {request.model}")
        
        recommendations = recommendation_service.get_recommendations(
            liked_movies=request.movies,
            model_type=request.model,
            num_recommendations=request.num_recommendations
        )
        
        return RecommendationResponse(
            success=True,
            recommendations=recommendations,
            model_used=request.model,
            input_movies=request.movies
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model information endpoint
@app.get("/model-info", response_model=ModelInfoResponse, tags=["Models"])
async def get_model_info():
    """Get information about the available models."""
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        info = recommendation_service.get_model_info()
        
        return ModelInfoResponse(
            sae_model=ModelInfo(**info["sae_model"]),
            rbm_model=ModelInfo(**info["rbm_model"]),
            total_movies=info["total_movies"]
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Popular movies endpoint
@app.get("/popular", response_model=List[MovieItem], tags=["Movies"])
async def get_popular_movies(limit: int = Query(20, ge=1, le=100, description="Number of popular movies to return")):
    """Get popular movies for quick selection."""
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if recommendation_service.movies_df is None:
            raise HTTPException(status_code=503, detail="Movie data not available")
        
        # Return first N movies as "popular" (in a real system, you'd have popularity scores)
        popular_movies = recommendation_service.movies_df.head(limit)
        
        return [
            MovieItem(
                movie_id=int(row['movie_id']),
                title=row['title']
            )
            for _, row in popular_movies.iterrows()
        ]
        
    except Exception as e:
        logger.error(f"Error getting popular movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint with API information
@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Root endpoint with API information."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MovieRec AI - FastAPI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .api-links { display: flex; justify-content: space-around; margin: 30px 0; }
            .api-link { padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
            .api-link:hover { background: #0056b3; }
            .endpoints { margin-top: 30px; }
            .endpoint { margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #007bff; }
            .method { font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 MovieRec AI - FastAPI</h1>
            <p>High-performance movie recommendation API using SAE and RBM models</p>
            
            <div class="api-links">
                <a href="/docs" class="api-link">📚 Swagger Docs</a>
                <a href="/redoc" class="api-link">📖 ReDoc</a>
                <a href="/health" class="api-link">❤️ Health Check</a>
            </div>
            
            <div class="endpoints">
                <h3>Available Endpoints:</h3>
                <div class="endpoint">
                    <span class="method">POST</span> /recommend - Get movie recommendations
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> /search - Search movies by title
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> /model-info - Get model information
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> /popular - Get popular movies
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> /health - Health check
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Development server runner
def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
