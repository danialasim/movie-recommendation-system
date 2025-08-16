#!/usr/bin/env python3
"""
Enhanced FastAPI app with Prometheus metrics integration.
"""

import sys
import json
import logging
import traceback
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# FastAPI and Pydantic imports
from fastapi import FastAPI, HTTPException, Query, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Import monitoring
try:
    from src.monitoring.prometheus_metrics import (
        metrics_collector, monitor_api_request, monitor_model_prediction,
        start_metrics_server, REGISTRY
    )
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Prometheus metrics not available: {e}")
    METRICS_AVAILABLE = False
    
    # Create dummy decorators
    def monitor_api_request(func):
        return func
    def monitor_model_prediction(model_type):
        def decorator(func):
            return func
        return decorator

# Import models
try:
    import load_trained_models
    
    def load_all_models():
        """Compatibility function to load all models."""
        loader = load_trained_models.ModelLoader()
        models_dict = loader.load_both_models(use_best=True)
        
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
    sys.exit(1)

# Setup logging
import logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

setup_logging()
logger = logging.getLogger(__name__)

# Pydantic models (same as before)
class MovieRecommendationRequest(BaseModel):
    movies: List[int] = Field(..., description="List of movie IDs that the user likes", min_items=1, max_items=50)
    model: str = Field("sae", description="Model to use for recommendations", regex="^(sae|rbm)$")
    num_recommendations: int = Field(10, description="Number of recommendations to return", ge=1, le=50)

    @validator('movies')
    def validate_movies(cls, v):
        if not all(isinstance(movie_id, int) and movie_id > 0 for movie_id in v):
            raise ValueError('All movie IDs must be positive integers')
        return v

class MovieItem(BaseModel):
    movie_id: int = Field(..., description="Unique movie identifier")
    title: str = Field(..., description="Movie title")
    rating: Optional[float] = Field(None, description="Predicted rating (1-5 scale)")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")

class RecommendationResponse(BaseModel):
    success: bool = Field(..., description="Whether the request was successful")
    recommendations: List[MovieItem] = Field(..., description="List of recommended movies")
    model_used: str = Field(..., description="Model that was used for recommendations")
    input_movies: List[int] = Field(..., description="Original input movie IDs")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the recommendations were generated")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    metrics_enabled: bool = Field(..., description="Whether Prometheus metrics are enabled")
    uptime_seconds: float = Field(..., description="System uptime in seconds")

# Global service instance
recommendation_service = None
service_start_time = time.time()

class EnhancedRecommendationService:
    """Enhanced FastAPI Movie Recommendation Service with monitoring."""
    
    def __init__(self):
        """Initialize the recommendation service."""
        logger.info("Initializing Enhanced Movie Recommendation Service...")
        
        self.models = {}
        self.movies_df = None
        self.model_info = {}
        
        # Load models and data
        self._load_models()
        self._load_movie_data()
        
        # Update metrics
        if METRICS_AVAILABLE:
            metrics_collector.set_active_models(
                sae_active=self.models.get("sae") is not None,
                rbm_active=self.models.get("rbm") is not None
            )
        
        logger.info("Enhanced Movie Recommendation Service initialized successfully")
    
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
            if METRICS_AVAILABLE:
                metrics_collector.record_error("ModelLoadError", "service")
            raise
    
    def _load_movie_data(self):
        """Load movie metadata."""
        try:
            movies_path = project_root / "data" / "preprocessed" / "ml-100k" / "movies.csv"
            
            if not movies_path.exists():
                logger.warning(f"Movies file not found at {movies_path}")
                return
            
            import pandas as pd
            self.movies_df = pd.read_csv(movies_path)
            
            logger.info(f"Loaded {len(self.movies_df)} movies")
            
        except Exception as e:
            logger.error(f"Error loading movie data: {e}")
            self.movies_df = None
    
    def get_movie_title(self, movie_id: int) -> str:
        """Get movie title by ID."""
        if self.movies_df is not None:
            movie_row = self.movies_df[self.movies_df['movie_id'] == movie_id]
            if not movie_row.empty:
                return movie_row.iloc[0]['title']
        return f"Movie {movie_id}"
    
    @monitor_model_prediction("general")
    def get_recommendations(self, liked_movies: List[int], model_type: str = "sae", 
                          num_recommendations: int = 10) -> List[MovieItem]:
        """Get movie recommendations with monitoring."""
        
        # Validate model availability
        if model_type not in self.models or self.models[model_type] is None:
            raise ValueError(f"Model {model_type} is not available")
        
        model = self.models[model_type]
        
        try:
            import torch
            import numpy as np
            
            # Create user vector
            n_movies = 1682
            user_vector = torch.zeros(n_movies)
            
            # Set liked movies to positive rating
            for movie_id in liked_movies:
                if movie_id <= n_movies:
                    user_vector[movie_id - 1] = 4.0
            
            # Get predictions with model-specific monitoring
            start_time = time.time()
            
            model.eval()
            with torch.no_grad():
                if model_type == "sae":
                    predictions = model(user_vector.unsqueeze(0)).squeeze()
                else:  # RBM
                    # RBM uses forward pass for predictions
                    predictions = model(user_vector.unsqueeze(0)).squeeze()
            
            prediction_time = time.time() - start_time
            
            # Record model-specific metrics
            if METRICS_AVAILABLE:
                metrics_collector.record_model_prediction(model_type, prediction_time, 'success')
            
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
                movie_id = idx + 1
                raw_score = predictions[idx]
                
                # Calculate rating (1-5 scale)
                rating = max(1.0, min(5.0, float(raw_score * 5.0)))
                if rating <= 0 or np.isnan(rating):
                    rating = 3.0
                
                # Calculate confidence with enhanced scoring
                base_confidence = min(0.99, max(0.01, abs(float(raw_score))))
                
                if base_confidence > 0.3:
                    confidence = 0.75 + (base_confidence - 0.3) * 0.5
                else:
                    confidence = 0.5 + base_confidence * 0.8
                
                confidence = min(0.99, max(0.50, confidence))
                
                # Record recommendation metrics
                if METRICS_AVAILABLE:
                    metrics_collector.record_recommendation_metrics(model_type, confidence, rating)
                
                recommendations.append(MovieItem(
                    movie_id=movie_id,
                    title=self.get_movie_title(movie_id),
                    rating=round(rating, 2),
                    confidence=round(confidence, 3)
                ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting {model_type} recommendations: {e}")
            if METRICS_AVAILABLE:
                metrics_collector.record_error(type(e).__name__, f"model_{model_type}")
            raise

# Create FastAPI app
app = FastAPI(
    title="MovieRec AI - Enhanced FastAPI",
    description="High-performance movie recommendation API with Prometheus monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request monitoring
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Monitor all HTTP requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    if METRICS_AVAILABLE:
        duration = time.time() - start_time
        method = request.method
        endpoint = str(request.url.path)
        status = response.status_code
        
        metrics_collector.record_api_request(method, endpoint, status, duration)
    
    return response

# Initialize service on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation service on startup."""
    global recommendation_service
    try:
        recommendation_service = EnhancedRecommendationService()
        
        # Start metrics server if available
        if METRICS_AVAILABLE:
            start_metrics_server(port=8002)
        
        logger.info("Enhanced FastAPI Movie Recommendation Service started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise

# Prometheus metrics endpoint
@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def get_metrics():
    """Expose Prometheus metrics."""
    if not METRICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Metrics not available")
    
    return generate_latest(REGISTRY)

# Enhanced health check
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Enhanced health check with metrics."""
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    uptime = time.time() - service_start_time
    
    return HealthResponse(
        status="healthy",
        models_loaded={
            "sae": recommendation_service.models.get("sae") is not None,
            "rbm": recommendation_service.models.get("rbm") is not None
        },
        metrics_enabled=METRICS_AVAILABLE,
        uptime_seconds=round(uptime, 2)
    )

# Enhanced recommendations endpoint
@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: MovieRecommendationRequest):
    """Get movie recommendations with performance metrics."""
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = time.time()
    
    try:
        logger.info(f"Getting recommendations for {len(request.movies)} liked movies using {request.model}")
        
        recommendations = recommendation_service.get_recommendations(
            liked_movies=request.movies,
            model_type=request.model,
            num_recommendations=request.num_recommendations
        )
        
        processing_time = time.time() - start_time
        
        # Prepare metrics
        metrics = {
            "processing_time_seconds": round(processing_time, 3),
            "recommendations_count": len(recommendations),
            "model_type": request.model
        }
        
        if recommendations:
            avg_confidence = sum(r.confidence for r in recommendations) / len(recommendations)
            avg_rating = sum(r.rating for r in recommendations) / len(recommendations)
            metrics.update({
                "average_confidence": round(avg_confidence, 3),
                "average_rating": round(avg_rating, 2)
            })
        
        return RecommendationResponse(
            success=True,
            recommendations=recommendations,
            model_used=request.model,
            input_movies=request.movies,
            metrics=metrics
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced root endpoint
@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Enhanced root endpoint with monitoring links."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MovieRec AI - Enhanced FastAPI</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .status {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .status-item {{ padding: 15px; background: #e8f4fd; border-radius: 8px; text-align: center; }}
            .api-links {{ display: flex; justify-content: space-around; margin: 30px 0; flex-wrap: wrap; }}
            .api-link {{ padding: 12px 20px; margin: 5px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
            .api-link:hover {{ background: #0056b3; }}
            .monitoring {{ background: #28a745; }}
            .monitoring:hover {{ background: #1e7e34; }}
            .endpoints {{ margin-top: 30px; }}
            .endpoint {{ margin: 10px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #007bff; }}
            .method {{ font-weight: bold; color: #007bff; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎬 MovieRec AI - Enhanced FastAPI</h1>
                <p>High-performance movie recommendation API with Prometheus monitoring</p>
            </div>
            
            <div class="status">
                <div class="status-item">
                    <h4>🚀 API Status</h4>
                    <p>Running</p>
                </div>
                <div class="status-item">
                    <h4>📊 Monitoring</h4>
                    <p>{'Enabled' if METRICS_AVAILABLE else 'Disabled'}</p>
                </div>
                <div class="status-item">
                    <h4>⏱️ Uptime</h4>
                    <p>{round(time.time() - service_start_time, 0)}s</p>
                </div>
            </div>
            
            <div class="api-links">
                <a href="/docs" class="api-link">📚 Swagger Docs</a>
                <a href="/redoc" class="api-link">📖 ReDoc</a>
                <a href="/health" class="api-link">❤️ Health Check</a>
                {'<a href="/metrics" class="api-link monitoring">📈 Prometheus Metrics</a>' if METRICS_AVAILABLE else ''}
            </div>
            
            <div class="endpoints">
                <h3>Available Endpoints:</h3>
                <div class="endpoint">
                    <span class="method">POST</span> /recommend - Get movie recommendations with metrics
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> /health - Enhanced health check with uptime
                </div>
                {'<div class="endpoint"><span class="method">GET</span> /metrics - Prometheus metrics</div>' if METRICS_AVAILABLE else ''}
            </div>
            
            {'<p><strong>Monitoring:</strong> Prometheus metrics available at <code>/metrics</code></p>' if METRICS_AVAILABLE else ''}
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_fastapi_app:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
