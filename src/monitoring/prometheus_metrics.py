#!/usr/bin/env python3
"""
Prometheus metrics collection for Movie Recommendation System.
"""

import time
import logging
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, CollectorRegistry
from functools import wraps
import threading

logger = logging.getLogger(__name__)

# Create custom registry
REGISTRY = CollectorRegistry()

# Define metrics
API_REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

API_REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

MODEL_PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['model_type', 'status'],
    registry=REGISTRY
)

MODEL_PREDICTION_DURATION = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction duration in seconds',
    ['model_type'],
    registry=REGISTRY
)

RECOMMENDATION_CONFIDENCE = Histogram(
    'recommendation_confidence',
    'Distribution of recommendation confidence scores',
    ['model_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    registry=REGISTRY
)

RECOMMENDATION_RATING = Histogram(
    'recommendation_rating',
    'Distribution of recommendation ratings',
    ['model_type'],
    buckets=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    registry=REGISTRY
)

ACTIVE_MODELS = Gauge(
    'active_models',
    'Number of active models',
    ['model_type'],
    registry=REGISTRY
)

SYSTEM_INFO = Info(
    'system_info',
    'System information',
    registry=REGISTRY
)

MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['type'],
    registry=REGISTRY
)

ERROR_COUNT = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'component'],
    registry=REGISTRY
)

class MetricsCollector:
    """Centralized metrics collection for the movie recommendation system."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.start_time = time.time()
        self._setup_system_info()
        
    def _setup_system_info(self):
        """Set up system information metrics."""
        import platform
        import sys
        
        SYSTEM_INFO.info({
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor() or 'unknown',
            'hostname': platform.node()
        })
    
    def record_api_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record API request metrics."""
        API_REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        API_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_model_prediction(self, model_type: str, duration: float, status: str = 'success'):
        """Record model prediction metrics."""
        MODEL_PREDICTION_COUNT.labels(model_type=model_type, status=status).inc()
        MODEL_PREDICTION_DURATION.labels(model_type=model_type).observe(duration)
    
    def record_recommendation_metrics(self, model_type: str, confidence: float, rating: float):
        """Record recommendation quality metrics."""
        RECOMMENDATION_CONFIDENCE.labels(model_type=model_type).observe(confidence)
        RECOMMENDATION_RATING.labels(model_type=model_type).observe(rating)
    
    def set_active_models(self, sae_active: bool = False, rbm_active: bool = False):
        """Set active model status."""
        ACTIVE_MODELS.labels(model_type='sae').set(1 if sae_active else 0)
        ACTIVE_MODELS.labels(model_type='rbm').set(1 if rbm_active else 0)
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        ERROR_COUNT.labels(error_type=error_type, component=component).inc()
    
    def update_memory_usage(self):
        """Update memory usage metrics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            MEMORY_USAGE.labels(type='rss').set(memory_info.rss)
            MEMORY_USAGE.labels(type='vms').set(memory_info.vms)
            
            # System memory
            sys_memory = psutil.virtual_memory()
            MEMORY_USAGE.labels(type='system_total').set(sys_memory.total)
            MEMORY_USAGE.labels(type='system_available').set(sys_memory.available)
            MEMORY_USAGE.labels(type='system_used').set(sys_memory.used)
            
        except ImportError:
            logger.warning("psutil not available, skipping memory metrics")
        except Exception as e:
            logger.error(f"Error updating memory metrics: {e}")

# Global metrics collector instance
metrics_collector = MetricsCollector()

def monitor_api_request(func):
    """Decorator to monitor API requests."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method = "unknown"
        endpoint = func.__name__
        status = 200
        
        try:
            # Try to extract request info if available
            if hasattr(args[0], 'method'):
                method = args[0].method
            if hasattr(args[0], 'url'):
                endpoint = str(args[0].url.path)
            
            result = await func(*args, **kwargs)
            
            # Extract status from result if it's a response object
            if hasattr(result, 'status_code'):
                status = result.status_code
            
            return result
            
        except Exception as e:
            status = 500
            metrics_collector.record_error(type(e).__name__, 'api')
            raise
            
        finally:
            duration = time.time() - start_time
            metrics_collector.record_api_request(method, endpoint, status, duration)
    
    return wrapper

def monitor_model_prediction(model_type: str):
    """Decorator to monitor model predictions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                
                # Record recommendation metrics if result contains recommendations
                if hasattr(result, '__iter__') and result:
                    for item in result[:5]:  # Sample first 5
                        if hasattr(item, 'confidence') and hasattr(item, 'rating'):
                            metrics_collector.record_recommendation_metrics(
                                model_type, item.confidence, item.rating
                            )
                
                return result
                
            except Exception as e:
                status = 'error'
                metrics_collector.record_error(type(e).__name__, f'model_{model_type}')
                raise
                
            finally:
                duration = time.time() - start_time
                metrics_collector.record_model_prediction(model_type, duration, status)
        
        return wrapper
    return decorator

def start_metrics_server(port: int = 8000):
    """Start the Prometheus metrics server."""
    try:
        start_http_server(port, registry=REGISTRY)
        logger.info(f"Prometheus metrics server started on port {port}")
        
        # Start background memory monitoring
        def update_memory_periodically():
            while True:
                metrics_collector.update_memory_usage()
                time.sleep(30)  # Update every 30 seconds
        
        memory_thread = threading.Thread(target=update_memory_periodically, daemon=True)
        memory_thread.start()
        
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

# Initialize metrics on import
def initialize_metrics():
    """Initialize metrics with default values."""
    try:
        # Set initial model status
        metrics_collector.set_active_models(sae_active=False, rbm_active=False)
        
        # Update initial memory usage
        metrics_collector.update_memory_usage()
        
        logger.info("Metrics collector initialized")
        
    except Exception as e:
        logger.error(f"Error initializing metrics: {e}")

# Call initialization
initialize_metrics()
