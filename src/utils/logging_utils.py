"""
Logging utility for movie recommendation system.

This module provides centralized logging configuration and utilities.
"""

import os
import sys
import logging
import logging.config
import yaml
from pathlib import Path


def setup_logging(default_path='config/logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration.
    
    Args:
        default_path: Path to the logging configuration file
        default_level: Default logging level if config file is not found
        env_key: Environment variable key for custom config path
    """
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / default_path
    
    # Check for environment variable override
    path = os.getenv(env_key, None)
    if path:
        config_path = Path(path)
    
    # Ensure log directories exist
    log_dirs = [
        project_root / 'logs' / 'training',
        project_root / 'logs' / 'evaluation', 
        project_root / 'logs' / 'models',
        project_root / 'logs' / 'api',
        project_root / 'logs' / 'data'
    ]
    
    for log_dir in log_dirs:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and apply logging configuration
    if config_path.exists():
        try:
            with open(config_path, 'rt') as f:
                config = yaml.safe_load(f.read())
            
            # Convert relative paths to absolute paths
            for handler_name, handler_config in config.get('handlers', {}).items():
                if 'filename' in handler_config:
                    filename = handler_config['filename']
                    if not os.path.isabs(filename):
                        handler_config['filename'] = str(project_root / filename)
            
            logging.config.dictConfig(config)
            
        except Exception as e:
            print(f"Error loading logging configuration from {config_path}: {e}")
            print("Using default logging configuration")
            logging.basicConfig(level=default_level,
                              format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        print(f"Logging configuration file not found at {config_path}")
        print("Using default logging configuration")
        logging.basicConfig(level=default_level,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_logger(name, logger_type='evaluation'):
    """
    Get a logger with the specified name and type.
    
    Args:
        name: Logger name (usually __name__)
        logger_type: Type of logger ('evaluation', 'training', 'models', 'api', 'data')
    
    Returns:
        Logger instance
    """
    
    # Setup logging if not already done
    if not logging.getLogger().handlers:
        setup_logging()
    
    # Get logger with appropriate type prefix
    logger_name = f"{logger_type}.{name}" if name != '__main__' else logger_type
    return logging.getLogger(logger_name)


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_mlflow_metrics(metrics, model_type, logger=None):
    """
    Log metrics to both logger and MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
        model_type: Type of model ('SAE', 'RBM')
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    # Log metrics to file
    logger.info(f"=== {model_type} EVALUATION METRICS ===")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{key}: {value}")
        elif isinstance(value, dict):
            logger.info(f"{key}: {value}")
        else:
            logger.info(f"{key}: {str(value)}")
    
    # Try to log to MLflow
    try:
        import mlflow
        
        with mlflow.start_run(run_name=f"{model_type}_evaluation"):
            # Log basic metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in ['hyperparameters', 'architecture']:
                    mlflow.log_metric(key, value)
                elif isinstance(value, str) and key != 'hyperparameters':
                    mlflow.log_param(key, value)
            
            # Log hyperparameters
            if 'hyperparameters' in metrics:
                for hp_key, hp_value in metrics['hyperparameters'].items():
                    if isinstance(hp_value, (int, float, str, bool)):
                        mlflow.log_param(f"hp_{hp_key}", hp_value)
            
            logger.info(f"Metrics logged to MLflow for {model_type}")
            
    except Exception as e:
        logger.warning(f"Failed to log metrics to MLflow: {e}")


# Initialize logging when module is imported
if __name__ != '__main__':
    setup_logging()
