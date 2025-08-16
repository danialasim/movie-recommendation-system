from setuptools import setup, find_packages

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
