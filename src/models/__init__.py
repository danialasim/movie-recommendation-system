"""
Models module for movie recommendation system.
Following MLOps Guide specifications exactly.

Provides:
- StackedAutoEncoder: Deep autoencoder for collaborative filtering
- RestrictedBoltzmannMachine: RBM for binary preference modeling
"""

from .autoencoder_model import (
    StackedAutoEncoder,
    create_sae_data_loader,
    normalize_ratings_for_sae
)

from .rbm_model import (
    RestrictedBoltzmannMachine,
    create_rbm_data_loader,
    convert_ratings_to_binary,
    evaluate_rbm_reconstruction
)

from .base_model import BaseRecommender

__all__ = [
    'StackedAutoEncoder',
    'RestrictedBoltzmannMachine', 
    'BaseRecommender',
    'create_sae_data_loader',
    'normalize_ratings_for_sae',
    'create_rbm_data_loader',
    'convert_ratings_to_binary',
    'evaluate_rbm_reconstruction'
]
