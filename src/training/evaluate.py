"""
Evaluation module for movie recommendation system.

This module provides comprehensive evaluation metrics for recommendation models
including RMSE, MAE, Precision@K, Recall@K, NDCG@K, and other metrics.

Author: Movie Recommendation System Team
Date: August 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Setup proper logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_utils import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__, 'training')


class ModelEvaluator:
    """
    Comprehensive evaluator for recommendation models.
    
    Provides various metrics for evaluating both rating prediction
    and ranking quality of recommendation systems.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def evaluate_sae(self, model, test_matrix: torch.Tensor, 
                    train_matrix: Optional[torch.Tensor] = None,
                    k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Comprehensive evaluation of Stacked AutoEncoder.
        
        Args:
            model: Trained SAE model
            test_matrix: Test data matrix
            train_matrix: Training data matrix (for recommendation evaluation)
            k_values: List of k values for top-k metrics
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info("Evaluating Stacked AutoEncoder...")
        
        model.eval()
        metrics = {}
        
        # Rating prediction metrics
        with torch.no_grad():
            # Get predictions for test data
            predictions = model.predict_ratings(test_matrix)
            
            # Create mask for rated items in test set
            test_mask = test_matrix > 0
            
            if test_mask.sum() > 0:
                # Denormalize ratings (assuming test data is normalized [0,1] -> [1,5])
                true_ratings = test_matrix[test_mask] * 4 + 1
                pred_ratings = predictions[test_mask] * 4 + 1
                
                # Clamp predictions to valid range
                pred_ratings = torch.clamp(pred_ratings, 1, 5)
                
                # Compute rating prediction metrics
                metrics['rmse'] = torch.sqrt(F.mse_loss(pred_ratings, true_ratings)).item()
                metrics['mae'] = F.l1_loss(pred_ratings, true_ratings).item()
                
                # R-squared
                ss_res = ((true_ratings - pred_ratings) ** 2).sum()
                ss_tot = ((true_ratings - true_ratings.mean()) ** 2).sum()
                metrics['r2_score'] = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
                
                # Mean prediction and actual ratings
                metrics['mean_prediction'] = pred_ratings.mean().item()
                metrics['mean_actual'] = true_ratings.mean().item()
                metrics['num_test_ratings'] = test_mask.sum().item()
            
            # Ranking metrics (if training data is provided)
            if train_matrix is not None:
                ranking_metrics = self._compute_ranking_metrics(
                    model, train_matrix, test_matrix, k_values
                )
                metrics.update(ranking_metrics)
        
        logger.info(f"SAE Evaluation completed. RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        return metrics
    
    def _compute_ranking_metrics(self, model, train_matrix: torch.Tensor,
                                test_matrix: torch.Tensor, k_values: List[int]) -> Dict[str, float]:
        """
        Compute ranking-based metrics for recommendation quality.
        
        Args:
            model: Trained model
            train_matrix: Training data
            test_matrix: Test data (held-out ratings)
            k_values: List of k values for evaluation
            
        Returns:
            Dict[str, float]: Ranking metrics
        """
        metrics = {}
        
        # Get predictions for all items
        with torch.no_grad():
            all_predictions = model.predict_ratings(train_matrix)
        
        precision_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        ndcg_scores = {k: [] for k in k_values}
        
        n_users = train_matrix.shape[0]
        users_evaluated = 0
        
        for user_idx in range(n_users):
            # Get user's training and test ratings
            user_train = train_matrix[user_idx]
            user_test = test_matrix[user_idx]
            user_pred = all_predictions[user_idx]
            
            # Skip users with no test ratings
            test_items = (user_test > 0).nonzero().squeeze()
            if test_items.numel() == 0:
                continue
            
            # Get items not seen during training (for recommendation)
            unrated_mask = user_train == 0
            if unrated_mask.sum() == 0:
                continue
            
            users_evaluated += 1
            
            # Get top-k recommendations among unrated items
            unrated_predictions = user_pred.clone()
            unrated_predictions[~unrated_mask] = float('-inf')  # Exclude rated items
            
            for k in k_values:
                if unrated_mask.sum() < k:
                    continue
                
                # Get top-k recommendations
                _, top_k_indices = torch.topk(unrated_predictions, k)
                
                # Check which recommended items are actually liked in test set
                # (assuming rating >= 3.5 means "liked")
                recommended_ratings = user_test[top_k_indices]
                relevant_items = (recommended_ratings >= 3.5).float()
                
                # Precision@K
                precision_k = relevant_items.sum().item() / k
                precision_scores[k].append(precision_k)
                
                # Recall@K
                total_relevant = (user_test >= 3.5).sum().item()
                if total_relevant > 0:
                    recall_k = relevant_items.sum().item() / total_relevant
                    recall_scores[k].append(recall_k)
                
                # NDCG@K
                ndcg_k = self._compute_ndcg_at_k(recommended_ratings.cpu().numpy(), k)
                ndcg_scores[k].append(ndcg_k)
        
        # Aggregate metrics
        for k in k_values:
            if precision_scores[k]:
                metrics[f'precision_at_{k}'] = np.mean(precision_scores[k])
            if recall_scores[k]:
                metrics[f'recall_at_{k}'] = np.mean(recall_scores[k])
            if ndcg_scores[k]:
                metrics[f'ndcg_at_{k}'] = np.mean(ndcg_scores[k])
        
        # Coverage metric
        all_predictions_flat = all_predictions.flatten()
        unrated_mask_flat = (train_matrix == 0).flatten()
        if unrated_mask_flat.sum() > 0:
            # What fraction of items could potentially be recommended?
            _, top_items = torch.topk(all_predictions_flat[unrated_mask_flat], 
                                    min(1000, unrated_mask_flat.sum()))
            metrics['catalog_coverage'] = len(torch.unique(top_items)) / train_matrix.shape[1]
        
        metrics['users_evaluated'] = users_evaluated
        
        return metrics
    
    def _compute_ndcg_at_k(self, relevances: np.ndarray, k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k.
        
        Args:
            relevances: Array of relevance scores
            k: Cut-off position
            
        Returns:
            float: NDCG@k score
        """
        def dcg_at_k(r, k):
            r = np.asfarray(r)[:k]
            if r.size:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            return 0.0
        
        dcg_max = dcg_at_k(sorted(relevances, reverse=True), k)
        if not dcg_max:
            return 0.0
        
        return dcg_at_k(relevances, k) / dcg_max
    
    def compute_diversity_metrics(self, model, user_item_matrix: torch.Tensor,
                                 movie_features: Optional[torch.Tensor] = None,
                                 k: int = 10) -> Dict[str, float]:
        """
        Compute diversity metrics for recommendations.
        
        Args:
            model: Trained model
            user_item_matrix: User-item interaction matrix
            movie_features: Movie feature matrix (optional)
            k: Number of recommendations to analyze
            
        Returns:
            Dict[str, float]: Diversity metrics
        """
        metrics = {}
        
        with torch.no_grad():
            predictions = model.predict_ratings(user_item_matrix)
        
        # Intra-list diversity (average pairwise distance within recommendation lists)
        if movie_features is not None:
            diversity_scores = []
            
            for user_idx in range(user_item_matrix.shape[0]):
                user_pred = predictions[user_idx]
                unrated_mask = user_item_matrix[user_idx] == 0
                
                if unrated_mask.sum() < k:
                    continue
                
                # Get top-k recommendations
                unrated_pred = user_pred.clone()
                unrated_pred[~unrated_mask] = float('-inf')
                _, top_k_indices = torch.topk(unrated_pred, k)
                
                # Compute pairwise similarities in feature space
                top_k_features = movie_features[top_k_indices]
                similarities = F.cosine_similarity(
                    top_k_features.unsqueeze(1), 
                    top_k_features.unsqueeze(0), 
                    dim=2
                )
                
                # Average diversity (1 - similarity)
                # Only consider upper triangle (excluding diagonal)
                mask = torch.triu(torch.ones_like(similarities), diagonal=1).bool()
                avg_similarity = similarities[mask].mean().item()
                diversity = 1 - avg_similarity
                
                diversity_scores.append(diversity)
            
            if diversity_scores:
                metrics['intra_list_diversity'] = np.mean(diversity_scores)
        
        # Novelty (inverse popularity)
        item_popularity = (user_item_matrix > 0).sum(dim=0).float()
        novelty_scores = []
        
        for user_idx in range(user_item_matrix.shape[0]):
            user_pred = predictions[user_idx]
            unrated_mask = user_item_matrix[user_idx] == 0
            
            if unrated_mask.sum() < k:
                continue
            
            unrated_pred = user_pred.clone()
            unrated_pred[~unrated_mask] = float('-inf')
            _, top_k_indices = torch.topk(unrated_pred, k)
            
            # Average inverse popularity of recommended items
            rec_popularity = item_popularity[top_k_indices]
            # Add small constant to avoid division by zero
            novelty = (1.0 / (rec_popularity + 1e-8)).mean().item()
            novelty_scores.append(novelty)
        
        if novelty_scores:
            metrics['novelty'] = np.mean(novelty_scores)
        
        return metrics
    
    def evaluate_cold_start(self, model, cold_start_users: torch.Tensor,
                           cold_start_test: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model performance on cold-start users.
        
        Args:
            model: Trained model
            cold_start_users: Cold-start user data (few ratings)
            cold_start_test: Test data for cold-start users
            
        Returns:
            Dict[str, float]: Cold-start evaluation metrics
        """
        logger.info("Evaluating cold-start performance...")
        
        model.eval()
        metrics = {}
        
        with torch.no_grad():
            predictions = model.predict_ratings(cold_start_users)
            
            # Rating prediction metrics for cold-start users
            test_mask = cold_start_test > 0
            
            if test_mask.sum() > 0:
                true_ratings = cold_start_test[test_mask] * 4 + 1  # Denormalize
                pred_ratings = predictions[test_mask] * 4 + 1
                pred_ratings = torch.clamp(pred_ratings, 1, 5)
                
                metrics['cold_start_rmse'] = torch.sqrt(F.mse_loss(pred_ratings, true_ratings)).item()
                metrics['cold_start_mae'] = F.l1_loss(pred_ratings, true_ratings).item()
                metrics['cold_start_coverage'] = (predictions > 0).float().mean().item()
                metrics['num_cold_start_users'] = cold_start_users.shape[0]
        
        return metrics
    
    def create_evaluation_report(self, metrics: Dict[str, float], 
                               model_name: str, save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            metrics: Dictionary of evaluation metrics
            model_name: Name of the model
            save_path: Path to save the report (optional)
            
        Returns:
            str: Formatted report
        """
        report_lines = [
            f"Evaluation Report for {model_name}",
            "=" * 50,
            "",
            "Rating Prediction Metrics:",
            "-" * 30
        ]
        
        # Rating prediction metrics
        rating_metrics = ['rmse', 'mae', 'r2_score', 'mean_prediction', 'mean_actual', 'num_test_ratings']
        for metric in rating_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    report_lines.append(f"{metric.upper().replace('_', ' ')}: {value:.4f}")
                else:
                    report_lines.append(f"{metric.upper().replace('_', ' ')}: {value}")
        
        # Ranking metrics
        ranking_metrics = [k for k in metrics.keys() if any(x in k for x in ['precision', 'recall', 'ndcg'])]
        if ranking_metrics:
            report_lines.extend([
                "",
                "Ranking Metrics:",
                "-" * 20
            ])
            for metric in sorted(ranking_metrics):
                value = metrics[metric]
                report_lines.append(f"{metric.upper().replace('_', ' ')}: {value:.4f}")
        
        # Diversity metrics
        diversity_metrics = [k for k in metrics.keys() if any(x in k for x in ['diversity', 'novelty', 'coverage'])]
        if diversity_metrics:
            report_lines.extend([
                "",
                "Diversity Metrics:",
                "-" * 20
            ])
            for metric in sorted(diversity_metrics):
                value = metrics[metric]
                report_lines.append(f"{metric.upper().replace('_', ' ')}: {value:.4f}")
        
        # Cold-start metrics
        cold_start_metrics = [k for k in metrics.keys() if 'cold_start' in k]
        if cold_start_metrics:
            report_lines.extend([
                "",
                "Cold-Start Metrics:",
                "-" * 20
            ])
            for metric in sorted(cold_start_metrics):
                value = metrics[metric]
                if isinstance(value, float):
                    report_lines.append(f"{metric.upper().replace('_', ' ')}: {value:.4f}")
                else:
                    report_lines.append(f"{metric.upper().replace('_', ' ')}: {value}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                               save_path: Optional[str] = None):
        """
        Plot comparison of metrics across different models.
        
        Args:
            metrics_dict: Dictionary of {model_name: metrics}
            save_path: Path to save the plot
        """
        # Extract common metrics for comparison
        common_metrics = set()
        for model_metrics in metrics_dict.values():
            common_metrics.update(model_metrics.keys())
        
        # Filter to important metrics
        important_metrics = [m for m in common_metrics if any(x in m for x in 
                           ['rmse', 'mae', 'precision_at_10', 'recall_at_10', 'ndcg_at_10'])]
        
        if not important_metrics:
            logger.warning("No common important metrics found for comparison")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in metrics_dict.items():
            for metric in important_metrics:
                if metric in metrics:
                    comparison_data.append({
                        'Model': model_name,
                        'Metric': metric,
                        'Value': metrics[metric]
                    })
        
        if not comparison_data:
            return
        
        df = pd.DataFrame(comparison_data)
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(important_metrics[:6]):  # Plot up to 6 metrics
            if i >= len(axes):
                break
                
            metric_data = df[df['Metric'] == metric]
            if not metric_data.empty:
                sns.barplot(data=metric_data, x='Model', y='Value', ax=axes[i])
                axes[i].set_title(f'{metric.upper().replace("_", " ")}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(important_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison plot saved to {save_path}")
        
        plt.show()


def calculate_recommendation_metrics(true_ratings: torch.Tensor, 
                                   predicted_ratings: torch.Tensor,
                                   k: int = 10) -> Dict[str, float]:
    """
    Calculate recommendation metrics for a single user.
    
    Args:
        true_ratings: True user ratings
        predicted_ratings: Predicted user ratings
        k: Number of top recommendations
        
    Returns:
        Dict[str, float]: Metrics dictionary
    """
    metrics = {}
    
    # Get top-k recommendations
    _, top_k_indices = torch.topk(predicted_ratings, k)
    
    # Binary relevance (rating >= 3.5 means relevant)
    relevant_items = (true_ratings >= 3.5).float()
    recommended_relevant = relevant_items[top_k_indices]
    
    # Precision@K
    metrics['precision'] = recommended_relevant.sum().item() / k
    
    # Recall@K
    total_relevant = relevant_items.sum().item()
    if total_relevant > 0:
        metrics['recall'] = recommended_relevant.sum().item() / total_relevant
    else:
        metrics['recall'] = 0.0
    
    # F1@K
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0.0
    
    return metrics
