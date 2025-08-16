#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for Movie Recommendation System.

This script evaluates both SAE and RBM models with detailed metrics,
visualizations, and performance comparisons.

Usage:
    python evaluate_models.py [--models sae rbm] [--dataset ml-100k] [--visualize]
"""

import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

from data.data_preprocessing import MovieLensPreprocessor
from data.data_utils import DataValidator
from models.autoencoder_model import StackedAutoEncoder
from models.rbm_model import RestrictedBoltzmannMachine
from training.evaluate import ModelEvaluator
from load_trained_models import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluation system for trained models."""
    
    def __init__(self, data_path: str = "data", models_path: str = "models", 
                 results_path: str = "evaluation_results"):
        """
        Initialize the evaluator.
        
        Args:
            data_path: Path to data directory
            models_path: Path to models directory
            results_path: Path to save evaluation results
        """
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.model_loader = ModelLoader(str(models_path))
        self.evaluator = ModelEvaluator()
        
        # Initialize data containers
        self.train_matrix = None
        self.test_matrix = None
        self.val_matrix = None
        self.preprocessor = None
        
    def load_data(self, dataset: str = "ml-100k") -> None:
        """Load and preprocess evaluation data."""
        
        logger.info(f"Loading {dataset} dataset for evaluation...")
        
        try:
            # Load preprocessed data if available
            preprocessed_path = self.data_path / "preprocessed"
            
            # Check for SAE format files first
            if (preprocessed_path / "test_matrix_sae.pt").exists():
                logger.info("Loading preprocessed matrices (SAE/RBM format)...")
                self.train_matrix = torch.load(preprocessed_path / "train_matrix_sae.pt")
                self.test_matrix = torch.load(preprocessed_path / "test_matrix_sae.pt")
                
                logger.info(f"Loaded matrices - Train: {self.train_matrix.shape}, Test: {self.test_matrix.shape}")
                return
                
            # Check for general format files
            elif (preprocessed_path / "test_matrix.pt").exists():
                logger.info("Loading preprocessed matrices (general format)...")
                self.train_matrix = torch.load(preprocessed_path / "train_matrix.pt")
                self.test_matrix = torch.load(preprocessed_path / "test_matrix.pt")
                
                if (preprocessed_path / "val_matrix.pt").exists():
                    self.val_matrix = torch.load(preprocessed_path / "val_matrix.pt")
                
                logger.info(f"Loaded matrices - Train: {self.train_matrix.shape}, Test: {self.test_matrix.shape}")
                return
                
            else:
                raise FileNotFoundError("No preprocessed matrices found. Please run training first to generate preprocessed data.")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def evaluate_sae_model(self, use_best: bool = True, k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """Evaluate SAE model with comprehensive metrics."""
        
        logger.info("🧠 Evaluating Stacked AutoEncoder (SAE) model...")
        
        try:
            # Load SAE model
            sae_model, sae_params = self.model_loader.load_sae_model(use_best=use_best)
            sae_model.eval()
            
            # Prepare data for SAE (check if already normalized)
            if self.test_matrix.max() <= 1.0 and self.test_matrix.min() >= 0.0:
                # Data is already normalized
                test_sae = self.test_matrix
                train_sae = self.train_matrix if self.train_matrix is not None else None
            else:
                # Need to normalize
                from models.autoencoder_model import normalize_ratings_for_sae
                train_sae = normalize_ratings_for_sae(self.train_matrix) if self.train_matrix is not None else None
                test_sae = normalize_ratings_for_sae(self.test_matrix)
            
            # Get predictions
            with torch.no_grad():
                predictions = sae_model(test_sae)
                
                # Don't denormalize - work in normalized space
                # predictions already in [0,1] range
            
            # Calculate comprehensive metrics
            metrics = self.evaluator.evaluate_sae(sae_model, test_sae, train_sae, k_values)
            
            # Add additional metrics
            test_mask = (test_sae > 0)
            if test_mask.sum() > 0:
                true_ratings = test_sae[test_mask]
                pred_ratings = predictions[test_mask]
                
                # Basic metrics on normalized scale
                rmse = torch.sqrt(torch.mean((true_ratings - pred_ratings) ** 2)).item()
                mae = torch.mean(torch.abs(true_ratings - pred_ratings)).item()
                
                # Convert to original scale for interpretability
                rmse_original = rmse * 4
                mae_original = mae * 4
                
                # R² score
                ss_res = torch.sum((true_ratings - pred_ratings) ** 2)
                ss_tot = torch.sum((true_ratings - torch.mean(true_ratings)) ** 2)
                r2_score = (1 - ss_res / ss_tot).item()
                
                metrics.update({
                    'rmse': rmse_original,  # Report in original scale
                    'rmse_normalized': rmse,
                    'mae': mae_original,  # Report in original scale
                    'mae_normalized': mae,
                    'r2_score': r2_score,
                    'mean_prediction': pred_ratings.mean().item(),
                    'mean_actual': true_ratings.mean().item(),
                    'std_prediction': pred_ratings.std().item(),
                    'std_actual': true_ratings.std().item(),
                    'num_test_ratings': test_mask.sum().item()
                })
            
            # Model information
            total_params = sum(p.numel() for p in sae_model.parameters())
            metrics.update({
                'model_type': 'SAE',
                'architecture': f"{sae_params['n_movies']} → {' → '.join(map(str, sae_params['hidden_dims']))} → {sae_params['n_movies']}",
                'total_parameters': total_params,
                'hyperparameters': sae_params
            })
            
            logger.info(f"SAE Evaluation completed - RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"SAE evaluation failed: {e}")
            return {'error': str(e)}
    
    def evaluate_rbm_model(self, use_best: bool = True) -> Dict[str, Any]:
        """Evaluate RBM model with comprehensive metrics."""
        
        logger.info("⚡ Evaluating Restricted Boltzmann Machine (RBM) model...")
        
        try:
            # Load RBM model
            rbm_model, rbm_params = self.model_loader.load_rbm_model(use_best=use_best)
            rbm_model.eval()
            
            # Prepare data for RBM (binary ratings)
            from models.rbm_model import convert_ratings_to_binary, evaluate_rbm_reconstruction
            
            # Check if we need to load RBM-specific data
            rbm_test_path = self.data_path / "preprocessed" / "test_matrix_rbm.pt"
            if rbm_test_path.exists():
                test_rbm = torch.load(rbm_test_path)
            else:
                # Convert from existing data
                if self.test_matrix.max() <= 1.0:
                    # Data is normalized, need to convert back first
                    original_scale = self.test_matrix * 4 + 1
                    test_rbm = convert_ratings_to_binary(original_scale)
                else:
                    test_rbm = convert_ratings_to_binary(self.test_matrix)
            
            # Evaluate reconstruction
            metrics = evaluate_rbm_reconstruction(rbm_model, test_rbm)
            
            # Additional RBM-specific metrics
            with torch.no_grad():
                # Use forward pass for reconstruction
                reconstructed = rbm_model(test_rbm)
                
                # Binary accuracy
                binary_accuracy = (reconstructed.round() == test_rbm).float().mean().item()
                
                # Free energy statistics
                free_energies = []
                batch_size = 100
                for i in range(0, test_rbm.size(0), batch_size):
                    batch = test_rbm[i:i+batch_size]
                    if batch.size(0) > 0:
                        fe = rbm_model.free_energy(batch)
                        free_energies.extend(fe.cpu().numpy())
                
                metrics.update({
                    'binary_accuracy': binary_accuracy,
                    'mean_free_energy': np.mean(free_energies),
                    'std_free_energy': np.std(free_energies),
                    'min_free_energy': np.min(free_energies),
                    'max_free_energy': np.max(free_energies)
                })
            
            # Model information
            total_params = sum(p.numel() for p in rbm_model.parameters())
            metrics.update({
                'model_type': 'RBM',
                'architecture': f"{rbm_params['visible_units']} visible ↔ {rbm_params['n_hidden']} hidden",
                'total_parameters': total_params,
                'hyperparameters': rbm_params
            })
            
            logger.info(f"RBM Evaluation completed - Reconstruction Error: {metrics.get('reconstruction_error', 'N/A'):.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"RBM evaluation failed: {e}")
            return {'error': str(e)}
    
    def compare_models(self, sae_metrics: Dict[str, Any], rbm_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare SAE and RBM models."""
        
        logger.info("📊 Comparing model performance...")
        
        comparison = {
            'sae': sae_metrics,
            'rbm': rbm_metrics,
            'comparison': {}
        }
        
        # Parameter comparison
        if 'total_parameters' in sae_metrics and 'total_parameters' in rbm_metrics:
            comparison['comparison']['parameter_efficiency'] = {
                'sae_params': sae_metrics['total_parameters'],
                'rbm_params': rbm_metrics['total_parameters'],
                'ratio': rbm_metrics['total_parameters'] / sae_metrics['total_parameters']
            }
        
        # Architecture comparison
        if 'architecture' in sae_metrics and 'architecture' in rbm_metrics:
            comparison['comparison']['architectures'] = {
                'sae': sae_metrics['architecture'],
                'rbm': rbm_metrics['architecture']
            }
        
        # Performance summary
        performance_summary = []
        if 'rmse' in sae_metrics:
            performance_summary.append(f"SAE RMSE: {sae_metrics['rmse']:.4f}")
        if 'reconstruction_error' in rbm_metrics:
            performance_summary.append(f"RBM Reconstruction Error: {rbm_metrics['reconstruction_error']:.4f}")
        
        comparison['comparison']['performance_summary'] = performance_summary
        
        return comparison
    
    def create_visualizations(self, sae_metrics: Dict[str, Any], rbm_metrics: Dict[str, Any]) -> None:
        """Create comprehensive visualizations of model performance."""
        
        logger.info("📈 Creating performance visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Architecture Comparison
        ax1 = plt.subplot(2, 3, 1)
        models = ['SAE', 'RBM']
        params = [
            sae_metrics.get('total_parameters', 0),
            rbm_metrics.get('total_parameters', 0)
        ]
        
        bars = ax1.bar(models, params, color=['skyblue', 'lightcoral'])
        ax1.set_title('Model Parameter Count', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Parameters')
        
        # Add value labels on bars
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{param:,}', ha='center', va='bottom')
        
        # 2. Performance Metrics Comparison
        ax2 = plt.subplot(2, 3, 2)
        metrics_data = []
        
        if 'rmse' in sae_metrics:
            metrics_data.append(('SAE RMSE', sae_metrics['rmse']))
        if 'reconstruction_error' in rbm_metrics:
            metrics_data.append(('RBM Recon. Error', rbm_metrics['reconstruction_error']))
        
        if metrics_data:
            metric_names, metric_values = zip(*metrics_data)
            bars = ax2.bar(metric_names, metric_values, color=['green', 'orange'])
            ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Error Value')
            
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')
        
        # 3. Rating Distribution Analysis (if available)
        ax3 = plt.subplot(2, 3, 3)
        if 'mean_prediction' in sae_metrics and 'mean_actual' in sae_metrics:
            categories = ['Actual Ratings', 'SAE Predictions']
            means = [sae_metrics['mean_actual'], sae_metrics['mean_prediction']]
            stds = [sae_metrics.get('std_actual', 0), sae_metrics.get('std_prediction', 0)]
            
            bars = ax3.bar(categories, means, yerr=stds, capsize=5, color=['blue', 'red'], alpha=0.7)
            ax3.set_title('Rating Statistics (SAE)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Rating Value')
            
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.3f}', ha='center', va='bottom')
        
        # 4. Model Architecture Visualization
        ax4 = plt.subplot(2, 3, 4)
        ax4.text(0.1, 0.8, 'SAE Architecture:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.6, sae_metrics.get('architecture', 'N/A'), fontsize=10, transform=ax4.transAxes)
        ax4.text(0.1, 0.4, 'RBM Architecture:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.2, rbm_metrics.get('architecture', 'N/A'), fontsize=10, transform=ax4.transAxes)
        ax4.set_title('Model Architectures', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        # 5. Hyperparameter Summary
        ax5 = plt.subplot(2, 3, 5)
        sae_hp = sae_metrics.get('hyperparameters', {})
        rbm_hp = rbm_metrics.get('hyperparameters', {})
        
        hp_text = "SAE Hyperparameters:\n"
        for key, value in list(sae_hp.items())[:5]:  # Show first 5
            if key not in ['hidden_dims', 'n_movies', 'model_type']:
                hp_text += f"  {key}: {value}\n"
        
        hp_text += "\nRBM Hyperparameters:\n"
        for key, value in list(rbm_hp.items())[:5]:  # Show first 5
            if key not in ['visible_units', 'model_type']:
                hp_text += f"  {key}: {value}\n"
        
        ax5.text(0.05, 0.95, hp_text, fontsize=9, transform=ax5.transAxes, verticalalignment='top')
        ax5.set_title('Hyperparameters', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        # 6. Performance Summary
        ax6 = plt.subplot(2, 3, 6)
        summary_text = "Performance Summary:\n\n"
        
        if 'rmse' in sae_metrics:
            summary_text += f"SAE RMSE: {sae_metrics['rmse']:.4f}\n"
        if 'mae' in sae_metrics:
            summary_text += f"SAE MAE: {sae_metrics['mae']:.4f}\n"
        if 'r2_score' in sae_metrics:
            summary_text += f"SAE R²: {sae_metrics['r2_score']:.4f}\n"
        
        summary_text += "\n"
        
        if 'reconstruction_error' in rbm_metrics:
            summary_text += f"RBM Recon. Error: {rbm_metrics['reconstruction_error']:.4f}\n"
        if 'binary_accuracy' in rbm_metrics:
            summary_text += f"RBM Binary Acc.: {rbm_metrics['binary_accuracy']:.4f}\n"
        
        ax6.text(0.05, 0.95, summary_text, fontsize=10, transform=ax6.transAxes, verticalalignment='top')
        ax6.set_title('Performance Summary', fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.results_path / 'model_evaluation_dashboard.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {viz_path}")
        
        plt.show()
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to JSON file."""
        
        results_file = self.results_path / 'evaluation_results.json'
        
        # Convert any tensor values to regular Python types for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.dim() == 0 else obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_tensors(results)
        
        with open(results_file, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def run_comprehensive_evaluation(self, models: List[str] = ['sae', 'rbm'], 
                                   dataset: str = 'ml-100k', visualize: bool = True) -> Dict[str, Any]:
        """Run comprehensive evaluation of specified models."""
        
        logger.info("🚀 Starting Comprehensive Model Evaluation")
        logger.info("="*60)
        
        # Load data
        self.load_data(dataset)
        
        results = {
            'dataset': dataset,
            'evaluation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_evaluated': models
        }
        
        sae_metrics = {}
        rbm_metrics = {}
        
        # Evaluate SAE
        if 'sae' in models:
            sae_metrics = self.evaluate_sae_model()
            results['sae'] = sae_metrics
        
        # Evaluate RBM
        if 'rbm' in models:
            rbm_metrics = self.evaluate_rbm_model()
            results['rbm'] = rbm_metrics
        
        # Compare models if both evaluated
        if 'sae' in models and 'rbm' in models:
            comparison = self.compare_models(sae_metrics, rbm_metrics)
            results['comparison'] = comparison['comparison']
        
        # Create visualizations
        if visualize and sae_metrics and rbm_metrics:
            self.create_visualizations(sae_metrics, rbm_metrics)
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self.print_evaluation_summary(results)
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted summary of evaluation results."""
        
        print("\n" + "="*70)
        print("MODEL EVALUATION SUMMARY")
        print("="*70)
        
        if 'sae' in results and 'error' not in results['sae']:
            sae = results['sae']
            print(f"\n🧠 SAE (Stacked AutoEncoder):")
            print(f"   Architecture: {sae.get('architecture', 'N/A')}")
            print(f"   Parameters: {sae.get('total_parameters', 'N/A'):,}")
            print(f"   RMSE: {sae.get('rmse', 'N/A'):.4f}" if 'rmse' in sae else "   RMSE: N/A")
            print(f"   MAE: {sae.get('mae', 'N/A'):.4f}" if 'mae' in sae else "   MAE: N/A")
            print(f"   R² Score: {sae.get('r2_score', 'N/A'):.4f}" if 'r2_score' in sae else "   R² Score: N/A")
        
        if 'rbm' in results and 'error' not in results['rbm']:
            rbm = results['rbm']
            print(f"\n⚡ RBM (Restricted Boltzmann Machine):")
            print(f"   Architecture: {rbm.get('architecture', 'N/A')}")
            print(f"   Parameters: {rbm.get('total_parameters', 'N/A'):,}")
            print(f"   Reconstruction Error: {rbm.get('reconstruction_error', 'N/A'):.4f}" if 'reconstruction_error' in rbm else "   Reconstruction Error: N/A")
            print(f"   Binary Accuracy: {rbm.get('binary_accuracy', 'N/A'):.4f}" if 'binary_accuracy' in rbm else "   Binary Accuracy: N/A")
        
        if 'comparison' in results:
            comp = results['comparison']
            print(f"\n📊 Model Comparison:")
            if 'parameter_efficiency' in comp:
                pe = comp['parameter_efficiency']
                print(f"   Parameter Ratio (RBM/SAE): {pe.get('ratio', 'N/A'):.2f}")
        
        print(f"\n📁 Results saved to: {self.results_path}")
        print("="*70)


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--models', nargs='+', choices=['sae', 'rbm'], 
                       default=['sae', 'rbm'], help='Models to evaluate')
    parser.add_argument('--dataset', choices=['ml-100k', 'ml-1m'], 
                       default='ml-100k', help='Dataset to use')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visualization dashboard')
    parser.add_argument('--results-path', default='evaluation_results',
                       help='Path to save results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(results_path=args.results_path)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        models=args.models,
        dataset=args.dataset,
        visualize=args.visualize
    )
    
    return results


if __name__ == "__main__":
    results = main()
