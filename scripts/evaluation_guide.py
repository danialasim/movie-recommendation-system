#!/usr/bin/env python3
"""
Model Evaluation Usage Guide

This script demonstrates how to use the model evaluation system
and shows different ways to evaluate your trained models.
"""

import json
from pathlib import Path

def show_evaluation_usage():
    """Show how to use the evaluation system."""
    
    print("🎯 MODEL EVALUATION SYSTEM USAGE GUIDE")
    print("="*60)
    
    print("\n📋 Available Evaluation Scripts:")
    print("="*40)
    
    print("\n1️⃣ Quick Evaluation (Recommended for fast testing)")
    print("   Command: python quick_evaluate.py")
    print("   Features:")
    print("   • Fast evaluation of both SAE and RBM models")
    print("   • Key performance metrics")
    print("   • Model architecture and hyperparameter display")
    print("   • Loads saved models with best hyperparameters")
    
    print("\n2️⃣ Comprehensive Evaluation (Full analysis)")
    print("   Command: python evaluate_models.py --models sae rbm --visualize")
    print("   Features:")
    print("   • Detailed performance analysis")
    print("   • Visual dashboard with charts")
    print("   • Model comparison")
    print("   • Results saved to JSON file")
    print("   • Comprehensive metrics including precision, recall, NDCG")
    
    print("\n3️⃣ Individual Model Evaluation")
    print("   SAE only: python evaluate_models.py --models sae")
    print("   RBM only: python evaluate_models.py --models rbm")
    
    print("\n📊 Available Metrics:")
    print("="*30)
    
    print("\n🧠 SAE (Stacked AutoEncoder) Metrics:")
    print("   • RMSE (Root Mean Square Error)")
    print("   • MAE (Mean Absolute Error)")
    print("   • R² Score (Coefficient of Determination)")
    print("   • Precision@K, Recall@K, NDCG@K")
    print("   • Catalog Coverage")
    
    print("\n⚡ RBM (Restricted Boltzmann Machine) Metrics:")
    print("   • Reconstruction Error")
    print("   • Binary Accuracy")
    print("   • Free Energy Statistics")
    print("   • Average Free Energy")
    
    print("\n📈 Visualization Features:")
    print("="*35)
    print("   • Model architecture comparison")
    print("   • Parameter count comparison")
    print("   • Performance metrics charts")
    print("   • Hyperparameter displays")
    print("   • Training statistics")

def show_latest_results():
    """Show the latest evaluation results."""
    
    print("\n🏆 LATEST EVALUATION RESULTS")
    print("="*40)
    
    results_file = Path("evaluation_results/evaluation_results.json")
    
    if not results_file.exists():
        print("❌ No evaluation results found. Run evaluation first:")
        print("   python quick_evaluate.py")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n📅 Evaluation Date: {results.get('evaluation_timestamp', 'Unknown')}")
    print(f"📊 Dataset: {results.get('dataset', 'Unknown')}")
    
    if 'sae' in results:
        sae = results['sae']
        print(f"\n🧠 SAE Performance:")
        print(f"   Architecture: {sae.get('architecture', 'N/A')}")
        print(f"   Parameters: {sae.get('total_parameters', 'N/A'):,}")
        print(f"   RMSE: {sae.get('rmse', 'N/A'):.4f}")
        print(f"   MAE: {sae.get('mae', 'N/A'):.4f}")
        print(f"   R² Score: {sae.get('r2_score', 'N/A'):.4f}")
    
    if 'rbm' in results:
        rbm = results['rbm']
        print(f"\n⚡ RBM Performance:")
        print(f"   Architecture: {rbm.get('architecture', 'N/A')}")
        print(f"   Parameters: {rbm.get('total_parameters', 'N/A'):,}")
        print(f"   Reconstruction Error: {rbm.get('reconstruction_error', 'N/A'):.4f}")
        print(f"   Binary Accuracy: {rbm.get('binary_accuracy', 'N/A'):.4f}")
    
    if 'comparison' in results:
        comp = results['comparison']
        print(f"\n📊 Model Comparison:")
        if 'parameter_efficiency' in comp:
            pe = comp['parameter_efficiency']
            print(f"   RBM has {pe.get('ratio', 'N/A'):.2f}x more parameters than SAE")

def show_performance_interpretation():
    """Show how to interpret the performance metrics."""
    
    print("\n🔍 PERFORMANCE METRICS INTERPRETATION")
    print("="*50)
    
    print("\n📏 Rating Prediction Metrics (SAE):")
    print("   • RMSE (Root Mean Square Error):")
    print("     - Lower is better (0 = perfect)")
    print("     - Typical range: 0.8-1.2 for movie ratings")
    print("     - Heavily penalizes large errors")
    
    print("\n   • MAE (Mean Absolute Error):")
    print("     - Lower is better (0 = perfect)")
    print("     - More interpretable than RMSE")
    print("     - Average prediction error in rating points")
    
    print("\n   • R² Score (Coefficient of Determination):")
    print("     - Higher is better (1 = perfect, 0 = as good as mean)")
    print("     - Negative values = worse than predicting the mean")
    print("     - Measures how much variance is explained")
    
    print("\n🔄 Reconstruction Metrics (RBM):")
    print("   • Reconstruction Error:")
    print("     - Lower is better")
    print("     - Measures how well the model reconstructs input")
    
    print("\n   • Binary Accuracy:")
    print("     - Higher is better (1.0 = 100% accurate)")
    print("     - Percentage of correctly predicted binary preferences")
    
    print("\n🎯 Recommendation Quality Metrics:")
    print("   • Precision@K: Proportion of relevant items in top-K")
    print("   • Recall@K: Proportion of relevant items found in top-K")
    print("   • NDCG@K: Normalized Discounted Cumulative Gain")
    print("   • Catalog Coverage: Diversity of recommended items")

def main():
    """Main function."""
    
    show_evaluation_usage()
    show_latest_results()
    show_performance_interpretation()
    
    print(f"\n✨ QUICK START:")
    print("="*20)
    print("1. Run quick evaluation: python quick_evaluate.py")
    print("2. Run full evaluation: python evaluate_models.py --visualize")
    print("3. Check results in: evaluation_results/")
    
    print(f"\n📝 Notes:")
    print("• Models are automatically loaded with their best hyperparameters")
    print("• Evaluation uses the test set from training")
    print("• Results are saved for future reference")
    print("• Visualizations help compare model performance")

if __name__ == "__main__":
    main()
