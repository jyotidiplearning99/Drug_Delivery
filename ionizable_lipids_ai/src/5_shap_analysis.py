"""
SHAP Analysis Module
Interprets model decisions using SHAP (SHapley Additive exPlanations)
Following Nature Communications 2024 paper methodology
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from rdkit import Chem
from rdkit.Chem import Draw
from config import *


def load_models():
    """Load trained models"""
    
    model1 = joblib.load(MODEL_DIR / 'delivery_model1.pkl')
    model2 = joblib.load(MODEL_DIR / 'delivery_model2.pkl')
    pka_model = joblib.load(MODEL_DIR / 'pka_model.pkl')
    
    print("✓ Models loaded successfully")
    return model1, model2, pka_model


def load_test_data():
    """Load test features and metadata"""
    
    X_test = np.load(PROCESSED_DATA_DIR / 'X_test.npy')
    metadata = pd.read_csv(PROCESSED_DATA_DIR / 'metadata_test.csv')
    
    print(f"✓ Loaded test data: {X_test.shape}")
    return X_test, metadata


def calculate_shap_values(model, X_test, model_name):
    """
    Calculate SHAP values for model interpretability
    Paper: Uses TreeExplainer for LightGBM models
    """
    
    print(f"\nCalculating SHAP values for {model_name}...")
    
    # Create TreeExplainer (optimized for tree-based models)
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values (use subset for speed)
    sample_size = min(100, len(X_test))
    shap_values = explainer.shap_values(X_test[:sample_size])
    
    # For classification models, shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    print(f"✓ SHAP values calculated: {shap_values.shape}")
    
    return shap_values, explainer


def create_shap_summary_plot(shap_values, X_test, model_name):
    """
    Create SHAP summary plot showing feature importance
    Paper Figure: Feature importance visualization
    """
    
    print(f"\nCreating SHAP summary plot for {model_name}...")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # SHAP summary plot
    shap.summary_plot(
        shap_values,
        X_test[:len(shap_values)],
        max_display=20,
        show=False,
        plot_type='bar'
    )
    
    plt.title(f'SHAP Feature Importance - {model_name}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = f'shap_summary_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def create_shap_beeswarm_plot(shap_values, X_test, model_name):
    """
    Create SHAP beeswarm plot showing feature effects
    """
    
    print(f"\nCreating SHAP beeswarm plot for {model_name}...")
    
    plt.figure(figsize=(12, 10))
    
    shap.summary_plot(
        shap_values,
        X_test[:len(shap_values)],
        max_display=20,
        show=False
    )
    
    plt.title(f'SHAP Feature Effects - {model_name}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'shap_beeswarm_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def calculate_ecfp_scores(shap_values, X_test, metadata):
    """
    Calculate ECFP scores following paper Equation 5-6
    
    Paper methodology:
    Con_i = Σ(SHAP_i,j × t_ECFP_i,j)
    Score_j = Σ(Con_i × ECFP_i,j)
    
    Where t_ECFP = 1 if ECFP=1, else -1
    """
    
    print("\nCalculating ECFP scores (Paper Equation 5-6)...")
    
    # Calculate contribution of each ECFP bit
    # ECFP bits are first 1024 features
    ecfp_bits = X_test[:, :ECFP_BITS]
    ecfp_shap = shap_values[:, :ECFP_BITS]
    
    # Transform ECFP: t_ECFP = 1 if ECFP=1, else -1
    t_ecfp = np.where(ecfp_bits == 1, 1, -1)
    
    # Contribution of each bit (Equation 5)
    contributions = ecfp_shap * t_ecfp
    
    # Calculate ECFP score for each lipid (Equation 6)
    ecfp_scores = np.sum(contributions * ecfp_bits, axis=1)
    
    # Add to metadata
    metadata_subset = metadata.iloc[:len(ecfp_scores)].copy()
    metadata_subset['ecfp_score'] = ecfp_scores
    
    # Sort by score
    metadata_subset = metadata_subset.sort_values('ecfp_score', ascending=False)
    
    print(f"✓ ECFP scores calculated")
    print(f"  Score range: {ecfp_scores.min():.2f} to {ecfp_scores.max():.2f}")
    
    # Save
    metadata_subset.to_csv(RESULTS_DIR / 'ecfp_scores.csv', index=False)
    
    return metadata_subset


def visualize_ecfp_score_correlation(metadata_with_scores):
    """
    Visualize correlation between ECFP score and delivery efficiency
    Paper: Supplementary Figure 4b
    """
    
    print("\nVisualizing ECFP score correlation...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(
        metadata_with_scores['ecfp_score'],
        metadata_with_scores['fold_vs_mc3'],
        alpha=0.6,
        s=50,
        c='#3498db'
    )
    
    # Trend line
    z = np.polyfit(
        metadata_with_scores['ecfp_score'],
        metadata_with_scores['fold_vs_mc3'],
        1
    )
    p = np.poly1d(z)
    x_line = np.linspace(
        metadata_with_scores['ecfp_score'].min(),
        metadata_with_scores['ecfp_score'].max(),
        100
    )
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)
    
    # Labels
    ax.set_xlabel('ECFP Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fold vs MC3', fontsize=12, fontweight='bold')
    ax.set_title('ECFP Score vs Delivery Efficiency', 
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'ecfp_score_correlation.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: ecfp_score_correlation.png")
    plt.close()


def identify_important_substructures(shap_values, X_test, metadata, top_n=10):
    """
    Identify most important ECFP bits (substructures)
    Paper: Used to inform lipid generation
    """
    
    print(f"\nIdentifying top {top_n} important substructures...")
    
    # Calculate mean absolute SHAP value for each ECFP bit
    ecfp_shap = shap_values[:, :ECFP_BITS]
    mean_abs_shap = np.abs(ecfp_shap).mean(axis=0)
    
    # Get top indices
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    
    # Create results
    results = []
    for idx in top_indices:
        results.append({
            'ecfp_bit': idx,
            'mean_abs_shap': mean_abs_shap[idx],
            'positive_contribution': (ecfp_shap[:, idx] > 0).sum(),
            'negative_contribution': (ecfp_shap[:, idx] < 0).sum()
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / 'important_ecfp_bits.csv', index=False)
    
    print("✓ Important ECFP bits identified")
    print(results_df.to_string(index=False))
    
    return results_df


def create_feature_importance_comparison(model1, model2, pka_model):
    """
    Compare feature importance across all three models
    """
    
    print("\nCreating feature importance comparison...")
    
    # Get feature importances
    model1_importance = model1.feature_importances_
    model2_importance = model2.feature_importances_
    pka_importance = pka_model.feature_importances_
    
    # Top 20 features for each model
    top_n = 20
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, importance, title in zip(
        axes,
        [model1_importance, model2_importance, pka_importance],
        ['Model 1 (≥1× MC3)', 'Model 2 (≥2× MC3)', 'pKa Model']
    ):
        # Get top features
        top_idx = np.argsort(importance)[-top_n:]
        top_values = importance[top_idx]
        
        # Plot
        ax.barh(range(top_n), top_values, color='#3498db', alpha=0.7)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f'Feature {i}' for i in top_idx])
        ax.set_xlabel('Feature Importance', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_importance_comparison.png',
                dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance_comparison.png")
    plt.close()


def main():
    """Main SHAP analysis pipeline"""
    
    print("="*70)
    print("SHAP ANALYSIS MODULE")
    print("Interpreting AI models for ionizable lipid design")
    print("="*70)
    
    # Load models and data
    model1, model2, pka_model = load_models()
    X_test, metadata = load_test_data()
    
    # SHAP analysis for Model 1
    shap_values_m1, explainer_m1 = calculate_shap_values(
        model1, X_test, "Model 1 (≥1× MC3)"
    )
    create_shap_summary_plot(shap_values_m1, X_test, "Model 1")
    create_shap_beeswarm_plot(shap_values_m1, X_test, "Model 1")
    
    # SHAP analysis for Model 2
    shap_values_m2, explainer_m2 = calculate_shap_values(
        model2, X_test, "Model 2 (≥2× MC3)"
    )
    create_shap_summary_plot(shap_values_m2, X_test, "Model 2")
    create_shap_beeswarm_plot(shap_values_m2, X_test, "Model 2")
    
    # SHAP analysis for pKa model
    shap_values_pka, explainer_pka = calculate_shap_values(
        pka_model, X_test, "pKa Model"
    )
    create_shap_summary_plot(shap_values_pka, X_test, "pKa Model")
    
    # Calculate ECFP scores (Paper Equation 5-6)
    metadata_with_scores = calculate_ecfp_scores(
        shap_values_m1, X_test[:len(shap_values_m1)], metadata
    )
    
    # Visualize ECFP score correlation
    visualize_ecfp_score_correlation(metadata_with_scores)
    
    # Identify important substructures
    important_bits = identify_important_substructures(
        shap_values_m1, X_test, metadata, top_n=10
    )
    
    # Compare feature importance across models
    create_feature_importance_comparison(model1, model2, pka_model)
    
    # Save SHAP values for later use
    np.save(RESULTS_DIR / 'shap_values_model1.npy', shap_values_m1)
    np.save(RESULTS_DIR / 'shap_values_model2.npy', shap_values_m2)
    np.save(RESULTS_DIR / 'shap_values_pka.npy', shap_values_pka)
    
    print("\n" + "="*70)
    print("SHAP ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {RESULTS_DIR}")
    print("\nKey findings:")
    print(f"  - ECFP score range: {metadata_with_scores['ecfp_score'].min():.2f} to {metadata_with_scores['ecfp_score'].max():.2f}")
    print(f"  - Top 10 important ECFP bits identified")
    print(f"  - Feature importance compared across 3 models")
    print("\nNext: python src/6_virtual_screening.py")
    print("="*70)


if __name__ == '__main__':
    main()
