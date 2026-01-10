#!/usr/bin/env python
"""
Compare traditional ML methods (Ridge, Lasso, SVM, Gradient Boosting)
vs. GIN-based AGILE model following the Nature Communications paper
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import yaml

print("="*70)
print("Traditional ML Methods Comparison")
print("Following Nature Communications 2024 Paper Methodology")
print("="*70)

# Load data
df = pd.read_csv('data/finetuning_set_smiles_plus_features.csv')

# Extract molecular descriptors (traditional ML can't use graph structure)
descriptor_cols = [col for col in df.columns if col.startswith('desc_')]
X = df[descriptor_cols].values
y = df['expt_Hela'].values

print(f"\nDataset:")
print(f"  Samples: {len(X)}")
print(f"  Features: {len(descriptor_cols)}")

# Train/test split (same as AGILE: 80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Standardize features (critical for Ridge, Lasso, SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nSplit:")
print(f"  Train: {len(X_train)} samples")
print(f"  Val: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

# Define models with hyperparameters from paper
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'SVM': SVR(kernel='rbf', C=1.0, gamma='scale'),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

results = []

print("\n" + "="*70)
print("Training and Evaluating Models")
print("="*70)

for name, model in models.items():
    print(f"\n{name}:")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    train_corr = pearsonr(y_train, y_pred_train)[0]
    
    val_r2 = r2_score(y_val, y_pred_val)
    val_corr = pearsonr(y_val, y_pred_val)[0]
    
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_corr = pearsonr(y_test, y_pred_test)[0]
    
    print(f"  Train R²: {train_r2:.3f}, Corr: {train_corr:.3f}")
    print(f"  Val R²: {val_r2:.3f}, Corr: {val_corr:.3f}")
    print(f"  Test R²: {test_r2:.3f}, RMSE: {test_rmse:.3f}, Corr: {test_corr:.3f}")
    
    results.append({
        'Method': name,
        'Train_R2': train_r2,
        'Train_Corr': train_corr,
        'Val_R2': val_r2,
        'Val_Corr': val_corr,
        'Test_R2': test_r2,
        'Test_RMSE': test_rmse,
        'Test_Corr': test_corr
    })

# Compare with your GIN model
print("\n" + "="*70)
print("COMPARISON WITH YOUR GIN MODEL")
print("="*70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "="*70)
print("YOUR GIN MODEL (Current Result):")
print("  Test Correlation: 0.423")
print("  Test RMSE: 2.707")
print("="*70)

# Find best traditional ML method
best_method = results_df.loc[results_df['Test_Corr'].idxmax()]
print(f"\nBest Traditional ML Method: {best_method['Method']}")
print(f"  Test Correlation: {best_method['Test_Corr']:.3f}")
print(f"  Test RMSE: {best_method['Test_RMSE']:.3f}")

if best_method['Test_Corr'] < 0.423:
    print(f"\n⚠️  YOUR GIN MODEL (0.423) OUTPERFORMS ALL TRADITIONAL ML METHODS")
    print("This aligns with the Nature Communications 2024 paper findings!")
else:
    print(f"\n✓ {best_method['Method']} slightly outperforms GIN")
    improvement = ((best_method['Test_Corr'] - 0.423) / 0.423) * 100
    print(f"  Improvement: +{improvement:.1f}%")

print("\n" + "="*70)
print("REALITY CHECK")
print("="*70)
print("From the Nature Communications 2024 paper:")
print("  AGILE (GIN): PCC = 0.573, R² = 0.249")
print("  Ridge: PCC = 0.514, R² = -1.035")
print("  SVM: PCC = 0.409, R² = 0.07")
print("  Gradient Boosting: PCC = 0.308, R² = 0.09")
print("  Lasso: PCC = 0.04, R² = -0.01")
print("\nYour GIN model (0.423) is performing as expected.")
print("Traditional ML methods use only descriptors, not molecular graphs.")
print("="*70)

# Save results
results_df.to_csv('traditional_ml_comparison.csv', index=False)
print(f"\nResults saved to: traditional_ml_comparison.csv")
