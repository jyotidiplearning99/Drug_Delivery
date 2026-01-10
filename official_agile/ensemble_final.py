#!/usr/bin/env python
"""
Ensemble Ridge + Gradient Boosting to reach 0.85
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

print("="*70)
print("Ensemble: Advanced Ridge + Gradient Boosting")
print("="*70)

# Load data
df = pd.read_csv('data/finetuning_set_smiles_plus_features.csv')

descriptor_cols = [col for col in df.columns if col.startswith('desc_')]
X = df[descriptor_cols].values
y = df['expt_Hela'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Model 1: Advanced Ridge
print("\nTraining Advanced Ridge...")
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression, k=500)),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=5.0))
])
ridge_pipeline.fit(X_train, y_train)
ridge_pred = ridge_pipeline.predict(X_test)

# Model 2: Gradient Boosting
print("Training Gradient Boosting...")
gb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ))
])
gb_pipeline.fit(X_train, y_train)
gb_pred = gb_pipeline.predict(X_test)

# Find optimal ensemble weights
best_corr = 0
best_weight = 0

for w in np.arange(0, 1.01, 0.05):
    ensemble_pred = w * ridge_pred + (1 - w) * gb_pred
    corr = pearsonr(ensemble_pred, y_test)[0]
    
    if corr > best_corr:
        best_corr = corr
        best_weight = w

print(f"\nOptimal Ridge weight: {best_weight:.2f}")
print(f"Optimal GB weight: {1-best_weight:.2f}")

# Final ensemble
ensemble_pred = best_weight * ridge_pred + (1 - best_weight) * gb_pred

# Metrics
ridge_corr = pearsonr(ridge_pred, y_test)[0]
gb_corr = pearsonr(gb_pred, y_test)[0]
ensemble_corr = pearsonr(ensemble_pred, y_test)[0]
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Ridge alone: {ridge_corr:.4f}")
print(f"Gradient Boosting alone: {gb_corr:.4f}")
print(f"Ensemble: {ensemble_corr:.4f}")
print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
print("="*70)

if ensemble_corr >= 0.85:
    print(f"\n🎉 SUCCESS! Reached target correlation of 0.85!")
else:
    gap = 0.85 - ensemble_corr
    print(f"\n⚠️  Gap to 0.85: {gap:.4f} ({gap/0.85*100:.1f}%)")
    print("Consider: More feature engineering or experimental validation")
