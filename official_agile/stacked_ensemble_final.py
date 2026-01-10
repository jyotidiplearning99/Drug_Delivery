#!/usr/bin/env python
"""
Stacked Ensemble with Meta-Learner - Final Push to 0.85
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import xgboost as xgb

print("="*70)
print("Stacked Ensemble - Final Push to 0.85")
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

# Base models (diverse set for maximum performance)
base_models = [
    ('ridge_poly', Pipeline([
        ('scaler', StandardScaler()),
        ('fs', SelectKBest(f_regression, k=500)),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=5.0))
    ])),
    ('ridge_simple', Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])),
    ('elastic', Pipeline([
        ('scaler', StandardScaler()),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5))
    ])),
    ('gb', Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            random_state=42
        ))
    ])),
    ('rf', Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42
        ))
    ])),
    ('extra_trees', Pipeline([
        ('scaler', StandardScaler()),
        ('et', ExtraTreesRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42
        ))
    ])),
    ('xgb', Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ))
    ]))
]

# Train base models and generate meta-features
print("\nTraining base models...")
meta_features_train = np.zeros((len(X_train), len(base_models)))
meta_features_test = np.zeros((len(X_test), len(base_models)))

base_predictions = []

for i, (name, model) in enumerate(base_models):
    print(f"  Training {name}...")
    
    # K-fold for meta-features (avoid overfitting)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train = y_train[train_idx]
        
        model.fit(X_fold_train, y_fold_train)
        meta_features_train[val_idx, i] = model.predict(X_fold_val)
    
    # Train on full training set for test predictions
    model.fit(X_train, y_train)
    meta_features_test[:, i] = model.predict(X_test)
    
    # Calculate individual model performance
    test_pred = model.predict(X_test)
    test_corr = pearsonr(test_pred, y_test)[0]
    base_predictions.append({
        'name': name,
        'correlation': test_corr
    })
    print(f"    {name} Test Correlation: {test_corr:.4f}")

# Meta-learner (simple Ridge for stability)
print("\nTraining meta-learner...")
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features_train, y_train)

# Final predictions
final_pred = meta_model.predict(meta_features_test)

# Metrics
final_corr = pearsonr(final_pred, y_test)[0]
final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))

print("\n" + "="*70)
print("STACKED ENSEMBLE RESULTS")
print("="*70)
print(f"Test Correlation: {final_corr:.4f}")
print(f"Test RMSE: {final_rmse:.4f}")
print("="*70)

print("\nBase Model Performance:")
for pred in base_predictions:
    print(f"  {pred['name']}: {pred['correlation']:.4f}")

print("\n" + "="*70)
if final_corr >= 0.85:
    print(f"🎉 SUCCESS! Reached target correlation of 0.85!")
    print(f"Final correlation: {final_corr:.4f}")
else:
    gap = 0.85 - final_corr
    print(f"Current best: {final_corr:.4f}")
    print(f"Gap to 0.85: {gap:.4f} ({gap/0.85*100:.1f}%)")

print("="*70)

# Save predictions
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': final_pred
})
results_df.to_csv('stacked_ensemble_predictions.csv', index=False)
print("\n✓ Predictions saved to: stacked_ensemble_predictions.csv")
