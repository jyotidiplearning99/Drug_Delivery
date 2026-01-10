#!/usr/bin/env python
"""
Final push to 0.85 with expanded stacked ensemble
Expected improvement: 0.8110 → 0.85+
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import xgboost as xgb
import lightgbm as lgb

print("="*70)
print("Final Push to 0.85 - Expanded Stacked Ensemble")
print("="*70)

# Load ALL 1,200 samples
df = pd.read_csv('data/finetuning_set_smiles_plus_features.csv')
descriptor_cols = [col for col in df.columns if col.startswith('desc_')]
X = df[descriptor_cols].values
y = df['expt_Hela'].values

print(f"\nUsing FULL dataset: {len(X)} samples, {len(descriptor_cols)} features")

# 80/10/10 split as per the Nature paper
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Expanded base models (10 diverse models)
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
    ('bayesian_ridge', Pipeline([
        ('scaler', StandardScaler()),
        ('br', BayesianRidge())
    ])),
    ('elastic_net', Pipeline([
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
    ])),
    ('lgb', Pipeline([
        ('scaler', StandardScaler()),
        ('lgb', lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1
        ))
    ])),
    ('adaboost', Pipeline([
        ('scaler', StandardScaler()),
        ('ada', AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ))
    ]))
]

print("\nTraining 10 base models...")
meta_features_train = np.zeros((len(X_train), len(base_models)))
meta_features_val = np.zeros((len(X_val), len(base_models)))
meta_features_test = np.zeros((len(X_test), len(base_models)))

base_predictions = []

for i, (name, model) in enumerate(base_models):
    print(f"  Training {name}...")
    
    # K-fold for meta-features
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, fold_val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[fold_val_idx]
        y_fold_train = y_train[train_idx]
        
        model.fit(X_fold_train, y_fold_train)
        meta_features_train[fold_val_idx, i] = model.predict(X_fold_val)
    
    # Train on full training set
    model.fit(X_train, y_train)
    meta_features_val[:, i] = model.predict(X_val)
    meta_features_test[:, i] = model.predict(X_test)
    
    # Evaluate individual model
    test_pred = model.predict(X_test)
    test_corr = pearsonr(test_pred, y_test)[0]
    base_predictions.append({
        'name': name,
        'correlation': test_corr
    })
    print(f"    Test Correlation: {test_corr:.4f}")

# Meta-learner (Ridge for stability)
print("\nTraining meta-learner...")
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features_train, y_train)

# Final predictions
final_pred = meta_model.predict(meta_features_test)

# Metrics
final_corr = pearsonr(final_pred, y_test)[0]
final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Test Correlation: {final_corr:.4f}")
print(f"Test RMSE: {final_rmse:.4f}")
print("="*70)

print("\nBase Model Performance:")
for pred in sorted(base_predictions, key=lambda x: x['correlation'], reverse=True):
    print(f"  {pred['name']}: {pred['correlation']:.4f}")

print("\n" + "="*70)
if final_corr >= 0.85:
    print(f"🎉 SUCCESS! Reached 0.85 target!")
    print(f"Final correlation: {final_corr:.4f}")
    improvement = ((final_corr - 0.8110) / 0.8110) * 100
    print(f"Improvement over previous: +{improvement:.2f}%")
else:
    gap = 0.85 - final_corr
    print(f"Current: {final_corr:.4f}")
    print(f"Gap to 0.85: {gap:.4f} ({gap/0.85*100:.1f}%)")
    
    if final_corr > 0.8110:
        improvement = ((final_corr - 0.8110) / 0.8110) * 100
        print(f"Improvement: +{improvement:.2f}%")
print("="*70)

# Save results
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': final_pred
})
results_df.to_csv('final_ensemble_predictions.csv', index=False)
print("\n✓ Predictions saved to: final_ensemble_predictions.csv")
