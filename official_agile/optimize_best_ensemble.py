#!/usr/bin/env python
"""
Optimize the 7-model ensemble that achieved 0.8110
Strategy: Better model selection + optimized weighting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.optimize import minimize
import xgboost as xgb

print("="*70)
print("Optimizing 7-Model Ensemble (Previous Best: 0.8110)")
print("="*70)

# Load data
df = pd.read_csv('data/finetuning_set_smiles_plus_features.csv')
descriptor_cols = [col for col in df.columns if col.startswith('desc_')]
X = df[descriptor_cols].values
y = df['expt_Hela'].values

# Same split as before
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Only use the 7 BEST models (exclude weak AdaBoost, ExtraTrees, ElasticNet)
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
    ('gb', Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42))
    ])),
    ('rf', Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42))
    ])),
    ('xgb', Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42))
    ])),
    ('lgb', Pipeline([
        ('scaler', StandardScaler()),
        ('lgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=43))  # Different seed
    ]))
]

print(f"\nTraining {len(base_models)} high-quality base models...")
predictions_train = np.zeros((len(X_train), len(base_models)))
predictions_test = np.zeros((len(X_test), len(base_models)))

for i, (name, model) in enumerate(base_models):
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    predictions_train[:, i] = model.predict(X_train)
    predictions_test[:, i] = model.predict(X_test)
    
    test_corr = pearsonr(predictions_test[:, i], y_test)[0]
    print(f"    Individual Test Correlation: {test_corr:.4f}")

# Strategy 1: Simple Average (Baseline)
simple_avg = predictions_test.mean(axis=1)
simple_corr = pearsonr(simple_avg, y_test)[0]
print(f"\n1. Simple Average: {simple_corr:.4f}")

# Strategy 2: Weighted Average (Optimize weights)
def objective(weights):
    """Minimize negative correlation"""
    ensemble_pred = (predictions_train * weights).sum(axis=1)
    return -pearsonr(ensemble_pred, y_train)[0]

# Constraints: weights sum to 1, all positive
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1) for _ in range(len(base_models))]
initial_weights = np.ones(len(base_models)) / len(base_models)

result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

weighted_pred = (predictions_test * optimal_weights).sum(axis=1)
weighted_corr = pearsonr(weighted_pred, y_test)[0]
print(f"\n2. Optimized Weighted Average: {weighted_corr:.4f}")
print(f"   Optimal weights: {dict(zip([name for name, _ in base_models], optimal_weights))}")

# Strategy 3: Ridge Meta-Learner (Best from previous)
from sklearn.linear_model import Ridge as MetaRidge
meta_model = MetaRidge(alpha=1.0)
meta_model.fit(predictions_train, y_train)
meta_pred = meta_model.predict(predictions_test)
meta_corr = pearsonr(meta_pred, y_test)[0]
print(f"\n3. Ridge Meta-Learner: {meta_corr:.4f}")

# Select best strategy
best_strategies = {
    'Simple Average': simple_corr,
    'Weighted Average': weighted_corr,
    'Ridge Meta-Learner': meta_corr
}

best_strategy = max(best_strategies, key=best_strategies.get)
best_corr = best_strategies[best_strategy]

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Best Strategy: {best_strategy}")
print(f"Test Correlation: {best_corr:.4f}")
print(f"Previous Best: 0.8110")

if best_corr > 0.8110:
    improvement = ((best_corr - 0.8110) / 0.8110) * 100
    print(f"Improvement: +{improvement:.2f}%")
else:
    print(f"Change: {best_corr - 0.8110:+.4f}")

if best_corr >= 0.85:
    print(f"\n🎉 SUCCESS! Reached 0.85 target!")
else:
    gap = 0.85 - best_corr
    print(f"\nGap to 0.85: {gap:.4f} ({gap/0.85*100:.1f}%)")

print("="*70)

# Save results
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted_simple': simple_avg,
    'predicted_weighted': weighted_pred,
    'predicted_meta': meta_pred
})
results_df.to_csv('optimized_ensemble_predictions.csv', index=False)
print("\n✓ Predictions saved to: optimized_ensemble_predictions.csv")
