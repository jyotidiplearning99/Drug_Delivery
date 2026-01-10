#!/usr/bin/env python
"""
Advanced Ridge with polynomial features and feature selection
Expected: 0.78-0.83 correlation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

print("="*70)
print("Advanced Ridge with Feature Engineering")
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

# Pipeline with feature selection and polynomial features
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression, k=500)),  # Select top 500 features
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Add interactions
    ('ridge', Ridge(alpha=5.0))
])

print("\nTraining advanced Ridge pipeline...")
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_corr = pearsonr(y_test, y_pred)[0]

print("\n" + "="*70)
print("ADVANCED RIDGE RESULTS")
print("="*70)
print(f"Test R²: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test Correlation: {test_corr:.4f}")
print("="*70)

print(f"\nImprovement over baseline Ridge (0.749): {((test_corr - 0.749) / 0.749 * 100):+.1f}%")
