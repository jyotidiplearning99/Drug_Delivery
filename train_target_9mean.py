#!/usr/bin/env python -u
"""
BEST APPROACH: Original NESTED_CV + CatBoost + 20 trials
Target: 1% improvement over 11.89% baseline
"""
import os
import sys
import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# SAFE unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

def print_flush(msg):
    print(msg, flush=True)

print_flush("="*70)
print_flush("BEST APPROACH: CatBoost + 20 Trials")
print_flush("Target: 1% improvement (10.9% MAE)")
print_flush("="*70)

# Add original code directory
sys.path.insert(0, 'data/pban-91-long-acting-injectables-0df9c4a/few_shot_models')

# Setup
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = results_dir / f'run_best_{timestamp}'
run_dir.mkdir(exist_ok=True)

print_flush(f"\nOutput: {run_dir}")

# Change to data directory
original_dir = os.getcwd()
os.chdir('data/pban-91-long-acting-injectables-0df9c4a/few_shot_models')

# Check dataset
print_flush("\nChecking dataset...")
if not os.path.exists("Dataset_17_feat.xlsx"):
    df = pd.read_csv("Dataset_17_feat.csv")
    df.to_excel("Dataset_17_feat.xlsx", index=False, engine='openpyxl')

# ============================================================================
# STRATEGY 1: CatBoost (Often 1-2% better than LGBM)
# ============================================================================
print_flush("\n" + "="*70)
print_flush("STRATEGY 1: CatBoost with 20 trials")
print_flush("="*70)

try:
    from catboost import CatBoostRegressor
    from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error
    
    data = pd.read_csv("Dataset_17_feat.csv")
    
    feature_cols = [col for col in data.columns if col not in 
                    ['Release', 'Experimental_index', 'DP_Group']]
    X = data[feature_cols]
    y = data['Release']
    groups = pd.Categorical(data['DP_Group']).codes
    
    catboost_maes = []
    
    print_flush("Training CatBoost (20 trials for better statistics)...")
    
    # OPTIMIZED CatBoost hyperparameter grid
    catboost_grid = {
        'iterations': [500, 600, 700, 800],
        'learning_rate': [0.02, 0.03, 0.05],
        'depth': [6, 8, 10],
        'l2_leaf_reg': [1, 3, 5],
        'random_strength': [1, 3, 5]
    }
    
    for trial in range(20):
        print_flush(f"  Trial {trial+1}/20...")
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42+trial)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        groups_train = groups[train_idx]
        
        # Scale data
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Inner CV for hyperparameter tuning
        gkf = GroupKFold(n_splits=10)
        
        model = CatBoostRegressor(random_seed=42, verbose=False)
        
        search = RandomizedSearchCV(
            model, catboost_grid, n_iter=100,
            scoring='neg_mean_absolute_error',
            cv=gkf, n_jobs=-1, random_state=42
        )
        
        search.fit(X_train_scaled, y_train, groups=groups_train)
        y_pred = search.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        catboost_maes.append(mae)
        print_flush(f"    MAE: {mae:.4f}")
    
    catboost_mean = float(np.mean(catboost_maes))
    catboost_std = float(np.std(catboost_maes))
    catboost_min = float(np.min(catboost_maes))
    
    print_flush(f"\n{'='*70}")
    print_flush("CatBoost Results:")
    print_flush(f"Mean MAE: {catboost_mean:.4f} ± {catboost_std:.4f} ({catboost_mean*100:.2f}%)")
    print_flush(f"Best MAE: {catboost_min:.4f} ({catboost_min*100:.2f}%)")
    
except Exception as e:
    print_flush(f"ERROR: {e}")
    catboost_mean = None

# ============================================================================
# STRATEGY 2: Original NESTED_CV with LGBM (20 trials, optimized HP)
# ============================================================================
print_flush("\n" + "="*70)
print_flush("STRATEGY 2: Original NESTED_CV + Optimized LGBM")
print_flush("="*70)

from NESTED_CV_ import NESTED_CV

# Monkey-patch for better hyperparameters
original_init = NESTED_CV.__init__

def optimized_init(self, datafile="Dataset_17_feat.xlsx", model_type=None):
    original_init(self, datafile, model_type)
    
    if model_type == 'LGBM':
        # REFINED grid (focus on successful ranges)
        self.p_grid = {
            "n_estimators": [600, 700, 800],  # Higher
            'boosting_type': ['gbdt', 'dart'],
            'num_leaves': [96, 128, 192],  # Medium-high range
            'learning_rate': [0.02, 0.03, 0.05],  # Lower
            'min_child_weight': [0.1, 0.5, 1.0],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_samples': [20, 30, 50],
            'reg_alpha': [0.01, 0.02, 0.03],  # More regularization
            'reg_lambda': [0.01, 0.02, 0.03],
            'max_depth': [10, 12, 15]
        }

NESTED_CV.__init__ = optimized_init

print_flush("Initializing NESTED_CV...")
cv_model = NESTED_CV(datafile="Dataset_17_feat.xlsx", model_type='LGBM')

print_flush("Loading data...")
cv_model.input_target()

print_flush(f"✓ Features: {cv_model.X.shape}")

print_flush("\nRunning nested CV (20 trials)...")
import time
start = time.time()

cv_model.cross_validation(input_value=20)  # 20 trials

elapsed = time.time() - start
print_flush(f"✓ Completed in {elapsed/60:.1f} minutes")

cv_model.results()

lgbm_mean = float(cv_model.CV_dataset['Test Score'].mean())
lgbm_std = float(cv_model.CV_dataset['Test Score'].std())
lgbm_min = float(cv_model.CV_dataset['Test Score'].min())

print_flush(f"\n{'='*70}")
print_flush("Optimized LGBM Results:")
print_flush(f"Mean MAE: {lgbm_mean:.4f} ± {lgbm_std:.4f} ({lgbm_mean*100:.2f}%)")
print_flush(f"Best MAE: {lgbm_min:.4f} ({lgbm_min*100:.2f}%)")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
os.chdir(original_dir)

print_flush("\n" + "="*70)
print_flush("FINAL RESULTS - BEST APPROACH")
print_flush("="*70)

results = []

if catboost_mean:
    results.append({
        'Algorithm': 'CatBoost (20 trials)',
        'Mean MAE': catboost_mean,
        'Mean %': catboost_mean * 100,
        'Best': catboost_min
    })

results.append({
    'Algorithm': 'Optimized LGBM (20 trials)',
    'Mean MAE': lgbm_mean,
    'Mean %': lgbm_mean * 100,
    'Best': lgbm_min
})

results_df = pd.DataFrame(results)
print_flush("\n" + results_df.to_string(index=False))

best = min(results, key=lambda x: x['Mean MAE'])

print_flush("\n" + "="*70)
print_flush("BEST APPROACH")
print_flush("="*70)
print_flush(f"Algorithm: {best['Algorithm']}")
print_flush(f"Mean MAE: {best['Mean MAE']:.4f} ({best['Mean %']:.2f}%)")
print_flush(f"Best trial: {best['Best']:.4f} ({best['Best']*100:.2f}%)")

print_flush(f"\nBaseline: 11.89%")
print_flush(f"Target: 10.9%")

improvement = ((0.1189 - best['Mean MAE']) / 0.1189) * 100

if best['Mean %'] <= 10.9:
    print_flush(f"\n🎉 TARGET ACHIEVED!")
    print_flush(f"✓ {improvement:.1f}% improvement over baseline")
elif best['Mean %'] < 11.5:
    print_flush(f"\n✓ CLOSE! Within 0.6% of target")
    print_flush(f"Improvement: {improvement:.1f}%")
else:
    print_flush(f"\nImprovement: {improvement:.1f}%")
    print_flush(f"Gap to target: {best['Mean %'] - 10.9:.2f}%")

# Save
summary = {
    'results': results,
    'best_algorithm': best['Algorithm'],
    'best_mae': best['Mean MAE'],
    'target_achieved': best['Mean %'] <= 10.9
}

with open(run_dir / 'best_approach_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print_flush(f"\n✓ Results saved to: {run_dir}")
print_flush("="*70)
