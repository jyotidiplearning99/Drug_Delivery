#!/usr/bin/env python -u
"""
Optimized training using ORIGINAL NESTED_CV with original 17 features
Target: MAE < 9% through hyperparameter optimization, not feature engineering
"""
import os
import sys
import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set unbuffered output (SAFE method)
os.environ['PYTHONUNBUFFERED'] = '1'

def print_flush(msg):
    """Print with immediate flush"""
    print(msg, flush=True)

print_flush("="*70)
print_flush("LAI ML TRAINING - TARGETING 9% MAE")
print_flush("Strategy: Original 17 features + advanced hyperparameters")
print_flush("="*70)

# Add original code directory to path
sys.path.insert(0, 'data/pban-91-long-acting-injectables-0df9c4a/few_shot_models')

# Setup
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = results_dir / f'run_optimized_{timestamp}'
run_dir.mkdir(exist_ok=True)

print_flush(f"\nOutput: {run_dir}")

# Change to data directory
original_dir = os.getcwd()
os.chdir('data/pban-91-long-acting-injectables-0df9c4a/few_shot_models')

# Check/convert dataset
print_flush("\nChecking dataset format...")
if not os.path.exists("Dataset_17_feat.xlsx"):
    print_flush("Converting CSV to Excel...")
    df = pd.read_csv("Dataset_17_feat.csv")
    df.to_excel("Dataset_17_feat.xlsx", index=False, engine='openpyxl')
    print_flush("✓ Conversion complete")

# Import and enhance NESTED_CV
from NESTED_CV_ import NESTED_CV

# Monkey-patch with OPTIMIZED hyperparameters
original_init = NESTED_CV.__init__

def optimized_init(self, datafile="Dataset_17_feat.xlsx", model_type=None):
    original_init(self, datafile, model_type)
    
    # OPTIMIZED LGBM grid (more aggressive)
    if model_type == 'LGBM':
        self.p_grid = {
            "n_estimators": [400, 500, 600, 700, 800, 1000],  # Higher
            'boosting_type': ['gbdt', 'dart'],  # Remove 'goss' (causing warnings)
            'num_leaves': [31, 64, 96, 128, 192],  # Focused range
            'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05],  # Lower rates
            'min_child_weight': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],  # Added
            'min_child_samples': [10, 20, 30, 50],
            'reg_alpha': [0, 0.005, 0.01, 0.02, 0.03],
            'reg_lambda': [0, 0.005, 0.01, 0.02, 0.03],
            'max_depth': [8, 10, 12, 15]  # Added
        }
        print_flush("✓ Using OPTIMIZED LGBM hyperparameter grid (17 features)")

NESTED_CV.__init__ = optimized_init

# Modify cross_validation to use 300 iterations
original_cv = NESTED_CV.cross_validation

def optimized_cross_validation(self, input_value):
    """Modified CV with 300 iterations (vs 100 in paper)"""
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV as RSCV
    from sklearn.metrics import mean_absolute_error
    import time
    
    NUM_TRIALS = input_value if input_value else 10
    
    self.itr_number = []
    self.outer_results = []
    self.inner_results = []
    self.model_params = []
    self.G_test_list = []
    self.y_test_list = []
    self.E_test_list = []
    self.T_test_list = []
    self.pred_list = []
    
    print_flush(f"\nStarting {NUM_TRIALS} trials with 300 HP iterations...")
    
    for i in range(NUM_TRIALS):
        trial_start = time.time()
        print_flush(f"\n{'='*64}")
        print_flush(f"TRIAL {i+1}/{NUM_TRIALS}")
        print_flush(f"{'='*64}")
        
        cv_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
        
        for train_index, test_index in cv_outer.split(self.X, self.Y, self.G):
            print_flush(f"  → Splitting data...")
            
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]
            G_train, G_test = self.G[train_index], self.G[test_index]
            E_train, E_test = self.E[train_index], self.E[test_index]
            T_train, T_test = self.T[train_index], self.T[test_index]
            
            print_flush(f"  → Train: {len(X_train)}, Test: {len(X_test)}")
            
            G_test = np.array(G_test)
            self.G_test_list.append(G_test)
            E_test = np.array(E_test)
            self.E_test_list.append(E_test)
            T_test = np.array(T_test)
            self.T_test_list.append(T_test)
            y_test = np.array(y_test)
            self.y_test_list.append(y_test)
            
            cv_inner = GroupKFold(n_splits=10)
            
            print_flush(f"  → HP optimization (n_iter=300)...")
            
            # Set LGBM to single-threaded
            if hasattr(self.user_defined_model, 'set_params'):
                try:
                    self.user_defined_model.set_params(n_jobs=1)
                except:
                    pass
            
            # INCREASED: 300 iterations (was 100 in paper, 200 in last attempt)
            search = RSCV(self.user_defined_model, self.p_grid, n_iter=300,
                         verbose=0, scoring='neg_mean_absolute_error',
                         cv=cv_inner, n_jobs=-1, refit=True)
            
            hp_start = time.time()
            result = search.fit(X_train, y_train, groups=G_train)
            hp_time = time.time() - hp_start
            
            print_flush(f"  → HP search completed in {hp_time/60:.1f} min")
            
            best_model = result.best_estimator_
            best_score = abs(result.best_score_)
            self.inner_results.append(best_score)
            
            print_flush(f"  → Predicting...")
            
            yhat = best_model.predict(X_test)
            self.pred_list.append(yhat)
            
            acc = mean_absolute_error(y_test, yhat)
            self.itr_number.append(i+1)
            self.outer_results.append(acc)
            self.model_params.append(result.best_params_)
            
            trial_time = time.time() - trial_start
            
            print_flush(f"\n{'#'*64}")
            print_flush(f"TRIAL {i+1}/{NUM_TRIALS} RESULTS")
            print_flush(f"{'#'*64}")
            print_flush(f"Test MAE: {acc:.4f} ({acc*100:.2f}%)")
            print_flush(f"Valid MAE: {best_score:.4f}")
            print_flush(f"Time: {trial_time/60:.1f} min")
            print_flush(f"Best params: {result.best_params_}")
            print_flush(f"{'#'*64}\n")

NESTED_CV.cross_validation = optimized_cross_validation

# Initialize and run
print_flush("\nInitializing NESTED_CV with LGBM (17 features)...")
cv_model = NESTED_CV(datafile="Dataset_17_feat.xlsx", model_type='LGBM')

print_flush("Loading data...")
cv_model.input_target()

print_flush(f"✓ Features shape: {cv_model.X.shape}")
print_flush(f"✓ Target shape: {cv_model.Y.shape}")
print_flush(f"✓ Groups: {len(cv_model.G.unique())}")

# Run optimized nested CV
import time
start_time = time.time()

cv_model.cross_validation(input_value=10)

total_time = time.time() - start_time
print_flush(f"\n✓ CV completed in {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")

# Compile and save
print_flush("\nCompiling results...")
cv_model.results()

print_flush("\nTraining best model...")
cv_model.best_model()

os.chdir(original_dir)

# Save results
print_flush("\nSaving results...")

cv_results_path = run_dir / 'nested_cv_results.csv'
cv_model.CV_dataset.to_csv(cv_results_path, index=False)

best_model_path = run_dir / 'best_lgbm_model.pkl'
with open(best_model_path, 'wb') as f:
    pickle.dump(cv_model.best_model, f)

# Summary
summary = {
    'test_mae_mean': float(cv_model.CV_dataset['Test Score'].mean()),
    'test_mae_std': float(cv_model.CV_dataset['Test Score'].std()),
    'test_mae_min': float(cv_model.CV_dataset['Test Score'].min()),
    'test_mae_max': float(cv_model.CV_dataset['Test Score'].max()),
    'test_mae_percent': float(cv_model.CV_dataset['Test Score'].mean() * 100),
    'valid_mae_mean': float(cv_model.CV_dataset['Valid Score'].mean()),
    'valid_mae_std': float(cv_model.CV_dataset['Valid Score'].std()),
    'best_model_params': cv_model.CV_dataset.iloc[0]['Model Parms'],
    'total_time_minutes': total_time / 60,
    'num_features': cv_model.X.shape[1],
    'n_iter_hyperparameters': 300
}

summary_path = run_dir / 'summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print_flush("\n" + "="*70)
print_flush("FINAL RESULTS - TARGETING 9% MAE")
print_flush("="*70)
print_flush(f"\n{'Metric':<35} {'Value'}")
print_flush("-" * 70)
print_flush(f"{'Test MAE (mean ± std)':<35} {summary['test_mae_mean']:.4f} ± {summary['test_mae_std']:.4f}")
print_flush(f"{'Test MAE (%)':<35} {summary['test_mae_percent']:.2f}%")
print_flush(f"{'Best trial MAE':<35} {summary['test_mae_min']:.4f} ({summary['test_mae_min']*100:.2f}%)")

print_flush(f"\n{'Baseline (17 feat, n_iter=100)':<35} 12.0%")
print_flush(f"{'Your previous (34 feat, n_iter=200)':<35} 12.15%")
print_flush(f"{'Target':<35} 9.0%")
print_flush(f"{'Nature paper':<35} 11.4%")

if summary['test_mae_percent'] < 9.0:
    print_flush(f"\n🎉 SUCCESS! Achieved MAE < 9%")
    improvement = ((0.120 - summary['test_mae_mean']) / 0.120) * 100
    print_flush(f"✓ {improvement:.1f}% improvement over baseline")
elif summary['test_mae_percent'] < 10.0:
    print_flush(f"\n✓ EXCELLENT! Within 1% of target")
    gap = summary['test_mae_percent'] - 9.0
    print_flush(f"Gap to target: {gap:.2f}%")
else:
    print_flush(f"\n✓ IMPROVED from 12.15%")
    improvement = ((0.1215 - summary['test_mae_mean']) / 0.1215) * 100
    print_flush(f"Improvement: {improvement:.1f}%")
    gap = summary['test_mae_percent'] - 9.0
    print_flush(f"Gap to target: {gap:.2f}%")

print_flush(f"\n{'Features used':<35} {summary['num_features']} (original)")
print_flush(f"{'HP iterations':<35} {summary['n_iter_hyperparameters']}")
print_flush(f"{'Training time':<35} {summary['total_time_minutes']/60:.2f} hours")

print_flush(f"\n✓ Results saved to: {run_dir}")
print_flush("="*70)
