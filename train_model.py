#!/usr/bin/env python -u
"""
Training using ORIGINAL NESTED_CV class from GitHub repository
Exact replication of Nature Communications 2023 methodology
"""
import os
import sys
import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Force unbuffered output
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
#sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

def print_flush(msg):
    """Print with immediate flush"""
    print(msg)
    sys.stdout.flush()

print_flush("="*70)
print_flush("LAI ML TRAINING - ORIGINAL GITHUB CODE")
print_flush("Predicting Fractional Drug Release from LAIs")
print_flush("="*70)

# Add original code directory to Python path
sys.path.insert(0, 'data/pban-91-long-acting-injectables-0df9c4a/few_shot_models')

# Setup output directory
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = results_dir / f'run_{timestamp}'
run_dir.mkdir(exist_ok=True)

print_flush(f"\nOutput: {run_dir}")

# Change to data directory
original_dir = os.getcwd()
os.chdir('data/pban-91-long-acting-injectables-0df9c4a/few_shot_models')

# Check/convert dataset format
print_flush("\nChecking dataset format...")
if not os.path.exists("Dataset_17_feat.xlsx"):
    print_flush("Converting CSV to Excel...")
    df = pd.read_csv("Dataset_17_feat.csv")
    df.to_excel("Dataset_17_feat.xlsx", index=False, engine='openpyxl')
    print_flush("✓ Conversion complete")
else:
    print_flush("✓ Excel file already exists")

# Import the original NESTED_CV class
from NESTED_CV_ import NESTED_CV

# Monkey-patch to add progress tracking and fix parallelism
original_cross_validation = NESTED_CV.cross_validation

def modified_cross_validation(self, input_value):
    """Modified version with progress tracking and fixed parallelism"""
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
    
    print_flush(f"\nStarting {NUM_TRIALS} trial(s)...")
    sys.stdout.flush()
    
    for i in range(NUM_TRIALS):
        trial_start = time.time()
        print_flush(f"\n{'='*64}")
        print_flush(f"TRIAL {i+1}/{NUM_TRIALS}")
        print_flush(f"{'='*64}")
        sys.stdout.flush()
        
        # Use random_state=i (varies per trial, as in original paper)
        cv_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
        
        for train_index, test_index in cv_outer.split(self.X, self.Y, self.G):
            print_flush(f"  → Splitting data...")
            sys.stdout.flush()
            
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]
            G_train, G_test = self.G[train_index], self.G[test_index]
            E_train, E_test = self.E[train_index], self.E[test_index]
            T_train, T_test = self.T[train_index], self.T[test_index]
            
            print_flush(f"  → Train: {len(X_train)}, Test: {len(X_test)}")
            sys.stdout.flush()
            
            G_test = np.array(G_test)
            self.G_test_list.append(G_test)
            E_test = np.array(E_test)
            self.E_test_list.append(E_test)
            T_test = np.array(T_test)
            self.T_test_list.append(T_test)
            y_test = np.array(y_test)
            self.y_test_list.append(y_test)
            
            cv_inner = GroupKFold(n_splits=10)
            
            print_flush(f"  → Starting hyperparameter search (n_iter=100)...")
            sys.stdout.flush()
            
            # Fix parallelism: Set LightGBM to single-threaded
            if hasattr(self.user_defined_model, 'set_params'):
                try:
                    self.user_defined_model.set_params(n_jobs=1)
                except:
                    pass
            
            # Original paper uses n_iter=100
            search = RSCV(self.user_defined_model, self.p_grid, n_iter=100,
                         verbose=0, scoring='neg_mean_absolute_error',
                         cv=cv_inner, n_jobs=-1, refit=True)
            
            hp_start = time.time()
            result = search.fit(X_train, y_train, groups=G_train)
            hp_time = time.time() - hp_start
            
            print_flush(f"  → HP search completed in {hp_time:.1f}s ({hp_time/60:.1f} min)")
            sys.stdout.flush()
            
            best_model = result.best_estimator_
            best_score = abs(result.best_score_)
            self.inner_results.append(best_score)
            
            print_flush(f"  → Predicting fractional drug release...")
            sys.stdout.flush()
            
            yhat = best_model.predict(X_test)
            self.pred_list.append(yhat)
            
            acc = mean_absolute_error(y_test, yhat)
            self.itr_number.append(i+1)
            self.outer_results.append(acc)
            self.model_params.append(result.best_params_)
            
            trial_time = time.time() - trial_start
            
            print_flush(f"\n{'#'*64}")
            print_flush(f"STATUS REPORT - Trial {i+1}/{NUM_TRIALS}")
            print_flush(f"{'#'*64}")
            print_flush(f"Test MAE: {acc:.4f} (prediction error: {acc*100:.1f}%)")
            print_flush(f"Valid MAE: {best_score:.4f}")
            print_flush(f"Trial time: {trial_time:.1f}s ({trial_time/60:.1f} min)")
            print_flush(f"\nBest params: {result.best_params_}")
            print_flush(f"{'#'*64}\n")
            sys.stdout.flush()

# Apply the monkey patch
NESTED_CV.cross_validation = modified_cross_validation

# Initialize NESTED_CV with LGBM model
print_flush("\nInitializing NESTED_CV with LGBM...")
print_flush("This predicts fractional drug release (0.0 to 1.0)")
sys.stdout.flush()

cv_model = NESTED_CV(datafile="Dataset_17_feat.xlsx", model_type='LGBM')

print_flush("Loading data...")
sys.stdout.flush()

cv_model.input_target()

print_flush(f"✓ Features shape: {cv_model.X.shape}")
print_flush(f"✓ Target shape: {cv_model.Y.shape}")
print_flush(f"✓ Drug-Polymer groups: {len(cv_model.G.unique())}")
sys.stdout.flush()

# Run nested cross-validation (10 trials)
print_flush("\n" + "="*70)
print_flush("STARTING NESTED CROSS-VALIDATION")
print_flush("Predicting fractional drug release at various timepoints")
print_flush("10 trials × n_iter=100 (exact paper methodology)")
print_flush("="*70)
sys.stdout.flush()

import time
start_time = time.time()

cv_model.cross_validation(input_value=10)

total_time = time.time() - start_time
print_flush(f"\n✓ Cross-validation completed in {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")
sys.stdout.flush()

print_flush("\nCompiling results...")
sys.stdout.flush()

cv_model.results()

print_flush("\nTraining best model on full dataset...")
sys.stdout.flush()

cv_model.best_model()

# Go back to project root
os.chdir(original_dir)

# Save results
print_flush("\nSaving results...")
sys.stdout.flush()

cv_results_path = run_dir / 'nested_cv_results.csv'
cv_model.CV_dataset.to_csv(cv_results_path, index=False)

best_model_path = run_dir / 'best_lgbm_model.pkl'
with open(best_model_path, 'wb') as f:
    pickle.dump(cv_model.best_model, f)

# Calculate summary statistics
summary = {
    'test_mae_mean': float(cv_model.CV_dataset['Test Score'].mean()),
    'test_mae_std': float(cv_model.CV_dataset['Test Score'].std()),
    'test_mae_percent': float(cv_model.CV_dataset['Test Score'].mean() * 100),
    'valid_mae_mean': float(cv_model.CV_dataset['Valid Score'].mean()),
    'valid_mae_std': float(cv_model.CV_dataset['Valid Score'].std()),
    'best_model_params': cv_model.CV_dataset.iloc[0]['Model Parms'],
    'total_time_minutes': total_time / 60,
    'total_time_hours': total_time / 3600,
}

summary_path = run_dir / 'summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print_flush("\n" + "="*70)
print_flush("FINAL RESULTS - FRACTIONAL DRUG RELEASE PREDICTION")
print_flush("="*70)
print_flush(f"Test MAE: {summary['test_mae_mean']:.3f} ± {summary['test_mae_std']:.3f}")
print_flush(f"  → Average prediction error: {summary['test_mae_percent']:.1f}%")
print_flush(f"Valid MAE: {summary['valid_mae_mean']:.3f} ± {summary['valid_mae_std']:.3f}")
print_flush(f"Total time: {summary['total_time_hours']:.2f} hours")
print_flush(f"\nExpected from Nature paper: MAE = 0.114 ± 0.036 (11.4%)")
print_flush(f"\n✓ Results saved to: {run_dir}")
print_flush(f"✓ CV results: {cv_results_path}")
print_flush(f"✓ Best model: {best_model_path}")
print_flush(f"✓ Summary: {summary_path}")
print_flush("="*70)
sys.stdout.flush()
