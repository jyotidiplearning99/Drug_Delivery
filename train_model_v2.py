#!/usr/bin/env python -u
"""
EXPERIMENT: Finding Optimal Number of Trials
Tests: 5, 8, 10, 12, 15, 20 trials
Goal: Find sweet spot where MAE is minimized
"""
import os
import sys
import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def print_flush(msg):
    """Print with immediate flush"""
    print(msg, flush=True)

print_flush("="*70)
print_flush("FINDING OPTIMAL NUMBER OF TRIALS")
print_flush("Testing: 5, 8, 10, 12, 15, 20 trials")
print_flush("="*70)

# Add original code directory
sys.path.insert(0, 'data/pban-91-long-acting-injectables-0df9c4a/few_shot_models')

# Setup
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_dir = results_dir / f'trial_optimization_{timestamp}'
experiment_dir.mkdir(parents=True, exist_ok=True)  # ← FIXED: Added parents=True

print_flush(f"\nOutput: {experiment_dir}")

# Change to data directory
original_dir = os.getcwd()
os.chdir('data/pban-91-long-acting-injectables-0df9c4a/few_shot_models')

# Check dataset
print_flush("\nChecking dataset...")
if not os.path.exists("Dataset_17_feat.xlsx"):
    df = pd.read_csv("Dataset_17_feat.csv")
    df.to_excel("Dataset_17_feat.xlsx", index=False, engine='openpyxl')

# Import original NESTED_CV
from NESTED_CV_ import NESTED_CV

# Monkey-patch for progress tracking
original_cv = NESTED_CV.cross_validation

def tracked_cross_validation(self, input_value):
    """Modified CV with progress tracking"""
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
    
    print_flush(f"\n  Starting {NUM_TRIALS} trials...")
    
    for i in range(NUM_TRIALS):
        trial_start = time.time()
        
        cv_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
        
        for train_index, test_index in cv_outer.split(self.X, self.Y, self.G):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]
            G_train, G_test = self.G[train_index], self.G[test_index]
            E_train, E_test = self.E[train_index], self.E[test_index]
            T_train, T_test = self.T[train_index], self.T[test_index]
            
            G_test = np.array(G_test)
            self.G_test_list.append(G_test)
            E_test = np.array(E_test)
            self.E_test_list.append(E_test)
            T_test = np.array(T_test)
            self.T_test_list.append(T_test)
            y_test = np.array(y_test)
            self.y_test_list.append(y_test)
            
            cv_inner = GroupKFold(n_splits=10)
            
            if hasattr(self.user_defined_model, 'set_params'):
                try:
                    self.user_defined_model.set_params(n_jobs=1)
                except:
                    pass
            
            search = RSCV(self.user_defined_model, self.p_grid, n_iter=100,
                         verbose=0, scoring='neg_mean_absolute_error',
                         cv=cv_inner, n_jobs=-1, refit=True)
            
            result = search.fit(X_train, y_train, groups=G_train)
            
            best_model = result.best_estimator_
            best_score = abs(result.best_score_)
            self.inner_results.append(best_score)
            
            yhat = best_model.predict(X_test)
            self.pred_list.append(yhat)
            
            acc = mean_absolute_error(y_test, yhat)
            self.itr_number.append(i+1)
            self.outer_results.append(acc)
            self.model_params.append(result.best_params_)
            
            trial_time = time.time() - trial_start
            
            print_flush(f"    Trial {i+1}/{NUM_TRIALS}: MAE = {acc:.4f} ({acc*100:.2f}%), Time = {trial_time/60:.1f} min")

NESTED_CV.cross_validation = tracked_cross_validation

# ============================================================================
# RUN EXPERIMENTS WITH DIFFERENT TRIAL COUNTS
# ============================================================================

trial_counts = [5, 8, 10, 12, 15, 20]
all_results = []

for n_trials in trial_counts:
    print_flush("\n" + "="*70)
    print_flush(f"EXPERIMENT: {n_trials} TRIALS")
    print_flush("="*70)
    
    # Initialize NESTED_CV
    print_flush(f"\nInitializing NESTED_CV with LGBM...")
    cv_model = NESTED_CV(datafile="Dataset_17_feat.xlsx", model_type='LGBM')
    
    print_flush("Loading data...")
    cv_model.input_target()
    
    print_flush(f"✓ Features: {cv_model.X.shape}")
    print_flush(f"✓ Target: {cv_model.Y.shape}")
    
    # Run nested CV
    import time
    start_time = time.time()
    
    cv_model.cross_validation(input_value=n_trials)
    
    total_time = time.time() - start_time
    
    # Compile results
    print_flush("\nCompiling results...")
    cv_model.results()
    
    # Calculate statistics
    test_mae_mean = float(cv_model.CV_dataset['Test Score'].mean())
    test_mae_std = float(cv_model.CV_dataset['Test Score'].std())
    test_mae_min = float(cv_model.CV_dataset['Test Score'].min())
    test_mae_max = float(cv_model.CV_dataset['Test Score'].max())
    
    # Store results
    result = {
        'n_trials': n_trials,
        'mae_mean': test_mae_mean,
        'mae_std': test_mae_std,
        'mae_min': test_mae_min,
        'mae_max': test_mae_max,
        'mae_percent': test_mae_mean * 100,
        'total_time_minutes': total_time / 60,
        'time_per_trial': (total_time / 60) / n_trials
    }
    
    all_results.append(result)
    
    print_flush(f"\n{'#'*70}")
    print_flush(f"RESULTS FOR {n_trials} TRIALS:")
    print_flush(f"Mean MAE: {test_mae_mean:.4f} ± {test_mae_std:.4f} ({test_mae_mean*100:.2f}%)")
    print_flush(f"Best MAE: {test_mae_min:.4f} ({test_mae_min*100:.2f}%)")
    print_flush(f"Total time: {total_time/60:.1f} min")
    print_flush(f"{'#'*70}\n")
    
    # Save individual results
    trial_dir = experiment_dir / f'trials_{n_trials}'
    trial_dir.mkdir(parents=True, exist_ok=True)  # ← FIXED: Added parents=True
    
    cv_model.CV_dataset.to_csv(trial_dir / 'cv_results.csv', index=False)
    
    with open(trial_dir / 'summary.json', 'w') as f:
        json.dump(result, f, indent=2)

# Go back to project root
os.chdir(original_dir)

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Save combined results
results_df.to_csv(experiment_dir / 'all_results.csv', index=False)

with open(experiment_dir / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

print_flush("\nCreating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure: MAE vs Number of Trials
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Mean MAE with error bars
ax1 = axes[0, 0]
ax1.errorbar(results_df['n_trials'], results_df['mae_percent'], 
             yerr=results_df['mae_std']*100, 
             marker='o', markersize=10, linewidth=2, capsize=5,
             color='#3498db', label='Mean MAE ± Std')
ax1.axhline(y=11.89, color='r', linestyle='--', linewidth=2, 
            label='Baseline (11.89%)', alpha=0.7)
ax1.set_xlabel('Number of Trials', fontsize=14, fontweight='bold')
ax1.set_ylabel('Mean MAE (%)', fontsize=14, fontweight='bold')
ax1.set_title('Mean MAE vs Number of Trials', fontsize=16, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Find minimum
min_idx = results_df['mae_percent'].idxmin()
min_trials = results_df.loc[min_idx, 'n_trials']
min_mae = results_df.loc[min_idx, 'mae_percent']
ax1.plot(min_trials, min_mae, 'r*', markersize=20)

# Plot 2: Best MAE per trial count
ax2 = axes[0, 1]
ax2.plot(results_df['n_trials'], results_df['mae_min']*100, 
         marker='s', markersize=10, linewidth=2, color='#2ecc71')
ax2.axhline(y=8.96, color='purple', linestyle='--', linewidth=2,
            label='Your Best Ever (8.96%)', alpha=0.7)
ax2.set_xlabel('Number of Trials', fontsize=14, fontweight='bold')
ax2.set_ylabel('Best MAE (%)', fontsize=14, fontweight='bold')
ax2.set_title('Best Trial MAE vs Number of Trials', fontsize=16, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

# Plot 3: Training time
ax3 = axes[1, 0]
ax3.bar(results_df['n_trials'], results_df['total_time_minutes'], 
        color='#e74c3c', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Number of Trials', fontsize=14, fontweight='bold')
ax3.set_ylabel('Total Training Time (minutes)', fontsize=14, fontweight='bold')
ax3.set_title('Training Time vs Number of Trials', fontsize=16, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# Plot 4: MAE Variance (stability)
ax4 = axes[1, 1]
ax4.plot(results_df['n_trials'], results_df['mae_std']*100, 
         marker='D', markersize=10, linewidth=2, color='#9b59b6')
ax4.set_xlabel('Number of Trials', fontsize=14, fontweight='bold')
ax4.set_ylabel('MAE Standard Deviation (%)', fontsize=14, fontweight='bold')
ax4.set_title('Model Stability vs Number of Trials', fontsize=16, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(experiment_dir / 'trial_optimization_analysis.png', dpi=300, bbox_inches='tight')
print_flush(f"✓ Saved: trial_optimization_analysis.png")
plt.close()

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print_flush("\n" + "="*70)
print_flush("FINAL ANALYSIS - FINDING THE SWEET SPOT")
print_flush("="*70)

print_flush("\nResults Summary:")
print_flush("-" * 70)
for _, row in results_df.iterrows():
    print_flush(f"{int(row['n_trials']):2d} trials: MAE = {row['mae_percent']:.2f}% ± {row['mae_std']*100:.2f}%, "
                f"Best = {row['mae_min']*100:.2f}%, Time = {row['total_time_minutes']:.1f} min")

# Find sweet spot
print_flush("\n" + "="*70)
print_flush("SWEET SPOT ANALYSIS")
print_flush("="*70)

# Best MAE
best_mae_idx = results_df['mae_percent'].idxmin()
best_mae_trials = int(results_df.loc[best_mae_idx, 'n_trials'])
best_mae_value = results_df.loc[best_mae_idx, 'mae_percent']

print_flush(f"\n✓ Lowest MAE: {best_mae_trials} trials → {best_mae_value:.2f}%")

# Best stability (lowest std)
best_stability_idx = results_df['mae_std'].idxmin()
best_stability_trials = int(results_df.loc[best_stability_idx, 'n_trials'])
best_stability_value = results_df.loc[best_stability_idx, 'mae_std']*100

print_flush(f"✓ Most stable: {best_stability_trials} trials → Std = {best_stability_value:.2f}%")

print_flush(f"\n✓ All results saved to: {experiment_dir}")
print_flush("="*70)
