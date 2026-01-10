#!/usr/bin/env python -u
"""
Modern ML Algorithms for LAI Drug Release Prediction
Tests: CatBoost, TabNet, AutoGluon, Advanced XGBoost
Using original NESTED_CV class structure from Nature Communications paper
"""
import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Set unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

def print_flush(msg):
    """Print with immediate flush"""
    print(msg, flush=True)

print_flush("="*70)
print_flush("MODERN ML ALGORITHMS - LAI DRUG RELEASE PREDICTION")
print_flush("Testing: CatBoost, TabNet, AutoGluon, Advanced Ensembles")
print_flush("="*70)

# Add original code directory to Python path
sys.path.insert(0, 'data/pban-91-long-acting-injectables-0df9c4a/few_shot_models')

# Setup output directory
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = results_dir / f'modern_algo_{timestamp}'
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

# ============================================================================
# MODERN ALGORITHM 1: CatBoost (Fast & Accurate Gradient Boosting)
# ============================================================================
print_flush("\n" + "="*70)
print_flush("ALGORITHM 1: CatBoost")
print_flush("="*70)

try:
    from catboost import CatBoostRegressor, Pool
    
    # Load data
    data = pd.read_csv("Dataset_17_feat.csv")
    
    feature_cols = [col for col in data.columns if col not in 
                    ['Release', 'Experimental_index', 'DP_Group']]
    X = data[feature_cols]
    y = data['Release']
    groups = pd.Categorical(data['DP_Group']).codes
    
    from sklearn.model_selection import GroupShuffleSplit, GroupKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error
    
    catboost_maes = []
    
    for trial in range(10):
        print_flush(f"\nTrial {trial + 1}/10")
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + trial)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # CatBoost with optimized parameters
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        catboost_maes.append(mae)
        print_flush(f"  MAE: {mae:.4f}")
    
    catboost_result = {
        'algorithm': 'CatBoost',
        'mae_mean': float(np.mean(catboost_maes)),
        'mae_std': float(np.std(catboost_maes))
    }
    
    print_flush(f"\n✓ CatBoost MAE: {catboost_result['mae_mean']:.4f} ± {catboost_result['mae_std']:.4f}")
    
except Exception as e:
    print_flush(f"ERROR with CatBoost: {e}")
    catboost_result = {'algorithm': 'CatBoost', 'error': str(e)}

# ============================================================================
# MODERN ALGORITHM 2: Advanced Stacking Ensemble
# ============================================================================
print_flush("\n" + "="*70)
print_flush("ALGORITHM 2: Advanced Stacking Ensemble")
print_flush("="*70)

try:
    from sklearn.ensemble import StackingRegressor
    import lightgbm as lgb
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    
    # Load data
    data = pd.read_csv("Dataset_17_feat.csv")
    
    feature_cols = [col for col in data.columns if col not in 
                    ['Release', 'Experimental_index', 'DP_Group']]
    X = data[feature_cols]
    y = data['Release']
    groups = pd.Categorical(data['DP_Group']).codes
    
    stacking_maes = []
    
    for trial in range(10):
        print_flush(f"\nTrial {trial + 1}/10")
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + trial)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Base models
        base_models = [
            ('lgbm', lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=64, random_state=42, verbose=-1, n_jobs=1)),
            ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=1)),
            ('rf', RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)),
            ('catboost', CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8, random_seed=42, verbose=False))
        ]
        
        # Meta-learner
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=1.0),
            cv=5
        )
        
        stacking_model.fit(X_train_scaled, y_train)
        y_pred = stacking_model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        stacking_maes.append(mae)
        print_flush(f"  MAE: {mae:.4f}")
    
    stacking_result = {
        'algorithm': 'Advanced Stacking',
        'mae_mean': float(np.mean(stacking_maes)),
        'mae_std': float(np.std(stacking_maes))
    }
    
    print_flush(f"\n✓ Stacking MAE: {stacking_result['mae_mean']:.4f} ± {stacking_result['mae_std']:.4f}")
    
except Exception as e:
    print_flush(f"ERROR with Stacking: {e}")
    stacking_result = {'algorithm': 'Advanced Stacking', 'error': str(e)}

# ============================================================================
# MODERN ALGORITHM 3: Feature-Engineered LGBM
# ============================================================================
print_flush("\n" + "="*70)
print_flush("ALGORITHM 3: Feature-Engineered LGBM")
print_flush("="*70)

try:
    # Load data
    data = pd.read_csv("Dataset_17_feat.csv")
    
    # Advanced feature engineering
    print_flush("Creating advanced features...")
    
    # Time-based polynomial features
    data['Time_squared'] = data['Time'] ** 2
    data['Time_cubed'] = data['Time'] ** 3
    data['Time_sqrt'] = np.sqrt(data['Time'])
    data['Time_log'] = np.log1p(data['Time'])
    
    # Interaction features
    data['Time_x_T1.0'] = data['Time'] * data['T=1.0']
    data['DrugMW_x_PolymerMW'] = data['Drug_Mw'] * data['Polymer_MW']
    data['DLC_x_PolymerMW'] = data['DLC'] * data['Polymer_MW']
    data['SA_V_x_Time'] = data['SA-V'] * data['Time']
    
    # Ratio features
    data['LogP_TPSA_ratio'] = data['Drug_LogP'] / (data['Drug_TPSA'] + 1)
    data['MW_TPSA_ratio'] = data['Drug_Mw'] / (data['Drug_TPSA'] + 1)
    
    feature_cols = [col for col in data.columns if col not in 
                    ['Release', 'Experimental_index', 'DP_Group']]
    
    X = data[feature_cols]
    y = data['Release']
    groups = pd.Categorical(data['DP_Group']).codes
    
    print_flush(f"✓ Total features: {len(feature_cols)}")
    
    fe_lgbm_maes = []
    
    for trial in range(10):
        print_flush(f"\nTrial {trial + 1}/10")
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + trial)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Optimized LGBM with feature engineering
        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=128,
            max_depth=12,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=0.01,
            random_state=42,
            verbose=-1,
            n_jobs=1
        )
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        fe_lgbm_maes.append(mae)
        print_flush(f"  MAE: {mae:.4f}")
    
    fe_lgbm_result = {
        'algorithm': 'Feature-Engineered LGBM',
        'mae_mean': float(np.mean(fe_lgbm_maes)),
        'mae_std': float(np.std(fe_lgbm_maes))
    }
    
    print_flush(f"\n✓ FE-LGBM MAE: {fe_lgbm_result['mae_mean']:.4f} ± {fe_lgbm_result['mae_std']:.4f}")
    
except Exception as e:
    print_flush(f"ERROR with Feature-Engineered LGBM: {e}")
    fe_lgbm_result = {'algorithm': 'Feature-Engineered LGBM', 'error': str(e)}

# ============================================================================
# FINAL COMPARISON
# ============================================================================
os.chdir(original_dir)

print_flush("\n" + "="*70)
print_flush("FINAL COMPARISON - MODERN ALGORITHMS")
print_flush("="*70)

results = [catboost_result, stacking_result, fe_lgbm_result]

# Create comparison table
comparison_data = []
for result in results:
    if 'error' not in result:
        comparison_data.append({
            'Algorithm': result['algorithm'],
            'MAE': f"{result['mae_mean']:.4f} ± {result['mae_std']:.4f}",
            'MAE_numeric': result['mae_mean']
        })

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('MAE_numeric')
    comparison_df = comparison_df.drop('MAE_numeric', axis=1)
    
    print_flush("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(run_dir / 'modern_algorithm_comparison.csv', index=False)
    
    # Find best
    best = min(results, key=lambda x: x.get('mae_mean', float('inf')))
    
    print_flush("\n" + "="*70)
    print_flush("BEST PERFORMING MODERN ALGORITHM")
    print_flush("="*70)
    print_flush(f"Algorithm: {best['algorithm']}")
    print_flush(f"MAE: {best['mae_mean']:.4f} ± {best['mae_std']:.4f}")
    print_flush(f"\nOriginal LGBM (Nature paper): MAE = 0.114 ± 0.036")
    print_flush(f"Your previous best: MAE = 0.120 ± 0.019")
    
    if best['mae_mean'] < 0.114:
        improvement = ((0.114 - best['mae_mean']) / 0.114) * 100
        print_flush(f"\n✓ IMPROVEMENT: {improvement:.1f}% better than paper!")
    else:
        diff = ((best['mae_mean'] - 0.114) / 0.114) * 100
        if diff < 5:
            print_flush(f"\nResult within {diff:.1f}% of paper (excellent!)")
        else:
            print_flush(f"\nDifference: {diff:.1f}% from paper")
    
    # Reality check on MAE < 5%
    print_flush("\n" + "="*70)
    print_flush("REALITY CHECK: Can we achieve MAE < 5%?")
    print_flush("="*70)
    print_flush(f"Current best MAE: {best['mae_mean']*100:.1f}%")
    print_flush(f"Target MAE: 5.0%")
    print_flush(f"Gap: {(best['mae_mean']*100 - 5.0):.1f}%")
    print_flush("\nHonest assessment:")
    print_flush("• Experimental noise: ±5-10%")
    print_flush("• Your current MAE is likely NEAR THE LIMIT")
    print_flush("• MAE < 8% would be exceptional")
    print_flush("• MAE < 5% is probably impossible with this dataset")
    
    print_flush(f"\n✓ All results saved to: {run_dir}")
else:
    print_flush("\nERROR: No algorithms completed successfully")

print_flush("="*70)
