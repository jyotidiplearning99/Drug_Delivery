"""
Model Training Module
Trains LightGBM models for:
1. Delivery efficiency classification (Model 1 & 2)
2. Apparent pKa regression
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
from config import *


def load_data(split):
    """Load features and labels for a given split"""
    
    X = np.load(PROCESSED_DATA_DIR / f'X_{split}.npy')
    y = pd.read_csv(PROCESSED_DATA_DIR / f'y_{split}.csv')
    
    return X, y


def train_delivery_model1():
    """
    Train Model 1: Delivery efficiency ≥1× MC3
    Used in first round of screening
    """
    
    print("\n" + "="*70)
    print("TRAINING MODEL 1: Delivery Efficiency (≥1× MC3)")
    print("="*70)
    
    # Load data
    X_train, y_train = load_data('train')
    X_val, y_val = load_data('val')
    X_test, y_test = load_data('test')
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize model with paper hyperparameters
    model = LGBMClassifier(**MODEL1_PARAMS)
    
    # Train
    print("\nTraining...")
    model.fit(
        X_train, 
        y_train['label_model1'],
        eval_set=[(X_val, y_val['label_model1'])],
        eval_metric='binary_logloss'
    )
    
    # Evaluate
    print("\nEvaluating...")
    
    # Training performance
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train['label_model1'], train_pred)
    
    # Validation performance
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val['label_model1'], val_pred)
    val_prec = precision_score(y_val['label_model1'], val_pred)
    val_recall = recall_score(y_val['label_model1'], val_pred)
    val_f1 = f1_score(y_val['label_model1'], val_pred)
    
    # Test performance
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test['label_model1'], test_pred)
    test_prec = precision_score(y_test['label_model1'], test_pred)
    test_recall = recall_score(y_test['label_model1'], test_pred)
    test_f1 = f1_score(y_test['label_model1'], test_pred)
    
    print(f"\nModel 1 Results:")
    print(f"{'Set':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 50)
    print(f"{'Train':<10} {train_acc:.4f}")
    print(f"{'Val':<10} {val_acc:.4f}     {val_prec:.4f}     {val_recall:.4f}     {val_f1:.4f}")
    print(f"{'Test':<10} {test_acc:.4f}     {test_prec:.4f}     {test_recall:.4f}     {test_f1:.4f}")
    
    # Save model
    model_path = MODEL_DIR / 'delivery_model1.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    return model


def train_delivery_model2():
    """
    Train Model 2: Delivery efficiency ≥2× MC3
    Used in second round of screening (stricter criterion)
    """
    
    print("\n" + "="*70)
    print("TRAINING MODEL 2: Delivery Efficiency (≥2× MC3)")
    print("="*70)
    
    # Load data
    X_train, y_train = load_data('train')
    X_val, y_val = load_data('val')
    X_test, y_test = load_data('test')
    
    # Initialize model
    model = LGBMClassifier(**MODEL2_PARAMS)
    
    # Train
    print("\nTraining...")
    model.fit(
        X_train,
        y_train['label_model2'],
        eval_set=[(X_val, y_val['label_model2'])],
        eval_metric='binary_logloss'
    )
    
    # Evaluate
    print("\nEvaluating...")
    
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val['label_model2'], val_pred)
    val_prec = precision_score(y_val['label_model2'], val_pred, zero_division=0)
    
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test['label_model2'], test_pred)
    test_prec = precision_score(y_test['label_model2'], test_pred, zero_division=0)
    
    print(f"\nModel 2 Results:")
    print(f"{'Set':<10} {'Accuracy':<10} {'Precision':<10}")
    print("-" * 30)
    print(f"{'Val':<10} {val_acc:.4f}     {val_prec:.4f}")
    print(f"{'Test':<10} {test_acc:.4f}     {test_prec:.4f}")
    print("\nNote: Model 2 is stricter - designed for high precision")
    
    # Save model
    model_path = MODEL_DIR / 'delivery_model2.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    return model


def train_pka_model():
    """
    Train pKa regression model
    Predicts apparent pKa of LNPs (target range: 6.0-7.0)
    """
    
    print("\n" + "="*70)
    print("TRAINING MODEL: Apparent pKa (Regression)")
    print("="*70)
    
    # Load data
    X_train, y_train = load_data('train')
    X_val, y_val = load_data('val')
    X_test, y_test = load_data('test')
    
    # Remove samples without pKa
    mask_train = y_train['apparent_pka'].notna()
    mask_val = y_val['apparent_pka'].notna()
    mask_test = y_test['apparent_pka'].notna()
    
    X_train_pka = X_train[mask_train]
    y_train_pka = y_train[mask_train]['apparent_pka']
    
    X_val_pka = X_val[mask_val]
    y_val_pka = y_val[mask_val]['apparent_pka']
    
    X_test_pka = X_test[mask_test]
    y_test_pka = y_test[mask_test]['apparent_pka']
    
    print(f"Train: {len(X_train_pka)}, Val: {len(X_val_pka)}, Test: {len(X_test_pka)}")
    
    # Initialize model
    model = LGBMRegressor(**PKA_MODEL_PARAMS)
    
    # Train
    print("\nTraining...")
    model.fit(
        X_train_pka,
        y_train_pka,
        eval_set=[(X_val_pka, y_val_pka)],
        eval_metric='mae'
    )
    
    # Evaluate
    print("\nEvaluating...")
    
    # Validation
    val_pred = model.predict(X_val_pka)
    val_mae = mean_absolute_error(y_val_pka, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_pka, val_pred))
    val_r2 = r2_score(y_val_pka, val_pred)
    
    # Test
    test_pred = model.predict(X_test_pka)
    test_mae = mean_absolute_error(y_test_pka, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_pka, test_pred))
    test_r2 = r2_score(y_test_pka, test_pred)
    
    print(f"\npKa Model Results:")
    print(f"{'Set':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 40)
    print(f"{'Val':<10} {val_mae:.4f}     {val_rmse:.4f}     {val_r2:.4f}")
    print(f"{'Test':<10} {test_mae:.4f}     {test_rmse:.4f}     {test_r2:.4f}")
    print(f"\nNote: pKa predictions most accurate in range 6.0-7.0")
    
    # Save model
    model_path = MODEL_DIR / 'pka_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    return model


def main():
    """Main training pipeline"""
    
    print("="*70)
    print("MODEL TRAINING")
    print("Training 3 models: Delivery Model 1, Model 2, and pKa Model")
    print("="*70)
    
    # Train Model 1
    model1 = train_delivery_model1()
    
    # Train Model 2
    model2 = train_delivery_model2()
    
    # Train pKa model
    pka_model = train_pka_model()
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"Models saved to: {MODEL_DIR}")
    print("  - delivery_model1.pkl (≥1× MC3)")
    print("  - delivery_model2.pkl (≥2× MC3)")
    print("  - pka_model.pkl (Apparent pKa)")
    print("\nNext: python src/5_shap_analysis.py")
    print("="*70)


if __name__ == '__main__':
    main()
