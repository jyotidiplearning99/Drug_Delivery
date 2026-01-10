"""
Data Preprocessing Module
Cleans and prepares ionizable lipid data for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def load_raw_data():
    """Load raw lipid data"""
    data_path = RAW_DATA_DIR / 'lipid_data.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Run: python src/1_data_collection.py first"
        )
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} lipid entries")
    return df


def clean_data(df):
    """
    Clean and validate data
    Remove invalid entries following paper's methodology
    """
    
    print("\nCleaning data...")
    initial_size = len(df)
    
    # 1. Remove entries without SMILES
    df = df[df['smiles'].notna()].copy()
    print(f"  After SMILES check: {len(df)}")
    
    # 2. Remove entries without delivery efficiency
    df = df[df['fold_vs_mc3'].notna()].copy()
    print(f"  After efficiency check: {len(df)}")
    
    # 3. Ensure required formulation info
    required_cols = ['molar_ratio_ionizable', 'helper_lipid', 'peg_lipid']
    for col in required_cols:
        df = df[df[col].notna()].copy()
    print(f"  After formulation check: {len(df)}")
    
    # 4. Ensure Mouse data (for consistency)
    df = df[df['animal_species'] == 'Mouse'].copy()
    print(f"  After species check: {len(df)}")
    
    # 5. Ensure IV administration (for initial model)
    df = df[df['administration_route'].isin(['IV', 'IM'])].copy()
    print(f"  After route check: {len(df)}")
    
    print(f"\n  Removed {initial_size - len(df)} invalid entries")
    print(f"  Final dataset: {len(df)} valid entries")
    
    return df


def create_classification_labels(df):
    """
    Create classification labels for delivery efficiency
    Model 1: ≥1× MC3 = positive
    Model 2: ≥2× MC3 = positive
    """
    
    # Model 1 labels (≥1× MC3)
    df['label_model1'] = (df['fold_vs_mc3'] >= MC3_THRESHOLD_MODEL1).astype(int)
    
    # Model 2 labels (≥2× MC3)
    df['label_model2'] = (df['fold_vs_mc3'] >= MC3_THRESHOLD_MODEL2).astype(int)
    
    print(f"\nClassification labels created:")
    print(f"  Model 1 (≥1× MC3): {df['label_model1'].sum()} positive / {len(df)} total ({df['label_model1'].mean()*100:.1f}%)")
    print(f"  Model 2 (≥2× MC3): {df['label_model2'].sum()} positive / {len(df)} total ({df['label_model2'].mean()*100:.1f}%)")
    
    return df


def split_data(df):
    """
    Split data into train/test sets
    Using stratified sampling to maintain class balance
    """
    
    # Split by Model 1 labels (more balanced)
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df['label_model1'],
        random_state=RANDOM_STATE
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")
    
    # Verify stratification
    print(f"\n  Train Model 1 positive rate: {train['label_model1'].mean()*100:.1f}%")
    print(f"  Test Model 1 positive rate:  {test['label_model1'].mean()*100:.1f}%")
    
    return train, test


def save_processed_data(train, test):
    """Save processed datasets"""
    
    train.to_csv(PROCESSED_DATA_DIR / 'train.csv', index=False)
    test.to_csv(PROCESSED_DATA_DIR / 'test.csv', index=False)
    
    print(f"\n✓ Processed data saved to: {PROCESSED_DATA_DIR}")


def main():
    """Main preprocessing pipeline"""
    
    print("="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Load data
    df = load_raw_data()
    
    # Clean data
    df = clean_data(df)
    
    # Create labels
    df = create_classification_labels(df)
    
    # Split data
    train, test = split_data(df)
    
    # Save
    save_processed_data(train, test)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print("Next: python src/3_feature_engineering.py")
    print("="*70)


if __name__ == '__main__':
    main()
