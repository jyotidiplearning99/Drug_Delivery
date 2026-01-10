"""
Feature Engineering Module
Converts molecular structures to ECFP fingerprints
Encodes formulation features
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from config import *


def smiles_to_ecfp(smiles, radius=ECFP_RADIUS, n_bits=ECFP_BITS):
    """
    Convert SMILES to Extended Connectivity Fingerprints (ECFP)
    Paper: radius=9, n_bits=1024
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES: {smiles}")
            return np.zeros(n_bits)
        
        # Morgan fingerprint = ECFP
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits
        )
        return np.array(fp)
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return np.zeros(n_bits)


def encode_helper_lipid(lipid_type):
    """
    Encode helper lipid as binary features
    Paper method: DSPC, DOPE, DOPC encoded as category variables
    """
    encoding_map = {
        'DSPC': [1, 1],  # DS=1, PC=1
        'DOPE': [0, 1],  # DO=0, PE=1
        'DOPC': [0, 0],  # DO=0, PC=0
    }
    return encoding_map.get(lipid_type, [0, 0])


def encode_peg_lipid(peg_type):
    """
    Encode PEG-lipid as categorical variable
    """
    peg_mapping = {
        'DMG-PEG2000': 0,
        'C14-PEG2000': 1,
        'C16-PEG2000': 2,
        'ALC-0159': 3,
    }
    return peg_mapping.get(peg_type, 0)


def engineer_features(df):
    """
    Generate all features for model training
    
    Features (1031 total):
    - ECFP fingerprints: 1024 bits
    - Helper lipid encoding: 2 features
    - PEG lipid encoding: 1 feature
    - Molar ratios: 4 features
    """
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    # Generate ECFP
    print("\nGenerating ECFP fingerprints...")
    ecfp_list = []
    
    for smiles in tqdm(df['smiles'], desc="Processing molecules"):
        ecfp = smiles_to_ecfp(smiles)
        ecfp_list.append(ecfp)
    
    ecfp_array = np.array(ecfp_list)
    print(f"✓ ECFP shape: {ecfp_array.shape}")
    
    # Encode helper lipids
    print("\nEncoding helper lipids...")
    helper_encoded = np.array([
        encode_helper_lipid(lipid) 
        for lipid in df['helper_lipid']
    ])
    
    # Encode PEG lipids
    print("Encoding PEG lipids...")
    peg_encoded = np.array([
        [encode_peg_lipid(peg)] 
        for peg in df['peg_lipid']
    ])
    
    # Extract molar ratios
    print("Extracting molar ratios...")
    molar_ratios = df[[
        'molar_ratio_ionizable',
        'molar_ratio_helper',
        'molar_ratio_cholesterol',
        'molar_ratio_peg'
    ]].values
    
    # Combine all features
    X_features = np.hstack([
        ecfp_array,      # 1024 features
        helper_encoded,  # 2 features
        peg_encoded,     # 1 feature
        molar_ratios     # 4 features
    ])
    
    print(f"\n✓ Final feature matrix: {X_features.shape}")
    print(f"  ECFP bits: 1024")
    print(f"  Helper lipid encoding: 2")
    print(f"  PEG lipid encoding: 1")
    print(f"  Molar ratios: 4")
    print(f"  Total: {X_features.shape[1]}")
    
    return X_features


def save_features(X_features, df, split_name):
    """Save engineered features and metadata"""
    
    # Save features as numpy array
    feature_path = PROCESSED_DATA_DIR / f'X_{split_name}.npy'
    np.save(feature_path, X_features)
    
    # Save labels for both models
    labels = df[[
        'lipid_id', 'smiles', 'fold_vs_mc3',
        'label_model1', 'label_model2'
    ]].copy()
    
    # Add pKa if available
    if 'apparent_pka' in df.columns:
        labels['apparent_pka'] = df['apparent_pka']
    
    labels_path = PROCESSED_DATA_DIR / f'metadata_{split_name}.csv'
    labels.to_csv(labels_path, index=False)
    
    print(f"✓ Saved features: {feature_path}")
    print(f"✓ Saved labels: {labels_path}")


def main():
    """Main feature engineering pipeline"""
    
    print("="*70)
    print("FEATURE ENGINEERING MODULE")
    print("="*70)
    
    # FIXED: Only process train and test (no val set)
    for split in ['train', 'test']:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} set")
        print(f"{'='*70}")
        
        # Check if file exists
        split_file = PROCESSED_DATA_DIR / f'{split}.csv'
        if not split_file.exists():
            print(f"✗ File not found: {split_file}")
            print(f"Skipping {split} set...")
            continue
        
        # Load data
        df = pd.read_csv(split_file)
        print(f"Loaded {len(df)} samples")
        
        # Engineer features
        X_features = engineer_features(df)
        
        # Save
        save_features(X_features, df, split)
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70)
    print("Next: python src/4_model_training.py")
    print("="*70)


if __name__ == '__main__':
    main()
