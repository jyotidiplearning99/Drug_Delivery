#!/usr/bin/env python
"""
Process Source Data.xlsx from Figshare
Extracts 387 LNP formulations for AI model training
Based on Nature Communications 2024 paper methodology
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from config import RAW_DATA_DIR

def load_source_data():
    """Load Source Data.xlsx from Figshare"""
    
    data_file = RAW_DATA_DIR / 'Source Data.xlsx'
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Source Data.xlsx not found in {RAW_DATA_DIR}\n"
            "Please download from: https://figshare.com/s/ad928807e1b4795b9b5e"
        )
    
    print(f"✓ Found: {data_file}")
    
    # Load all sheets to find the main data
    excel_file = pd.ExcelFile(data_file)
    print(f"✓ Excel file contains {len(excel_file.sheet_names)} sheet(s):")
    
    for sheet_name in excel_file.sheet_names:
        print(f"  - {sheet_name}")
    
    return excel_file


def identify_main_sheet(excel_file):
    """
    Identify which sheet contains the main dataset
    Based on paper: 387 LNP formulations
    """
    
    print("\nAnalyzing sheets to find main dataset...")
    
    candidates = []
    
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Check if this might be the main data
        # Look for key columns mentioned in paper
        key_columns = ['smiles', 'structure', 'lipid', 'pka', 'delivery', 'mc3', 'fold']
        
        column_str = ' '.join(df.columns.astype(str)).lower()
        matches = sum(1 for kw in key_columns if kw in column_str)
        
        candidates.append({
            'sheet_name': sheet_name,
            'rows': len(df),
            'columns': len(df.columns),
            'matches': matches,
            'score': matches * len(df)  # Prioritize sheets with more data and matching columns
        })
        
        print(f"  {sheet_name}: {len(df)} rows, {len(df.columns)} cols, {matches} keyword matches")
    
    # Select best candidate
    best = max(candidates, key=lambda x: x['score'])
    
    print(f"\n✓ Selected main sheet: '{best['sheet_name']}' ({best['rows']} rows)")
    
    return best['sheet_name']


def process_main_data(excel_file, sheet_name):
    """
    Process the main data sheet
    Extract and map columns according to paper's data structure
    """
    
    print(f"\nProcessing sheet: '{sheet_name}'...")
    
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    print(f"Original data: {df.shape}")
    print(f"\nOriginal columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Create processed dataset
    processed = pd.DataFrame()
    
    # Column mapping strategy
    # Try to find columns by common names/patterns
    
    def find_column(df, possible_names, default=None):
        """Find column by trying multiple possible names (case-insensitive)"""
        for name in possible_names:
            for col in df.columns:
                if name.lower() in str(col).lower():
                    return col
        return default
    
    # Map columns
    column_mapping = {
        'lipid_id': ['ID', 'Lipid_ID', 'Index', 'Number'],
        'lipid_name': ['Name', 'Lipid_Name', 'Ionizable_Lipid', 'Lipid'],
        'smiles': ['SMILES', 'Structure', 'Lipid_Structure'],
        'helper_lipid': ['Helper', 'Helper_Lipid', 'Phospholipid', 'DSPC'],
        'peg_lipid': ['PEG', 'PEG_Lipid', 'PEG-Lipid'],
        'molar_ratio_ionizable': ['Ionizable', 'Ionizable_Ratio', 'Lipid_Ratio', 'IL'],
        'molar_ratio_helper': ['Helper_Ratio', 'Phospholipid_Ratio', 'PL'],
        'molar_ratio_cholesterol': ['Cholesterol', 'Chol', 'Cholesterol_Ratio'],
        'molar_ratio_peg': ['PEG_Ratio'],
        'apparent_pka': ['pKa', 'Apparent_pKa', 'LNP_pKa'],
        'fold_vs_mc3': ['Fold', 'MC3', 'Delivery', 'Efficiency', 'Relative'],
        'particle_size_nm': ['Size', 'Diameter', 'Particle'],
        'pdi': ['PDI', 'Polydispersity'],
        'encapsulation_efficiency': ['EE', 'Encapsulation'],
    }
    
    print("\n" + "="*70)
    print("COLUMN MAPPING")
    print("="*70)
    
    for target_col, possible_names in column_mapping.items():
        found_col = find_column(df, possible_names)
        
        if found_col:
            processed[target_col] = df[found_col]
            print(f"✓ {target_col:<30} ← {found_col}")
        else:
            print(f"✗ {target_col:<30} (not found)")
    
    # Generate IDs if not present
    if 'lipid_id' not in processed.columns or processed['lipid_id'].isna().all():
        processed['lipid_id'] = [f'LIPID_{str(i).zfill(4)}' for i in range(len(processed))]
        print(f"✓ Generated lipid_id")
    
    # Fill defaults for missing columns
    defaults = {
        'source': 'Nature Communications 2024',
        'cholesterol': 'Yes',
        'animal_species': 'Mouse',
        'administration_route': 'IV',
        'mrna_type': 'Luciferase',
        'dose_mg_per_kg': 0.5,
    }
    
    for col, value in defaults.items():
        if col not in processed.columns:
            processed[col] = value
    
    # Set default molar ratios if missing (standard formulation: 50:10:38.5:1.5)
    if 'molar_ratio_ionizable' not in processed.columns or processed['molar_ratio_ionizable'].isna().all():
        processed['molar_ratio_ionizable'] = 50.0
    if 'molar_ratio_helper' not in processed.columns or processed['molar_ratio_helper'].isna().all():
        processed['molar_ratio_helper'] = 10.0
    if 'molar_ratio_cholesterol' not in processed.columns or processed['molar_ratio_cholesterol'].isna().all():
        processed['molar_ratio_cholesterol'] = 38.5
    if 'molar_ratio_peg' not in processed.columns or processed['molar_ratio_peg'].isna().all():
        processed['molar_ratio_peg'] = 1.5
    
    # Add optional columns with NaN if missing
    optional_cols = ['expression_level']
    for col in optional_cols:
        if col not in processed.columns:
            processed[col] = np.nan
    
    print(f"\n✓ Processed dataset: {processed.shape}")
    
    return processed


def validate_processed_data(df):
    """Validate the processed dataset"""
    
    print("\n" + "="*70)
    print("DATA VALIDATION")
    print("="*70)
    
    # Check required columns
    required_cols = ['smiles', 'fold_vs_mc3']
    missing_required = [col for col in required_cols if col not in df.columns]
    
    if missing_required:
        print(f"\n✗ ERROR: Missing critical columns: {missing_required}")
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        return False
    
    print("✓ All required columns present")
    
    # Check data completeness
    print(f"\nData completeness:")
    print(f"  SMILES: {df['smiles'].notna().sum():4d} / {len(df):4d} ({df['smiles'].notna().mean()*100:.1f}%)")
    print(f"  fold_vs_mc3: {df['fold_vs_mc3'].notna().sum():4d} / {len(df):4d} ({df['fold_vs_mc3'].notna().mean()*100:.1f}%)")
    
    if 'apparent_pka' in df.columns:
        print(f"  pKa: {df['apparent_pka'].notna().sum():4d} / {len(df):4d} ({df['apparent_pka'].notna().mean()*100:.1f}%)")
    
    if 'helper_lipid' in df.columns:
        print(f"\nHelper lipid types: {df['helper_lipid'].value_counts().to_dict()}")
    
    if 'peg_lipid' in df.columns:
        print(f"PEG lipid types: {df['peg_lipid'].value_counts().to_dict()}")
    
    # Statistics
    if 'fold_vs_mc3' in df.columns:
        print(f"\nDelivery efficiency (fold vs MC3):")
        print(f"  Mean: {df['fold_vs_mc3'].mean():.2f}")
        print(f"  Median: {df['fold_vs_mc3'].median():.2f}")
        print(f"  Min: {df['fold_vs_mc3'].min():.2f}")
        print(f"  Max: {df['fold_vs_mc3'].max():.2f}")
        print(f"  ≥1× MC3: {(df['fold_vs_mc3'] >= 1.0).sum()} ({(df['fold_vs_mc3'] >= 1.0).mean()*100:.1f}%)")
        print(f"  ≥2× MC3: {(df['fold_vs_mc3'] >= 2.0).sum()} ({(df['fold_vs_mc3'] >= 2.0).mean()*100:.1f}%)")
    
    if 'apparent_pka' in df.columns:
        pka_data = df['apparent_pka'].dropna()
        if len(pka_data) > 0:
            print(f"\nApparent pKa:")
            print(f"  Mean: {pka_data.mean():.2f}")
            print(f"  Range: {pka_data.min():.2f} - {pka_data.max():.2f}")
            print(f"  In 6.0-7.0 range: {((pka_data >= 6.0) & (pka_data <= 7.0)).sum()} ({((pka_data >= 6.0) & (pka_data <= 7.0)).mean()*100:.1f}%)")
    
    # Sample display
    print("\n" + "-"*70)
    print("Sample data (first 3 entries):")
    print("-"*70)
    
    display_cols = ['lipid_id', 'lipid_name', 'fold_vs_mc3', 'apparent_pka', 'helper_lipid']
    display_cols = [c for c in display_cols if c in df.columns]
    
    print(df[display_cols].head(3).to_string(index=False))
    
    return True


def save_processed_data(df):
    """Save processed data"""
    
    output_path = RAW_DATA_DIR / 'lipid_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"✓ Total entries: {len(df)}")
    
    return output_path


def main():
    """Main processing pipeline"""
    
    print("="*70)
    print("PROCESSING SOURCE DATA FROM FIGSHARE")
    print("Nature Communications 2024 - Ionizable Lipids Dataset")
    print("="*70)
    
    # Load Excel file
    print("\n1. Loading Source Data.xlsx...")
    excel_file = load_source_data()
    
    # Identify main data sheet
    print("\n2. Identifying main data sheet...")
    main_sheet = identify_main_sheet(excel_file)
    
    # Process data
    print("\n3. Processing and mapping columns...")
    processed_df = process_main_data(excel_file, main_sheet)
    
    # Validate
    print("\n4. Validating processed data...")
    if not validate_processed_data(processed_df):
        print("\n✗ Validation failed. Please check the data and column mapping.")
        return
    
    # Save
    print("\n5. Saving processed data...")
    output_path = save_processed_data(processed_df)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"✓ Dataset ready: {output_path}")
    print(f"✓ Total lipids: {len(processed_df)}")
    print(f"✓ Expected by paper: 387 formulations")
    
    if len(processed_df) >= 300:
        print("\n✓ Dataset size matches paper expectations!")
    else:
        print(f"\n⚠ Dataset smaller than expected ({len(processed_df)} vs 387)")
        print("  This might be a subset or need additional sheets processing")
    
    print("\nNext steps:")
    print("  1. Review: head data/raw/lipid_data.csv")
    print("  2. Run preprocessing: python src/2_preprocessing.py")
    print("  3. Continue pipeline: sbatch submit_pipeline.sh")
    print("="*70)


if __name__ == '__main__':
    main()
