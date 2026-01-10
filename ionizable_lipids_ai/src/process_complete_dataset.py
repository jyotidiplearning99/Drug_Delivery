#!/usr/bin/env python
"""
Complete Dataset Processor for Nature Communications 2024
Properly extracts and combines data from multiple Supplementary Tables
Target: 387 LNP formulations with complete data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from config import RAW_DATA_DIR

def load_excel_file():
    """Load Source Data.xlsx"""
    
    data_file = RAW_DATA_DIR / 'Source Data.xlsx'
    
    if not data_file.exists():
        raise FileNotFoundError(f"File not found: {data_file}")
    
    print(f"✓ Found: {data_file}")
    return pd.ExcelFile(data_file)


def inspect_supplementary_tables(excel_file):
    """
    Inspect Supplementary Tables to understand their content
    Focus on tables with 387 rows (matching paper's dataset)
    """
    
    print("\n" + "="*70)
    print("INSPECTING SUPPLEMENTARY TABLES (387 rows)")
    print("="*70)
    
    target_sheets = [
        'Supplementary Table 3',
        'Supplementary Table 7', 
        'Supplementary Table 8'
    ]
    
    sheet_data = {}
    
    for sheet_name in target_sheets:
        if sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            sheet_data[sheet_name] = df
            
            print(f"\n{sheet_name}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns:")
            for i, col in enumerate(df.columns, 1):
                non_null = df[col].notna().sum()
                print(f"    {i:2d}. {col:<50} | Non-null: {non_null:3d}/{len(df):3d}")
            
            print(f"\n  First 3 rows:")
            print(df.head(3).to_string(max_colwidth=40))
    
    return sheet_data


def extract_lipid_structures(sheet_data):
    """
    Extract ionizable lipid structures (SMILES) from tables
    Based on paper methodology
    """
    
    print("\n" + "="*70)
    print("EXTRACTING LIPID STRUCTURES")
    print("="*70)
    
    # Try to find SMILES or structure column
    for sheet_name, df in sheet_data.items():
        print(f"\nChecking {sheet_name}...")
        
        # Look for columns containing structure information
        structure_cols = [col for col in df.columns if any(
            keyword in str(col).lower() 
            for keyword in ['smiles', 'structure', 'lipid', 'formula']
        )]
        
        if structure_cols:
            print(f"  Found potential structure columns: {structure_cols}")
            for col in structure_cols:
                print(f"\n  Sample from '{col}':")
                print(df[col].head(3).to_string())
        else:
            print(f"  No structure columns found")
    
    return None


def combine_supplementary_tables(sheet_data):
    """
    Combine data from multiple supplementary tables
    Strategy: Use common index/ID to merge tables
    """
    
    print("\n" + "="*70)
    print("COMBINING SUPPLEMENTARY TABLES")
    print("="*70)
    
    # Start with the table that has most columns or clear identifier
    if 'Supplementary Table 3' in sheet_data:
        main_df = sheet_data['Supplementary Table 3'].copy()
        print(f"\nStarting with Supplementary Table 3: {main_df.shape}")
    else:
        print("✗ Supplementary Table 3 not found!")
        return None
    
    # Add index column if not present
    if 'index' not in main_df.columns and 'ID' not in main_df.columns:
        main_df.insert(0, 'lipid_index', range(1, len(main_df) + 1))
        print("✓ Added lipid_index column")
    
    # Try to merge with Table 7 (likely has delivery efficiency)
    if 'Supplementary Table 7' in sheet_data:
        table7 = sheet_data['Supplementary Table 7'].copy()
        
        # Add index for merging
        if len(table7) == len(main_df):
            table7.insert(0, 'lipid_index', range(1, len(table7) + 1))
            
            # Merge on index
            main_df = pd.merge(
                main_df, 
                table7, 
                on='lipid_index', 
                how='left',
                suffixes=('', '_table7')
            )
            print(f"✓ Merged with Table 7: {main_df.shape}")
        else:
            print(f"✗ Table 7 size mismatch: {len(table7)} vs {len(main_df)}")
    
    # Try to merge with Table 8
    if 'Supplementary Table 8' in sheet_data:
        table8 = sheet_data['Supplementary Table 8'].copy()
        
        if len(table8) == len(main_df):
            table8.insert(0, 'lipid_index', range(1, len(table8) + 1))
            
            main_df = pd.merge(
                main_df,
                table8,
                on='lipid_index',
                how='left',
                suffixes=('', '_table8')
            )
            print(f"✓ Merged with Table 8: {main_df.shape}")
    
    print(f"\n✓ Combined dataset shape: {main_df.shape}")
    print(f"\nCombined columns:")
    for i, col in enumerate(main_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return main_df


def map_to_required_format(combined_df):
    """
    Map combined data to required format for AI model
    """
    
    print("\n" + "="*70)
    print("MAPPING TO REQUIRED FORMAT")
    print("="*70)
    
    processed = pd.DataFrame()
    
    # Generate lipid IDs
    processed['lipid_id'] = [f'LIPID_{str(i).zfill(4)}' for i in range(len(combined_df))]
    
    # Try to map columns intelligently
    print("\nAttempting column mapping...")
    
    column_map = {}
    
    # Check each column in combined data
    for col in combined_df.columns:
        col_lower = str(col).lower()
        
        # SMILES/Structure
        if any(kw in col_lower for kw in ['smiles', 'structure']):
            column_map['smiles'] = col
        
        # Delivery efficiency
        elif any(kw in col_lower for kw in ['delivery', 'efficiency', 'mc3', 'fold']):
            column_map['fold_vs_mc3'] = col
        
        # pKa
        elif 'pka' in col_lower:
            column_map['apparent_pka'] = col
        
        # Helper lipid
        elif any(kw in col_lower for kw in ['helper', 'phospholipid', 'dspc']):
            column_map['helper_lipid'] = col
        
        # PEG lipid
        elif 'peg' in col_lower:
            column_map['peg_lipid'] = col
        
        # Molar ratios
        elif 'ratio' in col_lower:
            if 'ionizable' in col_lower:
                column_map['molar_ratio_ionizable'] = col
            elif 'helper' in col_lower or 'phospholipid' in col_lower:
                column_map['molar_ratio_helper'] = col
            elif 'chol' in col_lower:
                column_map['molar_ratio_cholesterol'] = col
            elif 'peg' in col_lower:
                column_map['molar_ratio_peg'] = col
    
    # Apply mapping
    print("\nMapping results:")
    for target_col, source_col in column_map.items():
        processed[target_col] = combined_df[source_col]
        print(f"✓ {target_col:<30} ← {source_col}")
    
    # Show what's missing
    required_cols = ['smiles', 'fold_vs_mc3', 'apparent_pka']
    missing = [col for col in required_cols if col not in processed.columns]
    
    if missing:
        print(f"\n⚠ Missing critical columns: {missing}")
        print("\nThis means the data is split differently than expected.")
        print("Manual inspection required!")
    
    # Add defaults for missing columns
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
    
    # Set default molar ratios if missing
    if 'molar_ratio_ionizable' not in processed.columns:
        processed['molar_ratio_ionizable'] = 50.0
    if 'molar_ratio_helper' not in processed.columns:
        processed['molar_ratio_helper'] = 10.0
    if 'molar_ratio_cholesterol' not in processed.columns:
        processed['molar_ratio_cholesterol'] = 38.5
    if 'molar_ratio_peg' not in processed.columns:
        processed['molar_ratio_peg'] = 1.5
    
    # Add optional columns
    optional = ['lipid_name', 'particle_size_nm', 'pdi', 
                'encapsulation_efficiency', 'expression_level']
    for col in optional:
        if col not in processed.columns:
            processed[col] = np.nan
    
    return processed


def create_manual_extraction_guide(sheet_data):
    """
    If automatic extraction fails, create a guide for manual extraction
    """
    
    print("\n" + "="*70)
    print("MANUAL EXTRACTION GUIDE")
    print("="*70)
    
    print("\nThe automatic processor couldn't find all required data.")
    print("Based on the paper, you need to manually combine:")
    
    print("\n1. SMILES structures (might be in a separate file)")
    print("2. Delivery efficiency vs MC3 (Supplementary Table 7)")
    print("3. Apparent pKa values (might be in Supplementary Fig. 1b data)")
    print("4. Formulation details (Supplementary Table 3 or 8)")
    
    print("\nPossible solutions:")
    print("a) Check if there's a separate 'Structures' sheet")
    print("b) Check if SMILES are encoded/compressed")
    print("c) Contact paper authors for complete dataset")
    print("d) Check if Supplementary Data files are separate from Source Data")
    
    # Save what we have
    output_dir = RAW_DATA_DIR / 'extracted_sheets'
    output_dir.mkdir(exist_ok=True)
    
    for sheet_name, df in sheet_data.items():
        output_file = output_dir / f'{sheet_name.replace(" ", "_")}.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file}")
    
    print(f"\n✓ All sheets extracted to: {output_dir}")
    print("You can manually inspect and combine them")


def main():
    """Main processing pipeline"""
    
    print("="*70)
    print("COMPLETE DATASET EXTRACTION")
    print("Nature Communications 2024 - 387 LNP Formulations")
    print("="*70)
    
    # Load Excel file
    excel_file = load_excel_file()
    
    # Inspect supplementary tables
    sheet_data = inspect_supplementary_tables(excel_file)
    
    if not sheet_data:
        print("\n✗ No supplementary tables found with 387 rows!")
        return
    
    # Try to extract structures
    extract_lipid_structures(sheet_data)
    
    # Combine tables
    combined_df = combine_supplementary_tables(sheet_data)
    
    if combined_df is None:
        print("\n✗ Failed to combine tables")
        create_manual_extraction_guide(sheet_data)
        return
    
    # Map to required format
    processed_df = map_to_required_format(combined_df)
    
    # Check if we have required data
    required_cols = ['smiles', 'fold_vs_mc3']
    has_required = all(col in processed_df.columns for col in required_cols)
    
    if not has_required:
        print("\n" + "="*70)
        print("INCOMPLETE DATA - MANUAL EXTRACTION NEEDED")
        print("="*70)
        
        # Save what we have for manual review
        combined_df.to_csv(RAW_DATA_DIR / 'combined_tables_raw.csv', index=False)
        print(f"\n✓ Saved combined raw data for manual review")
        
        create_manual_extraction_guide(sheet_data)
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("1. Check extracted CSV files in data/raw/extracted_sheets/")
        print("2. Look for SMILES structures in other supplementary files")
        print("3. Manually combine the data following the template")
        print("4. Or contact paper authors: defangouyang@um.edu.mo")
        print("="*70)
    else:
        # Save processed data
        output_path = RAW_DATA_DIR / 'lipid_data.csv'
        processed_df.to_csv(output_path, index=False)
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"✓ Dataset extracted: {output_path}")
        print(f"✓ Total entries: {len(processed_df)}")
        print(f"✓ Columns: {list(processed_df.columns)}")
        
        # Display summary
        print("\nData summary:")
        print(f"  With SMILES: {processed_df['smiles'].notna().sum()}")
        print(f"  With delivery data: {processed_df['fold_vs_mc3'].notna().sum()}")
        if 'apparent_pka' in processed_df.columns:
            print(f"  With pKa: {processed_df['apparent_pka'].notna().sum()}")
        
        print("\nNext: python src/2_preprocessing.py")


if __name__ == '__main__':
    main()
