#!/usr/bin/env python
"""
Convert downloaded dataset to required format
"""

import pandas as pd
from pathlib import Path

def convert_supplementary_to_format(input_file):
    """
    Convert Supplementary Table 1 to required format
    """
    
    print(f"Converting: {input_file}")
    
    # Read the supplementary table
    df = pd.read_excel(input_file)
    
    # Map columns to required format
    # (Adjust column names based on actual supplementary table)
    
    converted = pd.DataFrame()
    
    # Example mapping (adjust based on actual column names):
    try:
        converted['lipid_id'] = df.get('Lipid ID', '')
        converted['source'] = df.get('Reference', '')
        converted['lipid_name'] = df.get('Ionizable Lipid Name', '')
        converted['smiles'] = df.get('SMILES', '')
        converted['helper_lipid'] = df.get('Helper Lipid', 'DSPC')
        converted['peg_lipid'] = df.get('PEG Lipid', 'DMG-PEG2000')
        converted['cholesterol'] = 'Yes'
        converted['molar_ratio_ionizable'] = df.get('Ionizable Lipid Ratio', 50.0)
        converted['molar_ratio_helper'] = df.get('Helper Lipid Ratio', 10.0)
        converted['molar_ratio_cholesterol'] = df.get('Cholesterol Ratio', 38.5)
        converted['molar_ratio_peg'] = df.get('PEG Ratio', 1.5)
        converted['apparent_pka'] = df.get('Apparent pKa', '')
        converted['particle_size_nm'] = df.get('Size (nm)', '')
        converted['pdi'] = df.get('PDI', '')
        converted['encapsulation_efficiency'] = df.get('EE (%)', '')
        converted['animal_species'] = 'Mouse'
        converted['administration_route'] = df.get('Route', 'IV')
        converted['mrna_type'] = df.get('mRNA Type', 'Luciferase')
        converted['dose_mg_per_kg'] = df.get('Dose', 0.5)
        converted['expression_level'] = df.get('Expression Level', '')
        converted['fold_vs_mc3'] = df.get('Fold vs MC3', '')
        
        # Save
        output_path = Path("data/raw/lipid_data.csv")
        converted.to_csv(output_path, index=False)
        
        print(f"✓ Converted: {len(converted)} entries")
        print(f"✓ Saved to: {output_path}")
        
        return converted
    
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        print("\nPlease check column names in the downloaded file")
        print("Columns found:", list(df.columns))
        return None


if __name__ == '__main__':
    # Example usage
    input_file = "data/raw/supplementary_table1.xlsx"
    
    if Path(input_file).exists():
        convert_supplementary_to_format(input_file)
    else:
        print(f"File not found: {input_file}")
        print("Please download the dataset first")
