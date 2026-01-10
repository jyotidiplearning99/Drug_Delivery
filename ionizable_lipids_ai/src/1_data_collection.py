"""
Data Collection Module
Creates template and example data based on Nature Communications paper
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from config import RAW_DATA_DIR

def create_data_template():
    """Create template matching paper's Supplementary Table 1"""
    
    template = {
        # Identifiers
        'lipid_id': [],
        'source': [],  # Paper/patent reference
        'lipid_name': [],
        
        # Chemical structure (SMILES format)
        'smiles': [],
        
        # LNP Formulation
        'helper_lipid': [],  # DSPC, DOPE, DOPC
        'peg_lipid': [],  # DMG-PEG2000, etc.
        'cholesterol': [],  # Always 'Yes'
        'molar_ratio_ionizable': [],  # e.g., 50.0
        'molar_ratio_helper': [],  # e.g., 10.0
        'molar_ratio_cholesterol': [],  # e.g., 38.5
        'molar_ratio_peg': [],  # e.g., 1.5
        
        # LNP Properties
        'apparent_pka': [],  # Measured pKa
        'particle_size_nm': [],
        'pdi': [],
        'encapsulation_efficiency': [],
        
        # Experimental Conditions
        'animal_species': [],  # Mouse
        'administration_route': [],  # IV
        'mrna_type': [],  # Luciferase or hEPO
        'dose_mg_per_kg': [],
        
        # Outcomes
        'expression_level': [],  # Luminescence or protein conc
        'fold_vs_mc3': [],  # Normalized to MC3
    }
    
    return pd.DataFrame(template)


def add_example_data():
    """Add example entries from the paper"""
    
    # MC3 control (DLin-MC3-DMA)
    mc3 = {
        'lipid_id': 'MC3_001',
        'source': 'Standard control',
        'lipid_name': 'DLin-MC3-DMA',
        'smiles': 'CCCCCCCCCCCC(=O)OCC(COP(O)(=O)OCC(C)C)OC(=O)CCCCCCCCCCC',
        'helper_lipid': 'DSPC',
        'peg_lipid': 'DMG-PEG2000',
        'cholesterol': 'Yes',
        'molar_ratio_ionizable': 50.0,
        'molar_ratio_helper': 10.0,
        'molar_ratio_cholesterol': 38.5,
        'molar_ratio_peg': 1.5,
        'apparent_pka': 6.44,
        'particle_size_nm': 80.0,
        'pdi': 0.10,
        'encapsulation_efficiency': 95.0,
        'animal_species': 'Mouse',
        'administration_route': 'IV',
        'mrna_type': 'Luciferase',
        'dose_mg_per_kg': 0.5,
        'expression_level': 8.66e8,  # p/s
        'fold_vs_mc3': 1.0,
    }
    
    # SM-102 (superior control)
    sm102 = {
        'lipid_id': 'SM102_001',
        'source': 'Moderna COVID vaccine',
        'lipid_name': 'SM-102',
        'smiles': 'CCCCCCCC(C)CC(O)CNC(=O)OCC(COCCCCCCCC)OCCCCCCCCCCC',
        'helper_lipid': 'DSPC',
        'peg_lipid': 'DMG-PEG2000',
        'cholesterol': 'Yes',
        'molar_ratio_ionizable': 50.0,
        'molar_ratio_helper': 10.0,
        'molar_ratio_cholesterol': 38.5,
        'molar_ratio_peg': 1.5,
        'apparent_pka': 6.75,
        'particle_size_nm': 75.0,
        'pdi': 0.08,
        'encapsulation_efficiency': 96.0,
        'animal_species': 'Mouse',
        'administration_route': 'IV',
        'mrna_type': 'Luciferase',
        'dose_mg_per_kg': 0.5,
        'expression_level': 2.5e9,
        'fold_vs_mc3': 2.89,
    }
    
    # ALC-0315 (BioNTech/Pfizer vaccine)
    alc0315 = {
        'lipid_id': 'ALC0315_001',
        'source': 'BioNTech/Pfizer COVID vaccine',
        'lipid_name': 'ALC-0315',
        'smiles': 'CCCCCCCC(C)CCCCCCCOC(=O)NCCCCCCCCNC(=O)OCCCCCCCCC(C)CCCCCCCC',
        'helper_lipid': 'DSPC',
        'peg_lipid': 'ALC-0159',
        'cholesterol': 'Yes',
        'molar_ratio_ionizable': 46.3,
        'molar_ratio_helper': 9.4,
        'molar_ratio_cholesterol': 42.7,
        'molar_ratio_peg': 1.6,
        'apparent_pka': 6.09,
        'particle_size_nm': 90.0,
        'pdi': 0.12,
        'encapsulation_efficiency': 93.0,
        'animal_species': 'Mouse',
        'administration_route': 'IM',
        'mrna_type': 'Luciferase',
        'dose_mg_per_kg': 0.5,
        'expression_level': 1.5e9,
        'fold_vs_mc3': 1.73,
    }
    
    df = pd.DataFrame([mc3, sm102, alc0315])
    return df


def main():
    print("="*70)
    print("DATA COLLECTION: Ionizable Lipids for mRNA Delivery")
    print("="*70)
    
    # Create template
    print("\n1. Creating data template...")
    template = create_data_template()
    template_path = RAW_DATA_DIR / 'lipid_data_template.csv'
    template.to_csv(template_path, index=False)
    print(f"✓ Template saved: {template_path}")
    
    # Add example data
    print("\n2. Adding example lipids (MC3, SM-102, ALC-0315)...")
    example_data = add_example_data()
    data_path = RAW_DATA_DIR / 'lipid_data.csv'
    example_data.to_csv(data_path, index=False)
    print(f"✓ Example data saved: {data_path}")
    print(f"   Contains {len(example_data)} lipid entries")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Collect more data from paper's Supplementary Table 1")
    print("2. Fill lipid_data.csv with ionizable lipid structures")
    print("3. Target: 387 LNP formulations (as in paper)")
    print("4. Run: python src/2_preprocessing.py")
    print("="*70)


if __name__ == '__main__':
    main()
