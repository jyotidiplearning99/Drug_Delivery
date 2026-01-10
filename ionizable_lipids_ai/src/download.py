#!/usr/bin/env python
"""
Download Dataset: AI-Driven Ionizable Lipid Design
From: Nature Communications 2024
https://doi.org/10.1038/s41467-024-55072-6
"""

import os
import requests
import pandas as pd
from pathlib import Path

def download_figshare_data():
    """
    Download dataset from Figshare repository
    Paper states: "Source data deposited in figshare repository"
    URL: https://figshare.com/s/ad928807e1b4795b9b5e
    """
    
    print("="*70)
    print("DOWNLOADING DATASET FROM FIGSHARE")
    print("="*70)
    
    # Figshare direct download link
    figshare_url = "https://figshare.com/ndownloader/files/49712380"  # Main dataset
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download file
    print("\nDownloading from Figshare...")
    output_file = data_dir / "ionizable_lipids_dataset.xlsx"
    
    try:
        response = requests.get(figshare_url, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded: {output_file}")
        return output_file
    
    except Exception as e:
        print(f"✗ Error downloading from Figshare: {e}")
        return None


def download_supplementary_data():
    """
    Download Supplementary Table 1 from Nature Communications
    Contains: 387 LNP formulations with ionizable lipid structures
    """
    
    print("\n" + "="*70)
    print("DOWNLOADING SUPPLEMENTARY MATERIALS")
    print("="*70)
    
    # Nature Communications supplementary material URL
    # Format: https://static-content.springer.com/esm/art%3A[DOI]/MediaObjects/[file].xlsx
    
    supp_url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-55072-6/MediaObjects/41467_2024_55072_MOESM3_ESM.xlsx"
    
    data_dir = Path("data/raw")
    output_file = data_dir / "supplementary_table1.xlsx"
    
    try:
        print("\nDownloading Supplementary Table 1...")
        response = requests.get(supp_url, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded: {output_file}")
        return output_file
    
    except Exception as e:
        print(f"✗ Error downloading supplementary: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.nature.com/articles/s41467-024-55072-6#Sec25")
        return None


def process_downloaded_data(file_path):
    """
    Process downloaded data and convert to required format
    """
    
    if file_path is None or not file_path.exists():
        print("\n✗ No data file to process")
        return
    
    print("\n" + "="*70)
    print("PROCESSING DOWNLOADED DATA")
    print("="*70)
    
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        print(f"\n✓ Loaded data: {len(df)} entries")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Save as CSV for easier processing
        csv_path = file_path.parent / "lipid_data.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Saved as CSV: {csv_path}")
        
        # Display summary
        print("\n" + "-"*70)
        print("DATASET SUMMARY")
        print("-"*70)
        print(df.head())
        print("\n" + "-"*70)
        print(f"Total entries: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        
        return df
    
    except Exception as e:
        print(f"✗ Error processing data: {e}")
        return None


def create_template_from_paper():
    """
    Create data template based on paper's methodology
    Section: Methods - Data preparation
    """
    
    print("\n" + "="*70)
    print("CREATING DATA TEMPLATE FROM PAPER")
    print("="*70)
    
    # Template based on paper's data structure
    template = {
        # Identifiers
        'lipid_id': [],
        'source': [],  # Patent/paper reference
        'ionizable_lipid_name': [],
        
        # Chemical structure
        'smiles': [],  # SMILES notation
        
        # LNP Formulation (from paper methods)
        'helper_lipid': [],  # DSPC, DOPE, DOPC
        'peg_lipid': [],  # DMG-PEG2000, C14-PEG2000, C16-PEG2000, ALC-0159
        'cholesterol': [],  # Always 'Yes'
        'molar_ratio_ionizable': [],  # e.g., 50.0
        'molar_ratio_helper': [],  # e.g., 10.0
        'molar_ratio_cholesterol': [],  # e.g., 38.5
        'molar_ratio_peg': [],  # e.g., 1.5
        
        # LNP Properties
        'apparent_pka': [],  # Measured pKa (target: 6.0-7.0)
        'particle_size_nm': [],
        'pdi': [],  # Polydispersity index
        'encapsulation_efficiency': [],  # Percentage
        
        # Experimental Conditions
        'animal_species': [],  # Mouse (from paper: only mouse data used)
        'administration_route': [],  # IV or IM
        'mrna_type': [],  # Luciferase or hEPO
        'dose_mg_per_kg': [],
        
        # Outcomes
        'expression_level': [],  # Luminescence (p/s) or protein conc
        'fold_vs_mc3': [],  # Normalized to MC3 standard
    }
    
    template_df = pd.DataFrame(template)
    
    template_path = Path("data/raw/data_template_paper.csv")
    template_df.to_csv(template_path, index=False)
    
    print(f"✓ Created template: {template_path}")
    print("\nTemplate columns:")
    for col in template.keys():
        print(f"  - {col}")
    
    return template_path


def main():
    """
    Main function to download and process dataset
    """
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  DATASET DOWNLOAD: AI-DRIVEN IONIZABLE LIPID DESIGN         ║
    ║  Nature Communications 2024                                   ║
    ║  DOI: 10.1038/s41467-024-55072-6                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Method 1: Try Figshare
    figshare_file = download_figshare_data()
    if figshare_file and figshare_file.exists():
        df = process_downloaded_data(figshare_file)
        if df is not None:
            print("\n✓ SUCCESS: Dataset downloaded from Figshare")
            return
    
    # Method 2: Try Supplementary Materials
    supp_file = download_supplementary_data()
    if supp_file and supp_file.exists():
        df = process_downloaded_data(supp_file)
        if df is not None:
            print("\n✓ SUCCESS: Dataset downloaded from Supplementary Materials")
            return
    
    # Method 3: Create template for manual data entry
    print("\n" + "="*70)
    print("FALLBACK: CREATING TEMPLATE")
    print("="*70)
    print("\nAutomatic download failed. Creating template for manual data entry...")
    
    template_path = create_template_from_paper()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. OPTION A: Download manually from paper")
    print("   - Visit: https://www.nature.com/articles/s41467-024-55072-6")
    print("   - Click 'Supplementary Information'")
    print("   - Download: Supplementary Table 1 (41467_2024_55072_MOESM3_ESM.xlsx)")
    print(f"   - Save to: data/raw/")
    
    print("\n2. OPTION B: Download from Figshare")
    print("   - Visit: https://figshare.com/s/ad928807e1b4795b9b5e")
    print("   - Download all files")
    print(f"   - Save to: data/raw/")
    
    print("\n3. OPTION C: Contact authors")
    print("   - Email: defangouyang@um.edu.mo or linjinzhong@fudan.edu.cn")
    print("   - Request: Full dataset (387 LNP formulations)")
    
    print("\n4. Then run: python src/2_preprocessing.py")
    print("="*70)


if __name__ == '__main__':
    main()
