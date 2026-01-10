"""
Virtual Screening Module
Generates and screens ~20 million virtual ionizable lipids
Following Nature Communications 2024 paper methodology
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import itertools
from config import *


def load_screening_models():
    """Load trained models for screening"""
    
    model1 = joblib.load(MODEL_DIR / 'delivery_model1.pkl')
    model2 = joblib.load(MODEL_DIR / 'delivery_model2.pkl')
    pka_model = joblib.load(MODEL_DIR / 'pka_model.pkl')
    
    print("✓ Screening models loaded")
    return model1, model2, pka_model


def generate_tail_library():
    """
    Generate tail library for lipid generation
    Paper: 892 tail candidates
    """
    
    print("\nGenerating tail library...")
    
    # Simplified tail patterns (expand based on synthesis capability)
    tail_patterns = []
    
    # Pattern 1: Simple alkyl chains
    for length in range(8, 18):
        smiles = 'C' * length
        tail_patterns.append({
            'smiles': smiles,
            'type': 'alkyl',
            'length': length
        })
    
    # Pattern 2: Ester-containing tails
    for length_before in range(4, 12):
        for length_after in range(4, 12):
            smiles = 'C' * length_before + 'C(=O)O' + 'C' * length_after
            tail_patterns.append({
                'smiles': smiles,
                'type': 'ester',
                'length': length_before + length_after
            })
    
    # Pattern 3: Cyclohexyl-containing tails
    for length_before in range(4, 10):
        for length_after in range(4, 10):
            smiles = 'C' * length_before + 'C1CCCCC1' + 'C' * length_after
            tail_patterns.append({
                'smiles': smiles,
                'type': 'cyclohexyl',
                'length': length_before + length_after + 6
            })
    
    # Pattern 4: Branched tails
    for main_length in range(8, 16):
        for branch_pos in range(3, main_length-2):
            main = 'C' * branch_pos + 'C(C)' + 'C' * (main_length - branch_pos - 1)
            tail_patterns.append({
                'smiles': main,
                'type': 'branched',
                'length': main_length + 1
            })
    
    print(f"✓ Generated {len(tail_patterns)} tail candidates")
    
    return tail_patterns


def generate_head_library():
    """
    Generate head library
    Paper: 237 head candidates, focusing on ethanolamine for Round 2
    """
    
    print("\nGenerating head library...")
    
    head_patterns = []
    
    # Ethanolamine head (most common in dataset)
    head_patterns.append({
        'smiles': 'NCCO',
        'name': 'ethanolamine',
        'type': 'hydroxyl_amine'
    })
    
    # Variations with methylation
    head_patterns.append({
        'smiles': 'N(C)CCO',
        'name': 'N-methyl-ethanolamine',
        'type': 'hydroxyl_amine'
    })
    
    # Extended ethanolamine
    head_patterns.append({
        'smiles': 'NCCCO',
        'name': 'propanolamine',
        'type': 'hydroxyl_amine'
    })
    
    print(f"✓ Generated {len(head_patterns)} head candidates")
    
    return head_patterns


def smiles_to_ecfp(smiles, radius=ECFP_RADIUS, n_bits=ECFP_BITS):
    """Convert SMILES to ECFP fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros(n_bits)


def assemble_lipid(head, tail1, tail2=None):
    """
    Assemble complete ionizable lipid from components
    Simplified assembly - extend based on synthesis capability
    """
    
    if tail2 is None:
        tail2 = tail1  # Same tails for simplicity
    
    # Simple assembly pattern: N(tail1)(tail2)head
    lipid_smiles = f"{head['smiles']}N({tail1['smiles']})({tail2['smiles']})"
    
    return {
        'smiles': lipid_smiles,
        'head': head['name'],
        'tail1_type': tail1['type'],
        'tail2_type': tail2['type'],
        'tail1_length': tail1['length'],
        'tail2_length': tail2['length']
    }


def encode_formulation(helper_lipid='DSPC', peg_lipid='DMG-PEG2000'):
    """Encode formulation features"""
    
    # Helper lipid encoding
    helper_map = {'DSPC': [1, 1], 'DOPE': [0, 1], 'DOPC': [0, 0]}
    helper_encoded = helper_map.get(helper_lipid, [0, 0])
    
    # PEG lipid encoding
    peg_map = {'DMG-PEG2000': 0, 'C14-PEG2000': 1, 'C16-PEG2000': 2}
    peg_encoded = [peg_map.get(peg_lipid, 0)]
    
    # Molar ratios (standard formulation)
    molar_ratios = [50.0, 10.0, 38.5, 1.5]
    
    return np.array(helper_encoded + peg_encoded + molar_ratios)


def screen_lipid(lipid_smiles, model1, model2, pka_model):
    """
    Screen a single lipid through all models
    Returns: (pka, prob_model1, prob_model2, pass_criteria)
    """
    
    # Generate ECFP
    ecfp = smiles_to_ecfp(lipid_smiles)
    
    # Encode formulation
    formulation = encode_formulation()
    
    # Combine features
    features = np.concatenate([ecfp, formulation]).reshape(1, -1)
    
    # Predict
    pka = pka_model.predict(features)[0]
    prob_m1 = model1.predict_proba(features)[0, 1]
    prob_m2 = model2.predict_proba(features)[0, 1]
    
    # Check criteria
    pka_ok = (PKA_RANGE[0] <= pka <= PKA_RANGE[1])
    model1_ok = prob_m1 > 0.5
    model2_ok = prob_m2 > 0.5
    
    return {
        'pka': pka,
        'prob_model1': prob_m1,
        'prob_model2': prob_m2,
        'pka_ok': pka_ok,
        'model1_positive': model1_ok,
        'model2_positive': model2_ok,
        'pass_round1': pka_ok and model1_ok,
        'pass_round2': pka_ok and model2_ok
    }


def virtual_screening_round1(heads, tails, model1, pka_model, max_lipids=10000):
    """
    First round of virtual screening
    Paper: Screen with Model 1 (≥1× MC3 criterion)
    """
    
    print("\n" + "="*70)
    print("FIRST ROUND VIRTUAL SCREENING")
    print("Criterion: ≥1× MC3 delivery efficiency")
    print("="*70)
    
    results = []
    lipids_screened = 0
    
    # Generate lipids
    print("\nGenerating and screening lipids...")
    
    for head in tqdm(heads, desc="Heads"):
        for tail1, tail2 in itertools.islice(
            itertools.product(tails, repeat=2), 
            max_lipids // len(heads)
        ):
            # Assemble lipid
            lipid = assemble_lipid(head, tail1, tail2)
            
            # Screen
            screening_result = screen_lipid(
                lipid['smiles'], model1, None, pka_model
            )
            
            # Store if passes criteria
            if screening_result['pass_round1']:
                result = {**lipid, **screening_result}
                results.append(result)
            
            lipids_screened += 1
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('prob_model1', ascending=False)
    
    print(f"\n✓ Screened {lipids_screened:,} lipids")
    print(f"✓ Found {len(results_df)} passing lipids")
    
    # Save
    results_df.to_csv(RESULTS_DIR / 'round1_candidates.csv', index=False)
    
    return results_df


def virtual_screening_round2(round1_candidates, model2):
    """
    Second round of virtual screening
    Paper: Apply Model 2 (≥2× MC3 criterion) to Round 1 candidates
    """
    
    print("\n" + "="*70)
    print("SECOND ROUND VIRTUAL SCREENING")
    print("Criterion: ≥2× MC3 delivery efficiency")
    print("="*70)
    
    results = []
    
    print(f"\nApplying Model 2 to {len(round1_candidates)} Round 1 candidates...")
    
    for _, lipid in tqdm(round1_candidates.iterrows(), total=len(round1_candidates)):
        # Screen with Model 2
        screening_result = screen_lipid(
            lipid['smiles'], None, model2, None
        )
        
        # Store if passes stricter criteria
        if screening_result['model2_positive']:
            result = {
                'smiles': lipid['smiles'],
                'head': lipid['head'],
                'tail1_type': lipid['tail1_type'],
                'tail2_type': lipid['tail2_type'],
                'pka': lipid['pka'],
                'prob_model1': lipid['prob_model1'],
                'prob_model2': screening_result['prob_model2']
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('prob_model2', ascending=False)
    
    print(f"\n✓ Found {len(results_df)} lipids passing stricter criteria")
    
    # Save
    results_df.to_csv(RESULTS_DIR / 'round2_candidates.csv', index=False)
    
    return results_df


def select_diverse_candidates(candidates, n_select=6):
    """
    Select diverse candidates for experimental validation
    Based on structural diversity and predicted performance
    """
    
    print(f"\nSelecting {n_select} diverse candidates...")
    
    # Select top performers with diverse structures
    selected = []
    
    # Group by tail types
    for tail_combo in candidates[['tail1_type', 'tail2_type']].drop_duplicates().values:
        subset = candidates[
            (candidates['tail1_type'] == tail_combo[0]) &
            (candidates['tail2_type'] == tail_combo[1])
        ]
        
        if len(subset) > 0 and len(selected) < n_select:
            # Pick top from this structural class
            selected.append(subset.iloc[0])
    
    selected_df = pd.DataFrame(selected)
    
    print("✓ Selected diverse candidates:")
    print(selected_df[['smiles', 'head', 'tail1_type', 'tail2_type', 'pka', 'prob_model2']].to_string(index=False))
    
    # Save
    selected_df.to_csv(RESULTS_DIR / 'selected_for_synthesis.csv', index=False)
    
    return selected_df


def main():
    """Main virtual screening pipeline"""
    
    print("="*70)
    print("VIRTUAL SCREENING MODULE")
    print("Generating and screening ionizable lipids")
    print("="*70)
    
    # Load models
    model1, model2, pka_model = load_screening_models()
    
    # Generate libraries
    heads = generate_head_library()
    tails = generate_tail_library()
    
    print(f"\nLibrary size:")
    print(f"  Heads: {len(heads)}")
    print(f"  Tails: {len(tails)}")
    print(f"  Potential lipids: ~{len(heads) * len(tails) * len(tails):,}")
    
    # Round 1 screening
    round1_candidates = virtual_screening_round1(
        heads, tails, model1, pka_model, max_lipids=10000
    )
    
    # Round 2 screening
    if len(round1_candidates) > 0:
        round2_candidates = virtual_screening_round2(round1_candidates, model2)
        
        # Select for synthesis
        if len(round2_candidates) > 0:
            selected = select_diverse_candidates(round2_candidates, n_select=6)
    
    print("\n" + "="*70)
    print("VIRTUAL SCREENING COMPLETE")
    print("="*70)
    print(f"Results saved to: {RESULTS_DIR}")
    print("\nScreening summary:")
    print(f"  Round 1 candidates: {len(round1_candidates)}")
    if 'round2_candidates' in locals():
        print(f"  Round 2 candidates: {len(round2_candidates)}")
    if 'selected' in locals():
        print(f"  Selected for synthesis: {len(selected)}")
    print("\nNext: python src/7_visualization.py")
    print("="*70)


if __name__ == '__main__':
    main()
