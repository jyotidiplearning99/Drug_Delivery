"""
MASTER SCRIPT: Run Complete AI-Driven Ionizable Lipid Design Pipeline
Executes all steps from data collection to visualization
"""

import sys
import subprocess
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}")
        print(f"Error: {e}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  AI-DRIVEN IONIZABLE LIPID DESIGN FOR mRNA DELIVERY         ║
    ║  Based on: Nature Communications 2024                         ║
    ║  Complete End-to-End Pipeline                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    src_dir = Path(__file__).parent / 'src'
    
    # Pipeline steps
    steps = [
        (src_dir / '1_data_collection.py', 'Step 1: Data Collection'),
        (src_dir / '2_preprocessing.py', 'Step 2: Data Preprocessing'),
        (src_dir / '3_feature_engineering.py', 'Step 3: Feature Engineering (ECFP)'),
        (src_dir / '4_model_training.py', 'Step 4: Model Training (LightGBM)'),
        (src_dir / '5_shap_analysis.py', 'Step 5: SHAP Analysis'),
        (src_dir / '6_virtual_screening.py', 'Step 6: Virtual Screening'),
        (src_dir / '7_visualization.py', 'Step 7: Results Visualization'),
    ]
    
    # Run pipeline
    for script, description in steps:
        if not script.exists():
            print(f"✗ Script not found: {script}")
            continue
            
        success = run_script(script, description)
        
        if not success:
            print(f"\n✗ Pipeline stopped at: {description}")
            print("Please fix errors and re-run")
            return
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults saved to: results/")
    print("Models saved to: models/")
    print("="*70)


if __name__ == '__main__':
    main()
