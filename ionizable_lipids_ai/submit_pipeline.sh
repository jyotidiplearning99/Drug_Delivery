#!/bin/bash
#SBATCH --job-name=lipid_ai_pipeline
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --output=results/pipeline_%j.out
#SBATCH --error=results/pipeline_%j.err

# ==============================================================================
# COMPLETE AI-DRIVEN IONIZABLE LIPID DESIGN PIPELINE
# Based on Nature Communications 2024 paper
# ==============================================================================

export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "AI-DRIVEN IONIZABLE LIPID DESIGN FOR mRNA DELIVERY"
echo "Complete Pipeline Execution"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================================================"

# Load modules
module purge
module load python-data/3.10-24.04

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --quiet rdkit==2023.9.1 shap==0.43.0

# Navigate to project directory
cd ionizable_lipids_ai

# Run complete pipeline
python RUN_ALL.py

echo "========================================================================
echo "Pipeline Complete!"
echo "========================================================================"

