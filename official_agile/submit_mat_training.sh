#!/bin/bash -l
#SBATCH --job-name=agile_final
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=logs/agile_%j.out
#SBATCH --error=logs/agile_%j.err

set -e

mkdir -p logs

echo "=========================================="
echo "AGILE Fine-tuning - Final Working Solution"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Initialize module system (CRITICAL!)
source /appl/profile/zz-csc-env.sh

# Clean environment
module --force purge
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH
export PYTHONNOUSERSITE=1

# Load CSC Python module
module load python-data/3.10-24.04

# Activate YOUR venv from login node
source /scratch/project_2010751/venvs/agile_fixed/bin/activate

# Verify environment
echo "Python: $(which python3)"
python3 << 'PYEOF'
import torch
import torch_geometric
import pandas
print(f"✓ torch: {torch.__version__}")
print(f"✓ torch_geometric: {torch_geometric.__version__}")
print(f"✓ pandas: {pandas.__version__}")
PYEOF

# Navigate to project
cd /scratch/project_2010751/Drug_Delivery_Lai_project/official_agile

# Extract data if needed
if [ ! -d "data" ]; then
    unzip data.zip
fi

# Run fine-tuning with official AGILE code
python3 run_mat_agile.py

echo "=========================================="
echo "✅ AGILE Fine-tuning Complete!"
echo "=========================================="