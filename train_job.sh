#!/bin/bash
#SBATCH --job-name=lai_full
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --output=results/training_%j.out
#SBATCH --error=results/training_%j.err

# Make Python output unbuffered
export PYTHONUNBUFFERED=1

echo "Starting LAI ML Training (FULL RUN: 10 trials × n_iter=100)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module purge
module load python-data/3.10-24.04

# Activate venv
source venv/bin/activate

# Install openpyxl if needed
pip install openpyxl --quiet

# Run with unbuffered Python
python -u train_model.py

echo "Complete!"
EOF

chmod +x train_job.sh