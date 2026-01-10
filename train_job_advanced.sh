#!/bin/bash
#SBATCH --job-name=lai_modern
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --output=results/modern_%j.out
#SBATCH --error=results/modern_%j.err

# Set unbuffered output
export PYTHONUNBUFFERED=1

echo "Starting Modern Algorithm Comparison"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module purge
module load python-data/3.10-24.04

# Activate venv
source venv/bin/activate

# Install modern packages
pip install catboost openpyxl --quiet

# Run training
python -u train_model_advanced.py

echo "Complete!"
EOF

chmod +x train_job_advanced.sh