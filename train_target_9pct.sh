#!/bin/bash
#SBATCH --job-name=lai_best
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --output=results/best_%j.out
#SBATCH --error=results/best_%j.err

export PYTHONUNBUFFERED=1

echo "Starting Best Approach Training"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load python-data/3.10-24.04
source venv/bin/activate

pip install catboost openpyxl --quiet

python -u train_target_9mean.py

echo "Complete!"