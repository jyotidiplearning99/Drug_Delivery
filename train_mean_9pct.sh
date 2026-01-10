#!/bin/bash
#SBATCH --job-name=lai_mean_9
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --output=results/mean_9pct_%j.out
#SBATCH --error=results/mean_9pct_%j.err

export PYTHONUNBUFFERED=1

echo "Starting Mean MAE < 9% Training"
echo "Multi-model ensemble approach"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load python-data/3.10-24.04
source venv/bin/activate

pip install openpyxl --quiet

python -u train_target_9mean.py

echo "Complete!"
