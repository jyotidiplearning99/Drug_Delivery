#!/bin/bash
#SBATCH --job-name=trial_opt
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --output=results/trial_opt_%j.out
#SBATCH --error=results/trial_opt_%j.err

export PYTHONUNBUFFERED=1

echo "Starting Trial Optimization"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load python-data/3.10-24.04
source venv/bin/activate

pip install openpyxl matplotlib seaborn --quiet

python -u train_model_v2.py

echo "Complete!"