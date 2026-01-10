#!/bin/bash
#SBATCH --job-name=lipid_step4
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --output=results/step4_%j.out

module load python-data/3.10-24.04
source venv/bin/activate
cd ionizable_lipids_ai
python src/4_model_training.py