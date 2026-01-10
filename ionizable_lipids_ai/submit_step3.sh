#!/bin/bash
#SBATCH --job-name=lipid_step3
#SBATCH --account=project_2010751
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/step3_%j.out

module load python-data/3.10-24.04
source venv/bin/activate
cd ionizable_lipids_ai
pip install --quiet tqdm
python src/3_feature_engineering.py
