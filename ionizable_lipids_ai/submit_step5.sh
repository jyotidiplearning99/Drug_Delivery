#!/bin/bash
#SBATCH --job-name=lipid_step5
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=results/step5_%j.out

module load python-data/3.10-24.04
source venv/bin/activate
cd ionizable_lipids_ai
pip install --quiet shap
python src/5_shap_analysis.py