#!/bin/bash
#SBATCH --job-name=lipid_step7
#SBATCH --account=project_2010751
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/step7_%j.out

module load python-data/3.10-24.04
source venv/bin/activate
cd ionizable_lipids_ai
python src/7_visualization.py