#!/bin/bash
#SBATCH --job-name=lipid_step6
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --output=results/step6_%j.out

module load python-data/3.10-24.04
source venv/bin/activate
cd ionizable_lipids_ai
python src/6_virtual_screening.py