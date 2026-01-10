#!/bin/bash
#SBATCH --job-name=lipid_step1
#SBATCH --account=project_2010751
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=results/step1_%j.out

module load python-data/3.10-24.04
source venv/bin/activate
cd ionizable_lipids_ai
python src/1_data_collection.py