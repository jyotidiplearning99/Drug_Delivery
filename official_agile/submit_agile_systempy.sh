#!/bin/bash
#SBATCH --job-name=agile_finetune
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

mkdir -p logs

echo "=========================================="
echo "AGILE Fine-tuning - Nature Comms 2024"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load Python module from Mahti
module purge
module load python-data/3.10-24.04

# Verify Python
python3 --version
which python3

# Navigate to project
cd /scratch/project_2010751/Drug_Delivery_Lai_project/official_agile

# Install required packages to user location
pip install --user pyyaml scikit-learn torch==2.0.1 torch-geometric rdkit mordred

# Extract data if needed
if [ ! -d "data" ]; then
    echo "Extracting data..."
    unzip data.zip
fi

# Run fine-tuning
echo "Starting fine-tuning..."
python3 finetune.py config_finetune.yaml

echo "=========================================="
echo "Complete!"
echo "=========================================="
