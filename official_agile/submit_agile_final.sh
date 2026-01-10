#!/bin/bash
#SBATCH --job-name=agile_complete
#SBATCH --account=project_2010751
#SBATCH --partition=medium
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=logs/agile_%j.out
#SBATCH --error=logs/agile_%j.err

mkdir -p logs

echo "=========================================="
echo "AGILE Platform - Nature Comms 2024"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load Python module
module purge
module load python-data/3.10-24.04

# Install all required packages
echo "Installing packages..."
pip install --user --upgrade scikit-learn==1.3.2
pip install --user pyyaml tensorboard rdkit mordred skunk

# Install PyTorch 2.0.1
pip install --user torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install --user torch-geometric
pip install --user torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Navigate to project
cd /scratch/project_2010751/Drug_Delivery_Lai_project/official_agile

# Extract data if needed
if [ ! -d "data" ]; then
    echo "Extracting data..."
    unzip data.zip
fi

# Run fine-tuning
echo "Starting fine-tuning..."
python3 finetune.py config_finetune.yaml

echo "=========================================="
echo "AGILE Fine-tuning Complete!"
echo "=========================================="
