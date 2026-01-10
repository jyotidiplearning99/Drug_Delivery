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

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "AGILE Fine-tuning - Nature Comms 2024"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Initialize conda from scratch (proper way)
source /scratch/project_2010751/Drug_Delivery_Lai_project/agile_platform/miniconda3/etc/profile.d/conda.sh

# Activate agile environment
conda activate agile

# Verify activation
echo "Python location: $(which python3)"
python3 --version

# Navigate to project
cd /scratch/project_2010751/Drug_Delivery_Lai_project/official_agile

# Extract data if needed
if [ ! -d "data" ]; then
    echo "Extracting experimental data..."
    unzip data.zip
fi

# Run fine-tuning
echo "Starting fine-tuning on HeLa cells..."
python3 finetune.py config_finetune.yaml

echo "=========================================="
echo "Fine-tuning complete!"
echo "=========================================="
