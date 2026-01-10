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
echo "AGILE Fine-tuning - Singularity Container"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load singularity
module load singularity

# Pull PyTorch container
CONTAINER="/scratch/project_2010751/pytorch_container.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "Downloading PyTorch container..."
    singularity pull $CONTAINER docker://pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
fi

# Navigate to project
cd /scratch/project_2010751/Drug_Delivery_Lai_project/official_agile

# Extract data if needed
if [ ! -d "data" ]; then
    echo "Extracting data..."
    unzip data.zip
fi

# Run in container
echo "Running fine-tuning in container..."
singularity exec $CONTAINER bash << 'INNEREOF'
pip install --user pyyaml scikit-learn torch-geometric rdkit mordred
cd /scratch/project_2010751/Drug_Delivery_Lai_project/official_agile
python3 finetune.py config_finetune.yaml
INNEREOF

echo "=========================================="
echo "Complete!"
echo "=========================================="
