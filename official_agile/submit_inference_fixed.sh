#!/bin/bash
#SBATCH --job-name=agile_inference
#SBATCH --account=project_2010751
#SBATCH --partition=small
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

mkdir -p logs

echo "=========================================="
echo "AGILE Inference & Visualization"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load Python module
module purge
module load python-data/3.10-24.04

# Install skunk package (required for visualization)
echo "Installing skunk package..."
pip install --user skunk

# Navigate to project
cd /scratch/project_2010751/Drug_Delivery_Lai_project/official_agile

# Find latest fine-tuned model and extract just the directory name
LATEST_MODEL_FULL=$(ls -td finetune/*/ | head -1)
LATEST_MODEL=$(basename "$LATEST_MODEL_FULL")

if [ -z "$LATEST_MODEL" ]; then
    echo "✗ No fine-tuned model found!"
    echo "Please run fine-tuning first."
    exit 1
fi

echo "Running inference on model: $LATEST_MODEL"
echo "Full path: finetune/$LATEST_MODEL"

# Pass only the directory name to infer_vis.py
python3 infer_vis.py "$LATEST_MODEL"

echo "=========================================="
echo "✅ Inference complete!"
echo "=========================================="
