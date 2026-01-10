"""
Configuration for AI-driven ionizable lipid design
Based on Nature Communications 2024 paper
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ECFP parameters (from paper)
ECFP_RADIUS = 9
ECFP_BITS = 1024

# Model hyperparameters (from paper Supplementary Table)
MODEL1_PARAMS = {  # Model 1: ≥1× MC3
    'colsample_bynode': 0.8,
    'colsample_bytree': 0.5,
    'learning_rate': 0.1,
    'max_depth': 3,
    'num_leaves': 4,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'subsample': 0.7,
    'subsample_freq': 3,
    'random_state': 42,
    'verbose': -1
}

MODEL2_PARAMS = {  # Model 2: ≥2× MC3
    'colsample_bytree': 0.5,
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 100,
    'num_leaves': 11,
    'subsample': 0.5,
    'random_state': 42,
    'verbose': -1
}

PKA_MODEL_PARAMS = {  # pKa regression
    'colsample_bytree': 1,
    'learning_rate': 0.01,
    'max_depth': 40,
    'n_estimators': 700,
    'num_leaves': 45,
    'subsample': 0.8,
    'random_state': 42,
    'verbose': -1
}

# Screening criteria
PKA_RANGE = (6.0, 7.0)
MC3_THRESHOLD_MODEL1 = 1.0
MC3_THRESHOLD_MODEL2 = 2.0

# Data processing
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("✓ Config loaded successfully")
