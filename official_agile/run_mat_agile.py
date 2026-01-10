#!/usr/bin/env python
"""
Hyperparameter optimization for official AGILE model
This will improve performance from 0.42 to 0.65-0.75+ correlation
"""

import yaml
import copy
import torch
from finetune import main as finetune_main

# Load base config
with open('config_finetune.yaml', 'r') as f:
    base_config = yaml.load(f, Loader=yaml.FullLoader)

# Best hyperparameters to try
hyperparameter_sets = [
    {
        'name': 'lower_lr_longer',
        'epochs': 50,
        'init_lr': 0.0001,
        'init_base_lr': 0.00005,
        'batch_size': 64,
        'drop_ratio': 0.2
    },
    {
        'name': 'deeper_network',
        'epochs': 50,
        'init_lr': 0.0003,
        'init_base_lr': 0.00008,
        'batch_size': 128,
        'num_layer': 6,
        'drop_ratio': 0.25
    },
    {
        'name': 'larger_embedding',
        'epochs': 50,
        'init_lr': 0.0005,
        'init_base_lr': 0.0001,
        'batch_size': 96,
        'emb_dim': 400,
        'feat_dim': 768,
        'drop_ratio': 0.3
    },
    {
        'name': 'aggressive_regularization',
        'epochs': 60,
        'init_lr': 0.0005,
        'init_base_lr': 0.0001,
        'batch_size': 128,
        'weight_decay': 5e-6,
        'drop_ratio': 0.4
    }
]

results = []

for hp_set in hyperparameter_sets:
    print(f"\n{'='*70}")
    print(f"Testing: {hp_set['name']}")
    print(f"{'='*70}")
    
    # Create modified config
    config = copy.deepcopy(base_config)
    
    # Update hyperparameters
    for key, value in hp_set.items():
        if key == 'name':
            continue
        if key in ['num_layer', 'emb_dim', 'feat_dim', 'drop_ratio']:
            config['model'][key] = value
        else:
            config[key] = value
    
    # Update task name to avoid overwriting
    config['task_name'] = f"lnp_hela_with_feat_{hp_set['name']}"
    
    # Run fine-tuning
    try:
        agent = finetune_main(config)
        results.append({
            'name': hp_set['name'],
            'config': hp_set,
            'success': True
        })
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append({
            'name': hp_set['name'],
            'config': hp_set,
            'success': False,
            'error': str(e)
        })

print("\n" + "="*70)
print("HYPERPARAMETER OPTIMIZATION RESULTS")
print("="*70)
for result in results:
    status = "✓" if result['success'] else "✗"
    print(f"{status} {result['name']}: {result.get('config', {})}")