#!/usr/bin/env python3
"""Roihu-only: copy validated frozen models into a standalone Streamlit test app.

The exported live_preview folder contains NO source workbooks, raw formulation
measurements, patient data, experiment results, or original upstream repository.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC = PROJECT_ROOT / 'results/model_improvement/frozen_v1/candidate_bundle'
DEST = HERE / 'models'

BASE_FEATURES = [
    'Drug_MW', 'Drug_TPSA', 'Number_of_Heteroatoms', 'Drug_LogP',
    'Polymer_MW', 'LA_GA', 'Initial_Drug_to_Polymer_Ratio', 'Particle_Size',
    'Drug_Loading_Capacity', 'Drug_Encapsulation_Efficiency', 'Surfactant_Percent',
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing local preview bundle')
    args = parser.parse_args()
    frozen_manifest = SRC / 'SHA256SUMS.txt'
    if not frozen_manifest.exists():
        raise SystemExit('Frozen model bundle missing. Check candidate_bundle/ first.')
    for line in frozen_manifest.read_text(encoding='utf-8').splitlines():
        digest, relative = line.split('  ', 1)
        path = (SRC / relative).resolve()
        if not path.is_file() or not path.is_relative_to(SRC.resolve()) or sha256(path) != digest:
            raise SystemExit('Frozen model integrity failed for: ' + relative)
    if (DEST.exists() or (HERE / 'live_config.json').exists()) and not args.overwrite:
        raise SystemExit('Preview bundle already exists; refusing overwrite. Use --overwrite only intentionally.')

    freeze_script = PROJECT_ROOT / 'experiments/model_improvement/freeze_v1_candidate.py'
    if not freeze_script.exists():
        raise SystemExit('Cannot find original freeze_v1_candidate.py; required for exact same data mapping')
    spec = importlib.util.spec_from_file_location('frozen_training_loader', str(freeze_script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data, _, _ = mod.load_training_data()
    if len(data) != 3903 or data['Formulation_Index'].nunique() != 301:
        raise SystemExit('Unexpected data shape while building demo defaults')

    numeric_stats = {}
    for feature in BASE_FEATURES:
        v = data[feature].astype(float)
        if not np.isfinite(v).all():
            raise SystemExit('Non-finite source descriptor: ' + feature)
        numeric_stats[feature] = {
            'min': float(v.min()), 'max': float(v.max()), 'median': float(v.median())
        }
    drug_names = sorted(set(data['Drug_Name'].astype(str)))
    methods = sorted(set(data['Formulation_Method'].astype(str)))
    # Representative row, explicitly labeled as *training* data, not a new test case.
    demo = data.sort_values(['Formulation_Index', 'Time']).iloc[0]
    demo_inputs = {feature: float(demo[feature]) for feature in BASE_FEATURES}
    demo_inputs.update({
        'Drug_Name': str(demo['Drug_Name']),
        'Formulation_Method': str(demo['Formulation_Method']),
    })

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    checksums = {}
    for variant in ('descriptor_only', 'combined_v1'):
        (DEST / variant).mkdir()
        for filename in ('model.cbm', 'metadata.json'):
            source = SRC / variant / filename
            target = DEST / variant / filename
            shutil.copy2(source, target)
            checksums[target.relative_to(HERE).as_posix()] = sha256(target)

    cv_path = PROJECT_ROOT / 'results/model_improvement/metadata_v1/summary.csv'
    cv = {'baseline_pooled_mae': 0.20364502540131488,
          'combined_pooled_mae': 0.19326095382001993,
          'source': 'Repeatedly inspected publication-grouped development CV on 301 source-linked formulations'}
    if cv_path.exists():
        import pandas as pd
        summary = pd.read_csv(cv_path).set_index('variant')
        cv['baseline_pooled_mae'] = float(summary.loc['baseline_reproduction','pooled_mae'])
        cv['combined_pooled_mae'] = float(summary.loc['plus_method_and_drug','pooled_mae'])

    config = {
        'schema_version': 1,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'intended_use': 'exploratory research testing only',
        'training_formulations': 301,
        'numeric_stats': numeric_stats,
        'drug_names': drug_names,
        'methods': methods,
        'demo_profile_training_example': demo_inputs,
        'development_cv': cv,
        'artifact_sha256': checksums,
        'training_source': 'Bao-matched PLGA dataset; original training data not included in public package',
    }
    config_file = HERE / 'live_config.json'
    config_file.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Prepared:', HERE)
    print('Frozen models: 2; training formulations: 301; numeric inputs:', len(BASE_FEATURES))
    print('Drug names:', len(drug_names), '; preparation methods:', len(methods))
    print('No raw source workbooks or release measurements copied.')
    print('IMPORTANT: Review rights to redistribute model weights before publishing a public repo.')


if __name__ == '__main__':
    main()
