"""Inference for frozen PLGA model pair; no training or source data required."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
NUMERIC_FEATURES = [
    'Drug_MW', 'Drug_TPSA', 'Number_of_Heteroatoms', 'Drug_LogP',
    'Polymer_MW', 'LA_GA', 'Initial_Drug_to_Polymer_Ratio', 'Particle_Size',
    'Drug_Loading_Capacity', 'Drug_Encapsulation_Efficiency', 'Surfactant_Percent',
]
VARIANTS = ('combined_v1', 'descriptor_only')
CATEGORICAL = ('Formulation_Method', 'Drug_Name')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_live_models(root=ROOT):
    root = Path(root)
    conf_file = root / 'live_config.json'
    if not conf_file.is_file():
        raise FileNotFoundError('Missing live_config.json. Run prepare_bundle.py on Roihu first.')
    conf = json.loads(conf_file.read_text(encoding='utf-8'))
    if conf.get('schema_version') != 1:
        raise ValueError('Unsupported model bundle schema version')
    loaded = {}
    for variant in VARIANTS:
        vdir = root / 'models' / variant
        md_path, model_path = vdir / 'metadata.json', vdir / 'model.cbm'
        for path in (md_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(f'Model artifact not found: {path}')
            key = path.relative_to(root).as_posix()
            if sha256(path) != conf['artifact_sha256'][key]:
                raise ValueError('Model integrity check failed: ' + key)
        md = json.loads(md_path.read_text(encoding='utf-8'))
        expected = NUMERIC_FEATURES + ['Time']
        if variant == 'combined_v1':
            expected += list(CATEGORICAL)
        if md['feature_order'] != expected:
            raise ValueError(f'Unexpected feature order for {variant}')
        if md['categorical_features'] != (list(CATEGORICAL) if variant == 'combined_v1' else []):
            raise ValueError(f'Unexpected categorical features for {variant}')
        if md.get('training_formulations') != 301:
            raise ValueError('Unexpected training-set size')
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        loaded[variant] = (model, md)
    a, b = (loaded[k][1] for k in VARIANTS)
    if float(a['time_min']) != float(b['time_min']) or float(a['time_max']) != float(b['time_max']):
        raise ValueError('Models have different prediction time support')
    return loaded, conf


def canonical_inputs(numeric: Mapping, drug: str, method: str):
    values = {}
    for key in NUMERIC_FEATURES:
        try:
            values[key] = float(numeric[key])
        except (KeyError, ValueError, TypeError):
            raise ValueError('Invalid numeric input: ' + key)
        if not np.isfinite(values[key]):
            raise ValueError('Non-finite input: ' + key)

    for key in ('Drug_MW', 'Polymer_MW', 'Particle_Size', 'Initial_Drug_to_Polymer_Ratio'):
        if values[key] <= 0:
            raise ValueError(key + ' must be positive')
    for key in ('Drug_TPSA', 'Number_of_Heteroatoms', 'LA_GA', 'Surfactant_Percent'):
        if values[key] < 0:
            raise ValueError(key + ' cannot be negative')
    if values['Number_of_Heteroatoms'] != round(values['Number_of_Heteroatoms']):
        raise ValueError('Number_of_Heteroatoms must be a whole number')
    if not 0 <= values['Drug_Loading_Capacity'] <= 1:
        raise ValueError('Drug_Loading_Capacity must be a fraction between 0 and 1')
    if not 0 <= values['Drug_Encapsulation_Efficiency'] <= 100:
        raise ValueError('Encapsulation efficiency must be within 0–100%')

    drug = str(drug).strip().lower()
    method = str(method).strip()
    if not drug or not method or len(drug) > 160 or len(method) > 160:
        raise ValueError('Drug name and formulation method are required (maximum 160 characters each)')
    return {**values, 'Drug_Name': drug, 'Formulation_Method': method}


def predict_curve(loaded, form_values, variant='combined_v1', n_grid=256):
    if variant not in loaded:
        raise ValueError('Unknown model variant')
    model, meta = loaded[variant]
    n_grid = int(n_grid)
    if n_grid < 2 or n_grid > 1024:
        raise ValueError('Prediction grid size out of supported range')
    times = np.linspace(float(meta['time_min']), float(meta['time_max']), n_grid)
    sample = {}
    for feature in meta['feature_order']:
        if feature == 'Time':
            sample[feature] = times
        elif feature in CATEGORICAL:
            sample[feature] = [str(form_values[feature])] * n_grid
        else:
            sample[feature] = np.full(n_grid, float(form_values[feature]))
    frame = pd.DataFrame(sample)[meta['feature_order']]
    raw = np.asarray(model.predict(frame), dtype=float).reshape(-1)
    if raw.size != n_grid or not np.isfinite(raw).all():
        raise RuntimeError('Model returned invalid predictions')
    constrained = IsotonicRegression(increasing=True, out_of_bounds='clip').fit_transform(times, raw)
    constrained = np.clip(constrained, 0.0, 1.0)
    if np.any(np.diff(constrained) < -1e-10):
        raise RuntimeError('Monotonic projection failed')
    return pd.DataFrame({'Time_days': times, 'Release_fraction': constrained, 'Release_percent': 100.0 * constrained})


def out_of_range(form_values, conf):
    result = []
    for key in NUMERIC_FEATURES:
        value = form_values[key]
        stats = conf['numeric_stats'][key]
        if value < stats['min'] or value > stats['max']:
            result.append(key)
    return result


def familiar_categories(form_values, conf):
    return {
        'drug_seen_in_training': form_values['Drug_Name'] in conf['drug_names'],
        'method_seen_in_training': form_values['Formulation_Method'] in conf['methods'],
    }


def sample_at_days(curve, days):
    x, y = curve['Time_days'].to_numpy(), curve['Release_percent'].to_numpy()
    return {d: float(np.interp(d, x, y)) if x[0] <= d <= x[-1] else None for d in days}
