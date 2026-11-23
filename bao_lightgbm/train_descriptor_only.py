"""Train an independent Bao-only descriptor LightGBM, with grouped CV.

This is a research training script, not needed at Streamlit runtime.
Training uses source measurement time > 1 day; it never uses release-curve
kinetic features or future observed release as inputs.
"""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import GroupKFold
from inference import ROOT, FEATURES

SOURCE = ROOT/'data/published_profiles.csv'
MODEL = ROOT/'models/descriptor_only.txt'
SUMMARY = ROOT/'models/descriptor_validation.json'

def estimator():
    return LGBMRegressor(n_estimators=240, learning_rate=.04, num_leaves=15,
                         min_child_samples=8, colsample_bytree=.9, subsample=.9,
                         reg_lambda=1.0,random_state=42,verbosity=-1,n_jobs=2)

def main():
    df=pd.read_csv(SOURCE)
    expected=FEATURES+['formulation_id','time_days','release_fraction']
    assert not df[expected].isna().any().any()
    drops=[]
    for fid,g in df.groupby('formulation_id'):
        profile=g.groupby('time_days').release_fraction.mean().sort_index().to_numpy()
        if np.any(np.diff(profile)<-.08):drops.append(str(fid))
    good=df.loc[~df.formulation_id.astype(str).isin(drops)].copy()
    good=good.loc[good.time_days>1.].copy()
    groups=good.formulation_id.astype(str)
    features=FEATURES+['time_days']
    y=good.release_fraction.to_numpy(float)
    oof=np.full(len(good),np.nan)
    for tr,te in GroupKFold(n_splits=5).split(good[features],y,groups):
        model=estimator().fit(good.iloc[tr][features],y[tr])
        oof[te]=model.predict(good.iloc[te][features])
    clipped=np.clip(oof,0,1)
    # Per-formulation, at the actual observed timepoints, apply same monotonic rule as app.
    projected=np.full(len(good),np.nan)
    for fid,idx in good.groupby(groups,sort=False).indices.items():
        ix=np.asarray(idx,int)
        sorted_ix=ix[np.argsort(good.iloc[ix].time_days.to_numpy())]
        projected[sorted_ix]=np.maximum.accumulate(clipped[sorted_ix])
    final=estimator().fit(good[features],y)
    final.booster_.save_model(str(MODEL))
    metrics={
      'algorithm':'LightGBM descriptor-only, no first-day data',
      'source':'Bao Mendeley Data v2, 10.17632/zzvtdrcy76.2',
      'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      'n_source_profiles':int(df.formulation_id.nunique()),
      'n_training_profiles':int(groups.nunique()),
      'n_training_release_rows_later_than_day1':len(good),
      'excluded_large_cumulative_release_drop_profile_ids':drops,
      'excluded_profile_count':len(drops),
      'cv_folds':5,
      'cv_grouping':'formulation_id, no publication holdout',
      'cv_raw_clipped_mae':float(mean_absolute_error(y,clipped)),
      'cv_raw_clipped_rmse':float(np.sqrt(mean_squared_error(y,clipped))),
      'cv_monotone_projected_mae':float(mean_absolute_error(y,projected)),
      'cv_monotone_projected_rmse':float(np.sqrt(mean_squared_error(y,projected))),
      'features':features,
      'hyperparameters':'240 iterations, learning_rate 0.04, 15 leaves, min_child_samples 8, reg_lambda 1.0, seed 42',
      'limitations':'formulation CV only; development data; can share drug/source across folds; not independent external evaluation'
    }
    SUMMARY.write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
