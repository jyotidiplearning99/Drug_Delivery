from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from inference import ROOT, META, FEATURES, load_models, parse_early_csv, predict_curve, validate_descriptors, early_signatures

assert META['n_all_formulations']==321 and META['n_all_measurements']==4913
assert META['n_descriptor_formulations']==316 and META['n_assisted_formulations']==212
assert hashlib.sha256((ROOT/'data/published_profiles.csv').read_bytes()).hexdigest()==META['normalized_source_sha256']
assert hashlib.sha256((ROOT/'data/mp_dataset_processed.xlsx').read_bytes()).hexdigest()==META['original_workbook_sha256']
models=load_models()
df=pd.read_csv(ROOT/'data/published_profiles.csv',dtype={'formulation_id':str})
assert len(df)==4913 and df.formulation_id.nunique()==321
row=df.drop_duplicates('formulation_id').dropna(subset=FEATURES).iloc[0]
records=pd.DataFrame([{**{f:row[f] for f in FEATURES},'formulation_id':'A'}])
rec=validate_descriptors(records).iloc[0].to_dict()
times=np.array([1.01,2,7,14,30],float)
y,mode=predict_curve(models,rec,times)
assert mode=='descriptor_only' and np.isfinite(y).all() and (y>=0).all() and (y<=1).all() and (np.diff(y)>=-1e-10).all()
print('PASS: descriptor-only model; release % at days 7/14/30:',np.round(y[2:]*100,2).tolist())
# find a known profile with adequate first-day measurements
a=None
for fid,g in df.groupby('formulation_id'):
    e=g.loc[g.time_days.between(0,1),['time_days','release_fraction']]
    if e.time_days.nunique()>=2 and (g.time_days>1).any():
        a=(fid,g,e)
        break
assert a is not None
fid,g,e=a
rec=validate_descriptors(g.drop_duplicates('formulation_id')[['formulation_id']+FEATURES]).iloc[0].to_dict()
y2,mode2=predict_curve(models,rec,times,e)
assert mode2=='first_day_assisted' and np.isfinite(y2).all() and (y2>=0).all() and (y2<=1).all() and (np.diff(y2)>=-1e-10).all()
print('PASS: first-day-assisted model for published formulation',fid)
# fail-closed test: only one timepoint -> descriptor-only
_,mm=predict_curve(models,rec,times,e.iloc[:1])
assert mm=='descriptor_only'
try:
    validate_descriptors(records.assign(drug_mw=-1))
except ValueError:pass
else:raise AssertionError('Negative MW did not raise an error')
print('PASS: two distinct early times required; input validation, bounds and monotonicity OK')
