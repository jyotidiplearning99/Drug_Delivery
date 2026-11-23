# PLGA Release Studio — Bao-only LightGBM, multi-formulation edition

A **separate** interactive Streamlit app for the Bao et al. processed PLGA dataset. Not CatBoost and not the combined 347-formulation dataset.

## What this package contains

- `app.py`: multi-formulation editor, up to 25 independent parameter sets; interactive release curves, 15 published profile overlays, CSV export
- `inference.py`: predictor; guards around first-day-assisted versus descriptor-only inference
- `models/first_day_assisted.txt`: exported LightGBM Booster **from our previously saved trained** `lightgbm_trained_real_data.joblib`
- `models/descriptor_only.txt`: a **newly trained descriptor-only LightGBM Booster** using 316 of 321 Bao formulations, excluding five with large downward jumps in recorded cumulative release
- `data/published_profiles.csv`: all 4,913 normalized real data rows from the Bao processed workbook, with CC BY 4.0 attribution
- `data/mp_dataset_processed.xlsx`: original Bao processed workbook (retained for audit; app reads normalized CSV)
- `models/metadata.json`: input fields, feature order, data source and historical CV metrics
- `smoke_test.py`: model integrity/curve checks
- `train_descriptor_only.py` and `models/descriptor_validation.json`: reproducible all-eligible Bao descriptor-model training and grouped-CV metrics (requires scikit-learn on the training machine)

## Two valid modes

1. **Descriptor-only:** 10 numerical properties + time, no early measurements. 316 Bao formulations / 4,078 later-than-day-1 points; new 5-fold formulation-grouped *monotone-projected* MAE **0.15531** (release fraction). Not publication-held-out or independent testing. Historical comparator on the smaller 212-formulation early-data cohort was 0.16244.
2. **First-day-assisted:** requires >=2 distinct **actual** measurement times at or before 1 day. Feature engineering uses *only* these observations. Uses the previously saved LightGBM and prior formulation-grouped MAE 0.11934.

These validation MAEs come from **different evaluation cohorts** (descriptor-only 316 formulations, early-assisted 212) and different postprocessing protocols; they are **not directly comparable**. Neither is publication-held-out or independent external testing.

The first-day model excludes 105 profiles lacking sufficient early data and 4 with large reported cumulative-release decreases. The descriptor-only model does **not** require first-day measurements; it uses 316 formulations, excluding five with large release decreases.

## Run locally or on Roihu

```
python -m pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

On CSC, use the app for local testing or a compute-node session; do **not** expose a shared login node publicly.

## Deploy as a second public app without changing current CatBoost app

Copy this `bao_lightgbm` directory into the *root of your own* `Drug_Delivery` repository and push a separate branch. In Streamlit Community Cloud deploy:

- Repo `jyotidiplearning99/Drug_Delivery`
- Branch `bao-lightgbm-live`
- Main file `bao_lightgbm/app.py`
- Python version 3.12 recommended
- A **different** App URL, e.g. `plga-bao-lightgbm.streamlit.app` if available

Do not push `git add .` on Roihu; stage only `bao_lightgbm/`.

## Units and assumptions

Bao processed dataset is in days. Polymer MW is **kDa**, drug loading and encapsulation are **percent** (not fractions), lactide-glycolide ratio converted to **lactide percent**, and **Solubility Enhancer Concentration** is distinct from surfactant percentage. These fields are shown explicitly in the UI. Drug MW g/mol, TPSA Å², particle size µm, initial drug/polymer ratio is source mass ratio.

A monotonic projection and [0,100%] bounds are applied to model outputs. The training data and postprocessing may contain imperfect experimental curves; don't infer model accuracy from training-profile overlays.

## Source attribution and caveats

Bao Z et al. *A dataset on formulation parameters and characteristics of drug-loaded PLGA microparticles*. Mendeley Data v2: https://data.mendeley.com/datasets/zzvtdrcy76/2 — CC BY 4.0.

Research prototype only; no external prospective validation, no clinical or therapeutic claim. No trained CatBoost weights are included.
