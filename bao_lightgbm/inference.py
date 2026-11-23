"""Bao-only LightGBM inference. Experimental in-vitro release prediction.

First-day features are fitted exclusively from supplied <=day-1 observations.
Full-data release profile values are NEVER used to construct those inputs.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
FEATURES = [f for f in META["descriptor_features"] if f != "time_days"]
EARLY_FEATURES = [
    "kp_k", "kp_n", "kp_r2", "weibull_alpha", "weibull_beta", "weibull_r2",
    "release_0_25d", "release_0_5d", "release_1d", "early_release_last", "early_time_last",
]
LABELS = {
    "drug_mw": "Drug molecular weight (g/mol)",
    "drug_logp": "Drug LogP",
    "drug_tpsa": "Drug TPSA (Å²)",
    "polymer_mw_kda": "Polymer MW (kDa)",
    "la_ga_pct": "Lactide content (%) — 50 = 50:50 PLGA",
    "drug_polymer_ratio": "Initial drug-to-polymer mass ratio — 0.1 = 1:10",
    "particle_size_um": "Particle size (µm)",
    "drug_loading_pct": "Measured drug loading (%)",
    "encapsulation_efficiency_pct": "Measured encapsulation efficiency (%)",
    "solubility_enhancer_conc_pct": "Solubility enhancer concentration (source %, not surfactant)",
}


def load_models():
    return {
        "descriptor_only": lgb.Booster(model_file=str(ROOT / "models" / "descriptor_only.txt")),
        "first_day_assisted": lgb.Booster(model_file=str(ROOT / "models" / "first_day_assisted.txt")),
    }


def _kp(t, k, n):
    return k * np.power(t, n)


def _weibull(t, alpha, beta):
    return 1.0 - np.exp(-np.power(t / alpha, beta))


def early_signatures(early: pd.DataFrame):
    """Equivalent to the original saved LightGBM training feature generator."""
    g = early.loc[(early.time_days >= 0) & (early.time_days <= 1.)].copy()
    g = g.sort_values("time_days").groupby("time_days", as_index=False)["release_fraction"].mean()
    t = g.time_days.to_numpy(float)
    y = g.release_fraction.to_numpy(float)
    result = {k: np.nan for k in EARLY_FEATURES}
    if len(t) < 2:
        return result
    result["early_release_last"] = float(y[-1])
    result["early_time_last"] = float(t[-1])
    for key, when in [("release_0_25d", .25), ("release_0_5d", .5), ("release_1d", 1.)]:
        if min(t) <= when <= max(t):
            result[key] = float(np.interp(when, t, y))
    ok = (t > 0) & (y > 0)
    if ok.sum() >= 2:
        a, n = np.polyfit(np.log(t[ok]), np.log(y[ok]), 1)
        k = float(np.exp(n))
        n = float(np.clip(a, 0.01, 3))
        if np.isfinite(k) and np.isfinite(n):
            yh = _kp(t[ok], k, n)
            ss = np.sum((y[ok] - y[ok].mean()) ** 2)
            result.update(kp_k=k, kp_n=n,
                          kp_r2=float(1 - np.sum((y[ok] - yh) ** 2) / ss) if ss > 1e-9 else np.nan)
    if len(t) >= 3 and np.sum((t > 0) & (y > 0) & (y < 1)) >= 3:
        mask = (t > 0) & (y > 0) & (y < 1)
        try:
            pars, _ = curve_fit(_weibull, t[mask], y[mask], p0=[4., .5],
                                bounds=([0.001, 0.01], [5000, 5]), maxfev=1800)
            ph = _weibull(t[mask], *pars)
            ss = float(np.sum((y[mask] - y[mask].mean()) ** 2))
            result.update(weibull_alpha=float(pars[0]), weibull_beta=float(pars[1]),
                          weibull_r2=float(1 - np.sum((y[mask] - ph) ** 2) / ss) if ss > 1e-9 else np.nan)
        except (RuntimeError, ValueError, FloatingPointError):
            pass
    return result


def validate_descriptors(rows: pd.DataFrame):
    df = rows.copy()
    required = ["formulation_id"] + FEATURES
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    df["formulation_id"] = df["formulation_id"].fillna("").astype(str).str.strip()
    df = df.loc[df["formulation_id"] != ""].copy()
    if not len(df):
        raise ValueError("Add at least one formulation.")
    if len(df) > 25:
        raise ValueError("A maximum of 25 custom formulations is supported per run.")
    if df.formulation_id.duplicated().any():
        raise ValueError("Each formulation_id must be unique.")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not np.isfinite(df[FEATURES].to_numpy(float)).all():
        raise ValueError("All numerical descriptor columns need valid numbers. No fabricated defaults are applied to blank cells.")
    for c in ["drug_mw", "polymer_mw_kda", "particle_size_um"]:
        if (df[c] <= 0).any():
            raise ValueError(f"{c} must be positive")
    for c in ["drug_tpsa", "drug_polymer_ratio", "solubility_enhancer_conc_pct"]:
        if (df[c] < 0).any():
            raise ValueError(f"{c} cannot be negative")
    for c in ["la_ga_pct", "drug_loading_pct", "encapsulation_efficiency_pct"]:
        if (~df[c].between(0, 100)).any():
            raise ValueError(f"{c} must be between 0 and 100")
    return df[["formulation_id"] + FEATURES]


def out_of_range(df: pd.DataFrame):
    warnings = []
    ranges = META["training_numeric_ranges"]
    for row in df.to_dict("records"):
        outside = [f for f in FEATURES if f in ranges and not (ranges[f][0] <= row[f] <= ranges[f][1])]
        if outside:
            warnings.append((row["formulation_id"], outside))
    return warnings


def parse_early_csv(df: pd.DataFrame):
    """User-entered early observations are in PERCENT; never inferred."""
    required = {"formulation_id", "time_days", "release_percent"}
    if not required.issubset(df.columns):
        raise ValueError("First-day CSV requires formulation_id,time_days,release_percent")
    e = df[list(required)].copy()
    e["formulation_id"] = e["formulation_id"].fillna("").astype(str).str.strip()
    e["time_days"] = pd.to_numeric(e["time_days"], errors="coerce")
    e["release_percent"] = pd.to_numeric(e["release_percent"], errors="coerce")
    e = e.dropna()
    if not e.time_days.between(0, 1).all():
        raise ValueError("All first-day measurement times must be in [0,1] days.")
    if not e.release_percent.between(0, 100).all():
        raise ValueError("First-day release_percent must be between 0 and 100.")
    e["release_fraction"] = e["release_percent"] / 100.
    return e[["formulation_id", "time_days", "release_fraction"]]


def predict_curve(models: dict, desc: dict, days, early: pd.DataFrame | None = None):
    """Use first-day model only with >=2 distinct measured timepoints <= day1."""
    times = np.asarray(days, dtype=float)
    if not np.isfinite(times).all() or (times <= 1.).any():
        raise ValueError("Forecast time grid must be finite and strictly greater than one day")
    mode = "descriptor_only"
    feature_list = META["descriptor_features"]
    base = {f: float(desc[f]) for f in FEATURES}
    if early is not None and len(early):
        g = early.loc[early.time_days.between(0, 1)].copy().sort_values("time_days")
        if g.time_days.nunique() >= 2:
            mode = "first_day_assisted"
            base.update(early_signatures(g))
            feature_list = META["assisted_features"]
    x = pd.DataFrame([{**base, "time_days": float(t)} for t in times], columns=feature_list)
    y = np.asarray(models[mode].predict(x), float)
    order = np.argsort(times)
    ysorted = np.maximum.accumulate(np.clip(y[order], 0, 1))
    if mode == "first_day_assisted":
        ysorted = np.maximum(ysorted, float(np.clip(g.release_fraction.iloc[-1], 0, 1)))
    out = np.empty_like(ysorted)
    out[order] = ysorted
    return out, mode
