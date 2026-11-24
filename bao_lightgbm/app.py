"""PLGA Release Studio — Bao LightGBM, paper-aligned multi-formulation UI.

Uses only the existing trained Bao models in models/. It neither retrains
nor substitutes inputs from the 347-formulation Alabsi et al. (2026) study.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import streamlit as st

from inference import (
    ROOT, META, FEATURES, LABELS, load_models, parse_early_csv,
    predict_curve, validate_descriptors, out_of_range,
)

st.set_page_config(
    page_title="PLGA Release Studio | Bao LightGBM",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.block-container {max-width: 1320px; padding-top: 1.5rem; padding-bottom: 3rem;}
h1 {letter-spacing: -.035em;}
div[data-testid="stMetric"] {border: 1px solid #d5e0e7; border-radius: 10px; padding: 12px 14px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained LightGBM models…")
def cached_models():
    return load_models()

@st.cache_data(show_spinner="Loading experimental release data…")
def source_data():
    return pd.read_csv(ROOT / "data" / "published_profiles.csv", dtype={"formulation_id": str})

try:
    models = cached_models()
    data = source_data()
except Exception as exc:
    st.error("Could not load the Bao LightGBM models or source data. Verify the models/ and data/ folders in the deployed branch.")
    st.exception(exc)
    st.stop()

# These groups mirror the experimental workflow in the 2026 PLGA paper.
# Only the exact 10 descriptors accepted by the deployed Bao LightGBM models
# are rendered as inputs. The 2026 paper contains other engineered features.
GROUPS = [
    ("Drug descriptors", ["drug_mw", "drug_logp", "drug_tpsa"]),
    ("Polymer descriptors", ["polymer_mw_kda", "la_ga_pct"]),
    ("Formulation descriptors", [
        "drug_polymer_ratio", "particle_size_um", "drug_loading_pct",
        "encapsulation_efficiency_pct", "solubility_enhancer_conc_pct",
    ]),
]
FIELD_HELP = {
    "drug_mw": "Drug molecular weight (g/mol). Use the correct molecular or salt form.",
    "drug_logp": "Calculated LogP; negative values are possible.",
    "drug_tpsa": "Topological polar surface area (Å²).",
    "polymer_mw_kda": "Polymer molecular weight on the Bao dataset scale: kDa.",
    "la_ga_pct": "Lactide percentage (50 means 50:50 PLGA). Do not enter '50:50'.",
    "drug_polymer_ratio": "Initial drug-to-polymer mass ratio (0.1 means 1:10).",
    "particle_size_um": "Measured particle size in micrometres (µm).",
    "drug_loading_pct": "Measured drug loading capacity, percentage 0–100.",
    "encapsulation_efficiency_pct": "Measured encapsulation efficiency, percentage 0–100.",
    "solubility_enhancer_conc_pct": "Bao solubility-enhancer concentration, source percentage; not interchangeable with every surfactant measurement.",
}

st.title("🧪 PLGA Release Studio")
st.caption("Bao dataset · LightGBM edition · Formulation-based prediction and first-day-assisted forecasting")
st.warning(
    "**Experimental research prototype.** Predictions are intended for research exploration, "
    "not clinical decisions or manufacturing release. Models have no independent external validation."
)

with st.sidebar:
    st.subheader("Prediction settings")
    horizon = st.slider("Prediction horizon (days)", 7, 180, 60, 1, key="bao_horizon")
    target_day = st.slider("Inspect release at day", 2, horizon, min(14, horizon), 1, key="bao_target_day")
    n_grid = st.slider("Curve resolution", 60, 300, 160, 20, key="bao_curve_resolution")
    show_observed_window = st.checkbox("Mark model training-time limit", value=True, key="bao_training_window")
    st.divider()
    st.subheader("Model information")
    st.write("**Dataset:** Bao et al. (2025), 321 PLGA formulations")
    st.write("**Model:** LightGBM, cumulative release after day 1")
    st.metric("Descriptor-only CV MAE", f"{META['descriptor_oof_mae']:.3f}")
    st.metric("First-day-assisted CV MAE", f"{META['assisted_oof_mae']:.3f}")
    st.caption("Different validation cohorts; MAEs are not directly comparable.")
    st.markdown("[Bao original dataset](https://data.mendeley.com/datasets/zzvtdrcy76/2)")


def make_time_grid():
    return np.linspace(1.01, float(horizon), int(n_grid))


def prepare_early_map(early_edit: pd.DataFrame, valid_ids: list[str]):
    """Never invent or auto-fill early measurements."""
    if early_edit is None or early_edit.empty:
        return {}
    clean = parse_early_csv(early_edit)
    groups = {str(name): g for name, g in clean.groupby("formulation_id")}
    unused = sorted(set(groups) - set(valid_ids))
    if unused:
        st.warning("Early measurements ignored for unknown formulation names: " + ", ".join(unused[:10]))
    return {name: g for name, g in groups.items() if name in valid_ids}


def render_predictions(
    descriptors: pd.DataFrame,
    *,
    early_by_id: dict | None = None,
    observations: dict | None = None,
    show_observations: bool = False,
    widget_prefix: str,
):
    """Render identical outputs for custom and published tabs, with unique keys."""
    desc = validate_descriptors(descriptors)
    early_by_id = early_by_id or {}
    numeric_warnings = out_of_range(desc)
    if numeric_warnings:
        examples = "; ".join(f"{name}: {', '.join(fields)}" for name, fields in numeric_warnings[:5])
        st.warning("Outside observed training descriptor range (extrapolation): " + examples)

    time_grid = make_time_grid()
    curves = []
    summaries = []
    fig = go.Figure()
    colors = qualitative.Plotly + qualitative.Safe + qualitative.D3
    used_assisted = 0
    for idx, record in enumerate(desc.to_dict("records")):
        name = str(record["formulation_id"])
        y, mode = predict_curve(models, record, time_grid, early_by_id.get(name))
        if mode == "first_day_assisted":
            used_assisted += 1
        frame = pd.DataFrame({
            "formulation_id": name,
            "time_days": time_grid,
            "release_percent": y * 100.0,
            "prediction_mode": mode,
        })
        curves.append(frame)
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=time_grid, y=y * 100.0, mode="lines", name=name,
            line={"width": 3, "color": color},
            hovertemplate="Day %{x:.1f}<br>Predicted release %{y:.1f}%<extra>" + name + "</extra>",
        ))
        if show_observations and observations and name in observations:
            g = observations[name]
            fig.add_trace(go.Scatter(
                x=g["time_days"], y=g["release_fraction"] * 100.0,
                mode="markers", name=f"{name} · experimental",
                marker={"size": 7, "color": color, "symbol": "circle-open"},
                hovertemplate="Day %{x:.1f}<br>Measured release %{y:.1f}%<extra></extra>",
            ))
        summaries.append({
            "Formulation": name,
            "Mode": "First-day assisted" if mode == "first_day_assisted" else "Descriptors only",
            f"Day {target_day} release (%)": round(float(np.interp(target_day, time_grid, y) * 100.0), 2),
            f"Day {horizon} release (%)": round(float(y[-1] * 100.0), 2),
        })

    if show_observed_window:
        tmax = float(META["training_numeric_ranges"]["time_days"][1])
        if horizon > tmax:
            fig.add_vrect(
                x0=tmax, x1=horizon, fillcolor="orange", opacity=.07,
                line_width=0, annotation_text="Beyond training time",
            )
    fig.add_vline(x=1, line_dash="dot", line_color="#9ca3af")
    fig.add_vline(x=target_day, line_dash="dash", line_color="#64748b", opacity=.6)
    fig.update_layout(
        title="Predicted cumulative drug-release profiles",
        template="plotly_white", height=510,
        xaxis={"title": "Time (days)", "range": [0, horizon]},
        yaxis={"title": "Cumulative drug release (%)", "range": [0, 102]},
        legend={"orientation": "h", "y": -0.22},
        margin={"l": 10, "r": 10, "b": 15, "t": 55},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{widget_prefix}_release_plot")
    if early_by_id:
        st.caption(f"{used_assisted} of {len(desc)} profile(s) used actual first-day measurements; "
                   "profiles with fewer than two distinct early time points use descriptor-only LightGBM.")
    st.dataframe(pd.DataFrame(summaries), hide_index=True, use_container_width=True)

    all_predictions = pd.concat(curves, ignore_index=True)
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download predicted release curves (CSV)",
            all_predictions.to_csv(index=False).encode("utf-8"),
            file_name="bao_lightgbm_predicted_profiles.csv", mime="text/csv",
            key=f"{widget_prefix}_predictions_csv", use_container_width=True,
        )
    with right:
        st.download_button(
            "Download formulation inputs (CSV)",
            desc.to_csv(index=False).encode("utf-8"),
            file_name="bao_lightgbm_formulation_parameters.csv", mime="text/csv",
            key=f"{widget_prefix}_parameters_csv", use_container_width=True,
        )
    if horizon > float(META["training_numeric_ranges"]["time_days"][1]):
        st.warning("The selected horizon extends beyond the descriptor-only training time range; the model is extrapolating.")
    st.caption("Predicted values are bounded to 0–100% and forced to be nondecreasing. "
               "This does not provide a confidence interval or validate biological accuracy.")


published = data.drop_duplicates("formulation_id").dropna(subset=FEATURES).reset_index(drop=True)
if published.empty:
    st.error("No valid published formulation rows were found.")
    st.stop()

# Keep three concise tabs. The primary tab intentionally mirrors the first PLGA Studio.
tab_predict, tab_published, tab_about = st.tabs([
    "🧪 Predict formulations", "📈 Compare experimental profiles", "ℹ️ About",
])

with tab_predict:
    st.subheader("Describe your PLGA formulations")
    st.caption("Change the individual numerical fields, then click **Predict release profiles**. "
               "Each formulation is drawn as a separate curve on the same graph.")
    entry_mode = st.radio(
        "Input method", ["Manual fields", "Spreadsheet / CSV"],
        horizontal=True, key="bao_entry_mode_paper_ui",
    )
    valid_payload = st.session_state.get("bao_custom_prediction_payload")

    if entry_mode == "Manual fields":
        n_forms = int(st.number_input(
            "Number of formulations", min_value=1, max_value=25, value=2, step=1,
            help="Up to 25 formulations can be predicted and compared together.",
            key="bao_number_of_formulations",
        ))
        st.info("Only properties used by this trained Bao LightGBM model are shown. "
                "The 2026 PLGA paper included additional descriptors and engineered features.")
        with st.form("bao_manual_form", clear_on_submit=False, border=True):
            records = []
            for i in range(n_forms):
                example = published.iloc[i % len(published)]
                with st.expander(f"Formulation {i + 1}", expanded=(i == 0)):
                    name = st.text_input(
                        "Formulation name (curve legend)",
                        value=f"Formulation {i + 1}", key=f"bao_v4_name_{i}",
                    )
                    record = {"formulation_id": name}
                    for group_name, field_names in GROUPS:
                        st.markdown(f"**{group_name}**")
                        columns = st.columns(3)
                        for j, feature in enumerate(field_names):
                            with columns[j % 3]:
                                step = 1.0 if feature in {"drug_mw", "polymer_mw_kda", "particle_size_um"} else 0.1
                                record[feature] = float(st.number_input(
                                    LABELS[feature],
                                    value=float(example[feature]),
                                    step=step, format="%.4f",
                                    help=FIELD_HELP.get(feature),
                                    key=f"bao_v4_{i}_{feature}",
                                ))
                    records.append(record)
            with st.expander("Optional: measured release during the first 24 hours", expanded=False):
                st.caption(
                    "Enter **actual measured** release data, not model predictions. "
                    "At least two distinct observation times between day 0 and day 1 are required "
                    "to activate first-day-assisted LightGBM for a formulation. "
                    "Use the exact formulation name from above; otherwise leave this table blank."
                )
                early_manual = st.data_editor(
                    pd.DataFrame(columns=["formulation_id", "time_days", "release_percent"]),
                    num_rows="dynamic", hide_index=True, use_container_width=True,
                    key="bao_v4_early_manual_table",
                    column_config={
                        "formulation_id": st.column_config.TextColumn("Formulation name"),
                        "time_days": st.column_config.NumberColumn("Time (days)", format="%.3f"),
                        "release_percent": st.column_config.NumberColumn("Measured release (%)", format="%.2f"),
                    },
                )
            submitted_manual = st.form_submit_button(
                "Predict release profiles", type="primary", use_container_width=True,
            )
        if submitted_manual:
            try:
                desc = validate_descriptors(pd.DataFrame(records, columns=["formulation_id"] + FEATURES))
                e_map = prepare_early_map(early_manual, desc.formulation_id.tolist())
                st.session_state["bao_custom_prediction_payload"] = {
                    "descriptors": desc, "early_by_id": e_map, "source": "manual",
                }
                valid_payload = st.session_state["bao_custom_prediction_payload"]
            except Exception as exc:
                st.error(f"Correct the formulation inputs: {exc}")

    else:
        st.caption("Advanced entry: one formulation per row; CSV columns must match the model inputs.")
        initial = published.head(2)[FEATURES].copy().reset_index(drop=True)
        initial.insert(0, "formulation_id", ["Formulation 1", "Formulation 2"])
        blank = initial.head(0)
        st.download_button(
            "Download parameter template (CSV)", blank.to_csv(index=False).encode("utf-8"),
            "bao_parameter_template.csv", "text/csv", key="bao_v4_blank_template",
        )
        uploaded = st.file_uploader("Upload parameter sets (optional)", type=["csv"], key="bao_v4_uploaded_csv")
        try:
            table_source = pd.read_csv(uploaded) if uploaded is not None else initial
        except Exception as exc:
            st.error(f"Could not load CSV: {exc}")
            table_source = initial
        with st.form("bao_spreadsheet_form", clear_on_submit=False):
            edited = st.data_editor(
                table_source, num_rows="dynamic", hide_index=True, use_container_width=True,
                key="bao_v4_spreadsheet",
                column_config={
                    "formulation_id": st.column_config.TextColumn("Formulation name"),
                    **{feature: st.column_config.NumberColumn(LABELS[feature], format="%.4f")
                       for feature in FEATURES},
                },
            )
            st.caption("First-day release measurements can be entered below, if actually measured.")
            early_bulk = st.data_editor(
                pd.DataFrame(columns=["formulation_id", "time_days", "release_percent"]),
                num_rows="dynamic", hide_index=True, use_container_width=True,
                key="bao_v4_early_bulk",
            )
            submitted_bulk = st.form_submit_button("Predict release profiles", type="primary")
        if submitted_bulk:
            try:
                desc = validate_descriptors(edited)
                e_map = prepare_early_map(early_bulk, desc.formulation_id.tolist())
                st.session_state["bao_custom_prediction_payload"] = {
                    "descriptors": desc, "early_by_id": e_map, "source": "spreadsheet",
                }
                valid_payload = st.session_state["bao_custom_prediction_payload"]
            except Exception as exc:
                st.error(f"Correct the table or early-release entries: {exc}")

    if valid_payload is None:
        st.info("Enter your formulation parameters and click **Predict release profiles** to see the graph.")
    else:
        st.divider()
        st.subheader("Predicted cumulative drug release")
        st.caption("Showing the most recently submitted parameter set. Submit again after changing the inputs.")
        try:
            render_predictions(
                valid_payload["descriptors"], early_by_id=valid_payload["early_by_id"],
                widget_prefix="bao_custom",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab_published:
    st.subheader("Compare published Bao release profiles")
    st.caption("Compare observed experimental points against model predictions. These records contributed "
               "to model development and do **not** provide an independent validation set.")
    all_ids = published["formulation_id"].astype(str).tolist()
    examples = []
    for fid, group in data.groupby("formulation_id", sort=False):
        early = group.loc[group.time_days.between(0, 1), "time_days"]
        if early.nunique() >= 2 and (group.time_days > 1).any():
            examples.append(str(fid))
        if len(examples) == 3:
            break
    selection = st.multiselect(
        "Select published formulation IDs", all_ids, default=examples,
        max_selections=15, key="bao_v4_published_ids",
    )
    if selection:
        left, right = st.columns(2)
        with left:
            overlay = st.checkbox("Overlay measured experimental points", value=True, key="bao_v4_observed")
        with right:
            assisted = st.checkbox("Use available measured first-day observations", value=True, key="bao_v4_assisted")
        chosen = data[data.formulation_id.astype(str).isin(selection)].copy()
        rows = chosen.drop_duplicates("formulation_id")[["formulation_id"] + FEATURES].copy()
        obs_map = {str(fid): g[["time_days", "release_fraction"]].sort_values("time_days")
                   for fid, g in chosen.groupby("formulation_id")}
        first_map = {
            str(fid): g.loc[g.time_days.between(0, 1), ["time_days", "release_fraction"]].copy()
            for fid, g in chosen.groupby("formulation_id")
        } if assisted else {}
        try:
            render_predictions(
                rows, early_by_id=first_map, observations=obs_map,
                show_observations=overlay, widget_prefix="bao_published",
            )
            with st.expander("Published formulation descriptors"):
                st.dataframe(rows, hide_index=True, use_container_width=True)
            st.download_button(
                "Download selected measured release points (CSV)",
                chosen[["formulation_id", "time_days", "release_fraction"]].to_csv(index=False).encode("utf-8"),
                "bao_selected_experimental_release.csv", "text/csv",
                key="bao_v4_measured_csv",
            )
        except Exception as exc:
            st.error(f"Unable to display selected formulations: {exc}")
    else:
        st.info("Select at least one formulation.")

with tab_about:
    st.subheader("Dataset and prediction method")
    st.markdown(
        "**Source:** Bao et al. (2025), *A dataset on formulation parameters and characteristics of "
        "drug-loaded PLGA microparticles*. [Mendeley Data v2](https://data.mendeley.com/datasets/zzvtdrcy76/2) "
        "(CC BY 4.0)."
    )
    st.markdown(
        "**Models:** Descriptor-only LightGBM uses formulation properties and time. "
        "First-day-assisted LightGBM additionally uses actual release observations from day 0–1. "
        "Both forecast cumulative release after day 1."
    )
    st.markdown(
        f"**Internal validation:** Descriptor-only MAE **{META['descriptor_oof_mae']:.4f}** "
        f"(316 formulations); assisted MAE **{META['assisted_oof_mae']:.4f}** "
        "(212 formulations). Different cohorts; not directly comparable. "
        "Neither result establishes validation on unseen publications."
    )
    st.caption(
        "The interface follows the experimental input → first-day information → long-term prediction "
        "workflow described by Alabsi et al. (2026), but it is not a reproduction of that paper's "
        "347-formulation dataset, model feature set, or reported performance."
    )
    st.download_button(
        "Download normalized Bao release dataset (CSV)",
        data.to_csv(index=False).encode("utf-8"),
        "bao_release_measurements_321.csv", "text/csv", key="bao_v4_full_dataset_csv",
    )
