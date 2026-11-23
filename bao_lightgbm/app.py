"""PLGA Release Studio | Bao-only LightGBM, multi-formulation interactive prototype."""
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

st.set_page_config(page_title="PLGA Release Studio · Bao LightGBM", page_icon="🧪", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {max-width: 1350px; padding-top: 2rem;}
h1 {letter-spacing:-.04em;}
div[data-testid="stMetric"] {background:#f1f8fa; border-radius:12px; padding:12px 16px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading saved LightGBM models…")
def cached_models():
    return load_models()

@st.cache_data(show_spinner="Loading published Bao release profiles…")
def source_data():
    return pd.read_csv(ROOT / "data" / "published_profiles.csv", dtype={"formulation_id":str})

models = cached_models()
data = source_data()

st.title("PLGA Release Studio")
st.caption("**Bao dataset · LightGBM edition**  |  Multiple formulations · cumulative release curves · experimental research prototype")
st.info("**Two prediction modes:** formulation parameters only (descriptor-only LightGBM), or first-day-assisted LightGBM when actual measurements at two or more distinct times through day 1 are supplied. The assisted model must **not** be used without measured early-release data.", icon="ℹ️")

m1,m2,m3,m4=st.columns(4)
m1.metric("Source formulations",f'{META["n_all_formulations"]:,}')
m2.metric("Measured release points",f'{META["n_all_measurements"]:,}')
m3.metric("Descriptor-only CV MAE",f'{META["descriptor_oof_mae"]:.3f}')
m4.metric("First-day CV MAE",f'{META["assisted_oof_mae"]:.3f}')
st.caption("MAE in release fraction: descriptor-only **316** eligible Bao profiles, 5-fold MAE ~0.155; first-day-assisted **212** eligible profiles, 5-fold MAE ~0.119. **Different cohorts/projection rules: do not directly compare these MAEs.** Not publication-held-out testing.")

with st.sidebar:
    st.header("Plot options")
    horizon = st.slider("Prediction horizon (days)", min_value=7, max_value=180, value=60, step=1)
    target_day = st.slider("Inspect day", min_value=2, max_value=horizon, value=min(14,horizon))
    n_grid = st.slider("Curve resolution",min_value=60,max_value=300,value=160,step=20)
    show_band = st.checkbox("Show training horizon",value=True)
    st.divider()
    st.markdown("**Data provenance**")
    st.caption("Bao et al., Mendeley Data v2 (CC BY 4.0). Release time is in **days**. Polymer Mw is **kDa**. LA:GA entered as **lactide %**. Loading and encapsulation entered as percentages.")
    st.markdown("[Original Bao dataset](https://data.mendeley.com/datasets/zzvtdrcy76/2)")


def grid_days(horizon, n):
    return np.linspace(1.01, float(horizon), int(n))


def make_plot(curves, observations=None):
    fig = go.Figure()
    colors = qualitative.Plotly + qualitative.Safe + qualitative.D3
    for i, (name, frame) in enumerate(curves.items()):
        color=colors[i % len(colors)]
        fig.add_trace(go.Scatter(x=frame.time_days, y=frame.release_percent,
                                 mode="lines", name=f"{name} · {frame.model.iloc[0].replace('_',' ')}",
                                 line=dict(width=2.8,color=color)))
        if observations is not None and name in observations and observations[name] is not None:
            obs=observations[name]
            fig.add_trace(go.Scatter(x=obs.time_days, y=obs.release_fraction*100,
                                     mode="markers", name=f"{name} · observed",legendgroup=name,
                                     marker=dict(symbol="circle-open", size=6, color=color),opacity=.8))
    if show_band and horizon > float(META['training_numeric_ranges']['time_days'][1]):
        daymax=float(META['training_numeric_ranges']['time_days'][1])
        fig.add_vrect(x0=daymax, x1=horizon, fillcolor="orange", opacity=.08, line_width=0,
                      annotation_text="Beyond training range",annotation_position="top left")
    fig.add_vline(x=1, line_dash="dot",line_color="#88909b")
    fig.add_vline(x=target_day, line_dash="dash",line_color="#64748b",opacity=.65)
    fig.update_layout(template="plotly_white",height=505,margin=dict(l=10,r=10,t=20,b=10),
                      xaxis_title="Time (days)",yaxis_title="Cumulative drug release (%)",
                      xaxis=dict(range=[0,horizon]),yaxis=dict(range=[0,102]),
                      legend=dict(orientation="h",y=-.23),hovermode="x unified")
    return fig


def run_and_show(descriptors: pd.DataFrame, early_by_id: dict | None=None,
                 obs_by_id: dict | None=None, observations_enabled=False, ui_key: str='custom'):
    early_by_id = early_by_id or {}
    desc = validate_descriptors(descriptors)
    outside=out_of_range(desc)
    if outside:
        msg="; ".join(f"{name}: {', '.join(feats)}" for name,feats in outside[:6])
        st.warning(f"Some inputs are outside the training range (extrapolation): {msg}")
    days=grid_days(horizon,n_grid)
    curves={};long=[]; summaries=[]; assisted_count=0
    for record in desc.to_dict('records'):
        name=record['formulation_id']
        early=early_by_id.get(name)
        y,mode=predict_curve(models, record, days, early)
        if mode == 'first_day_assisted': assisted_count+=1
        frame=pd.DataFrame({'formulation_id':name,'time_days':days,
                            'release_percent':y*100,'model':mode})
        curves[name]=frame
        long.append(frame)
        summaries.append({'Formulation':name,'Prediction mode':'First-day assisted' if mode=='first_day_assisted' else 'Descriptors only',
                          f'At day {target_day} (%)':round(float(np.interp(target_day,days,y)*100),2),
                          f'At day {horizon} (%)':round(float(y[-1]*100),2)})
    fig=make_plot(curves, observations=obs_by_id if observations_enabled else None)
    st.plotly_chart(fig,use_container_width=True)
    if early_by_id and assisted_count < len(desc):
        st.caption(f"{assisted_count}/{len(desc)} curves used first-day-assisted LightGBM. The remainder used descriptor-only LightGBM because two valid early measurements were unavailable.")
    st.dataframe(pd.DataFrame(summaries),hide_index=True,use_container_width=True)
    export=pd.concat(long,ignore_index=True)
    c1,c2=st.columns(2)
    with c1:
        st.download_button('Download all predicted curves (CSV)',export.to_csv(index=False).encode('utf-8'),
                           'bao_lightgbm_multi_curves.csv','text/csv',use_container_width=True, key=f'{ui_key}_curves_csv')
    with c2:
        st.download_button('Download input parameter sets (CSV)',desc.to_csv(index=False).encode('utf-8'),
                           'bao_lightgbm_parameter_sets.csv','text/csv',use_container_width=True, key=f'{ui_key}_parameters_csv')
    if horizon > META['training_numeric_ranges']['time_days'][1]:
        st.warning('Predictions beyond the largest descriptor-only training time are extrapolations. The first-day-assisted model has a shorter training horizon; long-range assisted forecasts also require caution.')
    st.caption('Predictions are clipped to 0–100% and adjusted to be nondecreasing; this is a model output constraint, not a calibrated uncertainty bound.')


tab1,tab2,tab3=st.tabs(['🧬 Compare custom formulations','📈 Explore published release profiles','ℹ️ Model & dataset'])

with tab1:
    st.subheader('Compare your own PLGA formulations')
    st.caption('Enter parameters directly as individual numeric fields, just like the existing PLGA Studio. Each formulation produces its own release curve on the shared graph.')

    # Every starting value comes from a published Bao descriptor row, and remains editable.
    seed_rows=(data.drop_duplicates('formulation_id')
               .dropna(subset=FEATURES)
               .reset_index(drop=True))
    defaults=seed_rows.head(3)[FEATURES].copy().reset_index(drop=True)
    defaults.insert(0,'formulation_id',[f'Formulation {i+1}' for i in range(len(defaults))])
    template=defaults.head(0).copy()

    entry_mode=st.radio('How would you like to enter your formulations?',
                        options=['Manual parameter fields', 'Spreadsheet / CSV (advanced)'],
                        horizontal=True, key='bao_entry_mode')

    if entry_mode == 'Manual parameter fields':
        count=int(st.number_input('Number of formulations to compare',
                                  min_value=1, max_value=25, value=2, step=1,
                                  help='Choose up to 25 parameter sets; each becomes a separate predicted curve.'))
        st.caption('Change any number below. All 10 model input parameters are editable, with units shown in the labels. Expand each formulation to edit it.')
        groups=[
            ('Drug properties', ['drug_mw','drug_logp','drug_tpsa']),
            ('Polymer properties', ['polymer_mw_kda','la_ga_pct']),
            ('Formulation properties', ['drug_polymer_ratio','particle_size_um',
                                        'drug_loading_pct','encapsulation_efficiency_pct',
                                        'solubility_enhancer_conc_pct']),
        ]
        manual_records=[]
        for i in range(count):
            seed=seed_rows.iloc[i % len(seed_rows)]
            with st.expander(f'Formulation {i+1} — editable parameters', expanded=(i < 2)):
                name=st.text_input('Formulation name / ID', value=f'Formulation {i+1}',
                                   key=f'bao_manual_formulation_name_{i}')
                record={'formulation_id':name}
                for group_name, fields in groups:
                    st.markdown('**'+group_name+'**')
                    cols=st.columns(3)
                    for j, feature in enumerate(fields):
                        with cols[j % 3]:
                            record[feature]=float(st.number_input(
                                LABELS[feature], value=float(seed[feature]), step=0.1,
                                format='%.4f', key=f'bao_manual_{i}_{feature}'))
                manual_records.append(record)
        edited=pd.DataFrame(manual_records, columns=['formulation_id']+FEATURES)
        st.caption('These are editable demonstration inputs, not recommendations. Values outside the model training ranges trigger a warning below.')
    else:
        st.write('Edit the table directly or upload your own parameter sets. One row gives one predicted curve.')
        up=st.file_uploader('Optional: upload parameter sets CSV', type=['csv'],
                            key='custom_parameters')
        if up is not None:
            try:
                param_rows=pd.read_csv(up)
            except Exception as exc:
                st.error(f'Could not read parameter CSV: {exc}')
                param_rows=defaults
        else:
            param_rows=defaults
        st.download_button('Download blank input CSV template',
                           template.to_csv(index=False).encode('utf8'),
                           file_name='bao_lightgbm_input_template.csv', mime='text/csv')
        edited=st.data_editor(param_rows, num_rows='dynamic', use_container_width=True,
                              hide_index=True,
                              column_config={
                                  'formulation_id':st.column_config.TextColumn('Formulation name / ID'),
                                  **{f:st.column_config.NumberColumn(LABELS[f], format='%.4f')
                                     for f in FEATURES}
                              }, key='parameter_editor')

    with st.expander('Optional: add measured first-day release points for assisted prediction'):
        st.caption('Use actual laboratory measurements only. At least two distinct measurement times between day 0 and day 1 are required. Leave empty for descriptor-only LightGBM prediction.')
        early_blank=pd.DataFrame(columns=['formulation_id','time_days','release_percent'])
        early_edit=st.data_editor(early_blank, num_rows='dynamic',
                                  use_container_width=True, hide_index=True,
                                  key='early_editor')
        st.download_button('Download early-measurement CSV template',
                           early_blank.to_csv(index=False).encode('utf8'),
                           'bao_first_day_measurements_template.csv','text/csv')
    try:
        cleaned=edited.dropna(subset=['formulation_id']).copy()
        early_map={}
        if len(early_edit):
            early_clean=parse_early_csv(early_edit)
            early_map={str(k):v for k,v in early_clean.groupby('formulation_id')}
            unused=set(early_map)-set(cleaned.formulation_id.astype(str))
            if unused:
                st.warning('First-day measurements for unidentified formulations were not used: '
                           + ', '.join(sorted(unused)))
        run_and_show(cleaned, early_by_id=early_map, ui_key='custom')
    except Exception as exc:
        st.error(f'Please correct the input parameters: {exc}')

with tab2:
    st.subheader('Compare measured Bao profiles and model forecasts')
    st.caption('Select up to 15 real formulations. Their measured release points can be overlaid with the trained-model curves. These curves were included in model development and are **not independent test results**.')
    selection=data.drop_duplicates('formulation_id')
    all_ids=selection['formulation_id'].astype(str).tolist()
    # Prefer first three known eligible formulations with two distinct early timepoints and followup.
    d0=[]
    for fid,g in data.groupby('formulation_id',sort=False):
        if g.loc[g.time_days.between(0,1),'time_days'].nunique() >= 2 and (g.time_days>1).any():
            d0.append(str(fid))
        if len(d0)==3:break
    ids=st.multiselect('Choose published formulation IDs',options=all_ids,default=d0,max_selections=15)
    if ids:
        c1,c2=st.columns(2)
        with c1:
            show_obs=st.toggle('Overlay actual recorded release points',value=True)
        with c2:
            use_early=st.toggle('Use measured first-day data when eligible',value=True)
        profiles=data.loc[data.formulation_id.astype(str).isin(ids)].copy()
        rows=profiles.drop_duplicates('formulation_id')[['formulation_id']+FEATURES].copy()
        obs_by={str(fid):g[['time_days','release_fraction']].sort_values('time_days') for fid,g in profiles.groupby('formulation_id')}
        e_by={str(fid):g.loc[g.time_days.between(0,1),['time_days','release_fraction']].copy() for fid,g in profiles.groupby('formulation_id')} if use_early else {}
        try:
            run_and_show(rows,early_by_id=e_by,obs_by_id=obs_by,observations_enabled=show_obs, ui_key='published')
            with st.expander('View reported descriptors for selected formulations'):
                st.dataframe(rows,use_container_width=True,hide_index=True)
            st.download_button('Download selected observed release data',
                profiles[['formulation_id','time_days','release_fraction']].to_csv(index=False).encode('utf8'),
                'bao_selected_observed_release.csv','text/csv',key='published_observations_csv')
        except Exception as exc:
            st.error(f'Unable to display selected formulations: {exc}')
    else:
        st.info('Select at least one formulation to display curves.')

with tab3:
    st.subheader('About the LightGBM models')
    st.markdown('''
**Dataset:** Bao et al., *A dataset on formulation parameters and characteristics of drug-loaded PLGA microparticles*, Mendeley Data v2, DOI [10.17632/zzvtdrcy76.2](https://data.mendeley.com/datasets/zzvtdrcy76/2), **CC BY 4.0**.

**Descriptor-only model:** uses numerical drug and formulation properties plus time, without early measured release data. Trained on **316 formulations** with 4,078 release measurements after day 1. Five-fold formulation-grouped, monotone-projected out-of-fold MAE **0.1553**.

**First-day-assisted model:** uses the same descriptors plus kinetic/early-release signatures calculated **only from measured data at or before day 1**. It forecasts later values; out-of-fold MAE is **0.1193**.

The **first-day model** was fitted on 212 qualifying formulations with 2,704 release measurements after day 1, requiring two distinct measured time points at or before day 1. The **descriptor-only model** was subsequently fitted on 316 Bao profiles, excluding five profiles with large decreases in reported cumulative release. **These two MAEs have different validation cohorts and postprocessing; they cannot establish that assisted prediction is more accurate from the displayed numbers alone.** A previous same-cohort comparison on 212 formulations did favor early signatures (0.1624 vs 0.1193), but it was not an external test.

**Evaluation:** Five-fold grouped by formulation, not by publication or drug. These are exploratory development metrics; cross-formulation duplicates or shared studies may overstate generalization. The two reported MAEs use different subsets and are not directly comparable with the published LightGBM paper.

**Correct units:** Time in days; polymer MW in kDa; particle size in µm; LA:GA encoded as lactide percent (50 = 50:50); actual drug loading and encapsulation are percentages. The Bao variable *Solubility Enhancer Concentration* is not the same as *Surfactant %*. Training data has no measured temperature effect.

**Limitations:** Not tested on completely unseen publications, not a validated design or clinical decision system, and release-medium or process differences may cause large errors. Formulation selection and completeness restrictions can introduce bias. Published profile overlays include data used in fitting; they do not validate prediction generalization. Drug loading and encapsulation efficiency must be known from formulation characterization.
''')
    st.download_button('Download Bao normalized release measurements (CSV)',data.to_csv(index=False).encode('utf8'),
                       'bao_release_measurements_321.csv','text/csv')
