"""PLGA Release Studio — independent, deployable frozen-V1 testing preview.

Uses only models/ and live_config.json assembled by prepare_bundle.py on Roihu.
Neither retrains nor accesses the original source dataset or Roihu at runtime.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference import (
    NUMERIC_FEATURES, load_live_models, canonical_inputs, predict_curve,
    familiar_categories, out_of_range, sample_at_days,
)

st.set_page_config(page_title='PLGA Release Studio | V1 Testing', page_icon='🧪', layout='wide')

@st.cache_resource(show_spinner='Verifying frozen model files…')
def get_models():
    return load_live_models()

try:
    models, config = get_models()
except Exception as ex:
    st.error('The frozen model bundle is missing or has failed integrity checks.')
    st.code(str(ex))
    st.stop()

LABELS = {
    'Drug_MW': 'Drug molecular weight',
    'Drug_TPSA': 'Drug TPSA',
    'Number_of_Heteroatoms': 'Number of drug heteroatoms',
    'Drug_LogP': 'Drug LogP',
    'Polymer_MW': 'Polymer molecular weight (source scale)',
    'LA_GA': 'LA/GA (source encoding)',
    'Initial_Drug_to_Polymer_Ratio': 'Initial drug-to-polymer ratio (source scale)',
    'Particle_Size': 'Particle size (source scale)',
    'Drug_Loading_Capacity': 'Drug loading capacity (%)',
    'Drug_Encapsulation_Efficiency': 'Drug encapsulation efficiency (%)',
    'Surfactant_Percent': 'Solubility enhancer concentration (source scale)',
}
SOURCE_UNITS = {
    'Drug_MW': 'Drug molecular weight, typically g/mol. Confirm chemical form/salt.',
    'Drug_TPSA': 'TPSA on the source processed-data scale.',
    'Number_of_Heteroatoms': 'Non-negative whole-number count.',
    'Drug_LogP': 'Can be negative.',
    'Polymer_MW': 'Use the source processed-data scale. The physical unit has not been fully verified.',
    'LA_GA': 'Encoded numeric LA/GA descriptor from the training data; do not enter 50:50 literally.',
    'Initial_Drug_to_Polymer_Ratio': 'Use the numerical scale of the source dataset, not automatically a percentage.',
    'Particle_Size': 'Use source data scale; do not assume a physical unit without checking.',
    'Drug_Loading_Capacity': 'Enter 35.4 for 35.4%. Converted to 0.354 for inference.',
    'Drug_Encapsulation_Efficiency': 'Enter a percentage such as 88.3 for 88.3%.',
    'Surfactant_Percent': 'Bao source calls this Solubility Enhancer Concentration; units need verification.',
}

st.title('🧪 PLGA Release Studio')
st.caption('Frozen V1 testing preview · CatBoost + drug identity + formulation method · No release-curve-derived kinetic inputs')
st.warning('**Experimental research prototype.** Predictions are not independently externally validated and must not be used for treatment, manufacturing release or regulatory decisions.')

with st.sidebar:
    st.subheader('Model provenance')
    st.write('**Candidate:** combined CatBoost V1')
    st.write('**Training:** 301 Bao-linked formulations')
    st.write('**Comparison:** matched descriptor-only CatBoost trained on the same 301 formulations')
    st.write('**Inference:** capped 0–100% with a nondecreasing isotonic projection')
    st.write('**Prediction domain:** after day 1 only')
    st.divider()
    cv = config['development_cv']
    st.metric('Development study-held-out MAE — V1', f"{cv['combined_pooled_mae']:.4f}")
    st.caption(f"Descriptor-only: {cv['baseline_pooled_mae']:.4f}. These are historical CV results for the selected configurations, **not** independent test results of these final fitted files.")

numeric_default = config['demo_profile_training_example']
with st.form('plga_live_form'):
    st.subheader('Describe your formulation')
    st.caption('Fields use the original training-data numerical scales, except drug loading which is shown as a percentage. The default profile is from training data for demonstration, not for validation.')
    drug_options = [numeric_default['Drug_Name']] + [s for s in config['drug_names'] if s != numeric_default['Drug_Name']]
    drug_options += ['Other drug (enter name below)']
    method_options = [numeric_default['Formulation_Method']] + [s for s in config['methods'] if s != numeric_default['Formulation_Method']]
    method_options += ['Other method (enter name below)']
    drug_col, method_col = st.columns(2)
    with drug_col:
        known_drug = st.selectbox('Drug identity', drug_options, index=0)
        custom_drug = st.text_input('New / other drug name', value='', disabled=known_drug != 'Other drug (enter name below)')
    with method_col:
        known_method = st.selectbox('Formulation method', method_options, index=0)
        custom_method = st.text_input('Other preparation method', value='', disabled=known_method != 'Other method (enter name below)')

    groups = [
        ('Drug descriptors', NUMERIC_FEATURES[:4]),
        ('Polymer descriptors', NUMERIC_FEATURES[4:6]),
        ('Formulation descriptors', NUMERIC_FEATURES[6:]),
    ]
    vals = {}
    for heading, fields in groups:
        st.markdown('**' + heading + '**')
        columns = st.columns(3)
        for j, feature in enumerate(fields):
            with columns[j % 3]:
                factor = 100.0 if feature == 'Drug_Loading_Capacity' else 1.0
                initial = float(numeric_default[feature]) * factor
                if feature == 'Number_of_Heteroatoms':
                    shown = st.number_input(LABELS[feature], value=int(initial), step=1, help=SOURCE_UNITS[feature])
                else:
                    shown = st.number_input(LABELS[feature], value=initial, step=0.1, format='%.4f', help=SOURCE_UNITS[feature])
                vals[feature] = float(shown) / factor
    with st.expander('Source-unit warning', expanded=False):
        st.markdown('Polymer MW, particle size, LA/GA encoding and solubility-enhancer units require source-aligned inputs. A numerical prediction is **not** evidence that a custom input is physically feasible.')
    compare = st.checkbox('Show matched descriptor-only comparison', value=True)
    submitted = st.form_submit_button('Predict release curve', type='primary')

if submitted:
    drug = custom_drug if known_drug == 'Other drug (enter name below)' else known_drug
    method = custom_method if known_method == 'Other method (enter name below)' else known_method
    try:
        canonical = canonical_inputs(vals, drug, method)
        pred_v1 = predict_curve(models, canonical, 'combined_v1')
        baseline = predict_curve(models, canonical, 'descriptor_only') if compare else None
        st.session_state['live_predictions'] = {'combined': pred_v1, 'baseline': baseline, 'inputs': canonical}
    except (ValueError, RuntimeError, KeyError) as ex:
        st.error('Prediction inputs could not be processed: ' + str(ex))

record = st.session_state.get('live_predictions')
if record is None:
    st.info('Enter a formulation and select **Predict release curve** to start testing the frozen V1 model.')
else:
    values = record['inputs']
    status = familiar_categories(values, config)
    outside = out_of_range(values, config)
    st.divider()
    st.subheader('Predicted cumulative drug release')
    if outside:
        st.warning('**Outside the training descriptor ranges:** ' + ', '.join(outside) + '. Prediction reliability may be lower.')
    if not status['drug_seen_in_training']:
        st.warning('This drug name was not observed in the 301-formulation training subset. The model can still predict, but unseen-drug performance is uncertain.')
    if not status['method_seen_in_training']:
        st.warning('This preparation method was not observed during training.')
    if not outside and all(status.values()):
        st.caption('Numeric inputs are within observed ranges and categorical values are familiar. This is NOT a guarantee of reliable predictions.')

    curve = record['combined']
    baseline_curve = record.get('baseline')
    time_lo, time_hi = float(curve.Time_days.iloc[0]), float(curve.Time_days.iloc[-1])
    limit = min(time_hi, 60.0)
    if limit < time_lo:
        limit = time_hi
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve.Time_days, y=curve.Release_percent,
        name='Improved V1 (drug + method)', line={'width': 3}, mode='lines',
    ))
    if baseline_curve is not None:
        fig.add_trace(go.Scatter(
            x=baseline_curve.Time_days, y=baseline_curve.Release_percent,
            name='Descriptor-only matched baseline', mode='lines',
            line={'width': 2, 'dash': 'dash'},
        ))
    fig.update_layout(
        xaxis={'title': 'Days', 'range': [time_lo, limit]},
        yaxis={'title': 'Cumulative release (%)', 'range': [0, 100]},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.03, 'x': 0},
        margin={'l': 10, 'r': 10, 'b': 10, 't': 25},
        height=425,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f'Full prediction support: {time_lo:.2f} to {time_hi:.1f} days. Graph starts zoomed to 60 days where supported; downloads include the entire modeled interval. No prediction at day 0.')
    days = sample_at_days(curve, [7, 14, 30])
    cols = st.columns(3)
    for col, day in zip(cols, [7, 14, 30]):
        value = days[day]
        col.metric(f'Day {day}', 'Out of range' if value is None else f'{value:.1f}%')

    export = curve.copy()
    if baseline_curve is not None:
        export['Descriptor_only_release_percent'] = baseline_curve.Release_percent
        export['V1_minus_baseline_percentage_points'] = curve.Release_percent - baseline_curve.Release_percent
    download_col1, download_col2 = st.columns(2)
    download_col1.download_button('Download predictions CSV', export.to_csv(index=False).encode('utf-8'), file_name='plga_frozen_v1_curve.csv', mime='text/csv')
    output_record = {
        'model': 'combined_v1', 'dataset': '301 Bao-linked training formulations',
        'input_values_model_scale': values,
        'disclaimer': 'Exploratory development model; not externally validated',
    }
    download_col2.download_button('Download input configuration JSON', json.dumps(output_record, indent=2).encode('utf-8'), file_name='plga_frozen_v1_input.json', mime='application/json')

with st.expander('Important interpretation and limitations'):
    st.markdown('''
- **This is a separate preview of the 301-formulation frozen candidate**, not the 347-formulation model currently used by the original Streamlit app.
- The matched descriptor-only and V1 curves are two different trained models; differences do not establish the V1 prediction is correct for any individual formulation.
- The quoted 0.19326 versus 0.20365 MAEs are **previously examined study-held-out development results**, not independent external validation.
- The isotonic step ensures plotted curves are nondecreasing between supported points. It does not prove that the release mechanism is monotonic in the laboratory.
- Drug name is converted to lowercase to match training preprocessing. Categorical values unknown to training can still produce predictions, but those predictions require particular care.
- Drug loading is entered as a **percentage** and converted to a fraction; encapsulation efficiency is entered as a **percentage** and is not divided by 100.
- The model's numeric encodings and source-unit definitions should be checked before laboratory decisions. Some combinations of median descriptors or modified inputs may not be physically realizable.
- The model does not capture release-medium, temperature or other unrecorded study conditions. Do not treat generated curves as validated drug formulation guidance.
''')
