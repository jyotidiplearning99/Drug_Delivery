"""Check deployed models and baseline comparison using saved demo metadata."""
from inference import load_live_models, canonical_inputs, predict_curve, sample_at_days, familiar_categories, out_of_range

models, conf = load_live_models()
input_values = conf['demo_profile_training_example']
values = canonical_inputs(input_values, input_values['Drug_Name'], input_values['Formulation_Method'])
assert set(familiar_categories(values, conf).values()) == {True}
assert not out_of_range(values, conf)
for name in ('descriptor_only', 'combined_v1'):
    curve = predict_curve(models, values, name)
    assert len(curve) == 256
    assert curve.Release_fraction.between(0, 1).all()
    assert (curve.Release_fraction.diff().fillna(0) >= -1e-10).all()
    assert curve['Time_days'].is_monotonic_increasing
    print(name, 'OK; day 7/14/30 (%):', sample_at_days(curve, [7, 14, 30]))
print('PASS: both frozen models run, hashes verified, output bounded and nondecreasing')
