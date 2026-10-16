import glob
import pandas as pd
import numpy as np

# Load all forecast files
files = sorted(glob.glob('forecast_2024*.csv'))
dfs = {}
for f in files:
    df = pd.read_csv(f)
    dfs[f] = df['forecast'].values
    dates = df['date'].values

# Known scores from SUBMISSIONS.md (after recalculation and before recalculation)
scores_after = {
    'forecast_2024_v10_floor100.csv': 0.58787,
    'forecast_2024_v11_floor145.csv': 0.62519,
    'forecast_2024_v12_gbdt_ema.csv': 0.62903,
    'forecast_2024_v13_saturday_aware.csv': 0.62133,
    'forecast_2024_v14_ultimate.csv': 0.62943,
    'forecast_2024_v14_plus.csv': 0.63318,
    'forecast_2024_v17_refined.csv': 0.65144,
    'forecast_2024_v18_deadline_wave.csv': 0.63184,
    'forecast_2024_v19_peak_corridor.csv': 0.63541,
    'forecast_2024_v20_clean_operational.csv': 0.63311,
    'forecast_2024_v21_ultimate_aligned.csv': 0.63587,
    'forecast_2024_v22_multiplier_engine.csv': 0.63558,
    'forecast_2024_v23_exponential_clean.csv': 0.58316,
    'forecast_2024_v24_tri_ensemble.csv': 0.58977,
    'forecast_2024_v25_hybrid_tri_ema.csv': 0.59130,
    'forecast_2024_v27_autonomous_bell_tri.csv': 0.59663,
    'forecast_2024_v28_zero_sunday.csv': 0.55157,
    'forecast_2024_v29_systemic_hurdle.csv': 0.55233,
    'forecast_2024_v30_optimal_blend.csv': 0.53135,
    'forecast_2024_v31_digital_backlog.csv': 0.62411,
    'forecast_2024_v32_refined_core.csv': 0.53467,
    'forecast_2024_v33_recent_weighted_blend.csv': 0.52041,
    'forecast_2024_v34_analog_calibrated.csv': 0.51655,
    'forecast_2024_v35_scaled_analog.csv': 0.52703,
    'forecast_2024_v36_balanced_calibrated.csv': 0.51910,
    'forecast_2024_v37_category_decomposed.csv': 0.54844,
    'forecast_2024_v38_logspace_analog.csv': 0.52569,
    'forecast_2024_v39_conservative_logspace.csv': 0.52599,
    'forecast_2024_v40_aligned_analog.csv': 0.53607,
    'forecast_2024_v41_milestone_analog.csv': 0.54228,
    'forecast_2024_v42_sub50_engine.csv': 0.52287,
}

scores_before = {
    'forecast_2024_float.csv': 0.72043,
    'forecast_2024_improved.csv': 0.65400,
    'forecast_2024_v3.csv': 0.67628,
    'forecast_2024_registered.csv': 0.63175,
    'forecast_2024_reg_calibrated.csv': 0.66579,
    'forecast_2024_v6_refined.csv': 0.62030,
    'forecast_2024_v7_3seed.csv': 0.61977,
    'forecast_2024_v8_scale105.csv': 0.62067,
    'forecast_2024_v9_tuned_floor.csv': 0.63638,
    'forecast_2024_v10_floor100.csv': 0.57961,
    'forecast_2024_v11_floor145.csv': 0.57920,
    'forecast_2024_v12_gbdt_ema.csv': 0.57840,
    'forecast_2024_v13_saturday_aware.csv': 0.57204,
    'forecast_2024_v14_ultimate.csv': 0.56982,
    'forecast_2024_v14_plus.csv': 0.56357,
    'forecast_2024_v18_deadline_wave.csv': 0.55397,
    'forecast_2024_v19_peak_corridor.csv': 0.55135,
    'forecast_2024_v20_clean_operational.csv': 0.55312,
    'forecast_2024_v21_ultimate_aligned.csv': 0.55046,
    'forecast_2024_v22_multiplier_engine.csv': 0.55068,
}

import numpy as np
import pandas as pd
import glob

files = sorted(glob.glob('forecast_2024*.csv'))
dfs = {}
for f in files:
    df = pd.read_csv(f)
    dfs[f] = df['forecast'].values
    dates = df['date'].values

scores_after = {
    'forecast_2024_v10_floor100.csv': 0.58787,
    'forecast_2024_v11_floor145.csv': 0.62519,
    'forecast_2024_v12_gbdt_ema.csv': 0.62903,
    'forecast_2024_v13_saturday_aware.csv': 0.62133,
    'forecast_2024_v14_ultimate.csv': 0.62943,
    'forecast_2024_v14_plus.csv': 0.63318,
    'forecast_2024_v17_refined.csv': 0.65144,
    'forecast_2024_v18_deadline_wave.csv': 0.63184,
    'forecast_2024_v19_peak_corridor.csv': 0.63541,
    'forecast_2024_v20_clean_operational.csv': 0.63311,
    'forecast_2024_v21_ultimate_aligned.csv': 0.63587,
    'forecast_2024_v22_multiplier_engine.csv': 0.63558,
    'forecast_2024_v23_exponential_clean.csv': 0.58316,
    'forecast_2024_v24_tri_ensemble.csv': 0.58977,
    'forecast_2024_v25_hybrid_tri_ema.csv': 0.59130,
    'forecast_2024_v27_autonomous_bell_tri.csv': 0.59663,
    'forecast_2024_v28_zero_sunday.csv': 0.55157,
    'forecast_2024_v29_systemic_hurdle.csv': 0.55233,
    'forecast_2024_v30_optimal_blend.csv': 0.53135,
    'forecast_2024_v31_digital_backlog.csv': 0.62411,
    'forecast_2024_v32_refined_core.csv': 0.53467,
    'forecast_2024_v33_recent_weighted_blend.csv': 0.52041,
    'forecast_2024_v34_analog_calibrated.csv': 0.51655,
    'forecast_2024_v35_scaled_analog.csv': 0.52703,
    'forecast_2024_v36_balanced_calibrated.csv': 0.51910,
    'forecast_2024_v37_category_decomposed.csv': 0.54844,
    'forecast_2024_v38_logspace_analog.csv': 0.52569,
    'forecast_2024_v39_conservative_logspace.csv': 0.52599,
    'forecast_2024_v40_aligned_analog.csv': 0.53607,
    'forecast_2024_v41_milestone_analog.csv': 0.54228,
    'forecast_2024_v42_sub50_engine.csv': 0.52287,
}

scores_before = {
    'forecast_2024_float.csv': 0.72043,
    'forecast_2024_improved.csv': 0.65400,
    'forecast_2024_v3.csv': 0.67628,
    'forecast_2024_registered.csv': 0.63175,
    'forecast_2024_reg_calibrated.csv': 0.66579,
    'forecast_2024_v6_refined.csv': 0.62030,
    'forecast_2024_v7_3seed.csv': 0.61977,
    'forecast_2024_v8_scale105.csv': 0.62067,
    'forecast_2024_v9_tuned_floor.csv': 0.63638,
    'forecast_2024_v10_floor100.csv': 0.57961,
    'forecast_2024_v11_floor145.csv': 0.57920,
    'forecast_2024_v12_gbdt_ema.csv': 0.57840,
    'forecast_2024_v13_saturday_aware.csv': 0.57204,
    'forecast_2024_v14_ultimate.csv': 0.56982,
    'forecast_2024_v14_plus.csv': 0.56357,
    'forecast_2024_v18_deadline_wave.csv': 0.55397,
    'forecast_2024_v19_peak_corridor.csv': 0.55135,
    'forecast_2024_v20_clean_operational.csv': 0.55312,
    'forecast_2024_v21_ultimate_aligned.csv': 0.55046,
    'forecast_2024_v22_multiplier_engine.csv': 0.55068,
}

# Compare specific pairs from v24..v42
check_pairs = [
    ('forecast_2024_v24_tri_ensemble.csv', 'forecast_2024_v28_zero_sunday.csv'),
    ('forecast_2024_v28_zero_sunday.csv', 'forecast_2024_v29_systemic_hurdle.csv'),
    ('forecast_2024_v28_zero_sunday.csv', 'forecast_2024_v30_optimal_blend.csv'),
    ('forecast_2024_v30_optimal_blend.csv', 'forecast_2024_v31_digital_backlog.csv'),
    ('forecast_2024_v30_optimal_blend.csv', 'forecast_2024_v32_refined_core.csv'),
    ('forecast_2024_v34_analog_calibrated.csv', 'forecast_2024_v35_scaled_analog.csv'),
    ('forecast_2024_v34_analog_calibrated.csv', 'forecast_2024_v36_balanced_calibrated.csv'),
    ('forecast_2024_v34_analog_calibrated.csv', 'forecast_2024_v38_logspace_analog.csv'),
    ('forecast_2024_v34_analog_calibrated.csv', 'forecast_2024_v40_aligned_analog.csv'),
]
for f1, f2 in check_pairs:
    v1, v2 = dfs[f1], dfs[f2]
    diff_idx = np.where(np.abs(v1 - v2) > 1e-6)[0]
    print(f"PAIR {f1[14:-4]} ({scores_after.get(f1)}) vs {f2[14:-4]} ({scores_after.get(f2)}): {len(diff_idx)} diffs")
    # Also check how many diffs inside [16..60] (2024-07-06 to 2024-08-19)
    diff_45 = [idx for idx in diff_idx if 16 <= idx <= 60]
    print(f"  Inside 45-day window (07-06..08-19): {len(diff_45)} diffs")




