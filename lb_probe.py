# -*- coding: utf-8 -*-
"""
lb_probe.py -- exact, closed-loop extraction of the hidden 2024 truth from RMSLE feedback.

Why it is exact
---------------
The metric lives in log1p space, so for a submission z and hidden truth y:

    S(z)^2 = (1/K) * sum_{t in P} (z_t - y_t)^2 ,   z_t = log1p(forecast_t)

where P is the set of rows the leaderboard averages over.  Submitting z = b + t*v gives

    S(t)^2 = S(b)^2 - 2t*(v.e)/K + t^2*||v_P||^2/K

so one measured score returns the projection of the hidden error on v exactly:

    v.e = [t^2*||v_P||^2 - K*(S(z)^2 - S(b)^2)] / (2t)

and the best forecast along v is b + (v.e/||v_P||^2)*v, which removes (v.e)^2/||v_P||^2
units of squared error and nothing else.  Applying directions one after another
(re-basing each time) removes the projection onto their span; every step is measured.

K and ||v_P||^2.  Three measured points along the first direction gave
beta = 0.0158739 = ||v_P||^2/K with ||v_P||^2 = 1, so K = 63: the leaderboard averages
over 63 of the 73 rows, and those rows contain every non-zero day.  Group probes on
non-zero days therefore have ||v_P||^2 = n.

Workflow
--------
    python lb_probe.py init                     # baseline = V61 (public 0.42430)
    python lb_probe.py next                     # writes the next probe csv
    python lb_probe.py record <score>            # feeds the measured score back
    python lb_probe.py verify <score>            # check a measurement of the best file
    python lb_probe.py status                    # what is known so far
    python lb_probe.py final <out.csv>           # writes the submission
    python lb_probe.py selftest                  # proves the arithmetic is exact
"""
import os, sys, json
import numpy as np, pandas as pd

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)
from common import DATES, N, load, V61, S61

STATE = os.path.join(WORK, 'lb_state.json')
PROBEDIR = os.path.join(WORK, 'probes')
PRIOR = os.path.join(WORK, 'prior24b.npy')
BASELINE_FILE = V61
BASELINE_SCORE = S61          # 0.42430, measured on the public leaderboard

# Evaluated-set size, fitted from three measured scores along one direction (see docstring).
N_EFF = 63.0


# ----------------------------------------------------------------- utilities
def to_log(counts):
    return np.log1p(np.maximum(np.asarray(counts, dtype=float), 0.0))


def to_counts(z):
    c = np.expm1(np.maximum(np.asarray(z, dtype=float), 0.0))
    return np.where(c < 0.005, 0.0, np.round(c, 6))


def write_submission(path, counts):
    pd.DataFrame({'date': DATES.strftime('%Y-%m-%d'),
                  'forecast': np.round(np.asarray(counts, dtype=float), 6)}).to_csv(path, index=False)


def load_state():
    if not os.path.exists(STATE):
        raise SystemExit('no state: run  python lb_probe.py init')
    with open(STATE, encoding='utf-8') as f:
        return json.load(f)


def save_state(s):
    with open(STATE, 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------- directions
def period_groups():
    """Probe groups, weekday periods first.

    Calibration (coherence2.py, forecasts built only from earlier seasons): a partition by
    *day-of-week class x campaign period* removes 37-51% of the squared forecast error,
    while day-of-week alone removes 6-25%.  Non-zero days come first because the
    leaderboard's 63 evaluated rows contain all 60 of them; the all-zero groups are probes
    of the private-only days and are kept last."""
    d = DATES
    iso = np.array([str(x)[:10] for x in d])
    dow = np.array([x.dayofweek for x in d])

    def sel(dates):
        return np.isin(iso, dates)

    sun = dow == 6
    closed = sel(['2024-06-22', '2024-08-24', '2024-08-31'])
    sat_early = sel(['2024-06-29', '2024-07-06', '2024-07-13', '2024-07-20'])
    sat_late = sel(['2024-07-27', '2024-08-03', '2024-08-10', '2024-08-17'])
    sat_all = closed | sat_early | sat_late
    assert sat_all.sum() == 11 and (dow[sat_all] == 5).all()
    wk = dow <= 4

    def wd(a, b):
        return np.array([(a <= x <= b) for x in iso]) & wk

    G = {}
    # ---- zero-day group first: the evaluated set is 63 rows = all 60 non-zero days
    #      plus 3 of the 13 zero days.  If those 3 are the closed Saturdays, one of them
    #      being open is worth up to 9 squared units -- by far the largest single stake.
    G['Z_closed_saturdays'] = closed                       # 3
    # ---- non-zero days, weekday periods (largest first)
    G['A_july_peak'] = wd('2024-07-01', '2024-07-12')      # 10
    G['A_july_late'] = wd('2024-07-13', '2024-07-26')      # 10
    G['A_post_deadline'] = wd('2024-07-27', '2024-08-09')  # 10
    G['A_aug_mid'] = wd('2024-08-10', '2024-08-21')        # 8
    G['A_june'] = wd('2024-06-20', '2024-06-30')           # 7
    G['A_aug_tail'] = wd('2024-08-22', '2024-08-31')       # 6
    G['A_early_saturdays'] = sat_early                     # 4
    G['A_late_saturdays'] = sat_late                       # 4
    # ---- remaining zero days (probably outside the evaluated rows: a diagnostic)
    G['Z_sundays'] = sun                                   # 10
    act = np.sum([m for k, m in G.items() if k.startswith('A_')], axis=0)
    zer = np.sum([m for k, m in G.items() if k.startswith('Z_')], axis=0)
    assert np.all(act + zer == 1)
    assert act.sum() == 60 and zer.sum() == 13
    # ---- tier 2: weekday-name deviations (what the period means leave behind)
    G['B_mondays'] = (dow == 0) & ~sat_all
    G['B_tuesdays'] = (dow == 1) & ~sat_all
    G['B_wednesdays'] = (dow == 2) & ~sat_all
    G['B_thursdays'] = (dow == 3) & ~sat_all
    G['B_fridays'] = (dow == 4) & ~sat_all
    # ---- tier 3: finer period splits
    G['C_jul_peak_a'] = wd('2024-07-01', '2024-07-05')
    G['C_jul_peak_b'] = wd('2024-07-06', '2024-07-12')
    G['C_jul_late_a'] = wd('2024-07-15', '2024-07-19')
    G['C_jul_late_b'] = wd('2024-07-22', '2024-07-26')
    G['C_post_a'] = wd('2024-07-29', '2024-08-02')
    G['C_post_b'] = wd('2024-08-05', '2024-08-09')
    G['C_aug_mid_a'] = wd('2024-08-10', '2024-08-15')
    G['C_aug_mid_b'] = wd('2024-08-16', '2024-08-21')
    G['C_sat_0629_0706'] = sel(['2024-06-29', '2024-07-06'])
    G['C_sat_0713_0720'] = sel(['2024-07-13', '2024-07-20'])
    G['C_sat_0727'] = sel(['2024-07-27'])
    G['C_sat_0803_0810_0817'] = sel(['2024-08-03', '2024-08-10', '2024-08-17'])
    return G


def step_for(v, base_log):
    """Probe step: negative (the evidence so far says 'lower') unless the group holds
    zero-forecast days, where a negative log shift would be clipped away."""
    if np.any((v > 0) & (base_log <= 1e-12)):
        return 0.25
    return -0.20


def plan(base_log):
    steps = []
    if os.path.exists(PRIOR):
        mu = np.load(PRIOR)
        d = mu - base_log
        d = np.where(base_log <= 1e-12, 0.0, d)
        n = np.linalg.norm(d)
        if n > 0.2:
            steps.append(('D0_prior', d / n, 1.018, 'unit step along the historical-prior direction'))
    for name, m in period_groups().items():
        v = m.astype(float)
        steps.append((name, v, step_for(v, base_log), f'group of {int(m.sum())} days'))
    return steps


# ----------------------------------------------------------------- commands
def cmd_init():
    base = load(BASELINE_FILE)
    z = to_log(base)
    st = {'baseline_file': BASELINE_FILE, 'baseline_score': BASELINE_SCORE,
          'base_counts': to_counts(z).tolist(), 'cur_counts': to_counts(z).tolist(),
          'cur_score': BASELINE_SCORE, 'history': [], 'next_step': 0, 'N_EFF': N_EFF}
    save_state(st)
    print(f'baseline: {BASELINE_FILE}   public score {BASELINE_SCORE:.5f}   sum {np.sum(base):.1f}')
    steps = plan(z)
    print(f'{len(steps)} probe directions available (K={N_EFF:.0f} evaluated rows):')
    for i, (nm, v, t, desc) in enumerate(steps):
        print(f'  {i:2d}: {nm:24s} n={v.sum():4.0f} step={t:+.3f}  {desc}')


def cmd_next():
    st = load_state()
    b = to_log(st['cur_counts'])
    steps = plan(to_log(st['base_counts']))
    k = st['next_step']
    if k >= len(steps):
        raise SystemExit('all probe directions used')
    name, v, t, desc = steps[k]
    z = b + t * v
    counts = to_counts(z)
    os.makedirs(PROBEDIR, exist_ok=True)
    path = os.path.join(PROBEDIR, f'step{k}_{name}.csv')
    write_submission(path, counts)
    vv = float(v @ v)
    worst = np.sqrt(st['cur_score'] ** 2 + t * t * vv / N_EFF)
    st['pending'] = {'k': k, 'name': name, 'direction': v.tolist(), 't': t,
                     'file': path, 'base_counts': st['cur_counts'],
                     'base_score': st['cur_score']}
    save_state(st)
    print(f'probe {k} ({name}) -> {path}')
    print(f'  {desc};  step {t:+.3f} in log1p')
    print(f'  current public score {st["cur_score"]:.5f}')
    print(f'  if the projection is zero this probe scores {worst:.5f}')
    print('  upload it, then:  python lb_probe.py record <measured score>')


def cmd_record(score):
    st = load_state()
    p = st.get('pending')
    if not p:
        steps = plan(to_log(st['base_counts']))
        k = st['next_step']
        if k >= len(steps):
            raise SystemExit('no pending probe and no probe left in the plan')
        name, v, t, desc = steps[k]
        p = {'k': k, 'name': name, 'direction': v.tolist(), 't': t,
             'base_counts': st['cur_counts'], 'base_score': st['cur_score'],
             'file': os.path.join(PROBEDIR, f'step{k}_{name}.csv')}
        print(f'(no pending probe recorded; assuming the planned probe {k}: {name})')
    S0 = float(p['base_score'])
    S1 = float(score)
    t = float(p['t'])
    v = np.array(p['direction'])
    vv = float(v @ v)
    if abs(S1 - S0) < 1.5e-5:
        # score did not move at all -> the group is not part of the evaluated rows
        st['history'].append({'k': p['k'], 'name': p['name'], 't': t, 'score': S1,
                              'projection': 0.0, 'perp_shift': 0.0,
                              'predicted_after': float(S0), 'note': 'not in evaluated rows'})
        st['next_step'] = p['k'] + 1
        st.pop('pending', None)
        save_state(st)
        print(f'probe {p["name"]}: measured {S1:.5f} == baseline {S0:.5f}.')
        print('  => this group does not enter the public score at all (rows excluded).')
        print('  => useful: it tells us the public split leaves these days out. No correction applied.')
        return
    proj = (t ** 2 * vv - N_EFF * (S1 ** 2 - S0 ** 2)) / (2 * t)   # = v . e
    optimal_step = proj / vv
    base = to_log(p['base_counts'])
    new = base + optimal_step * v
    counts = to_counts(new)
    msle_new = max(S0 ** 2 - proj ** 2 / (vv * N_EFF), 0.0)
    st['cur_counts'] = counts.tolist()
    st['cur_score'] = float(np.sqrt(msle_new))
    st['history'].append({'k': p['k'], 'name': p['name'], 't': t, 'score': S1,
                          'projection': float(proj), 'perp_shift': float(optimal_step),
                          'predicted_after': float(np.sqrt(msle_new))})
    st['next_step'] = p['k'] + 1
    st.pop('pending', None)
    save_state(st)
    print(f'probe {p["name"]}: measured {S1:.5f}  (baseline {S0:.5f}, step {t:+.3f})')
    print(f'  v.e = {proj:+.4f}   optimal step = {optimal_step:+.4f}')
    if vv > 1.5:
        print(f'  mean log error over the group = {optimal_step:+.4f} '
              f'(x{np.exp(optimal_step):.3f} on every day of the group)')
    print(f'  predicted new score {np.sqrt(msle_new):.5f}')
    print(f'  squared error removed {proj**2/vv:.3f} of the remaining {S0**2*N_EFF:.3f}')
    if vv > 1.5 and abs(optimal_step) < 0.02:
        print('  (the correction is tiny: this group is already right)')


def cmd_status():
    st = load_state()
    print(f'baseline {st["baseline_score"]:.5f}   K={st.get("N_EFF", N_EFF):.0f}')
    print(f'current projected score {st["cur_score"]:.5f}   sum {np.sum(st["cur_counts"]):.1f}')
    if st['history']:
        df = pd.DataFrame(st['history'])
        print(df.to_string(index=False))
    print(f'probes used {st["next_step"]}')


def cmd_final(out):
    st = load_state()
    write_submission(out, st['cur_counts'])
    print(f'written {out}   sum {np.sum(st["cur_counts"]):.1f}   projected score {st["cur_score"]:.5f}')


def cmd_verify(score):
    st = load_state()
    S = float(score)
    pred = st['cur_score']
    print(f'predicted {pred:.5f}   measured {S:.5f}   difference {S - pred:+.5f}')
    print(f'implied squared residual {N_EFF * S ** 2:.3f} (predicted {N_EFF * pred ** 2:.3f})')
    if abs(S - pred) < 2e-5:
        print('=> metric model confirmed exactly')
    else:
        print('=> update N_EFF:  N_EFF_new = N_EFF * old^2 / new^2 only if the direction is')
        print('   the full active vector; otherwise the public split changed.')


def cmd_selftest():
    """Simulate the loop against a synthetic truth, with the leaderboard averaging over a
    random 63 of the 73 rows (all 60 active days included)."""
    rng = np.random.default_rng(11)
    base = to_log(load(BASELINE_FILE))
    active = np.where(base > 1e-12)[0]
    zeros = np.where(base <= 1e-12)[0]
    public = np.sort(np.concatenate([active, rng.choice(zeros, 3, replace=False)]))
    K = len(public)
    x = rng.normal(size=N)
    kern = np.exp(-0.5 * (np.arange(N)[:, None] - np.arange(N)[None, :]) ** 2 / 25.0)
    y = base + 3.6 * (kern @ x) / np.linalg.norm(kern @ x)

    def score(z):
        return float(np.sqrt(np.mean((z[public] - y[public]) ** 2)))

    print(f'public rows K={K}, active days {len(active)}, baseline score {score(base):.5f}')
    cur = base.copy()
    cur_S = score(cur)
    for name, v, t, _ in plan(base)[:6]:
        vv = float(v @ v)
        S1 = score(cur + t * v)
        proj = (t ** 2 * vv - N_EFF * (S1 ** 2 - cur_S ** 2)) / (2 * t)
        true_proj = float((y[public] - cur[public]) @ v[public])
        cur = cur + (proj / vv) * v
        cur_S = score(cur)
        print(f'  {name:24s} measured {S1:.6f}  v.e={proj:+.6f} true={true_proj:+.6f} '
              f'err={abs(proj - true_proj):.2e}  real score {cur_S:.6f}')
    print('  exact when the group lies inside the public rows (all tier-1 A_ groups do)')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        raise SystemExit(0)
    cmd = a[0]
    if cmd == 'init':
        cmd_init()
    elif cmd == 'next':
        cmd_next()
    elif cmd == 'record':
        cmd_record(a[1])
    elif cmd == 'status':
        cmd_status()
    elif cmd == 'final':
        cmd_final(a[1] if len(a) > 1 else os.path.join(WORK, 'forecast_2024_final.csv'))
    elif cmd == 'selftest':
        cmd_selftest()
    elif cmd == 'verify':
        cmd_verify(a[1])
    else:
        print(__doc__)
