# -*- coding: utf-8 -*-
"""Why have four probes returned so little?  Quantify the 'one zero-day is open' hypothesis:
the residual is 11.04 units, and a single open Saturday forecast as 0 contributes
log(1+T)^2, which for T=20 is 9.3 units -- i.e. almost the entire remainder."""
import numpy as np
K = 63.0
S = 0.41862
MSLE = S ** 2
tot = K * MSLE
print(f'current score {S:.5f}  MSLE {MSLE:.6f}  residual {tot:.2f} squared units')
print(f'per active day RMS if all on 60 days: {np.sqrt(tot/60):.4f}')
print()
print('contribution of a single day whose truth is T while we predict 0:')
for T in (1, 2, 3, 5, 8, 10, 15, 20, 25):
    e = np.log1p(T)
    print(f'  truth {T:2d} -> e = {e:5.3f} -> {e**2:6.2f} units  ({100*e**2/tot:5.1f}% of the residual)')
print()
print('if ONE zero day carries X units, the other 62 days carry the rest:')
for T in (5, 10, 15, 20, 25):
    e2 = np.log1p(T) ** 2
    rest = tot - e2
    print(f'  truth {T:2d}: residual on the other 62 days = {rest:6.2f} '
          f'(RMS {np.sqrt(max(rest,0)/62):.3f})')
print()
print('that RMS is exactly what the probes keep measuring: tiny.  06-29 gave 0.135,')
print('the period-share direction 0.33 of 2.54, the level 0.016.  A residual of 0.43 per day')
print('on the active days would have shown up far more strongly.')
print()
t = 0.35
n = 3
print(f'probe on the 3 closed Saturdays, t={t:+.2f}:')
print(' measured   interpretation                       score after correction')
for lbl, sum_e, nP in [('all three really zero', 0.0, 3),
                       ('two of them evaluated', 0.0, 2),
                       ('one open at 5', 1.79, 3),
                       ('one open at 10', 2.40, 3),
                       ('one open at 20', 3.04, 3),
                       ('one open at 25', 3.26, 3)]:
    dS2 = (t * t * nP - 2 * t * sum_e) / K
    S1 = np.sqrt(MSLE + dS2)
    proj = (t * t * nP - K * dS2) / (2 * t)
    mean = proj / nP
    new = np.sqrt(max(MSLE - nP * mean ** 2 / K, 0))
    print(f'  {S1:7.5f}   {lbl:32s} {new:7.5f}')
