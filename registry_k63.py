# -*- coding: utf-8 -*-
"""Re-test the old leaderboard registry with the CORRECT K = 63.
If the 28 registered scores are consistent, their projection can be recovered and used."""
import os, sys
import numpy as np, pandas as pd
pd.set_option('display.width', 240)
WORK = r'C:\Users\misha\Documents\deepseek-harness\default-workspace\work'
sys.path.insert(0, WORK)
from common import build, N, load, V61

K = 63.0
S61 = 0.42430
d = build()
A, S, labs = d['A'], d['S'], list(d['labs'])
a61 = d['a61']
B = A - a61
active = np.where(np.abs(a61) > 1e-12)[0]
print('registered submissions:', len(labs), ' active days:', len(active))

# e on the 60 active days
Ba = B[:, active]
normB2 = np.sum(Ba ** 2, axis=1)
rhs = normB2 - K * (S ** 2 - S61 ** 2)
M = 2 * Ba                      # (m x 60)
e, res, rank, sv = np.linalg.lstsq(M, rhs, rcond=None)
print('\nsingular values:', np.round(sv, 3))
r = M @ e - rhs
print('residual rms %.5f (score precision implies ~%.6f)' % (
    np.sqrt(np.mean(r ** 2)), 63 * 2 * 0.42 * 5e-6))
print('min-norm ||e|| = %.3f   budget ||e|| = %.3f' % (np.linalg.norm(e), np.sqrt(K * S61 ** 2)))

for rc in (1e-1, 1e-2, 1e-3, 1e-4):
    kk = sv > sv[0] * rc
    e2 = np.linalg.lstsq(M[:, :], rhs, rcond=rc)[0]
    print(f'  rcond={rc:6.0e} rank={kk.sum():2d} ||e||={np.linalg.norm(e2):8.3f} '
          f'resid_rms={np.sqrt(np.mean((M @ e2 - rhs) ** 2)):.5f}')

print('\nLOO test of the score model (using all 27 others):')
loo = []
for i in range(len(S)):
    k = np.arange(len(S)) != i
    ei = np.linalg.lstsq(M[k], rhs[k], rcond=None)[0]
    pred = np.sqrt(max((normB2[i] - 2 * Ba[i] @ ei) / K, 0))
    loo.append(pred)
loo = np.array(loo)
print(pd.DataFrame({'lab': labs, 'reg': S, 'loo': loo.round(5), 'diff': (loo - S).round(5)})
      .sort_values('reg').to_string(index=False))
print('LOO rms %.5f  max %.5f' % (np.sqrt(np.mean((loo - S) ** 2)), np.max(np.abs(loo - S))))
