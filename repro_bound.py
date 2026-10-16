"""Test of the explicit remainder bound of Theorem 1(b):
   |-ln<s> - beta^g Omega_g / g!| <= 3.1 (beta a)^{g+1} / g!   for 0 < beta a <= 1/2,
   a = || |V| || + max_c D_cc - min_c D_cc.
Random real and complex instances with nonzero diagonal, girths 3..6, 60 digits."""
import json, itertools
import numpy as np
import mpmath as mp
from repro_cycle import neglog_sign_mp, omega, girth

rng = np.random.default_rng(2026)
worst, rows = 0.0, []

def instances():
    # dense complex, sparse real, and single frustrated cycles with a diagonal
    for k in range(6):
        N = int(rng.integers(4, 8))
        A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
        H = (A + A.conj().T) / 2
        yield 'complex', H
    for k in range(6):
        N = 7; H = np.zeros((N, N))
        for i, j in itertools.combinations(range(N), 2):
            if rng.random() < 0.45:
                H[i, j] = H[j, i] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        H += np.diag(rng.normal(size=N))
        yield 'sparse', H
    for g in (3, 4, 5, 6):
        H = np.zeros((g, g))
        for i in range(g):
            j = (i + 1) % g; H[i, j] = H[j, i] = -rng.uniform(0.5, 1.5)
        H[0, g - 1] = H[g - 1, 0] = rng.uniform(0.5, 1.5)
        H += np.diag(rng.normal(size=g))
        yield f'cycle{g}', H

for kind, H in instances():
    g = girth(H)
    if g is None:
        continue
    D = np.diag(H).real; V = H - np.diag(D)
    a = np.linalg.norm(np.abs(V), 2) + D.max() - D.min()
    Om = omega(H, g)
    for ba in (0.5, 0.3, 0.1, 0.03):
        beta = float(ba / a)
        ex = neglog_sign_mp(H, beta)
        R = ex - beta ** g * Om / float(mp.factorial(g))
        scale = ba ** (g + 1) / float(mp.factorial(g))
        ratio = abs(R) / scale
        worst = max(worst, ratio)
        rows.append(dict(kind=kind, g=g, beta_a=ba, R=R, bound=3.1 * scale, ratio=ratio))
json.dump(dict(rows=rows, worst_ratio=worst), open('repro_bound.json', 'w'), indent=1)
print('instances x betas:', len(rows), ' max |R| / [(beta a)^{g+1}/g!] =', worst, '(bound 3.1)')
