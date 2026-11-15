#!/usr/bin/env python3
"""n-uniformity of the remainder for the finite-range resonant chain
(revision; numerical illustration of the uniform-remainder proposition).

Open chain, edges (i,i+1),(i,i+2), J > 0, optional uniform-amplitude staggered
detuning delta_a = d*(-1)^a.  T = n-2 triangles, Omega_3 = 3 J^3 T.
R = -ln<s> - (beta J)^3 T / 2.   Reports R/(n (beta J)^4).
Exact diagonalization sector by sector (excitation number conserved).
Writes repro_h3.json.
"""
import json, itertools
import numpy as np
from scipy.special import logsumexp

def sector_mats(n, k, edges, delta):
    states = [sum(1 << i for i in c) for c in itertools.combinations(range(n), k)]
    idx = {s: i for i, s in enumerate(states)}
    A = np.zeros((len(states), len(states)))
    D = np.zeros(len(states))
    for s, i in idx.items():
        D[i] = sum(delta[a] * (1 if (s >> a) & 1 else -1) for a in range(n))
        for a, b in edges:
            if ((s >> a) & 1) != ((s >> b) & 1):
                A[idx[s ^ (1 << a) ^ (1 << b)], i] = 1.0
    return A, D

def severity(n, bJ, bd):
    edges = [(i, i + 1) for i in range(n - 1)] + [(i, i + 2) for i in range(n - 2)]
    delta = [bd * (-1) ** a for a in range(n)]
    lz, lzs = [], []
    for k in range(n + 1):
        A, D = sector_mats(n, k, edges, delta)
        lz.append(logsumexp(-np.linalg.eigvalsh(bJ * A + np.diag(D))))
        lzs.append(logsumexp(-np.linalg.eigvalsh(-bJ * A + np.diag(D))))
    return logsumexp(lzs) - logsumexp(lz)

out = []
for bd in (0.0, 0.1):
    for bJ in (0.1, 0.2, 0.3):
        row = {"beta_delta": bd, "beta_J": bJ, "R_over_n_bJ4": {}, "rel_dev": {}}
        for n in (6, 8, 10, 12, 14):
            lead = 0.5 * bJ**3 * (n - 2)
            R = severity(n, bJ, bd) - lead
            row["R_over_n_bJ4"][str(n)] = R / (n * bJ**4)
            row["rel_dev"][str(n)] = R / lead
        out.append(row); print(row, flush=True)
json.dump(out, open("repro_h3.json", "w"), indent=1)
