"""Davies population-chain gap for the resonant chain, for three standard
detailed-balance rate functions; tests which one reproduces the published
n=5 value 0.932467 and reports the full n=5..12 sequence for each."""
import json, sys
import numpy as np
from repro_slq import build

J, beta = 0.6, 2.0
rates = {
    'glauber': lambda w: 1.0 / (1.0 + np.exp(np.clip(beta * w, -700, 700))),
    'metropolis': lambda w: np.minimum(1.0, np.exp(np.clip(-beta * w, -700, 700))),
    'sqrt': lambda w: np.exp(np.clip(-beta * w / 2, -700, 700)),
}
nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 12
out = {k: {} for k in rates}
for n in range(5, nmax + 1):
    H = build(n, J).toarray()
    E, U = np.linalg.eigh(H)
    N = 1 << n
    x = np.arange(N)
    M2 = np.zeros((N, N))
    for a in range(n):
        Xa = U.T @ U[x ^ (1 << a)]          # <c|X_a|d> in eigenbasis (real)
        M2 += Xa ** 2
    np.fill_diagonal(M2, 0.0)
    p = np.exp(-beta * (E - E.min())); p /= p.sum()
    for name, g in rates.items():
        W = g(E[:, None] - E[None, :]) * M2     # W[c,d] = rate d -> c
        L = W - np.diag(W.sum(0))
        S = np.diag(p ** -0.5) @ L @ np.diag(p ** 0.5)
        S = (S + S.T) / 2
        ev = np.sort(np.linalg.eigvalsh(S))[::-1]
        out[name][n] = float(-ev[1])
    print(n, {k: round(v[n], 6) for k, v in out.items()}, flush=True)
json.dump(out, open('repro_davies.json', 'w'), indent=1)
