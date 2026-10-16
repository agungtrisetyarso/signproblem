"""Remaining reproductions: Corollary 2(ii) brute force, twist stiffness
cross-check by finite differences, and data for Fig. 1(a,c)."""
import json, itertools
import numpy as np
import mpmath as mp
from repro_cycle import neglog_sign_mp, omega
from repro_chain import sector_H, lse
import scipy.linalg as la

rng = np.random.default_rng(42)
out = {}

# ---- Corollary 2(ii): balance vs brute-force curability in G_loc ----
X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1]); I2 = np.eye(2)
def op(n, ops):
    M = np.array([[1.0]])
    for k in range(n):
        M = np.kron(M, ops.get(k, I2))
    return M
def balanced(n, E, sig):
    return any(all(sig[e] == s[a] * s[b] for e, (a, b) in enumerate(E))
               for s in itertools.product([-1, 1], repeat=n))
graphs = {3: [(0, 1), (1, 2), (0, 2)], 4: [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]}
agree = total = 0
for trial in range(48):
    n = 3 if trial % 2 == 0 else 4
    E = graphs[n]
    Jc = rng.choice([-1, 1], len(E)) * rng.uniform(0.5, 1.5, len(E))
    Kc = rng.choice([-1, 1], len(E)) * rng.uniform(0.5, 1.5, len(E))
    H = sum(Jc[e] * op(n, {a: Z, b: Z}) + Kc[e] * op(n, {a: X, b: X}) for e, (a, b) in enumerate(E))
    Had = op(n, {k: np.array([[1, 1], [1, -1]]) / np.sqrt(2) for k in range(n)})
    curable = False
    for useH in (False, True):
        H1 = Had @ H @ Had if useH else H
        for eps in itertools.product([0, 1], repeat=n):
            U = op(n, {k: Z for k in range(n) if eps[k]})
            H2 = U @ H1 @ U
            off = H2 - np.diag(np.diag(H2))
            if off.max() <= 1e-9:
                curable = True
    pred = balanced(n, E, list(-np.sign(Kc))) or balanced(n, E, list(-np.sign(Jc)))
    total += 1; agree += int(pred == curable)
out['cor2'] = dict(agree=agree, total=total)

# ---- twist: kappa_2 by finite differences of ln Z(theta) vs Duhamel ----
import importlib.util, sys
spec = importlib.util.spec_from_file_location('fig_twist_src', 'fig_twist.py')
src = open('fig_twist.py').read().split('data = {}')[0]
ns = {}; exec(src, ns)
build, stiff, lz = ns['build'], ns['stiff'], ns['lz']
L = 10
H = build(L, 5, [(i, (i + 1) % L, 1.0 / L) for i in range(L)])
fd = []
for beta in (2.0, 8.0):
    (Cd,), _ = stiff(H, [beta])
    h = 1e-3
    l0, lp, lm = (lz(la.eigvalsh(H(t)), beta) for t in (0.0, h, -h))
    Cf = -(lp - 2 * l0 + lm) / h ** 2
    fd.append(dict(beta=beta, duhamel=float(Cd), finite_diff=float(Cf)))
out['twist_kappa2_crosscheck'] = fd

# ---- Fig. 1(a): single g-cycles at three temperatures, 60 digits ----
pts = []
for g in (3, 4, 5, 6):
    Hc = np.zeros((g, g))
    for i in range(g):
        j = (i + 1) % g; Hc[i, j] = Hc[j, i] = -1.0
    Hc[0, g - 1] = Hc[g - 1, 0] = 1.0
    for beta in (0.01, 0.03, 0.1):
        pts.append(dict(kind=f'g={g}', beta=beta, pred=float(beta ** g * omega(Hc, g) / mp.factorial(g)),
                        exact=neglog_sign_mp(Hc, beta)))
for k in range(4):
    A = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)); Hc = (A + A.conj().T) / 2
    for beta in (0.003, 0.01, 0.03):
        pts.append(dict(kind='complex', beta=beta, pred=float(beta ** 3 * omega(Hc, 3) / 6),
                        exact=neglog_sign_mp(Hc, beta)))
for n in (6, 8, 10):
    T = n - 2
    for bJ in (0.02, 0.05, 0.1):
        lzv, lzs = [], []
        for k in range(n + 1):
            lzv.append(lse(-la.eigvalsh(sector_H(n, k, bJ, [0.0] * n, 1, False))))
            lzs.append(lse(-la.eigvalsh(sector_H(n, k, bJ, [0.0] * n, 1, True))))
        pts.append(dict(kind='resonant', beta=bJ, n=n, pred=0.5 * bJ ** 3 * T,
                        exact=float(lse(np.array(lzs)) - lse(np.array(lzv)))))
out['fig1a'] = pts

# ---- Fig. 1(c): extensive (triangle chain, delta=0) vs dilute (one frustrated n-cycle) ----
ext, dil = [], []
bJ = 0.5
for n in range(5, 13):
    lzv, lzs = [], []
    for k in range(n + 1):
        lzv.append(lse(-la.eigvalsh(sector_H(n, k, bJ, [0.0] * n, 1, False))))
        lzs.append(lse(-la.eigvalsh(sector_H(n, k, bJ, [0.0] * n, 1, True))))
    ext.append(dict(n=n, rate=float((lse(np.array(lzs)) - lse(np.array(lzv))) / n),
                    leading=0.5 * bJ ** 3 * (n - 2) / n))
    Hc = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n; Hc[i, j] = Hc[j, i] = -1.0
    Hc[0, n - 1] = Hc[n - 1, 0] = 1.0
    dil.append(dict(n=n, rate=neglog_sign_mp(Hc, bJ) / n))
out['fig1c'] = dict(extensive=ext, dilute=dil, beta=bJ)

json.dump(out, open('repro_misc.json', 'w'), indent=1)
print(out['cor2'], out['twist_kappa2_crosscheck'])
for r in ext: print(r)
for r in dil: print(r)
