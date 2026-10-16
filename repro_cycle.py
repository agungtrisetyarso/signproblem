"""Independent reproduction of the cycle-law checks (Theorem 1, Table A1, Fig. 1).

Fresh implementation, written without reference to the original scripts.
All quantities that are smaller than ~1e-10 are evaluated in 60-digit
arithmetic (mpmath), because -ln<s> is obtained from a ratio of traces that
agree to that many digits and double precision floors at ~1e-16 * N.
"""
import json, itertools
import numpy as np
import mpmath as mp

mp.mp.dps = 60
rng = np.random.default_rng(20261016)


def mp_matrix(A):
    return mp.matrix([[mp.mpc(complex(x)) for x in row] for row in A])


def neglog_sign_mp(H, beta):
    """-ln <s> = ln Tr e^{-beta Hs} - ln Tr e^{-beta H}, 60 digits."""
    H = np.asarray(H, complex)
    D = np.diag(np.diag(H)); V = H - D
    Hs = D - np.abs(V)
    Z = mp.re(sum(mp.expm(-beta * mp_matrix(H))[i, i] for i in range(H.shape[0])))
    Zs = mp.re(sum(mp.expm(-beta * mp_matrix(Hs))[i, i] for i in range(H.shape[0])))
    return float(mp.log(Zs) - mp.log(Z))


def neglog_sign(H, beta):
    H = np.asarray(H, complex)
    D = np.diag(np.diag(H)); V = H - D
    Hs = D - np.abs(V)
    def lz(M):
        e = np.linalg.eigvalsh(M); m = (-beta * e).max()
        return m + np.log(np.exp(-beta * e - m).sum())
    return lz(Hs) - lz(H)


def omega(H, m):
    H = np.asarray(H, complex)
    V = H - np.diag(np.diag(H)); N = H.shape[0]
    a = np.trace(np.linalg.matrix_power(np.abs(V), m)).real
    b = np.trace(np.linalg.matrix_power(-V, m)).real
    return (a - b) / N


def girth(H, mmax=12, tol=1e-9):
    for m in range(3, mmax + 1):
        if omega(H, m) > tol:
            return m
    return None


out = {}

# ---- (1) single frustrated g-cycle, |V|=1, one negative-sign edge, D=0 ----
beta = 0.01
rows = []
for g in (3, 4, 5, 6):
    H = np.zeros((g, g))
    for i in range(g):
        j = (i + 1) % g
        H[i, j] = H[j, i] = -1.0           # -V = +1, phase 0
    H[0, g - 1] = H[g - 1, 0] = +1.0       # one edge with phase pi
    pred = beta ** g * omega(H, g) / mp.factorial(g)
    ex = neglog_sign_mp(H, beta)
    exd = neglog_sign(H, beta)
    rows.append(dict(g=g, girth_detected=girth(H), Omega_g=omega(H, g), pred=float(pred),
                     exact_mp=ex, exact_double=float(exd), rel=(ex - float(pred)) / float(pred)))
out['single_cycles'] = rows

# ---- (2) random complex Hermitian, dim 7, beta=0.01, 4 instances ----
rows = []
for k in range(4):
    A = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    H = (A + A.conj().T) / 2
    g = girth(H)
    pred = beta ** g * omega(H, g) / float(mp.factorial(g))
    ex = neglog_sign_mp(H, beta)
    rows.append(dict(g=g, pred=pred, exact=ex, rel=(ex - pred) / pred))
out['random_complex'] = rows

# ---- (3) random sparse real, dim 7, beta=0.01 ----
rows = []
for k in range(4):
    H = np.zeros((7, 7))
    for i, j in itertools.combinations(range(7), 2):
        if rng.random() < 0.45:
            H[i, j] = H[j, i] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
    H += np.diag(rng.normal(size=7)) * 0.0
    g = girth(H)
    if g is None:
        continue
    pred = beta ** g * omega(H, g) / float(mp.factorial(g))
    ex = neglog_sign_mp(H, beta)
    rows.append(dict(g=g, pred=pred, exact=ex, rel=(ex - pred) / pred))
out['random_sparse_real'] = rows

# ---- (4) flux sweep on a triangle, uniform edge phase Phi/3 ----
def tri(Phi):
    H = np.zeros((3, 3), complex)
    for i in range(3):
        j = (i + 1) % 3
        H[i, j] = -np.exp(1j * Phi / 3)    # -V_ij = e^{i Phi/3}
        H[j, i] = np.conj(H[i, j])
    return H
sweep = []
for Phi in np.linspace(0, np.pi, 21)[1:]:
    H = tri(Phi)
    pred = 0.08 ** 3 * omega(H, 3) / 6
    ex = neglog_sign_mp(H, 0.08)
    sweep.append(dict(Phi=Phi, pred=pred, exact=ex, ratio=ex / pred))
out['flux_sweep'] = sweep
out['flux_zero'] = [dict(beta=b, sign=float(mp.exp(-neglog_sign_mp(tri(0.0), b)))) for b in (0.5, 2, 5)]

# ---- (5) gauge invariance, random complex H, N=6, random D, beta=1.5 ----
A = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
H = (A + A.conj().T) / 2
s0 = np.exp(-neglog_sign(H, 1.5)); dev = 0.0
for k in range(4):
    U = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, 6)))
    dev = max(dev, abs(np.exp(-neglog_sign(U @ H @ U.conj().T, 1.5)) - s0))
out['gauge_max_dev'] = dev

# ---- (6) flux-free graph with mixed signs and random D ----
u = rng.choice([-1, 1], size=8)
H = np.zeros((8, 8))
for i, j in itertools.combinations(range(8), 2):
    if rng.random() < 0.5:
        H[i, j] = H[j, i] = -abs(rng.normal()) * u[i] * u[j]   # balanced signs
H += np.diag(rng.normal(size=8))
out['flux_free'] = dict(n_negative_offdiag=int((H - np.diag(np.diag(H)) > 0).sum()),
                        max_dev=max(abs(1 - np.exp(-neglog_sign(H, b))) for b in (0.1, 0.5, 1, 2, 3)))

# ---- (7) Tr A^3 / 2^n = (3/2) T for random graphs ----
def hop_matrix(n, edges):
    N = 2 ** n; A = np.zeros((N, N))
    for x in range(N):
        for a, b in edges:
            if ((x >> a) & 1) != ((x >> b) & 1):
                A[x ^ (1 << a) ^ (1 << b), x] = 1.0
    return A
tra = []
while len(tra) < 9:
    n = int(rng.integers(5, 10))
    E = [e for e in itertools.combinations(range(n), 2) if rng.random() < 0.5]
    T = sum(1 for a, b, c in itertools.combinations(range(n), 3)
            if (a, b) in E and (b, c) in E and (a, c) in E)
    if T == 0:
        continue
    A = hop_matrix(n, E)
    tra.append(dict(n=n, T=T, trA3_over_2n=float(np.trace(A @ A @ A) / 2 ** n), pred=1.5 * T))
out['trA3'] = tra

json.dump(out, open('repro_cycle.json', 'w'), indent=1, default=float)
for k, v in out.items():
    print(k, json.dumps(v, default=float)[:900])
