import numpy as np, itertools, scipy.linalg as la, json, sys

def basis(nsite, npart):
    states = [s for s in itertools.combinations(range(nsite), npart)]
    idx = {s: i for i, s in enumerate(states)}
    return states, idx

def build(nsite, npart, bonds, t=1.0, mu=None):
    """bonds: list of (i,j,phase_weight) ; hopping -t e^{i theta w} b_i^dag b_j + h.c.
    returns function H(theta) and the bond-structure."""
    states, idx = basis(nsite, npart)
    N = len(states)
    rows, cols, w = [], [], []
    for a, s in enumerate(states):
        occ = set(s)
        for (i, j, wt) in bonds:
            # b_i^dag b_j : move particle j -> i
            for (p, q, sg) in ((i, j, +1), (j, i, -1)):
                if q in occ and p not in occ:
                    ns = tuple(sorted((occ - {q}) | {p}))
                    rows.append(idx[ns]); cols.append(a); w.append(sg * wt)
    rows = np.array(rows); cols = np.array(cols); w = np.array(w, float)
    diag = np.zeros(N)
    if mu is not None:
        for a, s in enumerate(states):
            diag[a] = sum(mu[k] for k in s)
    def H(theta):
        M = np.zeros((N, N), complex)
        np.add.at(M, (rows, cols), -t * np.exp(1j * theta * w))
        return M + np.diag(diag)
    return H

def lnZ(M, beta):
    e = la.eigvalsh(M)
    m = (-beta * e).max()
    return m + np.log(np.exp(-beta * e - m).sum())

def avg_sign(M, beta):
    D = np.diag(np.diag(M)); V = M - D
    Hs = D - np.abs(V)
    return np.exp(lnZ(M, beta) - lnZ(Hs, beta))

def kubo_C(H, beta, h=1e-4):
    """C = -d^2 lnZ/dtheta^2 at 0 via Duhamel (exact), using J=dH/dth, K=d2H/dth2."""
    H0 = H(0.0)
    J = (H(h) - H(-h)) / (2 * h)
    K = (H(h) - 2 * H0 + H(-h)) / h**2
    e, U = la.eigh(H0)
    p = np.exp(-beta * (e - e.min())); Z = p.sum(); p /= Z
    Jt = U.conj().T @ J @ U; Kt = U.conj().T @ K @ U
    avgK = np.real(np.sum(p * np.diag(Kt)))
    de = e[:, None] - e[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        F = np.where(np.abs(de) > 1e-10, (p[None, :] - p[:, None]) / de, beta * p[:, None])
    duh = np.real(np.sum(F * np.abs(Jt) ** 2))   # int_0^beta <J(tau) J>/...
    # d2 lnZ/dth2 = -beta<K> + duh - beta^2<J>^2 ; <J>=0 by reversal symmetry
    return beta * avgK - beta * duh

out = {}
# ---- 1D ring, L=10, N=5, twist theta spread uniformly ----
L, Np = 10, 5
bonds = [(i, (i + 1) % L, 1.0 / L) for i in range(L)]
H = build(L, Np, bonds)
res1 = []
for beta in (0.5, 2.0, 8.0):
    C = kubo_C(H, beta)
    for th in (0.05, 0.2, np.pi / 2, np.pi):
        s = avg_sign(H(th), beta)
        zr = np.exp(lnZ(H(th), beta) - lnZ(H(0), beta))
        res1.append(dict(beta=beta, theta=th, sign=s.real, Zratio=zr.real,
                         pred_quad=float(np.exp(-th**2 * C / 2)), C=C, vill=float(sum(np.exp(-C*(th-2*np.pi*k)**2/2) for k in range(-6,7))/sum(np.exp(-C*(2*np.pi*k)**2/2) for k in range(-6,7)))))
    # evenness check
    sp = avg_sign(H(0.3), beta); sm = avg_sign(H(-0.3), beta)
    res1[-1]['even_diff'] = float(abs(sp - sm))
out['ring'] = res1

# ---- 2D 4x4 torus, half filling, twist along x ----
Lx = Ly = 4
site = lambda x, y: x + Lx * y
b2 = []
for x in range(Lx):
    for y in range(Ly):
        b2.append((site((x + 1) % Lx, y), site(x, y), 1.0 / Lx))  # x-bond carries twist
        b2.append((site(x, (y + 1) % Ly), site(x, y), 0.0))
H2 = build(16, 4, b2)
H0 = H2(0.0); Hpi = H2(np.pi)
e0 = la.eigvalsh(H0); epi = la.eigvalsh(Hpi)
# also need stoq reference of Hpi equals H0 ; check:
Dg = np.diag(np.diag(Hpi)); Vg = Hpi - Dg
ref_err = float(np.abs((Dg - np.abs(Vg)) - H0).max())
def lz(e, b):
    m = (-b * e).max(); return m + np.log(np.exp(-b * e - m).sum())
res2 = []
for T in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.3, 2.0, 4.0):
    b = 1 / T
    s = np.exp(lz(epi, b) - lz(e0, b))
    C = kubo_C(H2, b)
    m = np.arange(-6, 7)
    vill = np.sum(np.exp(-C * (np.pi - 2 * np.pi * m) ** 2 / 2)) / np.sum(np.exp(-C * (2 * np.pi * m) ** 2 / 2))
    res2.append(dict(T=T, sign_pi=float(s), W2=float(C), villain=float(vill)))
out['torus4x4'] = res2
out['ref_err'] = ref_err
# universal value
m = np.arange(-6, 7); sig2 = 2 / np.pi
out['universal_BKT'] = float(np.sum(np.exp(-sig2 * (np.pi - 2 * np.pi * m) ** 2 / 2)) / np.sum(np.exp(-sig2 * (2 * np.pi * m) ** 2 / 2)))
json.dump(out, open('twist.json', 'w'), indent=1)
for k in res1: print(k)
for k in res2: print(k)
print('ref_err', ref_err, 'universal', out['universal_BKT'])
