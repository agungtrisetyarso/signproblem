import numpy as np, itertools, scipy.linalg as la, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build(nsite, npart, bonds, t=1.0):
    states = list(itertools.combinations(range(nsite), npart))
    idx = {s: i for i, s in enumerate(states)}
    N = len(states); rows, cols, w = [], [], []
    for a, s in enumerate(states):
        occ = set(s)
        for (i, j, wt) in bonds:
            for (p, q, sg) in ((i, j, 1), (j, i, -1)):
                if q in occ and p not in occ:
                    rows.append(idx[tuple(sorted((occ - {q}) | {p}))]); cols.append(a); w.append(sg * wt)
    rows, cols, w = np.array(rows), np.array(cols), np.array(w, float)
    def H(th):
        M = np.zeros((N, N), complex); np.add.at(M, (rows, cols), -t * np.exp(1j * th * w)); return M
    return H

def lz(e, b):
    m = (-b * e).max(); return m + np.log(np.exp(-b * e - m).sum())

def stiff(H, betas, h=1e-4):
    H0 = H(0.0); J = (H(h) - H(-h)) / (2 * h); K = (H(h) - 2 * H0 + H(-h)) / h**2
    e, U = la.eigh(H0); Jt = np.abs(U.conj().T @ J @ U) ** 2; Kd = np.real(np.diag(U.conj().T @ K @ U))
    de = e[:, None] - e[None, :]; out = []
    for b in betas:
        p = np.exp(-b * (e - e.min())); p /= p.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            F = np.where(np.abs(de) > 1e-10, (p[None, :] - p[:, None]) / de, b * p[:, None])
        out.append(b * np.sum(p * Kd) - b * np.sum(F * Jt))
    return np.array(out), e

_Wg = np.arange(-60, 61)
def _dg(s2):
    p = np.exp(-_Wg ** 2 / (2 * s2)); return p / p.sum()
def matched_sigma2(C):
    """Parameter of the discrete Gaussian whose variance equals C (Hypothesis 1)."""
    from scipy.optimize import brentq
    return brentq(lambda s: (_Wg ** 2 * _dg(s)).sum() - C, 1e-4, 60)
def villain(C, th):
    """<s>(theta) under Hypothesis 1: discrete Gaussian winding with variance C."""
    return float((_dg(matched_sigma2(C)) * np.cos(th * _Wg)).sum())

data = {}
# (a) ring L=10, N=5, beta=8: theta sweep
L, Np, beta = 10, 5, 8.0
H = build(L, Np, [(i, (i + 1) % L, 1.0 / L) for i in range(L)])
(C1,), e0 = stiff(H, [beta])
ths = np.linspace(0, 2 * np.pi, 41)
exact_a = [np.exp(lz(la.eigvalsh(H(t)), beta) - lz(e0, beta)) for t in ths]
data['ring'] = dict(beta=beta, C=float(C1), theta=ths.tolist(), sign=[float(x) for x in exact_a])
# (b) 4x4 torus N=4, theta=pi, T sweep
site = lambda x, y: x + 4 * y
b2 = []
for x in range(4):
    for y in range(4):
        b2 += [(site((x + 1) % 4, y), site(x, y), 0.25), (site(x, (y + 1) % 4), site(x, y), 0.0)]
H2 = build(16, 4, b2)
Ts = np.linspace(0.25, 3.0, 23)
C2, e0b = stiff(H2, 1 / Ts)
epi = la.eigvalsh(H2(np.pi))
exact_b = np.array([np.exp(lz(epi, 1 / T) - lz(e0b, 1 / T)) for T in Ts])
data['torus'] = dict(T=Ts.tolist(), W2=C2.tolist(), sign=exact_b.tolist())
json.dump(data, open('fig_twist.json', 'w'), indent=1)

c1, c2, c3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
a = ax[0]
fine = np.linspace(0, 2 * np.pi, 400)
a.plot(fine / np.pi, np.exp(-fine**2 * C1 / 2), color=c2, lw=1.6, ls='--', label=r'small flux, $e^{-\theta^2\kappa_2/2}$')
a.plot(fine / np.pi, [villain(C1, t) for t in fine], color=c3, lw=1.6, label='Hyp. 1 (Gaussian winding)')
a.plot(ths / np.pi, exact_a, 'o', ms=4, color=c1, label='exact')
a.set_xlabel(r'twist $\theta/\pi$'); a.set_ylabel(r'$\langle s\rangle$')
a.set_title(r'(a) ring $L=10$, $N=5$, $\beta t=8$', fontsize=10)
a.grid(alpha=0.25, lw=0.5); a.legend(frameon=False, fontsize=8)
b = ax[1]
Tf = Ts
b.plot(Tf, [villain(c, np.pi) for c in C2], color=c3, lw=1.6, label='Hyp. 1 (Gaussian winding)')
msk = C2 < 0.3
b.plot(Tf[msk], (1 - 2 * C2)[msk], color=c2, lw=1.6, ls='--', label=r'exact bound $1-2\kappa_2$')
b.plot(Ts, exact_b, 'o', ms=4, color=c1, label=r'exact, $\theta=\pi$')
b.axhline(0.0864, color='0.45', lw=0.8, ls=':')
b.text(2.95, 0.105, r'0.086: Hyp. 1 at $T_c$, $L\to\infty$ (conditional)', ha='right', fontsize=8, color='0.3')
b.set_ylim(-0.02, 1.05); b.set_xlabel(r'$T/t$'); b.set_ylabel(r'$\langle s\rangle$')
b.set_title(r'(b) $4\times4$ torus, $N=4$, antiperiodic', fontsize=10)
b.grid(alpha=0.25, lw=0.5); b.legend(frameon=False, fontsize=8, loc='center right', bbox_to_anchor=(1.0, 0.42))
fig.tight_layout()
fig.savefig('fig_twist.pdf'); fig.savefig('fig_twist.png', dpi=150)
print('C1', C1)
for T, s, c in zip(Ts, exact_b, C2): print(f'{T:.3f} {s:.5f} {c:.5f} {villain(c,np.pi):.5f} {1-2*c:.5f}')

print('ring matched: pi/2', villain(C1, np.pi/2), 'pi', villain(C1, np.pi))
for T in (0.5, 1.0, 2.0):
    i = int(np.argmin(abs(Ts - T))); print('torus T', Ts[i], 'exact', exact_b[i], 'hyp1', villain(C2[i], np.pi))
