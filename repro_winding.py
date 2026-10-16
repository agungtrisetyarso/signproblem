"""Exact winding-number distributions of twisted hard-core bosons versus the
discrete Gaussian of Hypothesis 1.

P(W) is obtained exactly by Fourier inversion of Proposition 3(i):
    <s>(theta) = Z(theta)/Z(0) = sum_W P(W) e^{i theta W},
    P(W) = (1/M) sum_k Z(theta_k)/Z(0) e^{-i theta_k W},  theta_k = 2 pi k / M,
which is exact up to aliasing from |W| >= M/2 (M = 64 here; P(32) is negligible).
"""
import json, itertools
import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq

src = open('fig_twist.py').read().split('data = {}')[0]
ns = {}; exec(src, ns)
build, stiff, lz = ns['build'], ns['stiff'], ns['lz']

M = 64
thetas = 2 * np.pi * np.arange(M) / M
Wmax = 6


def exact_PW(H, betas):
    e0 = la.eigvalsh(H(0.0))
    specs = [la.eigvalsh(H(t)) for t in thetas]
    out = {}
    for b in betas:
        r = np.array([np.exp(lz(e, b) - lz(e0, b)) for e in specs])
        P = {W: float(np.real(np.mean(r * np.exp(-1j * thetas * W)))) for W in range(-Wmax, Wmax + 1)}
        out[b] = P
    return out


def dgauss(s2):
    Ws = np.arange(-60, 61)
    p = np.exp(-Ws ** 2 / (2 * s2)); p /= p.sum()
    return dict(zip(Ws.tolist(), p.tolist()))


def summarize(P, kappa2):
    Ws = range(-Wmax, Wmax + 1)
    var = sum(W * W * P[W] for W in Ws)
    sign = sum(((-1) ** abs(W)) * P[W] for W in Ws)
    G = dgauss(kappa2)                                     # Hypothesis 1 with sigma^2 = kappa2
    s2m = brentq(lambda s: sum(W * W * p for W, p in dgauss(s).items()) - var, 1e-4, 50)
    Gm = dgauss(s2m)                                       # moment-matched discrete Gaussian
    tv = 0.5 * sum(abs(P[W] - G.get(W, 0)) for W in Ws)
    tvm = 0.5 * sum(abs(P[W] - Gm.get(W, 0)) for W in Ws)
    return dict(P={W: P[W] for W in range(0, 4)}, G={W: G[W] for W in range(0, 4)},
                Gm={W: Gm[W] for W in range(0, 4)}, var_exact=var, kappa2=kappa2,
                min_P=min(P.values()), sum_P=sum(P.values()),
                sign_exact=sign, sign_G=sum(((-1) ** abs(W)) * p for W, p in G.items()),
                sign_Gm=sum(((-1) ** abs(W)) * p for W, p in Gm.items()),
                sigma2_matched=s2m, TV=tv, TV_matched=tvm)


out = {}
L = 10
Hr = build(L, 5, [(i, (i + 1) % L, 1.0 / L) for i in range(L)])
betas_r = [2.0, 8.0]
Pr = exact_PW(Hr, betas_r)
Cr, _ = stiff(Hr, betas_r)
for b, C in zip(betas_r, Cr):
    out[f'ring_beta{b:g}'] = summarize(Pr[b], float(C))

site = lambda x, y: x + 4 * y
b2 = []
for x in range(4):
    for y in range(4):
        b2 += [(site((x + 1) % 4, y), site(x, y), 0.25), (site(x, (y + 1) % 4), site(x, y), 0.0)]
Ht = build(16, 4, b2)
Ts = [0.5, 1.0, 2.0]
Pt = exact_PW(Ht, [1 / T for T in Ts])
Ct, _ = stiff(Ht, [1 / T for T in Ts])
for T, C in zip(Ts, Ct):
    out[f'torus_T{T:g}'] = summarize(Pt[1 / T], float(C))

# the discrete Gaussian at the Nelson-Kosterlitz parameter
G = dgauss(2 / np.pi)
out['NK_parameter'] = dict(sigma2=2 / np.pi, variance=sum(W * W * p for W, p in G.items()),
                          sign_pi=sum(((-1) ** abs(W)) * p for W, p in G.items()),
                          two_exp_minus_pi=2 * np.exp(-np.pi))
json.dump(out, open('repro_winding.json', 'w'), indent=1)
for k, v in out.items():
    print(k, json.dumps(v, default=float)[:600])
