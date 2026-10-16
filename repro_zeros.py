"""Complex-flux zeros of Z(theta) for twisted hard-core bosons.

By Proposition 3(i), Z(theta)/Z(0) = sum_W P(W) z^W with z = e^{i theta}, a
Laurent polynomial in z with real, symmetric, non-negative coefficients.  Its
zeros set the radius of convergence of the small-flux series of -ln<s>.
P(W) is obtained exactly by Fourier inversion on M twists; roots of the
truncated Laurent polynomial are then validated by evaluating Z(theta) directly
at the complex twist (non-Hermitian H(theta), trace of the matrix exponential).
Hypothesis 1 predicts the nearest zero at theta = pi + i/(2 sigma^2).
"""
import json
import numpy as np
import scipy.linalg as la

src = open('fig_twist.py').read().split('data = {}')[0]
ns = {}; exec(src, ns)
build, stiff, lz, matched_sigma2 = ns['build'], ns['stiff'], ns['lz'], ns['matched_sigma2']

M = 128
thetas = 2 * np.pi * np.arange(M) / M


def zeros_for(H, beta, label):
    e0 = la.eigvalsh(H(0.0))
    r = np.array([np.exp(lz(la.eigvalsh(H(t)), beta) - lz(e0, beta)) for t in thetas])
    P = np.real(np.fft.ifft(r))                      # P[W] for W = 0..M-1 (negative W wrap)
    Wmax = max(w for w in range(1, M // 2) if P[w] > 1e-12)
    coeff = np.array([P[w % M] for w in range(-Wmax, Wmax + 1)])   # z^{-Wmax} .. z^{Wmax}
    roots = np.roots(coeff[::-1])                    # polynomial in z of degree 2 Wmax
    th = -1j * np.log(roots.astype(complex))         # theta = -i ln z, Re in (-pi, pi]
    th = np.where(th.real < 0, th + 2 * np.pi, th)   # map to Re in [0, 2pi)
    k = np.argmin(np.abs(th.imag) + 10 * (np.abs(th.real - np.pi) > 0.5))
    t0 = th[k]
    # direct validation at the complex twist
    Z0 = np.exp(lz(e0, beta))
    Zc = np.trace(la.expm(-beta * H(t0)))
    Zr = np.trace(la.expm(-beta * H(t0.real)))
    on_neg_axis = bool(np.max(np.abs(np.angle(-roots[np.abs(roots) < 1e6])) ) < 1e-6)
    kappa2 = float(stiff(H, [beta])[0][0])
    s2 = matched_sigma2(kappa2)
    out = dict(label=label, beta=beta, Wmax=int(Wmax), theta0_re=float(t0.real), theta0_im=float(abs(t0.imag)),
               radius=float(abs(t0)), check_absZ_ratio=float(abs(Zc) / abs(Zr)),
               all_roots_on_negative_real_z_axis=on_neg_axis, kappa2=kappa2,
               gauss_im=float(1 / (2 * s2)), gauss_radius=float(np.hypot(np.pi, 1 / (2 * s2))),
               roots_theta=[[float(x.real), float(x.imag)] for x in th if abs(x.imag) < 8])
    print(label, beta, f"theta*={t0.real:.6f}+{abs(t0.imag):.4f}i |Z(theta*)|/|Z(Re)|={out['check_absZ_ratio']:.1e}",
          f"neg-axis={on_neg_axis} gauss Im={out['gauss_im']:.4f}", flush=True)
    return out


res = []
L = 10
Hr = build(L, 5, [(i, (i + 1) % L, 1.0 / L) for i in range(L)])
for b in (1.0, 2.0, 4.0, 8.0, 16.0):
    res.append(zeros_for(Hr, b, 'ring'))
site = lambda x, y: x + 4 * y
b2 = []
for x in range(4):
    for y in range(4):
        b2 += [(site((x + 1) % 4, y), site(x, y), 0.25), (site(x, (y + 1) % 4), site(x, y), 0.0)]
Ht = build(16, 4, b2)
for T in (2.0, 1.0, 0.5):
    res.append(zeros_for(Ht, 1 / T, 'torus'))
json.dump(res, open('repro_zeros.json', 'w'), indent=1)
