"""Fig. 3 (new): exact winding distributions vs variance-matched discrete Gaussians."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open('repro_winding.json'))
c1, c2 = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def dg(s2, Ws):
    W = np.arange(-60, 61); p = np.exp(-W ** 2 / (2 * s2)); p /= p.sum()
    return np.array([p[W == w][0] for w in Ws])


cases = [('ring_beta8', r'(a) ring, $\beta t=8$'), ('torus_T1', r'(b) $4\times4$, $T/t=1$'),
         ('torus_T0.5', r'(c) $4\times4$, $T/t=0.5$')]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for a, (key, title) in zip(ax, cases):
    v = d[key]
    Ws = np.arange(0, 4)
    Pex = np.array([v['P'][str(w)] for w in Ws])
    Pg = dg(v['sigma2_matched'], Ws)
    a.bar(Ws - 0.18, Pex, width=0.34, color=c1, label='exact')
    a.bar(Ws + 0.18, Pg, width=0.34, color=c2, label='Gaussian, same variance')
    a.set_yscale('log'); a.set_ylim(1e-5, 60)
    a.set_yticks([1e-5, 1e-3, 1e-1, 1]); a.set_xticks(Ws); a.set_xlabel(r'$|W|$'); a.set_title(title, fontsize=10)
    a.text(0.97, 0.97, f"TV = {v['TV_matched']:.4f}\n" + r"$\langle s\rangle(\pi)$: " +
           f"{v['sign_exact']:.3f} vs {v['sign_Gm']:.3f}", transform=a.transAxes,
           ha='right', va='top', fontsize=8)
ax[0].set_ylabel(r'$P(W)$')
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, ncol=2, loc='upper center')
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig('fig_winding.pdf'); fig.savefig('fig_winding.png', dpi=150)
