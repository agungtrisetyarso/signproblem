"""Fig. 5: distance of the nearest complex-flux zero from the real axis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = json.load(open('repro_zeros.json'))
c1, c2 = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for lab, mk, name in (('ring', 'o', r'ring $L=10$, $N=5$'), ('torus', 's', r'$4\times4$ torus, $N=4$')):
    b = np.array([x['beta'] for x in r if x['label'] == lab])
    ex = np.array([x['theta0_im'] for x in r if x['label'] == lab])
    g = np.array([x['gauss_im'] for x in r if x['label'] == lab])
    ax.loglog(b, ex, mk, color=c1, ms=6, label=f'exact, {name}')
    ax.loglog(b, g, mk, color=c2, ms=6, mfc='none', mew=1.4, label=f'Hyp. 1, {name}')
bb = np.array([0.8, 20])
ax.loglog(bb, 7.84 / bb, color='0.55', lw=0.9, ls='--')
ax.text(10, 7.84 / 10 * 1.25, r'$\propto 1/\beta$', color='0.35', fontsize=9)
ax.set_xlabel(r'$\beta t$'); ax.set_ylabel(r'$\mathrm{Im}\,\theta^\ast$  (Re $\theta^\ast=\pi$)')
ax.legend(frameon=False, fontsize=7.5, loc='upper right')
fig.tight_layout(); fig.savefig('fig_zeros.pdf'); fig.savefig('fig_zeros.png', dpi=150)
