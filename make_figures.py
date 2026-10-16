"""Regenerate Figs. 1, 3 and A1 from the reproduction JSON files only."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

c1, c2, c3, c4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
cyc = json.load(open('repro_cycle.json'))
misc = json.load(open('repro_misc.json'))
chain = json.load(open('repro_chain_16.json'))
slq = json.load(open('repro_slq_15_16_18.json'))
dav = json.load(open('repro_davies.json'))
summary = {}

# ---------------- Fig. 1 ----------------
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9))
a = ax[0]
pts = misc['fig1a']
mk = {'g=3': ('o', c1), 'g=4': ('s', c1), 'g=5': ('^', c1), 'g=6': ('v', c1),
      'complex': ('x', c2), 'resonant': ('D', c3)}
seen = set()
for p in pts:
    m, c = mk[p['kind']]
    lab = None
    key = 'cycles' if p['kind'].startswith('g=') else p['kind']
    if key not in seen:
        lab = {'cycles': 'cycles, $g=3$-$6$', 'complex': 'complex $H$',
               'resonant': 'resonant'}[key]
        seen.add(key)
    a.loglog(p['pred'], p['exact'], m, color=c, ms=4, mfc='none' if m != 'x' else c, label=lab)
lo, hi = min(p['pred'] for p in pts) / 3, max(p['pred'] for p in pts) * 3
a.loglog([lo, hi], [lo, hi], color='0.5', lw=0.8)
a.set_xlabel(r'$\beta^g\Omega_g/g!$'); a.set_ylabel(r'exact $-\ln\langle s\rangle$')
a.set_title('(a) cycle law', fontsize=10); a.legend(frameon=False, fontsize=8, loc='upper left', handletextpad=0.2)
ratios = [p['exact'] / p['pred'] for p in pts]
summary['fig1a'] = dict(npts=len(pts), median_ratio=float(np.median(ratios)),
                        min_ratio=float(min(ratios)), max_ratio=float(max(ratios)),
                        decades=float(np.log10(max(p['pred'] for p in pts) / min(p['pred'] for p in pts))))
b = ax[1]
sw = cyc['flux_sweep']
Ph = np.array([s['Phi'] for s in sw]); ex = np.array([s['exact'] for s in sw]); pr = np.array([s['pred'] for s in sw])
ff = np.linspace(0, np.pi, 200)
b.plot(ff / np.pi, 1e4 * pr[-1] * (1 - np.cos(ff)) / 2, color=c2, lw=1.4, label=r'$\beta^3\Omega_3/3!$')
b.plot(np.r_[0, Ph / np.pi], 1e4 * np.r_[0, ex], 'o', color=c1, ms=3.5, label='exact')
b.set_xlabel(r'flux $\Phi/\pi$'); b.set_ylabel(r'$-\ln\langle s\rangle$ ($10^{-4}$)')
b.set_title(r'(b) triangle, $\beta=0.08$', fontsize=10); b.legend(frameon=False, fontsize=8, loc='upper left')
summary['flux_sweep_ratio'] = [float(min(ex / pr)), float(max(ex / pr))]
c = ax[2]
e = misc['fig1c']['extensive']; d = misc['fig1c']['dilute']
c.semilogy([r['n'] for r in e], [r['rate'] for r in e], 'o-', color=c1, ms=3.5, lw=1.2, label='triangle chain (extensive)')
c.semilogy([r['n'] for r in e], [r['leading'] for r in e], '--', color=c1, lw=0.9, label=r'$\frac{1}{2}(\beta J)^3T/n$')
c.semilogy([r['n'] for r in d], [r['rate'] for r in d], 's-', color=c2, ms=3.5, lw=1.2, label='one frustrated $n$-cycle')
c.set_xlabel('$n$'); c.set_ylabel(r'$-\ln\langle s\rangle/n$')
c.set_title(r'(c) density, $\beta J=0.5$', fontsize=10); c.legend(frameon=False, fontsize=8, loc='lower left'); c.set_ylim(1e-19, 1)
fig.tight_layout(); fig.savefig('fig_cycle_law.pdf'); fig.savefig('fig_cycle_law.png', dpi=150)

# ---------------- Fig. 3: evidence chain ----------------
sgn = {int(k): v for k, v in chain['sign'].items()}
ns = np.array(sorted(sgn)); s = np.array([sgn[n] for n in ns])
rat = s[1:] / s[:-1]
fits = {}
for lo in (8, 9, 10, 11):
    m = ns >= lo
    A = np.vstack([ns[m], np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A, np.log(s[m]), rcond=None)
    fits[lo] = float(-coef[0])
rate_mid = float(np.mean(list(fits.values())))
rate_unc = float((max(fits.values()) - min(fits.values())) / 2)
summary['chain'] = dict(n=ns.tolist(), sign=s.tolist(), ratios=rat.tolist(), fits=fits,
                        rate=rate_mid, rate_halfspread=rate_unc,
                        ratio_ge10_mean=float(rat[ns[1:] >= 10].mean()),
                        ratio_ge10_sd=float(rat[ns[1:] >= 10].std(ddof=1)),
                        slq={k: v for k, v in slq.items()})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
a = ax[0]
nn = np.linspace(5.5, 18.5, 50)
n0 = 12; a.semilogy(nn, sgn[n0] * np.exp(-rate_mid * (nn - n0)), color='0.55', lw=0.9, label=fr'$e^{{-{rate_mid:.4f}\,n}}$')
a.semilogy(ns, s, 'o', color=c1, ms=4, label='exact (ED)')
sn = sorted(int(k) for k in slq)
a.errorbar(sn, [slq[str(k)]['sign'] for k in sn], yerr=[slq[str(k)]['err'] for k in sn], fmt='s', color=c2,
           ms=4, capsize=2, mfc='none', label='SLQ')
a.set_xlabel('$n$'); a.set_ylabel(r'$\langle s\rangle$'); a.set_title('(a) average sign', fontsize=10)
a.legend(frameon=False, fontsize=8)
b = ax[1]
b.plot(ns[1:], rat, 'o-', color=c1, ms=4, lw=1)
b.axhline(np.exp(-rate_mid), color='0.55', lw=0.9, ls='--')
b.axhspan(np.exp(-rate_mid - rate_unc), np.exp(-rate_mid + rate_unc), color='0.85', lw=0)
b.set_xlabel('$n$'); b.set_ylabel(r'$\langle s\rangle_n/\langle s\rangle_{n-1}$'); b.set_title('(b) successive ratios', fontsize=10)
fig.tight_layout(); fig.savefig('fig_evidence.pdf'); fig.savefig('fig_evidence.png', dpi=150)

# ---------------- Fig. A1: Davies gap ----------------
g = dav['glauber']; nd = sorted(int(k) for k in g); gv = [g[str(k)] for k in nd]
summary['davies'] = dict(n=nd, gap=gv, mean=float(np.mean(gv)))
fig, a = plt.subplots(figsize=(3.6, 2.5))
a.plot(nd, gv, 'o-', color=c1, ms=4, lw=1)
a.axhline(np.mean(gv), color='0.55', lw=0.9, ls='--')
a.set_ylim(0, 1.2); a.set_xlabel('$n$'); a.set_ylabel(r'Davies gap $\lambda_{\rm gap}$')
fig.tight_layout(); fig.savefig('fig_gap_sm.pdf'); fig.savefig('fig_gap_sm.png', dpi=150)
json.dump(summary, open('summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1)[:3000])
