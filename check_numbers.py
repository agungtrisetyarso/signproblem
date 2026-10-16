"""Verify that the derived numbers quoted in the manuscript match the JSON outputs.

Usage: python3 check_numbers.py ../signproblem_PhysicaA.tex
Exits with status 1 and lists every mismatch if any quoted number is absent.
"""
import json, math, sys

tex = open(sys.argv[1]).read()
S = json.load(open('summary.json'))
W = json.load(open('repro_winding.json'))
B = json.load(open('repro_bound.json'))
expect = {}

c = S['chain']; sign = dict(zip(c['n'], c['sign'])); slq = c['slq']
expect['chain rate'] = f"{c['rate']:.5f}"
expect['signed-estimator cost'] = f"{2 * c['rate']:.3f}"
expect['SLQ n=18'] = f"{slq['18']['sign']:.5f}"
expect['SLQ n=18 error'] = f"{slq['18']['err']:.5f}"
expect['prediction n=18'] = f"{sign[max(c['n'])] * math.exp(-c['rate'] * (18 - max(c['n']))):.5f}"
for n, v in sign.items():
    expect[f'chain n={n}'] = f"{v:.7f}"
for k in ('15', '16'):
    expect[f'SLQ deviation n={k}'] = f"{abs(slq[k]['sign'] - sign[int(k)]) / slq[k]['err']:.1f}\\sigma"
for g in S['davies']['gap']:
    expect[f'Davies gap {g:.3f}'] = f"{g:.3f}"
expect['Davies mean'] = f"{S['davies']['mean']:.3f}"
expect['Fig.1 points'] = f"${S['fig1a']['npts']}$"
expect['Fig.1 decades'] = f"{S['fig1a']['decades']:.1f}"
expect['Fig.1 median'] = f"{S['fig1a']['median_ratio']:.4f}"
expect['Fig.1 min ratio'] = f"{S['fig1a']['min_ratio']:.4f}"
expect['Fig.1 max ratio'] = f"{S['fig1a']['max_ratio']:.4f}"
expect['bound worst'] = f"{B['worst_ratio']:.3f}"
expect['bound cases'] = f"{len(B['rows'])} random"
for key, label in (('ring_beta8', 'ring 8'), ('torus_T1', 'torus 1'), ('torus_T0.5', 'torus 0.5')):
    v = W[key]
    expect[f'{label} sign exact'] = f"{v['sign_exact']:.5f}"
    expect[f'{label} sign Gauss'] = f"{v['sign_Gm']:.5f}"
    expect[f'{label} P(0) exact'] = f"{v['P']['0']:.5f}"
    expect[f'{label} P(1) exact'] = f"{v['P']['1']:.5f}"
    expect[f'{label} TV'] = f"{v['TV_matched']:.4f}"
expect['BKT value'] = '0.08638'
Zr = json.load(open('repro_zeros.json'))
for x in Zr:
    expect[f"zero {x['label']} beta={x['beta']} exact"] = f"{x['theta0_im']:.4f}"
    expect[f"zero {x['label']} beta={x['beta']} Hyp1"] = f"{x['gauss_im']:.4f}"
    expect[f"zero {x['label']} beta={x['beta']} radius"] = f"{x['radius']:.3f}"
assert all(abs(x['theta0_re'] - math.pi) < 1e-6 for x in Zr), 'a nearest zero is off Re theta = pi'
assert max(x['check_absZ_ratio'] for x in Zr) <= 1.3e-5

missing = {k: v for k, v in expect.items() if v not in tex}
print(f'{len(expect) - len(missing)}/{len(expect)} quoted numbers found in the manuscript')
for k, v in missing.items():
    print('  MISSING', k, v)
sys.exit(1 if missing else 0)
