#!/usr/bin/env python3
"""Decay rate of the open chain of Sec. 'Severity is not hardness' with a
systematic error from the even-odd oscillation (revision, Reviewer 2 point 4).

Input: the exact average signs of the chain table (n = 6..16), as produced by
repro_chain.py.  Writes repro_rate_sys.json.
"""
import json, sys
import numpy as np

SIGN = {6: 0.2174669, 7: 0.1509983, 8: 0.1056727, 9: 0.0736982, 10: 0.0514958,
        11: 0.0359523, 12: 0.0251132, 13: 0.0175388, 14: 0.0122506,
        15: 0.0085567, 16: 0.0059769}
if len(sys.argv) > 1:                       # optional: repro_chain_16.json
    d = json.load(open(sys.argv[1]))
    if "sign" in d:
        SIGN = {int(k): float(v) for k, v in d["sign"].items()}

ns = np.array(sorted(SIGN)); y = np.log([SIGN[n] for n in ns])

def fit(mask):
    a, b = np.polyfit(ns[mask], y[mask], 1)
    return -a, b

out = {}
# (1) window fits, all n
wins = {n0: fit(ns >= n0)[0] for n0 in (8, 9, 10, 11)}
out["window_fits"] = wins
a_all, b_all = fit(ns >= 8)
out["a_fit_n>=8"], out["b_fit_n>=8"] = a_all, b_all
out["window_halfrange"] = (max(wins.values()) - min(wins.values())) / 2
# (2) local rates  -ln[s(n+1)/s(n)]
ratios = {int(n): SIGN[n + 1] / SIGN[n] for n in ns[:-1]}
local = {n: -np.log(r) for n, r in ratios.items()}
out["ratios"] = ratios
out["local_rates"] = local
loc8 = np.array([local[n] for n in local if n >= 8])
loc10 = np.array([local[n] for n in local if n >= 10])
out["local_rate_range_n>=8"] = [loc8.min(), loc8.max()]
out["local_rate_halfrange_n>=8"] = (loc8.max() - loc8.min()) / 2
out["local_rate_halfrange_n>=10"] = (loc10.max() - loc10.min()) / 2
out["ratio_range_n>=8"] = [min(r for n, r in ratios.items() if n >= 8),
                           max(r for n, r in ratios.items() if n >= 8)]
# (3) separate even and odd sublattice fits (n >= 8)
a_even = fit((ns >= 8) & (ns % 2 == 0))[0]
a_odd = fit((ns >= 8) & (ns % 2 == 1))[0]
out["a_even"], out["a_odd"] = a_even, a_odd
out["even_odd_halfdiff"] = abs(a_even - a_odd) / 2
# quoted value: central = fit n>=8; systematic = half-range of local rates n>=8
out["quoted"] = {"a": a_all, "stat_window": out["window_halfrange"],
                 "sys_evenodd": out["local_rate_halfrange_n>=8"]}
out["pred_n18"] = float(np.exp(-a_all * 18 + b_all))
out["pred_n18_range"] = [float(np.exp(-(a_all + s) * 18 + b_all + s * 12))
                         for s in (+out["local_rate_halfrange_n>=8"],
                                   -out["local_rate_halfrange_n>=8"])]
json.dump(out, open("repro_rate_sys.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
