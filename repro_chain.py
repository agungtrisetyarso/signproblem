"""Independent reproduction of the open finite-range resonant chain (Sec. severity, Fig. 3).

H = J sum_E (s+_a s-_b + h.c.) + sum_a delta_a Z_a, edges (i,i+1),(i,i+2), open.
Excitation number is conserved; each sector is diagonalized exactly.
Convention under test: delta_a = 0.1 cos(2 pi a / n + 0.3), a = 0..n-1 or 1..n,
Z_a = +1 or -1 on an excited site.  We report which convention reproduces the
published n=6 value 0.2174669.
"""
import sys, json, itertools
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la


def sector_H(n, k, J, delta, zsign, stoq):
    states = [sum(1 << i for i in c) for c in itertools.combinations(range(n), k)]
    idx = {s: i for i, s in enumerate(states)}
    E = [(i, i + 1) for i in range(n - 1)] + [(i, i + 2) for i in range(n - 2)]
    M = np.zeros((len(states), len(states)))
    for a, s in enumerate(states):
        M[a, a] = sum(delta[i] * (zsign if (s >> i) & 1 else -zsign) for i in range(n))
        for i, j in E:
            if ((s >> i) & 1) != ((s >> j) & 1):
                M[idx[s ^ (1 << i) ^ (1 << j)], a] = (-abs(J) if stoq else J)
    return M


def lse(x):
    m = x.max(); return m + np.log(np.exp(x - m).sum())


def avg_sign(n, J, beta, conv):
    off, zsign = conv
    delta = [0.1 * np.cos(2 * np.pi * (a + off) / n + 0.3) for a in range(n)]
    lz, lzs = [], []
    for k in range(n + 1):
        e = la.eigvalsh(sector_H(n, k, J, delta, zsign, False))
        es = la.eigvalsh(sector_H(n, k, J, delta, zsign, True))
        lz.append(lse(-beta * e)); lzs.append(lse(-beta * es))
    return float(np.exp(lse(np.array(lz)) - lse(np.array(lzs))))


if __name__ == '__main__':
    out = {}
    J, beta = 0.6, 2.0
    convs = {(0, 1): None, (1, 1): None, (0, -1): None, (1, -1): None}
    for c in convs:
        convs[c] = avg_sign(6, J, beta, c)
    out['n6_conventions'] = {f'offset{c[0]}_z{c[1]}': v for c, v in convs.items()}
    print(out['n6_conventions'])
    best = min(convs, key=lambda c: abs(convs[c] - 0.2174669))
    out['convention'] = dict(offset=best[0], zsign=best[1])
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    seq = {}
    for n in range(6, nmax + 1):
        seq[n] = avg_sign(n, J, beta, best)
        print(n, seq[n], flush=True)
    out['sign'] = seq
    # weak coupling check of Eq. (triangle): beta J = 0.05 with and without detuning
    wk = []
    for n in range(8, 15):
        for det in (0.0, 1.0):
            off, zs = best
            d = [det * 0.1 * np.cos(2 * np.pi * (a + off) / n + 0.3) for a in range(n)]
            b, Jw = 2.0, 0.025
            lz, lzs = [], []
            for k in range(n + 1):
                lz.append(lse(-b * la.eigvalsh(sector_H(n, k, Jw, d, zs, False))))
                lzs.append(lse(-b * la.eigvalsh(sector_H(n, k, Jw, d, zs, True))))
            ex = lse(np.array(lzs)) - lse(np.array(lz))
            wk.append(dict(n=n, detuning=det, exact=float(ex), pred=0.5 * (b * Jw) ** 3 * (n - 2)))
    out['weak'] = wk
    json.dump(out, open(f'repro_chain_{nmax}.json', 'w'), indent=1)
