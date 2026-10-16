"""Stochastic Lanczos quadrature for <s> of the resonant chain at n = 15, 16, 18.

Same probe vectors for H and H_stoq (correlated sampling); errors are
jackknife standard errors of the ratio estimator over probes.
"""
import sys, json
import numpy as np
import scipy.sparse as sp


def build(n, J, beta_unused=None, stoq=False):
    N = 1 << n
    x = np.arange(N)
    delta = 0.1 * np.cos(2 * np.pi * np.arange(n) / n + 0.3)
    bits = (x[:, None] >> np.arange(n)) & 1
    diag = (delta[None, :] * (2 * bits - 1)).sum(1)
    E = [(i, i + 1) for i in range(n - 1)] + [(i, i + 2) for i in range(n - 2)]
    rows, cols = [], []
    for i, j in E:
        sel = x[((x >> i) & 1) != ((x >> j) & 1)]
        rows.append(sel ^ ((1 << i) | (1 << j))); cols.append(sel)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    val = np.full(rows.size, -abs(J) if stoq else J)
    return (sp.csr_matrix((val, (rows, cols)), shape=(N, N)) + sp.diags(diag)).tocsr()


def quad(H, z, m, beta):
    """log of z^T e^{-beta H} z via m-step Lanczos with full reorthogonalization."""
    N = z.size; Q = np.zeros((m, N)); a = np.zeros(m); b = np.zeros(m - 1)
    q = z / np.linalg.norm(z); Q[0] = q
    for k in range(m):
        w = H @ Q[k]
        a[k] = Q[k] @ w
        w -= Q[:k + 1].T @ (Q[:k + 1] @ w)
        w -= Q[:k + 1].T @ (Q[:k + 1] @ w)
        if k < m - 1:
            b[k] = np.linalg.norm(w); Q[k + 1] = w / b[k]
    T = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
    th, U = np.linalg.eigh(T)
    t = -beta * th + 2 * np.log(np.abs(U[0]) + 1e-300)
    mx = t.max()
    return 2 * np.log(np.linalg.norm(z)) + mx + np.log(np.exp(t - mx).sum())


if __name__ == '__main__':
    J, beta = 0.6, 2.0
    sizes = [int(s) for s in sys.argv[1].split(',')]
    K = int(sys.argv[2]); m = int(sys.argv[3])
    rng = np.random.default_rng(7)
    res = {}
    for n in sizes:
        H, Hs = build(n, J), build(n, J, stoq=True)
        lh, ls = [], []
        for k in range(K):
            z = rng.choice([-1.0, 1.0], size=1 << n)
            lh.append(quad(H, z, m, beta)); ls.append(quad(Hs, z, m, beta))
        lh, ls = np.array(lh), np.array(ls)
        ref = ls.max()
        wh, ws = np.exp(lh - ref), np.exp(ls - ref)
        est = wh.sum() / ws.sum()
        jk = np.array([(wh.sum() - wh[i]) / (ws.sum() - ws[i]) for i in range(K)])
        err = np.sqrt((K - 1) / K * ((jk - jk.mean()) ** 2).sum())
        res[n] = dict(sign=float(est), err=float(err), K=K, m=m)
        print(n, est, err, flush=True)
    json.dump(res, open(f'repro_slq_{"_".join(map(str, sizes))}.json', 'w'), indent=1)
