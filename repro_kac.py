#!/usr/bin/env python3
"""Kac-normalized complete-graph resonant model (revision, Reviewer 1 point 3).

H = (c/n) sum_{a<b} (s+_a s-_b + s-_a s+_b) = (c/n) (S^2 - S_z^2 - n/2),
H_stoq = -H  (D = 0, J = c/n > 0).

Computes the exact average sign at finite n from the collective-spin
decomposition, checks it against brute-force diagonalization for small n, and
compares with
  (i)  beta*c < 2 :  -ln<s> -> 2[artanh(beta c/2) - beta c/2]      (bounded)
  (ii) beta*c > 2 :  -ln<s>/n -> beta*Delta f
                     = max_s [beta c s^2 + H2(1/2+s)] - ln 2       (extensive)
Writes repro_kac.json.
"""
import json, math
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize_scalar

def log_mult(n, twoS):
    """log of the multiplicity of total spin S = twoS/2 for n spins 1/2."""
    k = (n - twoS) // 2                       # n/2 - S
    a = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return a + math.log((twoS + 1) / (n - k + 1))   # C(n,k)-C(n,k-1)

def log_Z(n, bc, sign):
    """log Tr exp(-sign * (bc/n) (S(S+1) - m^2 - n/2))."""
    terms = []
    for twoS in range(n % 2, n + 1, 2):
        S = twoS / 2.0
        m = np.arange(-S, S + 0.5, 1.0)
        e = (S * (S + 1) - m**2 - n / 2.0) * bc / n
        terms.append(log_mult(n, twoS) + logsumexp(-sign * e))
    return logsumexp(terms)

def severity(n, bc):
    return log_Z(n, bc, -1) - log_Z(n, bc, +1)      # ln Z_stoq - ln Z

def brute(n, bc):
    """Dense diagonalization in the full 2^n space."""
    N = 2**n
    V = np.zeros((N, N))
    for x in range(N):
        for a in range(n):
            for b in range(a + 1, n):
                if ((x >> a) & 1) != ((x >> b) & 1):
                    V[x ^ (1 << a) ^ (1 << b), x] += 1.0
    J = (bc / n)                                    # beta*J
    w = np.linalg.eigvalsh(V)
    return logsumexp(J * w) - logsumexp(-J * w)

def bounded_limit(bc):
    x = bc / 2.0
    return 2.0 * (math.atanh(x) - x)

def beta_delta_f(bc):
    def neg(s):
        p = 0.5 + s
        H2 = -p * math.log(p) - (1 - p) * math.log(1 - p) if 0 < p < 1 else 0.0
        return -(bc * s * s + H2)
    r = minimize_scalar(neg, bounds=(0.0, 0.5 - 1e-12), method="bounded",
                        options={"xatol": 1e-13})
    return -r.fun - math.log(2.0), r.x

out = {"brute_check": [], "bounded": [], "extensive": []}
for n in (4, 6, 8, 10):
    for bc in (1.0, 3.0):
        a, b = severity(n, bc), brute(n, bc)
        out["brute_check"].append({"n": n, "bc": bc, "collective": a, "brute": b,
                                   "absdiff": abs(a - b)})
for bc in (0.5, 1.0, 1.5, 1.9):
    row = {"bc": bc, "limit": bounded_limit(bc), "cubic": bc**3 / 12.0,
           "finite_n": {str(n): severity(n, bc) for n in (100, 1000, 10000)}}
    out["bounded"].append(row)
for bc in (2.5, 3.0, 4.0, 8.0):
    bdf, sstar = beta_delta_f(bc)
    row = {"bc": bc, "beta_delta_f": bdf, "s_star": sstar,
           "finite_n_per_spin": {str(n): severity(n, bc) / n
                                 for n in (100, 1000, 10000)}}
    out["extensive"].append(row)
json.dump(out, open("repro_kac.json", "w"), indent=1)
print(json.dumps(out, indent=1))
