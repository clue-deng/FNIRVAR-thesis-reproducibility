#!/usr/bin/env python3
"""
Minimal, dependency-free Student-t and normal quantiles.

scipy is not installed in the execution environment for this re-analysis
(see ../DECISIONS.md E-09), so the two-sided 95% Student-t critical value is
computed here from the regularised incomplete beta function via the standard
modified-Lentz continued fraction, then inverted by bisection.

`self_test()` checks the implementation against published critical values and
is executed by both the runner and the validator; it must pass before any
interval is reported.
"""
from __future__ import annotations

import math
from statistics import NormalDist


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-16) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    p = 0.5 * betai(0.5 * df, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_ppf(q: float, df: float, tol: float = 1e-12) -> float:
    """Inverse Student-t CDF by bisection on t_cdf (monotone)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    lo, hi = -1e3, 1e3
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def z_ppf(q: float) -> float:
    return NormalDist().inv_cdf(q)


def self_test() -> dict:
    """Published two-sided 95% critical values; tolerance 1e-6."""
    cases = {1: 12.706205, 5: 2.570582, 10: 2.228139, 19: 2.093024,
             30: 2.042272, 100: 1.983972}
    out = {}
    ok = True
    for df, ref in cases.items():
        got = t_ppf(0.975, df)
        good = abs(got - ref) < 1e-6
        ok = ok and good
        out[f"t_0.975_df{df}"] = {"computed": got, "reference": ref, "pass": good}
    zgot = z_ppf(0.975)
    zgood = abs(zgot - 1.959964) < 1e-6
    out["z_0.975"] = {"computed": zgot, "reference": 1.959964, "pass": zgood}
    out["all_pass"] = bool(ok and zgood)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
