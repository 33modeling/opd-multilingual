"""Statistics of design §7 without scipy: Student t (one-sided "greater"/"less" and "two-sided"; R1 of §3.2),
non-inferiority, MDE from the noncentral t (§7.3), Spearman permutation test (exact for n <= 10, else 100,000
random permutations), Holm, item bootstrap, and the crossed random-intercept slope recorded for M3.
"""
from __future__ import annotations

from itertools import islice, permutations
import math
import random

import numpy as np

ALTERNATIVES = ("greater", "less", "two-sided")


# ---------------------------------------------------------------- special functions
def _betacf(a, b, x, eps=1e-15, iterations=10000):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1., a - 1.
    c, d = 1., 1. - qab * x / qap
    d = 1. / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1. + aa * d
        d = 1. / (d if abs(d) > tiny else tiny)
        c = 1. + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1. + aa * d
        d = 1. / (d if abs(d) > tiny else tiny)
        c = 1. + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.
    if x >= 1:
        return 1.
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.) / (a + b + 2.):
        return front * _betacf(a, b, x) / a
    return 1. - front * _betacf(b, a, 1. - x) / b


def t_cdf(t, df):
    if math.isinf(t):
        return 1. if t > 0 else 0.
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2., 0.5, x)
    return 1. - tail if t > 0 else tail


def t_sf(t, df):
    return 1. - t_cdf(t, df) if t <= 0 else t_cdf(-t, df)


def t_ppf(p, df):
    low, high = -1e3, 1e3
    for _ in range(200):
        mid = (low + high) / 2
        if t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def normal_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def nct_sf(c, df, ncp, points=20001):
    """P(T' > c) for the noncentral t with df and noncentrality ncp: T' = (Z + ncp) / S, S = sqrt(chi2_df / df).
    Integrates P(Z > c s - ncp) over the density of S with Simpson's rule."""
    s_max = 1. + 12. / math.sqrt(df) + 2.
    s = np.linspace(1e-12, s_max, points)
    log_f = (math.log(2.) + (df / 2.) * math.log(df / 2.) - math.lgamma(df / 2.)
             + (df - 1.) * np.log(s) - df * s * s / 2.)
    density = np.exp(log_f)
    tail = 0.5 * np.array([math.erfc((c * v - ncp) / math.sqrt(2)) for v in s])
    h = (s_max - 1e-12) / (points - 1)
    weights = np.ones(points)
    weights[1:-1:2], weights[2:-1:2] = 4., 2.
    return float(h / 3. * np.sum(weights * density * tail))


def mde(sd, n, *, alpha=0.05, power=0.8, alternative="greater"):
    """Smallest true mean difference detected with `power` by the paired t on n pairs with pair SD `sd`."""
    if n < 2 or sd is None:
        return None
    df = n - 1
    crit = t_ppf(1 - (alpha if alternative != "two-sided" else alpha / 2), df)
    low, high = 0., 50.
    for _ in range(80):
        mid = (low + high) / 2
        if nct_sf(crit, df, mid * math.sqrt(n)) < power:
            low = mid
        else:
            high = mid
    return (low + high) / 2 * sd


# ---------------------------------------------------------------- tests
def paired_t(diffs, alternative="greater"):
    """One-sample t on paired differences. Reports the chosen alternative's p, the opposite direction and two-sided."""
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}")
    x = [float(v) for v in diffs]
    n = len(x)
    m = sum(x) / n if n else None
    out = {"n": n, "mean": m, "alternative": alternative, "values": x}
    if n < 2:
        return {**out, "sd": None, "se": None, "t": None, "df": None, "p": None}
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    if se == 0:
        t = math.inf if m > 0 else (-math.inf if m < 0 else 0.)
    else:
        t = m / se
    greater = t_sf(t, df) if not math.isinf(t) else (0. if t > 0 else 1.)
    less = 1. - greater if not math.isinf(t) else (1. - greater)
    if t == 0 and se == 0:
        greater = less = 1.
    two = min(1., 2 * min(greater, less))
    p = {"greater": greater, "less": less, "two-sided": two}[alternative]
    opposite = {"greater": less, "less": greater, "two-sided": None}[alternative]
    return {**out, "sd": sd, "se": se, "t": t, "df": df, "p": p, "p_opposite": opposite, "p_two_sided": two,
            "p_greater": greater, "p_less": less,
            "positive": sum(v > 0 for v in x), "negative": sum(v < 0 for v in x)}


def noninferiority(diffs, margin):
    """H0: mean <= -margin; H1: mean > -margin (one-sided t on diffs + margin). margin in the units of diffs."""
    result = paired_t([v + margin for v in diffs], "greater")
    result["mean"] = sum(diffs) / len(diffs) if diffs else None
    result["margin"] = margin
    result["values"] = list(diffs)
    return result


def ranks(values):
    """Average ranks (1-based) with ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2. + 1.
        i = j + 1
    return out


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    denominator = math.sqrt(float((xc ** 2).sum() * (yc ** 2).sum()))
    return float((xc * yc).sum() / denominator) if denominator else None


def spearman(x, y):
    return pearson(ranks(list(x)), ranks(list(y)))


def spearman_permutation(x, y, *, alternative="greater", exact_max_n=10, permutations_count=100000, seed=0,
                         chunk=200000):
    """Permutation test of Spearman's rho: y is permuted against x. Exact enumeration when n <= exact_max_n
    (p = share of the n! permutations with rho at least as extreme, identity included); otherwise
    `permutations_count` random permutations with p = (1 + hits) / (1 + count)."""
    if alternative not in ALTERNATIVES:
        raise ValueError(alternative)
    n = len(x)
    if n != len(y) or n < 3:
        return {"n": n, "rho": None, "p": None}
    rx = np.asarray(ranks(list(x)), float)
    ry = np.asarray(ranks(list(y)), float)
    rx -= rx.mean()
    ry -= ry.mean()
    scale = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    if scale == 0:
        return {"n": n, "rho": None, "p": None, "note": "constant ranks"}
    observed = float((rx * ry).sum()) / scale
    tol = 1e-12

    def count(perm_matrix):
        stats = (perm_matrix * rx).sum(1) / scale  # elementwise: no multi-threaded BLAS call
        if alternative == "greater":
            return int((stats >= observed - tol).sum())
        if alternative == "less":
            return int((stats <= observed + tol).sum())
        return int((np.abs(stats) >= abs(observed) - tol).sum())

    if n <= exact_max_n:
        hits = total = 0
        iterator = permutations(range(n))
        while True:
            block = list(islice(iterator, chunk))
            if not block:
                break
            index = np.asarray(block, dtype=np.int64)
            hits += count(ry[index])
            total += len(block)
        return {"n": n, "rho": observed, "p": hits / total, "method": "exact", "permutations": total,
                "alternative": alternative}
    rng = random.Random(seed)
    hits = 0
    done = 0
    while done < permutations_count:
        size = min(chunk, permutations_count - done)
        block = np.asarray([rng.sample(range(n), n) for _ in range(size)], dtype=np.int64)
        hits += count(ry[block])
        done += size
    return {"n": n, "rho": observed, "p": (1 + hits) / (1 + permutations_count), "method": "monte-carlo",
            "permutations": permutations_count, "seed": seed, "alternative": alternative}


def holm(pvalues):
    """{name: p} -> {name: Holm-adjusted p}; None entries are left out."""
    items = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    m = len(items)
    adjusted, running = {}, 0.
    for i, (p, k) in enumerate(items):
        running = max(running, min(1., (m - i) * p))
        adjusted[k] = running
    return adjusted


def item_bootstrap(per_item_diffs, *, iterations=10000, seed=0, level=0.95):
    """per_item_diffs: array [pairs, items] of per-item differences (the same item columns in every pair).
    Resamples item columns with replacement; returns the percentile interval of the mean pair difference."""
    data = np.asarray(per_item_diffs, float)
    if data.ndim != 2 or data.shape[1] == 0:
        return None
    rng = np.random.default_rng(seed)
    items = data.shape[1]
    means = np.empty(iterations)
    pair_items = data.mean(0)
    for b in range(iterations):
        means[b] = pair_items[rng.integers(0, items, items)].mean()
    low, high = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"mean": float(pair_items.mean()), "low": float(low), "high": float(high), "iterations": iterations,
            "seed": seed}


# ---------------------------------------------------------------- mixed model (recorded secondary for M3)
def _nelder_mead(f, x0, step=0.5, iterations=2000, tol=1e-10):
    n = len(x0)
    simplex = [np.asarray(x0, float)] + [np.asarray(x0, float) + step * np.eye(n)[i] for i in range(n)]
    values = [f(x) for x in simplex]
    for _ in range(iterations):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        fr = f(reflected)
        if fr < values[0]:
            expanded = centroid + 2 * (centroid - simplex[-1])
            fe = f(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(contracted)
            if fc < values[-1]:
                simplex[-1], values[-1] = contracted, fc
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (s - simplex[0]) for s in simplex[1:]]
                values = [values[0]] + [f(s) for s in simplex[1:]]
    best = int(np.argmin(values))
    return simplex[best], values[best]


def mixed_slope(y, x, groups_a, groups_b):
    """REML fit of y = alpha + beta x + b_a + u_b + e with independent random intercepts for two crossed factors
    (M3: a = language, b = seed). Returns beta, its SE and a normal-approximation two-sided p (recorded only)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    levels_a = sorted(set(groups_a))
    levels_b = sorted(set(groups_b))
    Za = np.asarray([[g == l for l in levels_a] for g in groups_a], float)
    Zb = np.asarray([[g == l for l in levels_b] for g in groups_b], float)
    Ka, Kb, eye = Za @ Za.T, Zb @ Zb.T, np.eye(len(y))

    def parts(theta):
        va, vb, ve = np.exp(theta)
        V = va * Ka + vb * Kb + ve * eye
        Vi = np.linalg.inv(V)
        XtViX = X.T @ Vi @ X
        beta = np.linalg.solve(XtViX, X.T @ Vi @ y)
        return V, Vi, XtViX, beta

    def neg_reml(theta):
        try:
            V, Vi, XtViX, beta = parts(theta)
        except np.linalg.LinAlgError:
            return 1e300
        r = y - X @ beta
        return 0.5 * (np.linalg.slogdet(V)[1] + np.linalg.slogdet(XtViX)[1] + float(r @ Vi @ r))

    start = math.log(max(float(np.var(y)), 1e-8) / 3)
    theta, value = _nelder_mead(neg_reml, [start, start, start])
    _, _, XtViX, beta = parts(theta)
    cov = np.linalg.inv(XtViX)
    se = math.sqrt(max(float(cov[1, 1]), 0.))
    z = float(beta[1]) / se if se else None
    return {"beta": float(beta[1]), "alpha": float(beta[0]), "se": se,
            "p_two_sided_normal": (math.erfc(abs(z) / math.sqrt(2)) if z is not None else None),
            "variances": {"a": float(math.exp(theta[0])), "b": float(math.exp(theta[1])),
                          "residual": float(math.exp(theta[2]))}, "neg_reml": float(value), "n": len(y)}
