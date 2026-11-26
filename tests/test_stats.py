import random

import pytest

from mlopd import stats

R1 = [2.531, -0.156, 2.406, 0.719, -4.625, -1.625, -4.250, -2.875]


def test_r1_paired_t_all_alternatives():
    g = stats.paired_t(R1, "greater")
    assert round(g["mean"], 2) == -0.98 and round(g["sd"], 2) == 2.81 and round(g["t"], 2) == -0.99 and g["df"] == 7
    assert round(g["p"], 2) == 0.82 and round(g["p_opposite"], 2) == 0.18
    less = stats.paired_t(R1, "less")
    assert round(less["p"], 2) == 0.18
    two = stats.paired_t(R1, "two-sided")
    assert abs(two["p"] - 2 * less["p"]) < 1e-12
    assert round(stats.mde(g["sd"], 8), 2) == 2.75
    with pytest.raises(ValueError):
        stats.paired_t(R1, "bigger")


def test_t_distribution_against_known_values():
    assert abs(stats.t_ppf(0.95, 7) - 1.894579) < 1e-5
    assert abs(stats.t_ppf(0.975, 15) - 2.131450) < 1e-5
    assert abs(stats.t_cdf(2.0, 10) - 0.963306) < 1e-5


def test_mde_table_of_design_7_3():
    table = {4: (1.65, 3.30, 4.95, 7.42), 8: (0.98, 1.96, 2.93, 4.40), 16: (0.65, 1.30, 1.96, 2.93)}
    for n, row in table.items():
        for sd, expected in zip((1., 2., 3., 4.5), row):
            assert abs(stats.mde(sd, n) - expected) < 0.006, (n, sd)


def test_noninferiority():
    result = stats.noninferiority([-1., -0.5, 0., 0.5], 3.)
    assert result["mean"] == -0.25 and result["p"] < 0.01
    assert stats.noninferiority([-4., -3.5, -2.5, -3.], 3.)["p"] > 0.5


def test_spearman_exact_and_critical_value():
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert abs(stats.spearman([1, 2, 2, 3], [1, 3, 2, 4]) - 0.9487) < 1e-4  # ties: average ranks
    r = stats.spearman_permutation([1, 2, 3, 4, 5], [5, 6, 7, 8, 9])
    assert r["method"] == "exact" and r["permutations"] == 120 and abs(r["p"] - 1 / 120) < 1e-12
    # n = 10, one-sided 5%: the critical rho is about 0.56 (design §3.3 M3)
    x = list(range(10))
    def p_of(perm):
        return stats.spearman_permutation(x, perm)["p"]
    y_hi = [0, 2, 1, 4, 3, 6, 5, 9, 7, 8]
    assert p_of(y_hi) < 0.05
    # two permutations either side of the critical value
    above = [1, 0, 3, 2, 6, 4, 9, 5, 8, 7]
    below = [3, 0, 6, 1, 9, 2, 4, 8, 5, 7]
    assert stats.spearman(x, above) >= 0.60 and p_of(above) < 0.05
    assert stats.spearman(x, below) <= 0.52 and p_of(below) > 0.05


def test_spearman_monte_carlo_matches_exact_roughly():
    x = list(range(12))
    y = [1, 0, 3, 2, 5, 4, 7, 6, 11, 9, 8, 10]
    r = stats.spearman_permutation(x, y, permutations_count=20000, seed=1)
    assert r["method"] == "monte-carlo" and r["p"] < 0.001


def test_holm():
    adjusted = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_item_bootstrap_and_mixed_slope():
    matrix = [[1.] * 50, [3.] * 50]
    boot = stats.item_bootstrap(matrix, iterations=200, seed=0)
    assert boot["mean"] == 2.0 and boot["low"] == boot["high"] == 2.0
    rng = random.Random(0)
    langs = [f"l{i}" for i in range(10)]
    seeds = [1, 2, 3, 4]
    g = {l: i / 10 for i, l in enumerate(langs)}
    b = {l: rng.gauss(0, 0.5) for l in langs}
    u = {s: rng.gauss(0, 0.5) for s in seeds}
    y, x, la, se = [], [], [], []
    for l in langs:
        for s in seeds:
            y.append(1. + 5. * g[l] + b[l] + u[s] + rng.gauss(0, 0.3)); x.append(g[l]); la.append(l); se.append(s)
    fit = stats.mixed_slope(y, x, la, se)
    assert abs(fit["beta"] - 5.) < 2.5 and fit["se"] > 0
