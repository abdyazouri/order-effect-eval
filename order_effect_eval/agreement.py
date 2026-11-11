"""Gwet's AC1 agreement coefficient, implemented two independent ways, plus a
paired bootstrap confidence interval.

Why AC1 instead of Cohen's/weighted kappa? Kappa's expected-agreement term is
computed from each rater's own marginal distribution, which makes kappa
unstable -- sometimes paradoxically *low* -- when one category is rare (the
well-documented "kappa paradox"). Gwet's AC1 (Gwet, 2008) replaces that with
an expected-agreement term based on the average prevalence of each category
across both raters, which stays well-behaved under exactly this kind of
imbalance -- the situation this project expects to sit in, since most
judgments on these claims are "reject" (false premise) by design.
"""
from __future__ import annotations

import numpy as np

CATEGORIES = ("accept", "reject")


def gwet_ac1_categorical(ratings_a, ratings_b, categories=CATEGORIES) -> float:
    """General form, works for any number of categories q >= 2.

        p_a       = observed agreement
        pi_k      = (count_a_k + count_b_k) / (2n)   -- average prevalence of category k
        p_e_gwet  = (1/(q-1)) * sum_k pi_k * (1 - pi_k)
        AC1       = (p_a - p_e_gwet) / (1 - p_e_gwet)
    """
    a = np.asarray(ratings_a)
    b = np.asarray(ratings_b)
    n = len(a)
    if n == 0 or len(b) != n:
        return float("nan")

    observed_agree = float(np.mean(a == b))

    q = len(categories)
    pi = np.array([(np.sum(a == c) + np.sum(b == c)) / (2 * n) for c in categories])
    pe_gwet = (1.0 / (q - 1)) * float(np.sum(pi * (1 - pi)))

    if pe_gwet == 0:
        return 1.0 if observed_agree == 1.0 else float("nan")
    return (observed_agree - pe_gwet) / (1 - pe_gwet)


def gwet_ac1_binary_closed_form(ratings_a, ratings_b, positive_category: str = "accept") -> float:
    """Independent re-derivation for the binary case only, used purely as a
    cross-check in tests. For two categories, pi_reject = 1 - pi_accept, so
    the general formula's expected-agreement term collapses to a single
    closed-form expression in pi_accept alone:

        p_e_gwet = 2 * pi_accept * (1 - pi_accept)

    `tests/test_agreement.py` asserts this always matches
    `gwet_ac1_categorical` on random binary data -- a genuine arithmetic
    cross-check, not just the same code path called twice.
    """
    a = np.asarray(ratings_a)
    b = np.asarray(ratings_b)
    n = len(a)
    if n == 0 or len(b) != n:
        return float("nan")

    observed_agree = float(np.mean(a == b))
    pi_accept = (np.sum(a == positive_category) + np.sum(b == positive_category)) / (2 * n)
    pe_gwet = 2.0 * pi_accept * (1.0 - pi_accept)

    if pe_gwet == 0:
        return 1.0 if observed_agree == 1.0 else float("nan")
    return (observed_agree - pe_gwet) / (1 - pe_gwet)


# Default public entry point uses the general categorical form.
gwet_ac1 = gwet_ac1_categorical


def bootstrap_ac1_ci(
    ratings_a,
    ratings_b,
    categories=CATEGORIES,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired (item-level) percentile bootstrap CI for Gwet's AC1."""
    a = np.asarray(ratings_a)
    b = np.asarray(ratings_b)
    n = len(a)
    point = gwet_ac1_categorical(a, b, categories)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = gwet_ac1_categorical(a[idx], b[idx], categories)
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return point, float("nan"), float("nan")
    lower = float(np.quantile(boots, alpha / 2))
    upper = float(np.quantile(boots, 1 - alpha / 2))
    return point, lower, upper
