import numpy as np
import pytest

from order_effect_eval.agreement import (
    bootstrap_ac1_ci,
    gwet_ac1_binary_closed_form,
    gwet_ac1_categorical,
)


def test_perfect_agreement_is_one():
    a = ["accept", "reject", "accept", "reject", "reject"]
    assert gwet_ac1_categorical(a, a) == pytest.approx(1.0)
    assert gwet_ac1_binary_closed_form(a, a) == pytest.approx(1.0)


def test_categorical_and_binary_closed_form_agree_on_random_data():
    rng = np.random.default_rng(42)
    for _ in range(30):
        n = rng.integers(5, 100)
        a = rng.choice(["accept", "reject"], size=n)
        b = rng.choice(["accept", "reject"], size=n)
        k1 = gwet_ac1_categorical(a, b)
        k2 = gwet_ac1_binary_closed_form(a, b)
        assert k1 == pytest.approx(k2, abs=1e-9), (k1, k2, a, b)


def test_constant_category_with_perfect_agreement_is_one():
    a = ["reject"] * 10
    assert gwet_ac1_categorical(a, a) == 1.0


def test_near_random_ratings_give_low_ac1():
    rng = np.random.default_rng(1)
    a = rng.choice(["accept", "reject"], size=5000)
    b = rng.choice(["accept", "reject"], size=5000)
    ac1 = gwet_ac1_categorical(a, b)
    assert abs(ac1) < 0.05


def test_systematic_disagreement_lowers_ac1():
    rng = np.random.default_rng(2)
    a = rng.choice(["accept", "reject"], size=500, p=[0.3, 0.7])
    b_matched = a.copy()
    b_offset = np.where(a == "accept", "reject", "accept")
    assert gwet_ac1_categorical(a, b_matched) > gwet_ac1_categorical(a, b_offset)


def test_bootstrap_ci_contains_point_and_has_sane_width():
    rng = np.random.default_rng(3)
    a = rng.choice(["accept", "reject"], size=300, p=[0.3, 0.7])
    noise = rng.random(300) < 0.1
    b = np.where(noise, np.where(a == "accept", "reject", "accept"), a)
    point, lo, hi = bootstrap_ac1_ci(a, b, n_boot=800, seed=0)
    assert lo <= point <= hi
    assert 0 < (hi - lo) < 1.0
