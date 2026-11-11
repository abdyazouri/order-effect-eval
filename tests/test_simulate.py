from order_effect_eval.agreement import bootstrap_ac1_ci
from order_effect_eval.flip import flip_rate
from order_effect_eval.simulate import generate


def test_shapes_and_categories():
    data = generate(n_items=100, seed=0)
    assert len(data["true_label"]) == 100
    assert len(data["is_ambiguous"]) == 100
    assert len(data["judgment_first"]) == 100
    assert len(data["reasoning_first"]) == 100
    assert set(data["judgment_first"]) <= {"accept", "reject"}
    assert set(data["reasoning_first"]) <= {"accept", "reject"}


def test_deterministic_for_same_seed():
    d1 = generate(n_items=50, seed=7)
    d2 = generate(n_items=50, seed=7)
    assert d1 == d2


def test_zero_order_effect_means_zero_flip_rate():
    data = generate(n_items=500, order_effect_size=0.0, seed=1)
    rate, _ = flip_rate(data["reasoning_first"], data["judgment_first"])
    assert rate == 0.0


def test_order_effect_recovered_by_flip_rate_and_ac1_gap():
    """The core validation: a larger injected effect should produce a higher
    flip rate AND a larger AC1 gap (reasoning-first worse than judgment-first).
    This is checked on size, not an exact value, since the generator is
    stochastic -- but the direction must hold, every time, at this sample size."""
    small = generate(n_items=1000, order_effect_size=0.1, seed=5)
    large = generate(n_items=1000, order_effect_size=0.5, seed=5)

    rate_small, _ = flip_rate(small["reasoning_first"], small["judgment_first"])
    rate_large, _ = flip_rate(large["reasoning_first"], large["judgment_first"])
    assert rate_large > rate_small

    ac1_rf_small, *_ = bootstrap_ac1_ci(small["true_label"], small["reasoning_first"], n_boot=200, seed=0)
    ac1_jf_small, *_ = bootstrap_ac1_ci(small["true_label"], small["judgment_first"], n_boot=200, seed=0)
    ac1_rf_large, *_ = bootstrap_ac1_ci(large["true_label"], large["reasoning_first"], n_boot=200, seed=0)
    ac1_jf_large, *_ = bootstrap_ac1_ci(large["true_label"], large["judgment_first"], n_boot=200, seed=0)

    gap_small = ac1_jf_small - ac1_rf_small
    gap_large = ac1_jf_large - ac1_rf_large
    assert gap_large > gap_small
    assert gap_small >= -1e-9  # judgment-first should never be worse than reasoning-first here
