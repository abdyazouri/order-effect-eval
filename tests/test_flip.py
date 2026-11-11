import pytest

from order_effect_eval.flip import flip_rate


def test_identical_conditions_zero_flip_rate():
    rate, idx = flip_rate(["accept", "reject", "reject"], ["accept", "reject", "reject"])
    assert rate == 0.0
    assert idx == []


def test_fully_different_conditions_full_flip_rate():
    rate, idx = flip_rate(["accept", "reject"], ["reject", "accept"])
    assert rate == 1.0
    assert idx == [0, 1]


def test_partial_flip_rate_and_indices():
    a = ["reject", "reject", "reject", "accept"]
    b = ["reject", "accept", "reject", "reject"]
    rate, idx = flip_rate(a, b)
    assert rate == 0.5
    assert idx == [1, 3]


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        flip_rate(["accept"], ["accept", "reject"])


def test_empty_input():
    rate, idx = flip_rate([], [])
    assert rate != rate  # NaN
    assert idx == []
