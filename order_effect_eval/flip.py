"""Flip rate: how often a judgment changes when the SAME case is evaluated
under a different ordering (reasoning-first vs. judgment-first), as opposed
to agreement between two different raters on the same ordering. This is the
number that directly answers "does order matter" -- a case can have perfect
agreement with ground truth under both orderings and still flip between them
if, say, ground truth is unanimous but the model is inconsistent across
orderings in a way that happens to net out the same -- which is exactly why
flip rate and AC1-vs-ground-truth are reported as two separate numbers, not
collapsed into one.
"""
from __future__ import annotations

import numpy as np


def flip_rate(judgments_condition_a, judgments_condition_b) -> tuple[float, list[int]]:
    a = np.asarray(judgments_condition_a)
    b = np.asarray(judgments_condition_b)
    if len(a) != len(b):
        raise ValueError(f"conditions must have equal length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        return float("nan"), []
    flips = a != b
    flipped_indices = [i for i, f in enumerate(flips) if f]
    return float(np.mean(flips)), flipped_indices
