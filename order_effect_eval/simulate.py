"""Generate a SYNTHETIC set of judgments under two evaluation orderings, with
a known, controllable "order effect" injected, so the agreement/flip-rate
pipeline in this repo can be validated against ground truth before it is
trusted on real model output.

Nothing here is real model behavior. It models a specific, nameable failure
mode: on ambiguous items where the correct judgment is "reject" (the claim
relies on a false premise), reasoning-first evaluation has an extra chance of
flipping an otherwise-correct "reject" into an incorrect "accept" --
operationalizing the hypothesis this whole project exists to test, so the
measurement pipeline can be checked against a case where the answer is known.
"""
from __future__ import annotations

import numpy as np

CATEGORIES = ("accept", "reject")


def generate(
    n_items: int = 200,
    p_ambiguous: float = 0.3,
    p_true_reject: float = 0.7,
    base_error_rate: float = 0.05,
    order_effect_size: float = 0.0,
    seed: int = 0,
) -> dict[str, list]:
    """Returns a dict with keys: true_label, is_ambiguous, judgment_first,
    reasoning_first -- one entry per item, in the same order."""
    rng = np.random.default_rng(seed)

    true_label = rng.choice(CATEGORIES, size=n_items, p=[1 - p_true_reject, p_true_reject])
    is_ambiguous = rng.random(n_items) < p_ambiguous

    # judgment-first: baseline noisy evaluator, same for every item
    noisy_flip = rng.random(n_items) < base_error_rate
    judgment_first = np.where(
        noisy_flip,
        np.where(true_label == "accept", "reject", "accept"),
        true_label,
    )

    # reasoning-first: starts identical to judgment-first, then the injected
    # effect can additionally flip a correct "reject" into "accept" -- but
    # only on ambiguous items where judgment-first got it right.
    reasoning_first = judgment_first.copy()
    eligible = is_ambiguous & (true_label == "reject") & (judgment_first == "reject")
    extra_flip = rng.random(n_items) < order_effect_size
    flip_mask = eligible & extra_flip
    reasoning_first[flip_mask] = "accept"

    return {
        "true_label": true_label.tolist(),
        "is_ambiguous": is_ambiguous.tolist(),
        "judgment_first": judgment_first.tolist(),
        "reasoning_first": reasoning_first.tolist(),
    }
