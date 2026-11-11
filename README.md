# order-effect-eval

Does the **order** in which a model is asked to evaluate a claim change whether it accepts a
conclusion built on a false premise? Specifically: does prompting a model to *reason first, then
judge* make it more likely to accept a false-premise conclusion than prompting it to *give a
provisional judgment first, then justify it*?

This is a known failure mode worth checking for directly: a system that builds out reasoning
before committing to a verdict can end up reasoning its way into accepting something a quicker,
judgment-first pass would have rejected outright. This repo is a small, honest toolkit for testing
that: two independent implementations of **Gwet's AC1** agreement coefficient, a paired bootstrap
confidence interval, a flip-rate measurement between the two orderings, a synthetic generator used
to validate the whole pipeline against a known, injected effect size *before* trusting it on real
model output, and a real (small, partial) multi-model pilot.

## Why Gwet's AC1 instead of Cohen's/weighted kappa

Kappa's expected-agreement term is computed from each rater's own marginal distribution, which
makes it unstable -- sometimes paradoxically *low* -- when one category is rare (the documented
"kappa paradox"). Most of the claims tested here are deliberately built on a false premise, so
"reject" dominates by design. Gwet's AC1 (Gwet, 2008) replaces kappa's expected-agreement term with
one based on the *average* prevalence of each category across both raters, and stays well-behaved
under exactly this kind of imbalance.

## Why two independent implementations

`gwet_ac1_categorical` is the general confusion-matrix-style formulation, used everywhere in the
pipeline. `gwet_ac1_binary_closed_form` re-derives the same quantity a different way for the binary
case specifically (exploiting `pi_reject = 1 - pi_accept` to collapse the formula to a single
closed-form expression). `tests/test_agreement.py` asserts the two agree to floating-point
precision on random data every run -- a genuine cross-check that the general implementation is
correct, not just internally self-consistent.

## Validating the methodology before trusting it

`order_effect_eval/simulate.py` generates synthetic judgments with a **known, controllable**
injected effect: on ambiguous items where the correct judgment is "reject," reasoning-first
evaluation gets an extra, adjustable chance of flipping an otherwise-correct "reject" into an
incorrect "accept." Running `make validate` confirms the measurement pipeline recovers this
correctly -- flip rate rises and the AC1 gap (judgment-first minus reasoning-first) widens as the
injected effect size increases:

```
injected order_effect_size = 0.0  -> flip rate 0.000, AC1 gap ~0.000
injected order_effect_size = 0.3  -> flip rate 0.060, AC1 gap ~0.109
injected order_effect_size = 0.6  -> flip rate 0.134, AC1 gap ~0.254
```

(exact numbers from `make validate`, which fixes the random seed -- yours will match if you run it unmodified)

This is the check that matters before any real pilot result is reported: if the pipeline could not
recover an effect we ourselves injected, nothing it says about real model output would be
trustworthy either.

## The real pilot: two rounds, two models, complete

**Round 1** (10 clear-cut false premises: geography, basic physics, well-known historical facts,
Arabic orthography) and **round 2** (10 deliberately borderline cases: correlation-vs-causation,
base-rate neglect, a historically-true premise with a non-sequitur conclusion, a valid-but-unsound
classical syllogism) were each put to two models under both prompt orderings, in separate
conversations per ordering to avoid the model carrying its answer over from one condition to the
other. Both rounds are now complete for both models.

```bash
make report   # renders reports/pilot_report.md from data/pilot_results.csv + data/special_cases.csv
```

**Round 1 result:** both models rejected every claim under both orderings (AC1 = 1.00, flip rate =
0% for both). Every false premise was a single, clear-cut, checkable fact, so a model that simply
checks facts correctly rejects all of them regardless of prompt order -- a null result on
judgments, which is exactly why round 2 exists.

**Round 2 result -- the first flip in the whole pilot:**

| model | condition | AC1 vs. truth | flip rate |
|---|---|---|---|
| claude-sonnet-4.6 | reasoning-first | 1.00 | 0% |
| claude-sonnet-4.6 | judgment-first | 1.00 | |
| qwen2.5-7b | reasoning-first | 1.00 | 11.1% |
| qwen2.5-7b | judgment-first | **0.876** | (1 of 9 paired cases) |

`qwen2.5-7b`'s verdict on the language-evolution-analogy case (does language change by the same
mechanism as biological natural selection?) flipped from a correct `reject` under reasoning-first to
an incorrect `accept` under judgment-first -- the judgment itself changed, not just the quality of
the reasoning behind it, and it came with fabricated support: an invented citation to a
"linguistic evolution theory" and surface-level analogies (shared vocabulary across languages
described as equivalent to genetic inheritance) presented as if they were established science.
`claude-sonnet-4.6` stayed flawless across every case in both rounds and both orderings.

One round-2 case deliberately does not fit the accept/reject scale at all: "all birds fly, the
ostrich is a bird, therefore the ostrich can fly" is a textbook *valid* syllogism built on a *false*
premise. `claude-sonnet-4.6` raised the validity/soundness distinction explicitly in both
orderings (reasoning-first gave a genuinely dual-aspect verdict; judgment-first folded the same
distinction into a clean "reject" with the nuance as a trailing note). `qwen2.5-7b` never raised the
distinction in either ordering, and introduced a real factual error in each: misclassifying the
ostrich's family under reasoning-first, and -- worse -- naming a big cat as an example of a
flightless bird under judgment-first. These four responses are logged in
`data/special_cases.csv` and rendered in their own report section rather than forced into the
binary scale, exactly as agreed before they were collected.

## A judgment can be correct for the wrong reason

Every one of the 76 regular-case responses across both rounds was annotated by hand as `sound` /
`flawed_but_coherent` (a real, identifiable error) / `fabricated` (invented terminology or rules
with no actual basis), logged in the `reasoning_quality` column of `data/pilot_results.csv` and
tallied per model per round by `make report`:

| model | round | condition | sound | flawed | fabricated |
|---|---|---|---|---|---|
| claude-sonnet-4.6 | round 1 | reasoning-first | 10 | 0 | 0 |
| claude-sonnet-4.6 | round 1 | judgment-first | 10 | 0 | 0 |
| claude-sonnet-4.6 | round 2 | reasoning-first | 9 | 0 | 0 |
| claude-sonnet-4.6 | round 2 | judgment-first | 9 | 0 | 0 |
| qwen2.5-7b | round 1 | reasoning-first | 5 | 5 | 0 |
| qwen2.5-7b | round 1 | judgment-first | 4 | 4 | 2 |
| qwen2.5-7b | round 2 | reasoning-first | 6 | 3 | 0 |
| qwen2.5-7b | round 2 | judgment-first | 5 | 3 | 1 |

**Read this for exactly what it is.** Single-digit case counts per cell are nowhere near enough for
a significance test, and this is two models, hand-coded by one person. Three things stand out,
honestly:

1. **`claude-sonnet-4.6` was flawless across both rounds, both orderings -- 38/38 regular cases
   sound** (plus the two valid/sound-aware responses to the special case). This pilot, across both
   difficulty levels it tried, found no order effect for this model at all; capability dominates
   whatever effect exists, at least at this task difficulty.
2. **`qwen2.5-7b` is not simply "worse under judgment-first across the board."** Several round-2
   cases were handled *better* under judgment-first than reasoning-first (e.g. the fire-stations
   confound case). The pattern that held up across both rounds is narrower and more specific:
   fabrication rate specifically rises under judgment-first (0→2 in round 1, 0→1 in round 2, plus
   the one verdict flip), while plain factual slips happen under both orderings at similar rates.
3. **The round-1 "reasoning-first is error-free" impression did not survive a full, systematic
   review** -- it had real errors too (wrong dates, a self-contradictory CO2 explanation) -- and
   round 2 confirms that reading: reasoning-first is not immune, it is specifically more resistant
   to full confabulation, which is a narrower and more defensible claim.

## What's next

- Run further open-weight models on both rounds for a genuinely cross-model comparison rather than
  a two-model pilot, now that the round-2 case set and the reasoning_quality coding scheme both
  exist and are validated.
- A real independent human-rater column (this pilot used a single, deliberately-constructed ground
  truth per case, chosen by the author) would let the pipeline report agreement with a second human
  rater as well as with the models -- useful particularly for the subset of round-2 cases that are
  genuinely debatable rather than clear-cut.
- Investigate whether the one verdict flip found here replicates: rerun the language-evolution-
  analogy case on `qwen2.5-7b` multiple times per ordering to see whether it is a reliable order
  effect or a single stochastic sample.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
make setup     # pip install -r requirements.txt
make test      # 19 tests, incl. the two-AC1-implementations cross-check
make validate  # synthetic check: does the pipeline recover a known injected effect?
make report    # render the real (partial) pilot report
```

## Layout

```
order_effect_eval/
  agreement.py   Gwet's AC1 (two independent implementations) + bootstrap CI
  flip.py        flip rate between two orderings on the same case
  simulate.py    synthetic data with a known, injected order effect, for validation
  report.py      loads data/pilot_results.csv, summarizes per model, honest about gaps
  cli.py         `validate` and `report` subcommands
data/
  pilot_results.csv   the real multi-model pilot data (round 1 + round 2, both complete)
  special_cases.csv   responses whose correct answer isn't a clean accept/reject (e.g. valid-but-unsound syllogisms)
tests/           19 tests across all of the above
```

## References

- Gwet, K. L. (2008). Computing inter-rater reliability and its variance in the presence of high
  agreement. *British Journal of Mathematical and Statistical Psychology*, 61(1), 29-48.
