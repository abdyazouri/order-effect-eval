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

## The real pilot (small, partial, and reported as such)

Ten short claims were built, each relying on a single, checkable false premise (geography, basic
physics, well-known historical facts, and one about Arabic orthography), with "reject" as the
correct judgment in every case. Each claim was put to two models under both prompt orderings, in
separate conversations per ordering to avoid the model carrying over its answer from one condition
to the other.

```bash
make report   # renders reports/pilot_report.md from data/pilot_results.csv
```

**Result:** both models rejected every claim under both orderings (AC1 = 1.00, flip rate = 0%).
`claude-sonnet-4.6` completed all 10 cases under both orderings. `qwen2.5-7b` is **incomplete**
(9/10 reasoning-first, 6/10 judgment-first -- one case did not return a response locally) --
reported exactly as that, not padded or assumed.

**Read this result for what it is, not more.** A null result here does not mean order has no
effect in general -- it means these ten claims were too unambiguous to show one. Every false
premise was a single, clear-cut, checkable fact (not a borderline or contestable case), so a model
that simply checks facts correctly will reject all of them regardless of prompt order. The
documented failure mode this project is built to detect is specifically expected to show up on
*ambiguous* cases, where reasoning can plausibly talk itself into an unwarranted conclusion -- which
is exactly what the synthetic validation above models (`is_ambiguous`), and exactly what this real
pilot's case set did not include enough of.

## A judgment can be correct for the wrong reason

Agreement and flip rate both treat a judgment as one of two outcomes -- correct or not -- which
misses something the pilot surfaced directly: on the Arabic-orthography case (`arabic_no_vowels`),
`qwen2.5-7b` reached the *correct* judgment ("reject") under **both** orderings, but the reasoning
behind it was not sound either time, and degraded differently depending on order:

- **reasoning-first**: conflated *harakat* (short vowel diacritics -- fatha, damma, kasra) with
  *huruf al-illah* (the actual vowel letters, alif/waw/ya') -- a real, identifiable grammatical
  error, but at least pointing at genuine concepts.
- **judgment-first**: invented grammatical terms and rules with no basis in Arabic grammar at all
  (e.g. treating "fi'l," "ayn," and "hamza" as categories of vowel letter) -- not a recognizable
  error anymore, closer to confabulation.

This is logged in the report as a qualitative annotation (`reasoning_quality` column in
`data/pilot_results.csv`: `sound` / `flawed_but_coherent` / `fabricated`), reviewed case by case,
not aggregated into a rate -- **this is one case, reviewed in this much detail, not a finding**.
But it reframes the original question usefully: maybe prompt order doesn't flip *judgments* much
(which is what AC1 and flip rate measure), but it can still degrade the *reasoning quality behind*
an unchanged judgment -- a real distinction the all-reject pilot above, read only as judgments,
would have missed entirely. Systematically scoring `reasoning_quality` across every case and model,
then testing whether flawed or fabricated reasoning predicts a *higher* flip rate on the harder,
borderline cases below, is the natural next step.

## What's next

- A second round of cases built specifically to be borderline -- a false premise embedded in an
  otherwise plausible-sounding argument, not a single clear-cut fact -- is the natural next step,
  since that is where an order effect, if real, should actually appear.
- Complete the `qwen2.5-7b` run (currently partial) and add further open-weight models for a
  genuinely cross-model comparison, not just a two-model pilot.
- A real independent human judgment column (this pilot used a single, deliberately unambiguous
  ground truth per case) would let the pipeline report agreement with a second human rater as well
  as with the models.

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
  pilot_results.csv   the real (partial) multi-model pilot data
tests/           19 tests across all of the above
```

## References

- Gwet, K. L. (2008). Computing inter-rater reliability and its variance in the presence of high
  agreement. *British Journal of Mathematical and Statistical Psychology*, 61(1), 29-48.
