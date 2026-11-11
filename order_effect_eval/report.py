"""Load a long-format pilot-results CSV and produce a markdown report:
per-model, per-round agreement with ground truth under each ordering, and the
flip rate between orderings -- computed only from cases where that model has
a judgment for both orderings, with the available case count reported
explicitly rather than silently padding missing cells. Round 1 and round 2
are reported separately since they differ deliberately in difficulty
(clear-cut false premises vs. borderline fallacies) and conflating them would
hide that difference rather than reveal it.

Expected CSV columns: round, case_id, case_text, ground_truth, model, condition, judgment, reasoning_quality
  - condition is one of "reasoning_first" / "judgment_first"
  - judgment and ground_truth are one of "accept" / "reject"
A model/case/condition combination with no row is treated as missing, not as
any particular judgment -- this project never invents a result for a case
that was not actually run. Cases whose correct answer does not reduce to a
clean accept/reject (e.g. a valid-but-unsound syllogism) belong in
data/special_cases.csv instead, not forced into this scale.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from order_effect_eval.agreement import bootstrap_ac1_ci
from order_effect_eval.flip import flip_rate


def load_pilot_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_special_cases(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _round_key(row: dict) -> str:
    return row.get("round") or "round1"


def _index(rows: list[dict]) -> tuple[dict, dict]:
    """(model, round) -> condition -> case_id -> judgment, plus (round, case_id) -> ground_truth."""
    by_model_round: dict = defaultdict(lambda: defaultdict(dict))
    ground_truth: dict = {}
    for row in rows:
        key = (row["model"], _round_key(row))
        by_model_round[key][row["condition"]][row["case_id"]] = row["judgment"]
        ground_truth[(_round_key(row), row["case_id"])] = row["ground_truth"]
    return by_model_round, ground_truth


def reasoning_quality_notes(rows: list[dict]) -> list[dict]:
    """Every row that has been manually annotated with a reasoning_quality
    label (sound / flawed_but_coherent / fabricated), regardless of whether
    the judgment itself was correct. This is a qualitative log, not a
    statistic: with only a handful of cases reviewed this closely so far, it
    is reported as individual annotated examples, not aggregated into a rate
    that would overstate how much has actually been checked."""
    return [
        {
            "round": _round_key(r),
            "model": r["model"],
            "case_id": r["case_id"],
            "case_text": r["case_text"],
            "condition": r["condition"],
            "judgment": r["judgment"],
            "reasoning_quality": r["reasoning_quality"],
        }
        for r in rows
        if r.get("reasoning_quality")
    ]


def reasoning_quality_counts(rows: list[dict]) -> dict:
    """(model, round) -> condition -> {quality_label: count}, over only the
    rows that have been annotated. This is a plain count, not a statistical
    test -- at typical pilot sample sizes (tens of cases) a formal test
    would overstate the confidence this count can actually support."""
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        q = r.get("reasoning_quality")
        if q:
            counts[(r["model"], _round_key(r))][r["condition"]][q] += 1
    return counts


def per_model_summary(rows: list[dict]) -> list[dict]:
    by_model_round, ground_truth = _index(rows)
    summaries = []

    for (model, round_), conditions in by_model_round.items():
        rf = conditions.get("reasoning_first", {})
        jf = conditions.get("judgment_first", {})

        for condition_name, judgments in [("reasoning_first", rf), ("judgment_first", jf)]:
            if not judgments:
                continue
            case_ids = sorted(judgments.keys())
            truth = [ground_truth[(round_, c)] for c in case_ids]
            model_j = [judgments[c] for c in case_ids]
            point, lo, hi = bootstrap_ac1_ci(truth, model_j, n_boot=5000, seed=0)
            summaries.append({
                "model": model,
                "round": round_,
                "condition": condition_name,
                "n_cases": len(case_ids),
                "ac1": round(point, 3),
                "ci_low": round(lo, 3) if lo == lo else None,  # NaN check
                "ci_high": round(hi, 3) if hi == hi else None,
            })

        paired_ids = sorted(set(rf.keys()) & set(jf.keys()))
        if paired_ids:
            rf_vals = [rf[c] for c in paired_ids]
            jf_vals = [jf[c] for c in paired_ids]
            rate, flipped = flip_rate(rf_vals, jf_vals)
            summaries.append({
                "model": model,
                "round": round_,
                "condition": "flip_rate (paired)",
                "n_cases": len(paired_ids),
                "ac1": round(rate, 3),
                "ci_low": None,
                "ci_high": None,
                "flipped_case_ids": [paired_ids[i] for i in flipped],
            })

    return summaries


def render_markdown(rows: list[dict], special_cases: list[dict] | None = None) -> str:
    summaries = per_model_summary(rows)
    by_model_round: dict = defaultdict(list)
    for s in summaries:
        by_model_round[(s["model"], s["round"])].append(s)

    lines = ["# Order-Effect Pilot Report\n"]
    lines.append(
        "Does evaluation order (reasoning-first vs. judgment-first) change whether a model "
        "accepts a conclusion built on a false premise? Reported per model and per round "
        "(round 1: clear-cut false premises; round 2: deliberately borderline fallacies), using "
        "only the cases actually run for that model/condition/round -- a missing cell means the "
        "case was not run, never an assumed result.\n"
    )

    for (model, round_), items in by_model_round.items():
        lines.append(f"## {model} — {round_}\n")
        lines.append("| condition | n cases | AC1 vs. ground truth | 95% CI |")
        lines.append("|---|---|---|---|")
        for item in items:
            if item["condition"] == "flip_rate (paired)":
                continue
            ci = f"[{item['ci_low']}, {item['ci_high']}]" if item["ci_low"] is not None else "n/a"
            lines.append(f"| {item['condition']} | {item['n_cases']} | {item['ac1']} | {ci} |")
        flip_items = [i for i in items if i["condition"] == "flip_rate (paired)"]
        if flip_items:
            f = flip_items[0]
            lines.append("")
            lines.append(f"**Flip rate** (cases run under both conditions, n={f['n_cases']}): **{f['ac1']:.1%}**")
            if f.get("flipped_case_ids"):
                lines.append(f"Flipped cases: {', '.join(f['flipped_case_ids'])}")
        lines.append("")

    notes = reasoning_quality_notes(rows)
    if notes:
        lines.append("## Reasoning-quality annotations\n")
        lines.append(
            "A judgment can be *correct* while the reasoning behind it is not. Each case is "
            "annotated by hand as `sound` / `flawed_but_coherent` (a real, identifiable error) / "
            "`fabricated` (invented terminology or rules with no actual basis). **These are plain "
            "counts, not a statistical test** -- at this sample size (tens of cases, one human "
            "annotator), a formal significance test would overstate the confidence these numbers "
            "can support. Read the direction, not the precision.\n"
        )

        counts = reasoning_quality_counts(rows)
        for (model, round_), by_condition in counts.items():
            lines.append(f"**{model} — {round_}**\n")
            lines.append("| condition | sound | flawed_but_coherent | fabricated | n annotated |")
            lines.append("|---|---|---|---|---|")
            for condition, cq in by_condition.items():
                total = sum(cq.values())
                lines.append(
                    f"| {condition} | {cq.get('sound', 0)} | {cq.get('flawed_but_coherent', 0)} "
                    f"| {cq.get('fabricated', 0)} | {total} |"
                )
            lines.append("")

        lines.append("**All annotated cases:**\n")
        lines.append("| round | model | case | condition | judgment | reasoning quality |")
        lines.append("|---|---|---|---|---|---|")
        for n in notes:
            lines.append(f"| {n['round']} | {n['model']} | {n['case_text']} | {n['condition']} | {n['judgment']} | {n['reasoning_quality']} |")
        lines.append("")

    if special_cases:
        lines.append("## Special cases (excluded from accept/reject scoring)\n")
        lines.append(
            "A case whose correct answer does not reduce to a clean accept/reject is logged "
            "here instead of forced into the binary scale above.\n"
        )
        lines.append("| round | model | case | condition | judgment | note |")
        lines.append("|---|---|---|---|---|---|")
        for sc in special_cases:
            lines.append(
                f"| {sc['round']} | {sc['model']} | {sc['case_text']} | {sc['condition']} "
                f"| {sc['judgment']} | {sc['note']} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_report(path: str | Path, markdown: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(markdown)
