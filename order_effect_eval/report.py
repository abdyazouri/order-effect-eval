"""Load a long-format pilot-results CSV and produce a markdown report:
per-model agreement with ground truth under each ordering, and the flip rate
between orderings -- computed only from cases where that model has a
judgment for both orderings, with the available case count reported
explicitly rather than silently padding missing cells.

Expected CSV columns: case_id, case_text, ground_truth, model, condition, judgment
  - condition is one of "reasoning_first" / "judgment_first"
  - judgment and ground_truth are one of "accept" / "reject"
A model/case/condition combination with no row is treated as missing, not as
any particular judgment -- this project never invents a result for a case
that was not actually run.
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


def _index(rows: list[dict]) -> dict:
    """model -> condition -> case_id -> judgment, plus case_id -> ground_truth."""
    by_model: dict = defaultdict(lambda: defaultdict(dict))
    ground_truth: dict = {}
    for row in rows:
        by_model[row["model"]][row["condition"]][row["case_id"]] = row["judgment"]
        ground_truth[row["case_id"]] = row["ground_truth"]
    return by_model, ground_truth


def reasoning_quality_notes(rows: list[dict]) -> list[dict]:
    """Every row that has been manually annotated with a reasoning_quality
    label (sound / flawed_but_coherent / fabricated), regardless of whether
    the judgment itself was correct. This is a qualitative log, not a
    statistic: with only a handful of cases reviewed this closely so far, it
    is reported as individual annotated examples, not aggregated into a rate
    that would overstate how much has actually been checked."""
    return [
        {
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


def per_model_summary(rows: list[dict]) -> list[dict]:
    by_model, ground_truth = _index(rows)
    summaries = []

    for model, conditions in by_model.items():
        rf = conditions.get("reasoning_first", {})
        jf = conditions.get("judgment_first", {})

        for condition_name, judgments in [("reasoning_first", rf), ("judgment_first", jf)]:
            if not judgments:
                continue
            case_ids = sorted(judgments.keys())
            truth = [ground_truth[c] for c in case_ids]
            model_j = [judgments[c] for c in case_ids]
            point, lo, hi = bootstrap_ac1_ci(truth, model_j, n_boot=5000, seed=0)
            summaries.append({
                "model": model,
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
                "condition": "flip_rate (paired)",
                "n_cases": len(paired_ids),
                "ac1": round(rate, 3),
                "ci_low": None,
                "ci_high": None,
                "flipped_case_ids": [paired_ids[i] for i in flipped],
            })

    return summaries


def render_markdown(rows: list[dict]) -> str:
    summaries = per_model_summary(rows)
    by_model: dict = defaultdict(list)
    for s in summaries:
        by_model[s["model"]].append(s)

    lines = ["# Order-Effect Pilot Report\n"]
    lines.append(
        "Does evaluation order (reasoning-first vs. judgment-first) change whether a model "
        "accepts a conclusion built on a false premise? Reported per model, using only the "
        "cases actually run for that model/condition -- a missing cell means the case was not "
        "run, never an assumed result.\n"
    )

    for model, items in by_model.items():
        lines.append(f"## {model}\n")
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
        lines.append("## Reasoning-quality annotations (qualitative, not a statistic)\n")
        lines.append(
            "A judgment can be *correct* while the reasoning behind it is not -- this is logged "
            "per case as it is reviewed, not aggregated into a rate, since only a small, uneven "
            "subset of cases has been reviewed this closely so far. `flawed_but_coherent` means a "
            "real but identifiable error; `fabricated` means invented terminology or rules with no "
            "actual basis.\n"
        )
        lines.append("| model | case | condition | judgment | reasoning quality |")
        lines.append("|---|---|---|---|---|")
        for n in notes:
            lines.append(f"| {n['model']} | {n['case_text']} | {n['condition']} | {n['judgment']} | {n['reasoning_quality']} |")
        lines.append("")

    return "\n".join(lines)


def write_report(path: str | Path, markdown: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(markdown)
