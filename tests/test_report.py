import pytest

from order_effect_eval.report import (
    load_special_cases,
    per_model_summary,
    reasoning_quality_counts,
    reasoning_quality_notes,
    render_markdown,
)


def _rows(model, reasoning_judgments, judgment_judgments, ground_truth="reject"):
    rows = []
    for i, j in enumerate(reasoning_judgments):
        rows.append({"case_id": f"c{i}", "case_text": "x", "ground_truth": ground_truth,
                      "model": model, "condition": "reasoning_first", "judgment": j})
    for i, j in enumerate(judgment_judgments):
        rows.append({"case_id": f"c{i}", "case_text": "x", "ground_truth": ground_truth,
                      "model": model, "condition": "judgment_first", "judgment": j})
    return rows


def test_complete_model_both_conditions_present():
    rows = _rows("model-a", ["reject"] * 5, ["reject"] * 5)
    summaries = per_model_summary(rows)
    conditions = {s["condition"]: s for s in summaries}
    assert conditions["reasoning_first"]["n_cases"] == 5
    assert conditions["judgment_first"]["n_cases"] == 5
    assert conditions["flip_rate (paired)"]["n_cases"] == 5
    assert conditions["flip_rate (paired)"]["ac1"] == 0.0


def test_partial_model_uses_only_available_cases():
    # 5 cases under reasoning_first, only 3 under judgment_first (c0,c1,c2)
    rows = _rows("model-b", ["reject"] * 5, [])
    rows += [{"case_id": f"c{i}", "case_text": "x", "ground_truth": "reject",
              "model": "model-b", "condition": "judgment_first", "judgment": "reject"} for i in range(3)]
    summaries = per_model_summary(rows)
    conditions = {s["condition"]: s for s in summaries}
    assert conditions["reasoning_first"]["n_cases"] == 5
    assert conditions["judgment_first"]["n_cases"] == 3
    # flip rate only over the 3 cases present in BOTH conditions
    assert conditions["flip_rate (paired)"]["n_cases"] == 3


def test_flip_detected_and_case_ids_reported():
    rows = _rows("model-c", ["reject", "accept", "reject"], ["reject", "reject", "reject"])
    summaries = per_model_summary(rows)
    flip = [s for s in summaries if s["condition"] == "flip_rate (paired)"][0]
    assert flip["n_cases"] == 3
    assert abs(flip["ac1"] - (1 / 3)) < 1e-3  # report.py rounds to 3 decimals
    assert flip["flipped_case_ids"] == ["c1"]


def test_reasoning_quality_notes_only_includes_annotated_rows():
    rows = [
        {"case_id": "c0", "case_text": "x", "ground_truth": "reject", "model": "m",
         "condition": "reasoning_first", "judgment": "reject", "reasoning_quality": "flawed_but_coherent"},
        {"case_id": "c1", "case_text": "y", "ground_truth": "reject", "model": "m",
         "condition": "judgment_first", "judgment": "reject", "reasoning_quality": ""},
        {"case_id": "c1", "case_text": "y", "ground_truth": "reject", "model": "m",
         "condition": "judgment_first", "judgment": "reject"},  # key missing entirely
    ]
    notes = reasoning_quality_notes(rows)
    assert len(notes) == 1
    assert notes[0]["case_id"] == "c0"
    assert notes[0]["reasoning_quality"] == "flawed_but_coherent"


def test_reasoning_quality_counts_tallies_per_model_round_and_condition():
    rows = [
        {"case_id": "c0", "model": "m", "round": "round1", "condition": "reasoning_first", "reasoning_quality": "sound"},
        {"case_id": "c1", "model": "m", "round": "round1", "condition": "reasoning_first", "reasoning_quality": "sound"},
        {"case_id": "c2", "model": "m", "round": "round1", "condition": "reasoning_first", "reasoning_quality": "flawed_but_coherent"},
        {"case_id": "c0", "model": "m", "round": "round1", "condition": "judgment_first", "reasoning_quality": "fabricated"},
        {"case_id": "c1", "model": "m", "round": "round1", "condition": "judgment_first", "reasoning_quality": ""},  # unannotated, excluded
    ]
    counts = reasoning_quality_counts(rows)
    key = ("m", "round1")
    assert counts[key]["reasoning_first"]["sound"] == 2
    assert counts[key]["reasoning_first"]["flawed_but_coherent"] == 1
    assert counts[key]["judgment_first"]["fabricated"] == 1
    assert "sound" not in counts[key]["judgment_first"] or counts[key]["judgment_first"]["sound"] == 0


def test_reasoning_quality_counts_keeps_rounds_separate():
    rows = [
        {"case_id": "c0", "model": "m", "round": "round1", "condition": "reasoning_first", "reasoning_quality": "sound"},
        {"case_id": "c0", "model": "m", "round": "round2", "condition": "reasoning_first", "reasoning_quality": "fabricated"},
    ]
    counts = reasoning_quality_counts(rows)
    assert counts[("m", "round1")]["reasoning_first"]["sound"] == 1
    assert counts[("m", "round2")]["reasoning_first"]["fabricated"] == 1


def test_two_models_kept_separate():
    rows = _rows("model-a", ["reject"] * 4, ["reject"] * 4)
    rows += _rows("model-b", ["accept"] * 4, ["reject"] * 4, ground_truth="reject")
    summaries = per_model_summary(rows)
    models = {s["model"] for s in summaries}
    assert models == {"model-a", "model-b"}


def _rows_with_round(model, round_, reasoning_judgments, judgment_judgments, ground_truth="reject"):
    rows = []
    for i, j in enumerate(reasoning_judgments):
        rows.append({"case_id": f"c{i}", "case_text": "x", "ground_truth": ground_truth,
                      "model": model, "round": round_, "condition": "reasoning_first", "judgment": j})
    for i, j in enumerate(judgment_judgments):
        rows.append({"case_id": f"c{i}", "case_text": "x", "ground_truth": ground_truth,
                      "model": model, "round": round_, "condition": "judgment_first", "judgment": j})
    return rows


def test_rounds_are_scored_independently_not_pooled():
    # Same model, same case ids, but round1 is clean and round2 has a flip.
    # If rounds were pooled instead of kept separate, the round1 n=5 clean
    # pair and the round2 n=5 flipped pair would merge into one n=10 flip
    # rate of 10%, hiding that the flip is confined entirely to round2.
    rows = _rows_with_round("m", "round1", ["reject"] * 5, ["reject"] * 5)
    rows += _rows_with_round("m", "round2", ["reject"] * 5, ["reject"] * 4 + ["accept"])
    summaries = per_model_summary(rows)
    flips = {s["round"]: s for s in summaries if s["condition"] == "flip_rate (paired)"}
    assert flips["round1"]["ac1"] == 0.0
    assert flips["round2"]["ac1"] == pytest.approx(0.2)
    assert flips["round2"]["flipped_case_ids"] == ["c4"]


def test_load_special_cases_missing_file_returns_empty(tmp_path):
    assert load_special_cases(tmp_path / "does_not_exist.csv") == []


def test_load_special_cases_reads_rows(tmp_path):
    p = tmp_path / "special.csv"
    p.write_text("round,case_id,case_text,model,condition,judgment,note\n"
                 "round2,r2c10,birds,model-a,reasoning_first,valid_but_unsound,some note\n")
    rows = load_special_cases(p)
    assert len(rows) == 1
    assert rows[0]["judgment"] == "valid_but_unsound"
    assert rows[0]["note"] == "some note"


def test_render_markdown_includes_special_cases_section():
    rows = _rows_with_round("m", "round1", ["reject"] * 3, ["reject"] * 3)
    special = [{"round": "round2", "model": "m", "case_text": "birds", "condition": "reasoning_first",
                "judgment": "valid_but_unsound", "note": "a note about validity vs soundness"}]
    md = render_markdown(rows, special)
    assert "Special cases" in md
    assert "valid_but_unsound" in md
    assert "a note about validity vs soundness" in md


def test_render_markdown_omits_special_cases_section_when_empty():
    rows = _rows_with_round("m", "round1", ["reject"] * 3, ["reject"] * 3)
    md = render_markdown(rows, [])
    assert "Special cases" not in md
