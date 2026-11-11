from order_effect_eval.report import per_model_summary, reasoning_quality_counts, reasoning_quality_notes


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


def test_reasoning_quality_counts_tallies_per_model_and_condition():
    rows = [
        {"case_id": "c0", "model": "m", "condition": "reasoning_first", "reasoning_quality": "sound"},
        {"case_id": "c1", "model": "m", "condition": "reasoning_first", "reasoning_quality": "sound"},
        {"case_id": "c2", "model": "m", "condition": "reasoning_first", "reasoning_quality": "flawed_but_coherent"},
        {"case_id": "c0", "model": "m", "condition": "judgment_first", "reasoning_quality": "fabricated"},
        {"case_id": "c1", "model": "m", "condition": "judgment_first", "reasoning_quality": ""},  # unannotated, excluded
    ]
    counts = reasoning_quality_counts(rows)
    assert counts["m"]["reasoning_first"]["sound"] == 2
    assert counts["m"]["reasoning_first"]["flawed_but_coherent"] == 1
    assert counts["m"]["judgment_first"]["fabricated"] == 1
    assert "sound" not in counts["m"]["judgment_first"] or counts["m"]["judgment_first"]["sound"] == 0


def test_two_models_kept_separate():
    rows = _rows("model-a", ["reject"] * 4, ["reject"] * 4)
    rows += _rows("model-b", ["accept"] * 4, ["reject"] * 4, ground_truth="reject")
    summaries = per_model_summary(rows)
    models = {s["model"] for s in summaries}
    assert models == {"model-a", "model-b"}
