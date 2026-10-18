from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.codex


def _script():
    path = Path(__file__).resolve().parents[1] / "figures/scripts/render_main_text_analogs.py"
    spec = importlib.util.spec_from_file_location("render_main_text_analogs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("selection_args, pair_count, row_count", [
    (("--all-prompts",), 14, 70),
    (("--all-models",), 11, 55),
])
def test_systematic_runs_report_missing_inputs_without_rendering(
    tmp_path, monkeypatch, selection_args, pair_count, row_count,
):
    results = tmp_path / "results"
    results.mkdir()
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "render_main_text_analogs.py", *selection_args,
        "--results-root", str(results),
        "--output-dir", str(output),
    ])
    script = _script()
    script.main()
    report = output / "skipped_groups.csv"
    with report.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == row_count
    assert all(r["group"] != "scaling" for r in rows)
    pairs = {(r["model"], r["prompt"]) for r in rows if r["group"] != "scaling"}
    assert len(pairs) == pair_count
    assert all(r["reason"].startswith("Missing files:") for r in rows)
    for row in rows:
        if row["group"] == "erasure":
            assert str(results / row["model"] / row["prompt"] / "iterative_erasure") in row["reason"]
    if selection_args == ("--all-prompts",):
        assert {model for model, _ in pairs} == {"gemma-27b"}
        assert {prompt for _, prompt in pairs} == set(script.PROMPTS)
    else:
        assert {prompt for _, prompt in pairs} == {"ABC_family_tree_yes_no"}
    before = report.read_bytes()
    with pytest.raises(FileExistsError):
        script.main()
    assert report.read_bytes() == before
    assert not list(output.rglob("*.svg"))
    assert not list(output.rglob("*.pdf"))


def test_default_model_and_prompt(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "render_main_text_analogs.py",
        "--group", "separability",
        "--results-root", str(results),
        "--output-dir", str(output),
    ])

    _script().main()

    with (output / "skipped_groups.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["model"] == "gemma-27b"
    assert rows[0]["prompt"] == "ABC_family_tree_yes_no"


def test_all_models_and_all_prompts_can_be_selected_explicitly(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "render_main_text_analogs.py",
        "--all-models",
        "--all-prompts",
        "--group", "separability",
        "--results-root", str(results),
        "--output-dir", str(output),
    ])

    script = _script()
    script.main()

    with (output / "skipped_groups.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 11 * len(script.PROMPTS)
    assert {row["model"] for row in rows} == {
        "gemma-4b", "gemma-12b", "gemma-27b",
        "llama-8b", "llama-70b",
        "qwen-0p6b", "qwen-1p7b", "qwen-4b", "qwen-8b", "qwen-14b", "qwen-32b",
    }
    assert {row["prompt"] for row in rows} == set(script.PROMPTS)


@pytest.mark.parametrize("failure, should_skip", [
    (KeyError("Query 'E --mother--> D?' is missing from record 0"), True),
    (ValueError("No records in test.ndjson matched trial axes {'patch_location': 'B1'}"), True),
    (RuntimeError("unexpected plotting failure"), False),
    (KeyError("unexpected field"), False),
    (ValueError("invalid data"), False),
    (KeyError("Benchmark query 'test' is missing from record 0"), True),
    (ValueError("No result records matched 2 benchmark case(s)"), True),
])
def test_missing_selections_are_skipped_but_bugs_stop(tmp_path, monkeypatch, failure, should_skip):
    import matplotlib.pyplot as plt
    from cognitive_maps_in_llms.analysis import main_text_analogs

    before = set(plt.get_fignums())

    def fail(*args, **kwargs):
        plt.subplots()  # Partial rendering must be cleaned up even on failure.
        raise failure
    monkeypatch.setattr(main_text_analogs, "render_separability", fail)
    path = tmp_path / "results/gemma-27b/ABC_family_tree_yes_no/exp_2_separability/results.ndjson"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "render_main_text_analogs.py", "--models", "gemma-27b",
        "--group", "separability",
        "--results-root", str(tmp_path / "results"), "--output-dir", str(output),
    ])
    if should_skip:
        _script().main()
    else:
        with pytest.raises(type(failure)) as caught:
            _script().main()
        assert caught.value is failure
    assert set(plt.get_fignums()) == before
    with (output / "skipped_groups.csv").open() as file:
        assert len(list(csv.DictReader(file))) == int(should_skip)


@pytest.mark.parametrize("scaling_only", [False, True])
def test_scaling_is_explicit_and_uses_prompt_directory(tmp_path, monkeypatch, scaling_only):
    import matplotlib.pyplot as plt
    from types import SimpleNamespace
    from cognitive_maps_in_llms.analysis import main_text_analogs, scaling_plots

    prompt = "ABC_family_tree_yes_no"
    root = tmp_path / "results"
    for preset in (prompt, "ABC_alien_family_tree_yes_no"):
        for models in scaling_plots.DEFAULT_MODEL_FAMILIES.values():
            for model in models:
                path = root / model / preset / "exp_1_behavior/results.ndjson"
                path.parent.mkdir(parents=True)
                path.write_text("{}\n")
    calls = []

    def plot():
        fig, ax = plt.subplots()
        return SimpleNamespace(fig=fig, ax=ax)

    def scaling(results, **kwargs):
        calls.append("scaling")
        assert kwargs["prompt_preset"] == prompt
        return plot()

    def behavior(results, **kwargs):
        calls.append("behavior")
        assert kwargs["include_scaling"] is False
        return {"behavior/example": plot()}

    monkeypatch.setattr(main_text_analogs, "render_behavior_scaling", scaling)
    monkeypatch.setattr(main_text_analogs, "render_behavior", behavior)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "render_main_text_analogs.py",
        *( ["--scaling-only"] if scaling_only else ["--group", "behavior"] ),
        "--results-root", str(root), "--output-dir", str(output),
    ])
    _script().main()
    assert calls == (["scaling"] if scaling_only else ["behavior"])
    expected = (output / prompt / "scaling.pdf" if scaling_only else
                output / "gemma-27b" / prompt / "behavior/example.pdf")
    assert list(output.rglob("*.pdf")) == [expected]
    assert expected.read_bytes().startswith(b"%PDF-")
    assert not list(output.rglob("*.svg"))

    original = expected.read_bytes()
    with pytest.raises(FileExistsError):
        _script().main()
    assert expected.read_bytes() == original

    expected.write_bytes(b"old panel")
    unrelated = output / "unrelated.txt"
    unrelated.write_text("keep")
    sys.argv.append("--overwrite")
    _script().main()
    assert expected.read_bytes().startswith(b"%PDF-")
    assert unrelated.read_text() == "keep"
    with (output / "skipped_groups.csv").open() as file:
        assert list(csv.DictReader(file)) == []

    # A later skipped render retains the old panel and flags it in the report.
    for path in root.rglob("results.ndjson"):
        path.unlink()
    original = expected.read_bytes()
    _script().main()
    assert expected.read_bytes() == original
    with (output / "skipped_groups.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert "may be stale" in rows[0]["reason"]


@pytest.mark.parametrize("extra", [
    ["--models", "gemma-27b"], ["--all-models"], ["--group", "behavior"],
])
def test_scaling_rejects_model_and_group_selection(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", [
        "render_main_text_analogs.py", "--scaling-only", *extra,
        "--results-root", str(tmp_path), "--output-dir", str(tmp_path / "output"),
    ])
    with pytest.raises(SystemExit) as exc:
        _script().main()
    assert exc.value.code == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("prompt", [
    "ABC_older_younger", "ABC_east_west", "ABC_north_south", "ABC_ahead_behind",
    "ABC_proposition_equivalence", "ABC_variable_equivalence",
    "ABC_family_trees_descendant_ancestor", "ABC_older_younger_unique_locations_yes_no",
])
def test_scaling_reports_missing_alien_input(tmp_path, monkeypatch, prompt):
    script = _script()
    root = tmp_path / "results"
    alien = prompt.replace("ABC_", "ABC_alien_", 1)
    missing = root / "gemma-27b" / alien / "exp_1_behavior/results.ndjson"
    for preset in (prompt, alien):
        for model in script.all_models():
            path = root / model / preset / "exp_1_behavior/results.ndjson"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path != missing:
                path.write_text("{}\n")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "render_main_text_analogs.py", "--scaling-only", "--prompts", prompt,
        "--results-root", str(root), "--output-dir", str(output),
    ])
    script.main()
    with (output / "skipped_groups.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["reason"] == "Missing files: " + str(missing)
    assert not list(output.rglob("*.pdf"))
