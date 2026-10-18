"""Tests for deterministic task scheduling annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import toml

from tools import populate_scheduling as scheduling


pytestmark = pytest.mark.codex


def _document(
    *,
    kind: str,
    prompt: str,
    n_trials: int,
    model: str = "gemma-27b",
    experiment: str | None = None,
    scheduling_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "kind": kind,
        "model_preset": model,
        "prompt_preset": prompt,
        "settings": {"n_trials": n_trials},
    }
    if experiment is not None:
        inputs["experiment"] = experiment
    return {
        "output_folder": "/tmp/output",
        "metadata": {},
        "inputs": inputs,
        "dependencies": {},
        "scheduling": scheduling_table or {},
    }


def _write_config(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        toml.dumps(document) + "# TODO: Estimate scheduling resources for this task.\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("prompt", "kind", "experiment", "n_trials", "expected_seconds"),
    [
        ("ABC_family_tree_who_is", "recording", None, 10, 8),
        ("ABC_family_tree_yes_no", "recording", None, 10, 8),
        ("ABC_family_tree_who_is", "erasure", None, 2, 840),
        ("ABC_family_tree_yes_no", "erasure", None, 2, 1680),
        ("ABC_family_tree_who_is", "mosaic_experiment", "exp_1_behavior", 10, 91),
        ("ABC_family_tree_yes_no", "mosaic_experiment", "exp_1_behavior", 10, 182),
        ("ABC_family_tree_who_is", "mosaic_experiment", "exp_2_separability", 10, 243),
        ("ABC_family_tree_yes_no", "mosaic_experiment", "exp_2_separability", 10, 1149),
        ("ABC_family_tree_who_is", "mosaic_experiment", "exp_3_additivity", 10, 55),
        ("ABC_family_tree_yes_no", "mosaic_experiment", "exp_3_additivity", 10, 100),
        ("ABC_family_tree_who_is", "mosaic_experiment", "exp_4_retrospective_prospective", 10, 412),
        ("ABC_family_tree_yes_no", "mosaic_experiment", "exp_4_retrospective_prospective", 10, 775),
        ("ABC_family_tree_who_is", "mosaic_experiment", "exp_5a_bottleneck", 10, 97),
        ("ABC_family_tree_yes_no", "mosaic_experiment", "exp_5a_bottleneck", 10, 188),
        ("ABC_family_tree_who_is", "mosaic_experiment", "exp_5b_sparse_mosaic", 10, 146),
        ("ABC_family_tree_yes_no", "mosaic_experiment", "exp_5b_sparse_mosaic", 10, 282),
        ("ABC_family_tree_who_is", "mosaic_experiment", "exp_5c_editing", 10, 678),
        ("ABC_family_tree_yes_no", "mosaic_experiment", "exp_5c_editing", 10, 3217),
        ("ABC_family_tree_who_is", "mosaic_experiment", "exp_s1_positions_rope", 10, 49),
        ("ABC_family_tree_yes_no", "mosaic_experiment", "exp_s1_positions_rope", 10, 94),
        ("ABC_family_tree_who_is", "mosaic_experiment", "exp_s2_positions_indirect_swap", 10, 52),
        ("ABC_family_tree_yes_no", "mosaic_experiment", "exp_s2_positions_indirect_swap", 10, 97),
    ],
)
def test_mini_run_regression_estimates(
    prompt: str,
    kind: str,
    experiment: str | None,
    n_trials: int,
    expected_seconds: int,
) -> None:
    document = _document(
        kind=kind,
        prompt=prompt,
        n_trials=n_trials,
        experiment=experiment,
    )

    runtime, profile = scheduling.estimate_runtime_seconds(
        document,
        path=Path("config.toml"),
    )

    assert runtime == expected_seconds
    assert profile == "1xgpu80"


def test_populate_scheduling_dry_run_write_and_idempotency(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "task_configs" / "exp_1_behavior_config.toml",
        _document(
            kind="mosaic_experiment",
            prompt="ABC_family_tree_who_is",
            n_trials=10,
            experiment="exp_1_behavior",
        ),
    )
    original = config.read_bytes()

    preview = scheduling.populate_scheduling(config.parent, dry_run=True)
    assert config.read_bytes() == original
    assert preview[0].replacement is not None

    written = scheduling.populate_scheduling(config.parent)
    assert written[0].replacement is not None
    parsed = scheduling._read_toml(config)
    assert parsed["scheduling"] == {
        "estimated_runtime_seconds": 91,
        "gpu_profile": "1xgpu80",
    }
    assert "TODO: Estimate scheduling resources" not in config.read_text()

    repeated = scheduling.populate_scheduling(config.parent)
    assert repeated[0].replacement is None


def test_conflict_preflight_leaves_every_config_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    first = _write_config(
        root / "a_config.toml",
        _document(
            kind="recording",
            prompt="ABC_family_tree_who_is",
            n_trials=10,
        ),
    )
    second = _write_config(
        root / "b_config.toml",
        _document(
            kind="recording",
            prompt="ABC_family_tree_who_is",
            n_trials=10,
            scheduling_table={
                "estimated_runtime_seconds": 999,
                "gpu_profile": "1xgpu80",
            },
        ),
    )
    before = {path: path.read_bytes() for path in (first, second)}

    with pytest.raises(ValueError, match="conflicting scheduling annotation"):
        scheduling.populate_scheduling(root)

    assert before == {path: path.read_bytes() for path in (first, second)}


def test_matching_skip_annotation_is_preserved(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "task_configs" / "exp_1_behavior_config.toml",
        _document(
            kind="mosaic_experiment",
            prompt="ABC_family_tree_who_is",
            n_trials=10,
            experiment="exp_1_behavior",
            scheduling_table={
                "estimated_runtime_seconds": 91,
                "gpu_profile": "1xgpu80",
                "skip": True,
            },
        ),
    )
    before = config.read_bytes()

    planned = scheduling.populate_scheduling(config.parent)

    assert planned[0].replacement is None
    assert config.read_bytes() == before


@pytest.mark.parametrize(
    ("prompt", "kind", "experiment", "expected"),
    [
        (
            "ABC_alien_older_younger",
            "mosaic_experiment",
            "exp_1_behavior",
            False,
        ),
        ("ABC_alien_older_younger", "recording", None, True),
        ("ABC_alien_older_younger", "erasure", None, True),
        (
            "ABC_alien_older_younger",
            "mosaic_experiment",
            "exp_2_separability",
            True,
        ),
        ("ABC_family_trees_descendant_ancestor", "recording", None, False),
        ("ABC_family_trees_descendant_ancestor", "erasure", None, False),
        (
            "ABC_older_younger_unique_locations_who_is",
            "mosaic_experiment",
            "exp_5a_bottleneck",
            True,
        ),
        (
            "ABC_older_younger_unique_locations_yes_no",
            "mosaic_experiment",
            "exp_2_separability",
            False,
        ),
        ("ABC_family_tree_who_is", "erasure", None, False),
    ],
)
def test_full_run_skip_policy(
    prompt: str,
    kind: str,
    experiment: str | None,
    expected: bool,
) -> None:
    document = _document(
        kind=kind,
        prompt=prompt,
        n_trials=1,
        experiment=experiment,
    )

    assert scheduling.should_skip_for_full_run(
        document,
        path=Path("config.toml"),
    ) is expected


@pytest.mark.parametrize("prompt", [
    "ABC_family_trees_descendant_ancestor",
    "ABC_older_younger_unique_locations_yes_no",
])
@pytest.mark.parametrize(("kind", "experiment"), [
    ("erasure", None),
    ("mosaic_experiment", "exp_5a_bottleneck"),
    ("mosaic_experiment", "exp_5b_sparse_mosaic"),
    ("mosaic_experiment", "exp_5c_editing"),
])
def test_additional_figure_five_prompts_are_included(
    prompt: str, kind: str, experiment: str | None,
) -> None:
    document = _document(kind=kind, prompt=prompt, experiment=experiment, n_trials=10)
    assert not scheduling.should_skip_for_full_run(document, path=Path("config.toml"))


def test_apply_full_run_skip_policy_adds_and_removes_skip(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    excluded = _write_config(
        root / "alien_recording_config.toml",
        _document(
            kind="recording",
            prompt="ABC_alien_older_younger",
            n_trials=10,
        ),
    )
    retained = _write_config(
        root / "new_erasure_config.toml",
        _document(
            kind="erasure",
            prompt="ABC_family_trees_descendant_ancestor",
            n_trials=10,
            scheduling_table={
                "estimated_runtime_seconds": 3360,
                "gpu_profile": "1xgpu80",
                "skip": True,
            },
        ),
    )

    planned = scheduling.populate_scheduling(
        root,
        apply_full_run_skip_policy=True,
    )

    assert [item.skip for item in planned] == [True, False]
    assert scheduling._read_toml(excluded)["scheduling"]["skip"] is True
    assert "skip" not in scheduling._read_toml(retained)["scheduling"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_preset", "unknown-model", "unsupported model preset"),
        ("prompt_preset", "unknown-prompt", "unsupported prompt preset"),
    ],
)
def test_unknown_presets_are_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    document = _document(
        kind="recording",
        prompt="ABC_family_tree_who_is",
        n_trials=1,
    )
    document["inputs"][field] = value
    _write_config(tmp_path / "task_config.toml", document)

    with pytest.raises(ValueError, match=message):
        scheduling.populate_scheduling(tmp_path)
