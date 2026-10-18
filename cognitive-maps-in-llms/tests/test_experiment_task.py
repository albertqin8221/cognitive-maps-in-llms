"""Tests for registered mosaic-experiment execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_neuroscience_tools.base_experiment import BaseExperiment, TrialState

from cognitive_maps_in_llms.experiments import (
    get_experiment_class,
    get_experiment_names,
)
from cognitive_maps_in_llms.tasks.experiment import (
    RESULTS_FILENAME,
    run_experiment,
)


pytestmark = pytest.mark.codex


class _FakeExperiment(BaseExperiment):
    calls: list[tuple[Any, ...]] = []

    def __init__(
        self,
        backend: Any,
        prompt_runtime: Any,
        n_trials: int,
        recordings: Path,
    ) -> None:
        self.calls.append((backend, prompt_runtime, n_trials, recordings))

    def run_ndjson(self, out_path: str | Path) -> None:
        Path(out_path).write_text('{"ok": true}\n', encoding="utf-8")

    def define_trial_axes(self) -> dict[str, list[Any]]:
        raise NotImplementedError

    def build_trial_state(self, trial_key: dict[str, Any]) -> TrialState:
        raise NotImplementedError

    def build_trial_queries(self, trial_state: TrialState) -> dict[str, Any]:
        raise NotImplementedError

    def run_query(self, trial_state: TrialState, query: Any) -> Any:
        raise NotImplementedError

    def serialize_result(
        self,
        trial_state: TrialState,
        query: Any,
        logits: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError


def test_registry_contains_only_runnable_experiment_classes() -> None:
    names = get_experiment_names()
    assert names
    for name in names:
        assert issubclass(get_experiment_class(name), BaseExperiment)


def test_unknown_experiment_name_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="Unknown experiment.*exp_1_behavior"):
        get_experiment_class("missing")


def test_run_experiment_forwards_arguments_and_writes_results(
    tmp_path: Path,
) -> None:
    _FakeExperiment.calls.clear()
    output_dir = tmp_path / "output"
    recordings = tmp_path / "recordings"

    run_experiment(
        "backend",  # type: ignore[arg-type]
        "prompt",  # type: ignore[arg-type]
        output_dir,
        experiment_cls=_FakeExperiment,
        n_trials=3,
        recordings=recordings,
    )

    assert _FakeExperiment.calls == [("backend", "prompt", 3, recordings)]
    assert (output_dir / RESULTS_FILENAME).read_text() == '{"ok": true}\n'


def test_run_experiment_refuses_existing_results(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    results_path = output_dir / RESULTS_FILENAME
    results_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_experiment(
            "backend",  # type: ignore[arg-type]
            "prompt",  # type: ignore[arg-type]
            output_dir,
            experiment_cls=_FakeExperiment,
            n_trials=1,
            recordings=tmp_path / "recordings",
        )

    assert results_path.read_text() == "existing\n"
