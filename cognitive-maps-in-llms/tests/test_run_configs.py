"""Tests for validated, grouped task-config execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import toml

from tools import run_configs as runner


pytestmark = pytest.mark.codex


def _write_config(
    path: Path,
    *,
    output_folder: Path,
    kind: str = "recording",
    model: str = "model_a",
    prompt: str = "ABC_family_tree_who_is",
    settings: dict[str, Any] | None = None,
    dependencies: dict[str, Path] | None = None,
    experiment: str | None = None,
) -> Path:
    """Write one task config using the public schema."""
    inputs: dict[str, Any] = {
        "kind": kind,
        "model_preset": model,
        "prompt_preset": prompt,
        "settings": settings or {},
    }
    if experiment is not None:
        inputs["experiment"] = experiment
    document = {
        "output_folder": str(output_folder),
        "metadata": {},
        "inputs": inputs,
        "dependencies": {
            name: str(dependency)
            for name, dependency in (dependencies or {}).items()
        },
        "scheduling": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml.dumps(document), encoding="utf-8")
    return path


def _install_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "get_preset_model_names", lambda: ["model_a", "model_b"])
    monkeypatch.setattr(
        runner,
        "get_preset_prompt_names",
        lambda: ["ABC_family_tree_who_is"],
    )


def test_run_configs_orders_dependencies_and_reuses_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_presets(monkeypatch)
    producer_output = tmp_path / "producer_output"
    consumer_output = tmp_path / "consumer_output"
    producer = _write_config(
        tmp_path / "producer.toml",
        output_folder=producer_output,
        settings={"n_trials": 2, "start_seed": 100},
    )
    consumer = _write_config(
        tmp_path / "consumer.toml",
        output_folder=consumer_output,
        kind="erasure",
        settings={"n_trials": 2, "start_seed": 200},
        dependencies={"recordings": producer},
    )

    calls: list[tuple[Any, ...]] = []
    model_loads: list[str] = []
    runtime_builds: list[str] = []

    def fake_load_model(model_name: str) -> object:
        model_loads.append(model_name)
        return object()

    def fake_build_runtime(model: object, prompt_name: str) -> Any:
        runtime_builds.append(prompt_name)
        return SimpleNamespace(backend="backend", prompt_runtime="prompt")

    def fake_recording(
        backend: Any,
        prompt_runtime: Any,
        output_dir: Path,
        *,
        n_trials: int,
        start_seed: int,
    ) -> None:
        calls.append(("recording", output_dir, n_trials, start_seed))

    def fake_erasure(
        backend: Any,
        prompt_runtime: Any,
        output_dir: Path,
        *,
        benchmark_name: str,
        recordings: Path,
        start_seed: int,
        n_trials: int,
    ) -> None:
        assert (recordings / runner.COMPLETION_FILENAME).is_file()
        calls.append(
            (
                "erasure",
                output_dir,
                benchmark_name,
                recordings,
                start_seed,
                n_trials,
            )
        )

    monkeypatch.setattr(runner, "load_preset_model", fake_load_model)
    monkeypatch.setattr(runner, "build_prompt_runtime", fake_build_runtime)
    monkeypatch.setattr(runner, "run_recording_task", fake_recording)
    monkeypatch.setattr(runner, "run_erasure_task", fake_erasure)

    runner.run_configs([consumer, producer])

    stdout = capsys.readouterr().out
    assert "completed recording: ABC_family_tree_who_is/producer" in stdout
    assert "completed erasure: ABC_family_tree_who_is/consumer" in stdout
    assert [call[0] for call in calls] == ["recording", "erasure"]
    assert calls[1][2] == "family_tree_who_is"
    assert calls[1][3] == producer_output
    assert calls[1][4:] == (200, 2)
    assert model_loads == ["model_a"]
    assert runtime_builds == ["ABC_family_tree_who_is"]
    for config, output in ((producer, producer_output), (consumer, consumer_output)):
        assert (output / runner.COPIED_CONFIG_FILENAME).read_bytes() == config.read_bytes()
        completion = json.loads(
            (output / runner.COMPLETION_FILENAME).read_text()
        )
        assert completion["config"] == runner.COPIED_CONFIG_FILENAME
        assert completion["config_sha256"] == hashlib.sha256(
            config.read_bytes()
        ).hexdigest()
        assert isinstance(completion["completed_at"], str)


def test_invalid_batch_loads_nothing_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_presets(monkeypatch)
    valid_output = tmp_path / "valid_output"
    invalid_output = tmp_path / "invalid_output"
    valid = _write_config(
        tmp_path / "valid.toml",
        output_folder=valid_output,
        settings={"n_trials": 2, "start_seed": 0},
    )
    invalid = _write_config(
        tmp_path / "invalid.toml",
        output_folder=invalid_output,
        settings={"n_trials": 2, "seed_start": 0},
    )
    monkeypatch.setattr(
        runner,
        "load_preset_model",
        lambda name: pytest.fail("model should not be loaded"),
    )

    with pytest.raises(ValueError, match="seed_start"):
        runner.run_configs([valid, invalid])

    assert not valid_output.exists()
    assert not invalid_output.exists()


@pytest.mark.parametrize("fraction", [0, 1, True, "two-thirds"])
def test_config_validation_rejects_invalid_dividing_layer_fraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fraction: object,
) -> None:
    _install_presets(monkeypatch)
    config = _write_config(
        tmp_path / "erasure.toml",
        output_folder=tmp_path / "output",
        kind="erasure",
        settings={
            "n_trials": 1,
            "start_seed": 0,
            "dividing_layer_fraction": fraction,
        },
        dependencies={"recordings": tmp_path / "recordings.toml"},
    )

    with pytest.raises(ValueError, match="dividing_layer_fraction"):
        runner._parse_config(config)


def test_dry_run_checks_external_dependency_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_presets(monkeypatch)
    producer_output = tmp_path / "old_output"
    producer_output.mkdir()
    (producer_output / runner.COMPLETION_FILENAME).write_text("{}")
    producer = _write_config(
        tmp_path / "old_producer.toml",
        output_folder=producer_output,
        settings={"n_trials": 1, "start_seed": 0},
    )
    consumer_output = tmp_path / "new_output"
    consumer = _write_config(
        tmp_path / "consumer.toml",
        output_folder=consumer_output,
        kind="erasure",
        settings={"n_trials": 1, "start_seed": 0},
        dependencies={"recordings": producer},
    )
    monkeypatch.setattr(
        runner,
        "load_preset_model",
        lambda name: pytest.fail("dry run must not load a model"),
    )

    runner.run_configs([consumer], dry_run=True)

    assert not consumer_output.exists()


def test_task_failure_leaves_config_without_completion_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_presets(monkeypatch)
    first_output = tmp_path / "first_output"
    second_output = tmp_path / "second_output"
    first = _write_config(
        tmp_path / "first.toml",
        output_folder=first_output,
        settings={"n_trials": 1, "start_seed": 0},
    )
    second = _write_config(
        tmp_path / "second.toml",
        output_folder=second_output,
        settings={"n_trials": 1, "start_seed": 1},
    )

    monkeypatch.setattr(runner, "load_preset_model", lambda name: object())
    monkeypatch.setattr(
        runner,
        "build_prompt_runtime",
        lambda model, prompt: SimpleNamespace(
            backend="backend",
            prompt_runtime="prompt",
        ),
    )

    def fail_recording(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("model pass failed")

    monkeypatch.setattr(runner, "run_recording_task", fail_recording)

    with pytest.raises(RuntimeError, match="model pass failed"):
        runner.run_configs([first, second])

    assert (first_output / runner.COPIED_CONFIG_FILENAME).is_file()
    assert not (first_output / runner.COMPLETION_FILENAME).exists()
    assert not second_output.exists()


def test_run_configs_executes_mosaic_experiment_with_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_presets(monkeypatch)
    producer_output = tmp_path / "recordings_output"
    producer_output.mkdir()
    (producer_output / runner.COMPLETION_FILENAME).write_text("{}")
    producer = _write_config(
        tmp_path / "recordings.toml",
        output_folder=producer_output,
        settings={"n_trials": 1, "start_seed": 0},
    )
    output = tmp_path / "mosaic_output"
    config = _write_config(
        tmp_path / "mosaic.toml",
        output_folder=output,
        kind="mosaic_experiment",
        settings={"n_trials": 2},
        dependencies={"recordings": producer},
        experiment="fake_experiment",
    )

    class FakeExperiment:
        def __init__(
            self,
            backend: Any,
            prompt_runtime: Any,
            n_trials: int,
            recordings: Path,
        ) -> None:
            pass

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        runner,
        "get_experiment_class",
        lambda name: FakeExperiment,
    )
    monkeypatch.setattr(runner, "load_preset_model", lambda name: "model")
    monkeypatch.setattr(
        runner,
        "build_prompt_runtime",
        lambda model, prompt: SimpleNamespace(
            backend="backend",
            prompt_runtime="prompt",
        ),
    )

    def fake_run_experiment(
        backend: Any,
        prompt_runtime: Any,
        output_dir: Path,
        *,
        experiment_cls: type[Any],
        n_trials: int,
        recordings: Path,
    ) -> None:
        calls.append(
            (
                backend,
                prompt_runtime,
                output_dir,
                experiment_cls,
                n_trials,
                recordings,
            )
        )

    monkeypatch.setattr(runner, "run_experiment_task", fake_run_experiment)

    runner.run_configs([config])

    assert calls == [
        (
            "backend",
            "prompt",
            output,
            FakeExperiment,
            2,
            producer_output,
        )
    ]
    assert (output / runner.COPIED_CONFIG_FILENAME).read_bytes() == config.read_bytes()
    assert (output / runner.COMPLETION_FILENAME).is_file()


@pytest.mark.parametrize(
    ("experiment", "settings", "match"),
    [
        ("missing", {"n_trials": 1}, "Unknown experiment"),
        ("fake_experiment", {}, "n_trials"),
        ("fake_experiment", {"n_trials": 1, "typo": True}, "typo"),
        ("fake_experiment", {"n_trials": 1, "output_dir": "bad"}, "reserved"),
    ],
)
def test_invalid_mosaic_config_loads_no_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment: str,
    settings: dict[str, Any],
    match: str,
) -> None:
    _install_presets(monkeypatch)

    class FakeExperiment:
        def __init__(
            self,
            backend: Any,
            prompt_runtime: Any,
            n_trials: int,
        ) -> None:
            pass

    real_get_experiment_class = runner.get_experiment_class

    def fake_get_experiment_class(name: str) -> type[Any]:
        if name == "fake_experiment":
            return FakeExperiment
        return real_get_experiment_class(name)

    monkeypatch.setattr(runner, "get_experiment_class", fake_get_experiment_class)
    monkeypatch.setattr(
        runner,
        "load_preset_model",
        lambda name: pytest.fail("model should not be loaded"),
    )
    config = _write_config(
        tmp_path / "mosaic.toml",
        output_folder=tmp_path / "output",
        kind="mosaic_experiment",
        settings=settings,
        experiment=experiment,
    )

    with pytest.raises(ValueError, match=match):
        runner.run_configs([config])

    assert not (tmp_path / "output").exists()


def test_run_configs_rejects_dependency_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_presets(monkeypatch)
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    _write_config(
        first,
        output_folder=tmp_path / "first_output",
        kind="erasure",
        settings={"n_trials": 1, "start_seed": 0},
        dependencies={"recordings": second},
    )
    _write_config(
        second,
        output_folder=tmp_path / "second_output",
        kind="erasure",
        settings={"n_trials": 1, "start_seed": 1},
        dependencies={"recordings": first},
    )

    with pytest.raises(ValueError, match="Dependency cycle"):
        runner.run_configs([first, second], dry_run=True)


@pytest.mark.parametrize(
    ("settings", "missing_setting"),
    [
        ({"n_trials": 1}, "start_seed"),
        ({"start_seed": 0}, "n_trials"),
    ],
)
def test_erasure_requires_seed_and_trial_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: dict[str, Any],
    missing_setting: str,
) -> None:
    _install_presets(monkeypatch)
    config = _write_config(
        tmp_path / "erasure.toml",
        output_folder=tmp_path / "output",
        kind="erasure",
        settings=settings,
        dependencies={"recordings": tmp_path / "recording.toml"},
    )
    monkeypatch.setattr(
        runner,
        "load_preset_model",
        lambda name: pytest.fail("invalid config must not load a model"),
    )

    with pytest.raises(ValueError, match=missing_setting):
        runner.run_configs([config], dry_run=True)

    assert not (tmp_path / "output").exists()
