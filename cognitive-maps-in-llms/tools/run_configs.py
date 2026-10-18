"""Validate and run one or more generated task configurations.

Usage:
    python run_configs.py CONFIG [CONFIG ...]
    python run_configs.py CONFIG [CONFIG ...] --dry-run

The complete batch is validated before any model is loaded or output is
written. Configurations are grouped by model, and dependencies within each
group run in topological order.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import torch
from transformers.utils import logging as transformers_logging

from llm_neuroscience_tools.base_experiment import BaseExperiment
from llm_neuroscience_tools.presets import (
    build_prompt_runtime,
    get_preset_model_names,
    get_preset_prompt_names,
    load_preset_model,
)

from cognitive_maps_in_llms.benchmarks import get_default_benchmark_name
from cognitive_maps_in_llms.experiments import get_experiment_class
from cognitive_maps_in_llms.tasks.erasure import run_erasure as run_erasure_task
from cognitive_maps_in_llms.tasks.experiment import (
    run_experiment as run_experiment_task,
)
from cognitive_maps_in_llms.tasks.recording import (
    record_activations_and_kv as run_recording_task,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib


COMPLETION_FILENAME = "completed.json"
COPIED_CONFIG_FILENAME = "config.toml"
TASK_KINDS = {"recording", "erasure", "mosaic_experiment"}
_TOP_LEVEL_FIELDS = {
    "output_folder",
    "metadata",
    "inputs",
    "dependencies",
    "scheduling",
}
_COMMON_INPUT_FIELDS = {"kind", "model_preset", "prompt_preset", "settings"}


def _configure_model_loading_progress() -> None:
    """Hide Transformers progress bars when stderr is not interactive."""
    if not sys.stderr.isatty():
        transformers_logging.disable_progress_bar()


@dataclass(frozen=True)
class _TaskConfig:
    """Validated information needed to execute one task config."""

    path: Path
    output_folder: Path
    kind: str
    model_preset: str
    prompt_preset: str
    experiment_cls: type[BaseExperiment] | None
    settings: dict[str, Any]
    dependencies: dict[str, Path]


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML document from disk.

    Args:
        path: Path to the TOML file.

    Returns:
        The parsed top-level TOML table.
    """
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _absolute_path(value: Any, *, context: str) -> Path:
    """Validate an absolute path read from a configuration.

    Args:
        value: Value that should contain an absolute path string.
        context: Human-readable field name used in validation errors.

    Returns:
        The validated path. The path need not exist yet.

    Raises:
        ValueError: If ``value`` is not an absolute path string.
    """
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{context} must be an absolute path string: {value!r}")
    return Path(value)


def _parse_config(path: Path) -> _TaskConfig:
    """Parse the fields that can be validated from one config in isolation.

    This checks the config schema, preset names, task arguments, and inexpensive
    setting constraints. Dependency existence and relationships are checked
    later, once every config in the requested batch has been parsed.

    Args:
        path: Absolute path to the task configuration.

    Returns:
        A normalized task configuration used internally by the runner.

    Raises:
        OSError: If the config cannot be read.
        ValueError: If the config schema or task arguments are invalid.
    """
    document = _read_toml(path)
    if set(document) != _TOP_LEVEL_FIELDS:
        raise ValueError(
            f"expected exactly these top-level fields: {sorted(_TOP_LEVEL_FIELDS)}"
        )
    for table_name in ("metadata", "inputs", "dependencies", "scheduling"):
        if not isinstance(document[table_name], dict):
            raise ValueError(f"{table_name} must be a table")

    inputs = document["inputs"]
    kind = inputs.get("kind")
    if kind not in TASK_KINDS:
        raise ValueError(f"inputs.kind must be one of {sorted(TASK_KINDS)}")
    expected_inputs = set(_COMMON_INPUT_FIELDS)
    if kind == "mosaic_experiment":
        expected_inputs.add("experiment")
    if set(inputs) != expected_inputs:
        raise ValueError(
            f"inputs for {kind!r} must contain exactly {sorted(expected_inputs)}"
        )

    model_preset = inputs["model_preset"]
    prompt_preset = inputs["prompt_preset"]
    if not isinstance(model_preset, str):
        raise ValueError("inputs.model_preset must be a string")
    if not isinstance(prompt_preset, str):
        raise ValueError("inputs.prompt_preset must be a string")
    if model_preset not in get_preset_model_names():
        raise ValueError(f"unknown model preset: {model_preset!r}")
    if prompt_preset not in get_preset_prompt_names():
        raise ValueError(f"unknown prompt preset: {prompt_preset!r}")

    settings = inputs["settings"]
    dependencies = document["dependencies"]
    if not isinstance(settings, dict):
        raise ValueError("inputs.settings must be a table")
    if settings.keys() & dependencies.keys():
        raise ValueError("settings and dependencies must use distinct names")

    dependency_paths = {
        name: _absolute_path(value, context=f"dependency {name!r}").resolve()
        for name, value in dependencies.items()
    }
    if not all(isinstance(name, str) and name for name in dependency_paths):
        raise ValueError("dependency names must be nonempty strings")

    experiment_cls = None
    if kind == "mosaic_experiment":
        experiment = inputs["experiment"]
        if not isinstance(experiment, str) or not experiment:
            raise ValueError("mosaic experiments require a nonempty experiment name")
        experiment_cls = get_experiment_class(experiment)

    config = _TaskConfig(
        path=path,
        output_folder=_absolute_path(
            document["output_folder"],
            context="output_folder",
        ),
        kind=kind,
        model_preset=model_preset,
        prompt_preset=prompt_preset,
        experiment_cls=experiment_cls,
        settings=dict(settings),
        dependencies=dependency_paths,
    )
    _validate_task_arguments(config)
    return config


def _validate_task_arguments(config: _TaskConfig) -> None:
    """Check settings and dependency names against the task's signature.

    Settings and resolved dependency folders are both passed to a task as
    keyword arguments. For erasure tasks, the benchmark argument is supplied
    automatically from the prompt preset and therefore cannot be configured.

    Args:
        config: Parsed task configuration to validate.

    Raises:
        ValueError: If an argument is unknown, required but missing, or cannot
            be bound to the task function.
    """
    keyword_arguments: dict[str, Any] = {
        **config.settings,
        **{name: Path("/dependency") for name in config.dependencies},
    }
    if config.kind == "mosaic_experiment":
        reserved = {
            "backend",
            "prompt_runtime",
            "output_dir",
            "experiment_cls",
        }
        collisions = reserved & keyword_arguments.keys()
        if collisions:
            raise ValueError(
                "invalid mosaic_experiment arguments: reserved names "
                f"{sorted(collisions)}"
            )
        if config.experiment_cls is None:  # Guarded by config parsing.
            raise ValueError("mosaic experiment class was not resolved")
        signature = inspect.signature(config.experiment_cls)
        positional_arguments = (object(), object())
        positional_count = 2
    else:
        task = run_recording_task if config.kind == "recording" else run_erasure_task
        signature = inspect.signature(task)
        positional_arguments = (object(), object(), Path("/output"))
        positional_count = 3

    if config.kind == "erasure":
        if "benchmark_name" in config.settings:
            raise ValueError(
                "benchmark_name is selected automatically from prompt_preset"
            )
        keyword_arguments["benchmark_name"] = get_default_benchmark_name(
            config.prompt_preset
        )

    parameters = tuple(signature.parameters.values())
    configurable_parameters = parameters[positional_count:]
    accepts_extra_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    allowed_keywords = {
        parameter.name
        for parameter in configurable_parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    required_keywords = {
        parameter.name
        for parameter in configurable_parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    problems = []
    unknown = set(keyword_arguments) - allowed_keywords
    if unknown and not accepts_extra_keywords:
        problems.append(f"unknown arguments {sorted(unknown)}")
    missing = required_keywords - set(keyword_arguments)
    if missing:
        problems.append(f"missing arguments {sorted(missing)}")
    if problems:
        raise ValueError(f"invalid {config.kind} arguments: {'; '.join(problems)}")

    try:
        signature.bind(
            *positional_arguments,
            **keyword_arguments,
        )
    except TypeError as exc:
        raise ValueError(f"invalid {config.kind} arguments: {exc}") from exc

    _validate_setting_values(config)


def _validate_setting_values(config: _TaskConfig) -> None:
    """Validate simple setting types and bounds before loading a model.

    Args:
        config: Parsed task configuration containing the settings.

    Raises:
        ValueError: If a recognized setting has an invalid type or value.
    """
    settings = config.settings
    integer_names = {"n_trials", "start_seed"}
    for name in integer_names & settings.keys():
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if "n_trials" in settings and settings["n_trials"] < 1:
        raise ValueError("n_trials must be positive")

    if "dividing_layer_fraction" in settings:
        value = settings["dividing_layer_fraction"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("dividing_layer_fraction must be a number")
        if not 0 < value < 1:
            raise ValueError("dividing_layer_fraction must fall within (0, 1)")

    if config.kind != "erasure":
        return
    numeric_bounds = {
        "coarse_fraction": (0, 1, False),
        "fidelity_threshold": (0, 1, True),
        "rewind_multiplier": (1, None, True),
    }
    for name, (minimum, maximum, include_minimum) in numeric_bounds.items():
        if name not in settings:
            continue
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        below_minimum = value < minimum if include_minimum else value <= minimum
        if below_minimum or (maximum is not None and value > maximum):
            interval_start = "[" if include_minimum else "("
            interval_end = "]" if maximum is not None else ")"
            upper = str(maximum) if maximum is not None else "infinity"
            raise ValueError(
                f"{name} must fall within {interval_start}{minimum}, {upper}{interval_end}"
            )


def _read_producer(config_path: Path) -> tuple[str, str, Path]:
    """Read the relevant fields from a dependency's producer config.

    Args:
        config_path: Path to the config that produced the dependency.

    Returns:
        The producer's model preset, prompt preset, and output folder.

    Raises:
        OSError: If the producer config cannot be read.
        ValueError: If required producer fields are absent or malformed.
    """
    document = _read_toml(config_path)
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"Dependency config has no inputs table: {config_path}")
    return (
        inputs["model_preset"],
        inputs["prompt_preset"],
        _absolute_path(
            document.get("output_folder"),
            context=f"{config_path}: output_folder",
        ),
    )


def _check_completion(folder: Path, *, dependency_name: str) -> None:
    """Verify that a dependency has a valid completion marker.

    Args:
        folder: Resolved output folder of the dependency.
        dependency_name: Logical dependency name used in error messages.

    Raises:
        FileNotFoundError: If ``completed.json`` is absent.
        ValueError: If the completion marker is not a JSON object.
        json.JSONDecodeError: If the completion marker is invalid JSON.
    """
    completion_path = folder / COMPLETION_FILENAME
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"Dependency {dependency_name!r} is incomplete: {completion_path}"
        )
    with completion_path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError(f"Completion record must contain an object: {completion_path}")


def _stable_topological_order(
    configs: Sequence[_TaskConfig],
    config_by_path: dict[Path, _TaskConfig],
    input_order: dict[Path, int],
) -> list[_TaskConfig]:
    """Order one model group so that internal producers run first.

    Configurations with no ordering relationship retain their command-line
    order, which keeps execution plans stable and easy to inspect.

    Args:
        configs: Configurations that share a model preset.
        config_by_path: Every config in the requested batch, indexed by path.
        input_order: Command-line position of each config path.

    Returns:
        Configurations in stable topological order.

    Raises:
        ValueError: If the dependency graph contains a cycle.
    """
    paths = {config.path for config in configs}
    prerequisites = {
        config.path: {
            dependency
            for dependency in config.dependencies.values()
            if dependency in paths
        }
        for config in configs
    }
    ordered: list[_TaskConfig] = []
    remaining = set(paths)
    while remaining:
        ready = sorted(
            (
                path
                for path in remaining
                if not (prerequisites[path] & remaining)
            ),
            key=input_order.__getitem__,
        )
        if not ready:
            cycle_paths = sorted(str(path) for path in remaining)
            raise ValueError(f"Dependency cycle among configs: {cycle_paths}")
        for path in ready:
            ordered.append(config_by_path[path])
            remaining.remove(path)
    return ordered


def _prepare_batch(
    config_paths: Sequence[str | Path],
) -> list[tuple[str, list[_TaskConfig]]]:
    """Validate an entire batch before loading models or writing outputs.

    The validation includes config schemas, task arguments, unique output
    folders, dependency compatibility, external dependency completion, and
    internal dependency cycles. Dependencies must use the same model and
    prompt presets as their consumers.

    Args:
        config_paths: Task config paths supplied by the caller or CLI.

    Returns:
        Model groups in first-seen order. Within each group, tasks are ordered
        so that internal dependencies run before their consumers.

    Raises:
        ValueError: If any config or dependency relationship is invalid.
    """
    if not config_paths:
        raise ValueError("At least one config path is required.")

    paths = [Path(path).resolve() for path in config_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("The config list contains duplicate paths.")

    configs: list[_TaskConfig] = []
    errors: list[str] = []
    for path in paths:
        try:
            configs.append(_parse_config(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise ValueError("Invalid task configs:\n- " + "\n- ".join(errors))

    output_folders = [config.output_folder for config in configs]
    if len(set(output_folders)) != len(output_folders):
        raise ValueError("Multiple configs use the same output_folder.")
    for config in configs:
        if config.output_folder.exists() or config.output_folder.is_symlink():
            errors.append(f"Output folder already exists: {config.output_folder}")

    config_by_path = {config.path: config for config in configs}
    for config in configs:
        for name, producer_path in config.dependencies.items():
            try:
                producer_model, producer_prompt, producer_output = _read_producer(
                    producer_path
                )
                if producer_model != config.model_preset:
                    raise ValueError("producer and consumer model presets differ")
                if producer_prompt != config.prompt_preset:
                    raise ValueError("producer and consumer prompt presets differ")
                if producer_path not in config_by_path:
                    _check_completion(producer_output, dependency_name=name)
            except Exception as exc:
                errors.append(f"{config.path}: dependency {name!r}: {exc}")
    if errors:
        raise ValueError("Batch is not execution-ready:\n- " + "\n- ".join(errors))

    input_order = {path: index for index, path in enumerate(paths)}
    grouped: dict[str, list[_TaskConfig]] = {}
    for config in configs:
        grouped.setdefault(config.model_preset, []).append(config)
    return [
        (
            model_name,
            _stable_topological_order(group, config_by_path, input_order),
        )
        for model_name, group in grouped.items()
    ]


def _dependency_folders(config: _TaskConfig) -> dict[str, Path]:
    """Resolve a task's logical dependencies to their output folders.

    Args:
        config: Consumer task configuration.

    Returns:
        Dependency argument names mapped to producer output folders.
    """
    return {
        name: _read_producer(path)[2]
        for name, path in config.dependencies.items()
    }


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_completion(output_folder: Path, *, config_sha256: str) -> None:
    """Atomically publish the completion marker after a successful task.

    Args:
        output_folder: Output folder containing the completed task's artifacts
            and copied configuration.
        config_sha256: Digest computed from the copied configuration before the
            task began.
    """
    completion = {
        "completed_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "config": COPIED_CONFIG_FILENAME,
        "config_sha256": config_sha256,
    }
    destination = output_folder / COMPLETION_FILENAME
    temporary = output_folder / f".{COMPLETION_FILENAME}.tmp"
    temporary.write_text(
        json.dumps(completion, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _print_plan(groups: Sequence[tuple[str, Sequence[_TaskConfig]]]) -> None:
    """Print the validated execution order and resolved output paths.

    Args:
        groups: Ordered model groups returned by :func:`_prepare_batch`.
    """
    for model_name, configs in groups:
        print(f"model: {model_name}")
        for config in configs:
            print(f"  {config.kind}: {config.path}")
            print(f"    output: {config.output_folder}")
            for name, folder in _dependency_folders(config).items():
                print(f"    dependency {name}: {folder}")


def run_configs(
    config_paths: Sequence[str | Path],
    *,
    dry_run: bool = False,
) -> None:
    """Validate and execute task configurations grouped by model.

    The complete batch is validated before execution begins. Each model is
    loaded once, prompt runtimes are reused within its model group, and model
    references are released before the next group is loaded. Immediately
    before each task, dependency completion and output nonexistence are checked
    again. A task receives its settings and resolved dependency folders as
    keyword arguments. Its completion marker is written only after it returns
    successfully.

    Args:
        config_paths: Task TOML files. Relative paths are resolved immediately.
        dry_run: If true, print the execution-ready plan without loading models
            or writing outputs.

    Raises:
        ValueError: If the batch or any config is invalid.
        FileExistsError: If an output folder already exists.
        Exception: Any exception raised by model loading or task execution is
            propagated. In that case, no completion marker is written for the
            failed task.
    """
    groups = _prepare_batch(config_paths)
    _print_plan(groups)
    if dry_run:
        return

    _configure_model_loading_progress()
    for model_name, configs in groups:
        model_started = monotonic()
        print(f"loading model: {model_name}", flush=True)
        model = load_preset_model(model_name)
        print(
            f"loaded model: {model_name} ({monotonic() - model_started:.1f}s)",
            flush=True,
        )
        runtimes: dict[str, Any] = {}
        runtime = None
        try:
            for config in configs:
                task_label = (
                    f"{config.prompt_preset}/"
                    f"{config.path.stem.removesuffix('_config')}"
                )
                dependency_folders = _dependency_folders(config)
                for name, folder in dependency_folders.items():
                    _check_completion(folder, dependency_name=name)
                if config.output_folder.exists() or config.output_folder.is_symlink():
                    raise FileExistsError(
                        f"Output folder already exists: {config.output_folder}"
                    )

                runtime = runtimes.get(config.prompt_preset)
                if runtime is None:
                    runtime = build_prompt_runtime(model, config.prompt_preset)
                    runtimes[config.prompt_preset] = runtime

                config.output_folder.mkdir(parents=True, exist_ok=False)
                copied_config = config.output_folder / COPIED_CONFIG_FILENAME
                shutil.copyfile(
                    config.path,
                    copied_config,
                )
                config_sha256 = _sha256_file(copied_config)
                arguments = {**config.settings, **dependency_folders}
                task_started = monotonic()
                print(f"starting {config.kind}: {task_label}", flush=True)
                if config.kind == "recording":
                    run_recording_task(
                        runtime.backend,
                        runtime.prompt_runtime,
                        config.output_folder,
                        **arguments,
                    )
                elif config.kind == "erasure":
                    run_erasure_task(
                        runtime.backend,
                        runtime.prompt_runtime,
                        config.output_folder,
                        benchmark_name=get_default_benchmark_name(
                            config.prompt_preset
                        ),
                        **arguments,
                    )
                else:
                    if config.experiment_cls is None:  # Guarded by parsing.
                        raise ValueError("mosaic experiment class was not resolved")
                    run_experiment_task(
                        runtime.backend,
                        runtime.prompt_runtime,
                        config.output_folder,
                        experiment_cls=config.experiment_cls,
                        **arguments,
                    )
                _write_completion(
                    config.output_folder,
                    config_sha256=config_sha256,
                )
                print(
                    f"completed {config.kind}: {task_label} "
                    f"({monotonic() - task_started:.1f}s)",
                    flush=True,
                )
        finally:
            runtimes.clear()
            runtime = None
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    """Parse CLI arguments and execute the requested task configs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="Task configuration files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the execution plan without running tasks.",
    )
    arguments = parser.parse_args()
    run_configs(arguments.configs, dry_run=arguments.dry_run)


if __name__ == "__main__":
    main()
