"""Partition annotated task configs and render a deterministic SLURM plan.

Usage:
    python tools/plan_slurm_jobs.py TASK_CONFIG_ROOT --dry-run
    python tools/plan_slurm_jobs.py TASK_CONFIG_ROOT --config-list CONFIGS.txt --dry-run
    python tools/plan_slurm_jobs.py TASK_CONFIG_ROOT
    python tools/plan_slurm_jobs.py TASK_CONFIG_ROOT --output-dir JOB_PLAN_DIR

The tool writes a machine-readable ``job_plan.json`` and explicit ``.slurm``
scripts. It never submits jobs. Dependency-connected tasks remain in the same
job so that ``run_configs.py`` can order them without SLURM job dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cognitive_maps_in_llms.benchmarks import (
    get_default_benchmark_name,
    load_benchmark,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib


COMPLETION_FILENAME = "completed.json"
PLAN_FILENAME = "job_plan.json"


@dataclass(frozen=True)
class ResourceConfig:
    """SLURM directives associated with a canonical GPU profile."""

    partition: str | None
    gpu_count: int
    constraint: str | None
    memory: str


RESOURCE_CONFIGS = {
    "mig": ResourceConfig("mig", 1, None, "32G"),
    "1xgpu40": ResourceConfig(None, 1, "gpu40", "64G"),
    "1xgpu80": ResourceConfig(None, 1, "gpu80", "64G"),
    "2xgpu80": ResourceConfig(None, 2, "gpu80", "64G"),
}


@dataclass(frozen=True)
class PlanningPolicy:
    """User-adjustable job packing and wall-time policy."""

    target_job_seconds: int = 4 * 3600
    maximum_walltime_seconds: int = 48 * 3600
    maximum_tasks_per_job: int = 50
    model_load_seconds: int = 120
    safety_factor: float = 1.5
    walltime_round_seconds: int = 15 * 60
    minimum_walltime_seconds: int = 62 * 60

    def validate(self) -> None:
        """Reject nonsensical planning parameters."""
        integer_values = {
            "target_job_seconds": self.target_job_seconds,
            "maximum_walltime_seconds": self.maximum_walltime_seconds,
            "maximum_tasks_per_job": self.maximum_tasks_per_job,
            "walltime_round_seconds": self.walltime_round_seconds,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_load_seconds < 0:
            raise ValueError("model_load_seconds must be nonnegative")
        if self.minimum_walltime_seconds < 0:
            raise ValueError("minimum_walltime_seconds must be nonnegative")
        if self.safety_factor < 1:
            raise ValueError("safety_factor must be at least 1")
        if self.minimum_walltime_seconds > self.maximum_walltime_seconds:
            raise ValueError("minimum wall time exceeds maximum wall time")


@dataclass(frozen=True)
class TaskConfig:
    """Scheduling and dependency fields read from one task config."""

    path: Path
    output_folder: Path
    kind: str
    model_preset: str
    prompt_preset: str
    estimated_runtime_seconds: int
    gpu_profile: str
    skip: bool
    dependencies: tuple[Path, ...]


@dataclass(frozen=True)
class TaskBundle:
    """Dependency-connected tasks that must remain in one job."""

    configs: tuple[TaskConfig, ...]
    estimated_runtime_seconds: int


@dataclass
class _MutableBin:
    """Internal bin used by deterministic LPT packing."""

    bundles: list[TaskBundle]
    estimated_runtime_seconds: int = 0
    task_count: int = 0


@dataclass(frozen=True)
class PlanResult:
    """Complete JSON document and rendered scripts."""

    document: dict[str, Any]
    scripts: dict[str, str]


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _required_table(
    document: dict[str, Any], name: str, *, path: Path
) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {name} must be a table")
    return value


def _required_string(
    table: dict[str, Any], name: str, *, path: Path
) -> str:
    value = table.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: {name} must be a nonempty string")
    return value


def _absolute_path(value: Any, *, field: str, path: Path) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{path}: {field} must be an absolute path")
    return Path(value).resolve()


def _validate_erasure_benchmark(prompt_preset: str) -> str:
    """Resolve and load the benchmark used by an erasure prompt preset."""
    benchmark_name = get_default_benchmark_name(prompt_preset)
    load_benchmark(benchmark_name)
    return benchmark_name


def _erasure_benchmark_is_unavailable(prompt_preset: str) -> bool:
    try:
        _validate_erasure_benchmark(prompt_preset)
    except Exception:
        return True
    return False


def read_config_list(path: Path) -> set[Path]:
    """Read a nonempty file containing one absolute config path per line."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config list is not a file: {path}")
    selected: set[Path] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        value = line.strip()
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            raise ValueError(
                f"{path}:{line_number}: config path must be absolute: {value!r}"
            )
        candidate = candidate.resolve()
        if candidate in selected:
            raise ValueError(
                f"{path}:{line_number}: duplicate config path: {candidate}"
            )
        selected.add(candidate)
    if not selected:
        raise ValueError(f"Config list is empty: {path}")
    return selected


def _parse_config(path: Path) -> TaskConfig:
    """Parse and validate the fields needed for job planning."""
    document = _read_toml(path)
    inputs = _required_table(document, "inputs", path=path)
    scheduling = _required_table(document, "scheduling", path=path)
    dependencies = _required_table(document, "dependencies", path=path)

    kind = _required_string(inputs, "kind", path=path)
    model = _required_string(inputs, "model_preset", path=path)
    prompt = _required_string(inputs, "prompt_preset", path=path)
    required_scheduling_fields = {"estimated_runtime_seconds", "gpu_profile"}
    allowed_scheduling_fields = required_scheduling_fields | {"skip"}
    scheduling_fields = set(scheduling)
    if (
        not required_scheduling_fields <= scheduling_fields
        or not scheduling_fields <= allowed_scheduling_fields
    ):
        raise ValueError(
            f"{path}: scheduling must contain estimated_runtime_seconds and "
            "gpu_profile, with optional skip"
        )
    skip = scheduling.get("skip", False)
    if not isinstance(skip, bool):
        raise ValueError(f"{path}: scheduling.skip must be Boolean")
    runtime = scheduling["estimated_runtime_seconds"]
    if isinstance(runtime, bool) or not isinstance(runtime, int) or runtime < 1:
        raise ValueError(f"{path}: estimated_runtime_seconds must be positive")
    profile = scheduling["gpu_profile"]
    if profile not in RESOURCE_CONFIGS:
        raise ValueError(f"{path}: unknown gpu_profile {profile!r}")

    dependency_paths = tuple(
        _absolute_path(value, field=f"dependency {name!r}", path=path)
        for name, value in sorted(dependencies.items())
    )
    return TaskConfig(
        path=path.resolve(),
        output_folder=_absolute_path(
            document.get("output_folder"),
            field="output_folder",
            path=path,
        ),
        kind=kind,
        model_preset=model,
        prompt_preset=prompt,
        estimated_runtime_seconds=runtime,
        gpu_profile=profile,
        skip=skip,
        dependencies=dependency_paths,
    )


def _read_external_producer(path: Path) -> tuple[str, str, Path]:
    """Read preset and output fields from an external producer config."""
    document = _read_toml(path)
    inputs = _required_table(document, "inputs", path=path)
    return (
        _required_string(inputs, "model_preset", path=path),
        _required_string(inputs, "prompt_preset", path=path),
        _absolute_path(
            document.get("output_folder"),
            field="output_folder",
            path=path,
        ),
    )


def load_task_configs(
    config_root: Path,
    *,
    selected_paths: set[Path] | None = None,
    allow_unsupported_erasure_prompts: bool = False,
) -> tuple[list[TaskConfig], list[TaskConfig], dict[Path, set[Path]]]:
    """Load a config tree and validate the selected dependency relationships."""
    config_root = config_root.resolve()
    if not config_root.is_dir():
        raise ValueError(f"Config root is not a directory: {config_root}")
    paths = sorted(path.resolve() for path in config_root.rglob("*_config.toml"))
    if not paths:
        raise ValueError(f"No *_config.toml files found below {config_root}")

    configs: list[TaskConfig] = []
    errors: list[str] = []
    for path in paths:
        try:
            configs.append(_parse_config(path))
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("Invalid task configs:\n- " + "\n- ".join(errors))

    output_folders = [config.output_folder for config in configs]
    if len(set(output_folders)) != len(output_folders):
        raise ValueError("Multiple configs use the same output_folder")

    skipped_configs = [config for config in configs if config.skip]
    config_by_path = {config.path: config for config in configs}
    if selected_paths is None:
        active_configs = [config for config in configs if not config.skip]
    else:
        selected_paths = {path.resolve() for path in selected_paths}
        unknown = selected_paths - set(config_by_path)
        if unknown:
            raise ValueError(
                "Config list contains paths outside the config tree: "
                + ", ".join(str(path) for path in sorted(unknown))
            )
        selected_skipped = [
            config_by_path[path]
            for path in sorted(selected_paths)
            if config_by_path[path].skip
        ]
        if selected_skipped:
            raise ValueError(
                "Config list selects skipped configs: "
                + ", ".join(str(config.path) for config in selected_skipped)
            )
        active_configs = [
            config for config in configs if config.path in selected_paths
        ]

    benchmark_errors = []
    for config in active_configs:
        if config.kind != "erasure":
            continue
        try:
            _validate_erasure_benchmark(config.prompt_preset)
        except Exception as exc:
            if not allow_unsupported_erasure_prompts:
                benchmark_errors.append(
                    f"{config.path}: erasure has no usable benchmark for prompt "
                    f"{config.prompt_preset!r}: {exc}"
                )
    if benchmark_errors:
        raise ValueError(
            "Invalid erasure benchmarks:\n- " + "\n- ".join(benchmark_errors)
        )

    profiles_by_model: dict[str, set[str]] = defaultdict(set)
    for config in active_configs:
        profiles_by_model[config.model_preset].add(config.gpu_profile)
    conflicts = {
        model: profiles
        for model, profiles in profiles_by_model.items()
        if len(profiles) != 1
    }
    if conflicts:
        raise ValueError(f"Models have conflicting GPU profiles: {conflicts!r}")

    active_paths = {config.path for config in active_configs}
    prerequisites: dict[Path, set[Path]] = {
        config.path: set() for config in active_configs
    }
    errors = []
    for config in active_configs:
        for dependency in config.dependencies:
            producer = config_by_path.get(dependency)
            try:
                if producer is None:
                    producer_model, producer_prompt, producer_output = (
                        _read_external_producer(dependency)
                    )
                    completion = producer_output / COMPLETION_FILENAME
                    if not completion.is_file():
                        raise FileNotFoundError(
                            f"external dependency is incomplete: {completion}"
                        )
                elif producer.path in active_paths:
                    if producer.skip:
                        raise ValueError("active config depends on skipped config")
                    producer_model = producer.model_preset
                    producer_prompt = producer.prompt_preset
                    prerequisites[config.path].add(producer.path)
                elif selected_paths is not None:
                    producer_model = producer.model_preset
                    producer_prompt = producer.prompt_preset
                    completion = producer.output_folder / COMPLETION_FILENAME
                    if not completion.is_file():
                        raise FileNotFoundError(
                            f"unselected dependency is incomplete: {completion}"
                        )
                else:
                    raise ValueError("active config depends on skipped config")
                if producer_model != config.model_preset:
                    raise ValueError("producer and consumer models differ")
                if producer_prompt != config.prompt_preset:
                    raise ValueError("producer and consumer prompts differ")
            except Exception as exc:
                errors.append(f"{config.path}: dependency {dependency}: {exc}")
    if errors:
        raise ValueError("Invalid dependencies:\n- " + "\n- ".join(errors))

    _stable_topological_order(
        {config.path for config in active_configs},
        prerequisites,
    )
    return active_configs, skipped_configs, prerequisites


def _stable_topological_order(
    paths: set[Path],
    prerequisites: dict[Path, set[Path]],
) -> list[Path]:
    """Topologically order paths with lexical tie-breaking."""
    ordered: list[Path] = []
    remaining = set(paths)
    while remaining:
        ready = sorted(
            path
            for path in remaining
            if not (prerequisites[path] & remaining)
        )
        if not ready:
            raise ValueError(
                "Dependency cycle among configs: "
                + ", ".join(str(path) for path in sorted(remaining))
            )
        ordered.extend(ready)
        remaining.difference_update(ready)
    return ordered


def build_dependency_bundles(
    configs: Sequence[TaskConfig],
    prerequisites: dict[Path, set[Path]],
) -> list[TaskBundle]:
    """Return undirected dependency components in deterministic order."""
    config_by_path = {config.path: config for config in configs}
    neighbors: dict[Path, set[Path]] = {
        config.path: set() for config in configs
    }
    for consumer, producers in prerequisites.items():
        for producer in producers:
            neighbors[consumer].add(producer)
            neighbors[producer].add(consumer)

    bundles: list[TaskBundle] = []
    unseen = set(config_by_path)
    while unseen:
        start = min(unseen)
        component: set[Path] = set()
        pending = [start]
        while pending:
            path = pending.pop()
            if path in component:
                continue
            component.add(path)
            pending.extend(sorted(neighbors[path] - component, reverse=True))
        unseen.difference_update(component)

        ordered_paths = _stable_topological_order(component, prerequisites)
        component_configs = tuple(config_by_path[path] for path in ordered_paths)
        models = {config.model_preset for config in component_configs}
        profiles = {config.gpu_profile for config in component_configs}
        if len(models) != 1 or len(profiles) != 1:
            raise ValueError(
                "Dependency bundle crosses model or GPU-profile boundaries: "
                f"{sorted(str(path) for path in component)}"
            )
        bundles.append(
            TaskBundle(
                configs=component_configs,
                estimated_runtime_seconds=sum(
                    config.estimated_runtime_seconds
                    for config in component_configs
                ),
            )
        )
    return bundles


def _requested_walltime(task_seconds: int, policy: PlanningPolicy) -> int:
    """Apply load overhead, safety margin, rounding, and minimum wall time."""
    padded = math.ceil(
        (task_seconds + policy.model_load_seconds) * policy.safety_factor
    )
    rounded = (
        math.ceil(padded / policy.walltime_round_seconds)
        * policy.walltime_round_seconds
    )
    return max(policy.minimum_walltime_seconds, rounded)


def _pack_group(
    bundles: Sequence[TaskBundle],
    policy: PlanningPolicy,
) -> list[_MutableBin]:
    """Pack one model/profile group using deterministic LPT greedy packing."""
    ordered = sorted(
        bundles,
        key=lambda bundle: (
            -bundle.estimated_runtime_seconds,
            str(bundle.configs[0].path),
        ),
    )
    bins: list[_MutableBin] = []
    for bundle in ordered:
        bundle_tasks = len(bundle.configs)
        if bundle_tasks > policy.maximum_tasks_per_job:
            raise ValueError(
                f"Dependency bundle has {bundle_tasks} tasks, exceeding the "
                f"maximum of {policy.maximum_tasks_per_job}: "
                f"{bundle.configs[0].path}"
            )
        bundle_walltime = _requested_walltime(
            bundle.estimated_runtime_seconds,
            policy,
        )
        if bundle_walltime > policy.maximum_walltime_seconds:
            raise ValueError(
                "Dependency bundle exceeds maximum wall time: "
                f"{bundle.configs[0].path} requires "
                f"{_format_walltime(bundle_walltime)}"
            )

        candidates = []
        for index, job_bin in enumerate(bins):
            combined_seconds = (
                job_bin.estimated_runtime_seconds
                + bundle.estimated_runtime_seconds
            )
            combined_tasks = job_bin.task_count + bundle_tasks
            if combined_tasks > policy.maximum_tasks_per_job:
                continue
            if (
                combined_seconds + policy.model_load_seconds
                > policy.target_job_seconds
            ):
                continue
            if (
                _requested_walltime(combined_seconds, policy)
                > policy.maximum_walltime_seconds
            ):
                continue
            candidates.append((job_bin.estimated_runtime_seconds, index))

        if candidates:
            _, selected_index = min(candidates)
            selected = bins[selected_index]
        else:
            selected = _MutableBin(bundles=[])
            bins.append(selected)
        selected.bundles.append(bundle)
        selected.estimated_runtime_seconds += bundle.estimated_runtime_seconds
        selected.task_count += bundle_tasks
    return bins


def _format_walltime(seconds: int) -> str:
    """Format seconds as a SLURM-compatible hours:minutes:seconds value."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _safe_job_component(value: str) -> str:
    """Normalize a model name for a concise SLURM job name."""
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-")
    return normalized or "model"


def _render_script(
    *,
    job_name: str,
    resource: ResourceConfig,
    walltime: str,
    log_path: Path,
    repository_root: Path,
    configs: Sequence[Path],
    conda_env: str,
) -> str:
    """Render one shell-valid SLURM script with explicit config paths."""
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
    ]
    if resource.partition is not None:
        directives.append(f"#SBATCH --partition={resource.partition}")
    directives.append(f"#SBATCH --gres=gpu:{resource.gpu_count}")
    if resource.constraint is not None:
        directives.append(f"#SBATCH --constraint={resource.constraint}")
    directives.extend(
        [
            f"#SBATCH --mem={resource.memory}",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --output={log_path}/%x-%j.out",
        ]
    )

    config_lines = [
        f"  {shlex.quote(str(path))}" + (" \\" if index < len(configs) - 1 else "")
        for index, path in enumerate(configs)
    ]
    body = [
        "",
        "set -eo pipefail",
        "",
        f"cd {shlex.quote(str(repository_root))}",
        "source ~/.bashrc",
        f"conda activate {shlex.quote(conda_env)}",
        "",
        "hostname",
        "date",
        "nvidia-smi",
        "",
        "python tools/run_configs.py \\",
        *config_lines,
        "",
        "date",
        "",
    ]
    return "\n".join(directives + body)


def _validate_shell(script_name: str, content: str) -> None:
    """Run bash's parser without executing a rendered script."""
    completed = subprocess.run(
        ["bash", "-n"],
        input=content,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"Generated script {script_name} is invalid: {completed.stderr.strip()}"
        )


def build_job_plan(
    config_root: Path,
    output_dir: Path,
    *,
    policy: PlanningPolicy | None = None,
    config_list: Path | None = None,
    allow_unsupported_erasure_prompts: bool = False,
    conda_env: str = "cognitive-maps",
) -> PlanResult:
    """Build and validate a complete in-memory job plan."""
    if not isinstance(conda_env, str) or not conda_env.strip():
        raise ValueError("conda_env must be a nonempty environment name or path")
    policy = policy or PlanningPolicy()
    policy.validate()
    config_root = config_root.resolve()
    output_dir = output_dir.resolve()
    repository_root = REPOSITORY_ROOT
    selected_paths = (
        read_config_list(config_list) if config_list is not None else None
    )
    tree_config_count = sum(1 for _ in config_root.rglob("*_config.toml"))

    configs, skipped_configs, prerequisites = load_task_configs(
        config_root,
        selected_paths=selected_paths,
        allow_unsupported_erasure_prompts=allow_unsupported_erasure_prompts,
    )
    unsupported_erasure_configs = [
        config
        for config in configs
        if (
            config.kind == "erasure"
            and _erasure_benchmark_is_unavailable(config.prompt_preset)
        )
    ]
    bundles = build_dependency_bundles(configs, prerequisites)
    bundles_by_group: dict[tuple[str, str], list[TaskBundle]] = defaultdict(list)
    for bundle in bundles:
        first = bundle.configs[0]
        bundles_by_group[(first.model_preset, first.gpu_profile)].append(bundle)

    jobs: list[dict[str, Any]] = []
    scripts: dict[str, str] = {}
    config_by_path = {config.path: config for config in configs}
    job_number = 0
    for (model, profile), group_bundles in sorted(bundles_by_group.items()):
        for job_bin in _pack_group(group_bundles, policy):
            job_number += 1
            job_id = f"job_{job_number:04d}"
            script_name = f"{job_id}__{_safe_job_component(model)}.slurm"
            script_path = output_dir / "slurm_scripts" / script_name
            component_paths = {
                config.path
                for bundle in job_bin.bundles
                for config in bundle.configs
            }
            ordered_paths = _stable_topological_order(
                component_paths,
                prerequisites,
            )
            requested_seconds = _requested_walltime(
                job_bin.estimated_runtime_seconds,
                policy,
            )
            if requested_seconds > policy.maximum_walltime_seconds:
                raise ValueError(
                    f"{job_id} exceeds maximum wall time after packing"
                )
            walltime = _format_walltime(requested_seconds)
            job_name = f"cmaps_{_safe_job_component(model)}_{job_number:04d}"
            content = _render_script(
                job_name=job_name,
                resource=RESOURCE_CONFIGS[profile],
                walltime=walltime,
                log_path=output_dir / "slurm_logs",
                repository_root=repository_root,
                configs=ordered_paths,
                conda_env=conda_env,
            )
            _validate_shell(script_name, content)
            scripts[script_name] = content

            jobs.append(
                {
                    "job_id": job_id,
                    "script": str(script_path),
                    "model_preset": model,
                    "gpu_profile": profile,
                    "task_estimated_runtime_seconds": (
                        job_bin.estimated_runtime_seconds
                    ),
                    "model_load_overhead_seconds": policy.model_load_seconds,
                    "estimated_runtime_seconds": (
                        job_bin.estimated_runtime_seconds
                        + policy.model_load_seconds
                    ),
                    "requested_walltime_seconds": requested_seconds,
                    "requested_walltime": walltime,
                    "configs": [str(path) for path in ordered_paths],
                    "internal_dependencies": {
                        str(path): [
                            str(dependency)
                            for dependency in sorted(
                                prerequisites[path] & component_paths
                            )
                        ]
                        for path in ordered_paths
                        if prerequisites[path] & component_paths
                    },
                }
            )

    planned_paths = [Path(path) for job in jobs for path in job["configs"]]
    expected_paths = set(config_by_path)
    if len(planned_paths) != len(set(planned_paths)):
        raise ValueError("The job plan contains duplicate configurations")
    if set(planned_paths) != expected_paths:
        missing = expected_paths - set(planned_paths)
        extra = set(planned_paths) - expected_paths
        raise ValueError(
            f"Job-plan coverage mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    document = {
        "schema_version": 1,
        "config_root": str(config_root),
        "config_list": (
            str(config_list.resolve()) if config_list is not None else None
        ),
        "output_dir": str(output_dir),
        "execution_ready": not unsupported_erasure_configs,
        "unsupported_erasure_configs": [
            str(config.path) for config in unsupported_erasure_configs
        ],
        "skipped_configs": [str(config.path) for config in skipped_configs],
        "policy": {
            "target_job_seconds": policy.target_job_seconds,
            "maximum_walltime_seconds": policy.maximum_walltime_seconds,
            "maximum_tasks_per_job": policy.maximum_tasks_per_job,
            "model_load_seconds": policy.model_load_seconds,
            "safety_factor": policy.safety_factor,
            "walltime_round_seconds": policy.walltime_round_seconds,
            "minimum_walltime_seconds": policy.minimum_walltime_seconds,
        },
        "summary": {
            "config_count": tree_config_count,
            "planned_config_count": len(configs),
            "skipped_config_count": len(skipped_configs),
            "dependency_bundle_count": len(bundles),
            "job_count": len(jobs),
            "task_estimated_runtime_seconds": sum(
                config.estimated_runtime_seconds for config in configs
            ),
            "skipped_task_estimated_runtime_seconds": sum(
                config.estimated_runtime_seconds for config in skipped_configs
            ),
        },
        "jobs": jobs,
    }
    if config_list is not None:
        document["summary"]["unselected_config_count"] = (
            tree_config_count - len(configs) - len(skipped_configs)
        )
    return PlanResult(document=document, scripts=scripts)


def write_job_plan(output_dir: Path, result: PlanResult) -> None:
    """Write a prevalidated plan to a new output directory."""
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    script_dir = output_dir / "slurm_scripts"
    log_dir = output_dir / "slurm_logs"
    script_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir()
    (output_dir / PLAN_FILENAME).write_text(
        json.dumps(result.document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for script_name, content in result.scripts.items():
        (script_dir / script_name).write_text(content, encoding="utf-8")


def _hours_to_seconds(value: float, *, name: str) -> int:
    """Convert a positive CLI duration in hours to seconds."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return math.ceil(value * 3600)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_root", type=Path)
    parser.add_argument(
        "--conda-env",
        default="cognitive-maps",
        help="Conda environment name or path activated by jobs (default: cognitive-maps).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Defaults to a job_plan directory alongside "
            "the task-config root."
        ),
    )
    parser.add_argument(
        "--config-list",
        type=Path,
        help=(
            "File containing one absolute task-config path per line. The full "
            "config root is validated, but only listed configs are planned."
        ),
    )
    parser.add_argument("--target-job-hours", type=float, default=4.0)
    parser.add_argument("--maximum-walltime-hours", type=float, default=48.0)
    parser.add_argument("--maximum-tasks-per-job", type=int, default=50)
    parser.add_argument("--model-load-seconds", type=int, default=120)
    parser.add_argument("--safety-factor", type=float, default=1.5)
    parser.add_argument("--walltime-round-minutes", type=int, default=15)
    parser.add_argument("--minimum-walltime-minutes", type=int, default=62)
    parser.add_argument(
        "--allow-unsupported-erasure-prompts",
        action="store_true",
        help=(
            "Build an analysis-only plan even when erasure benchmarks are "
            "missing; generated jobs are not execution-ready."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize the plan without writing files.",
    )
    return parser.parse_args(argv)


def resolve_output_dir(config_root: Path, output_dir: Path | None) -> Path:
    """Resolve an explicit output directory or the sibling default."""
    if output_dir is None:
        return config_root.resolve().parent / "job_plan"
    return output_dir.resolve()


def _print_summary(result: PlanResult, *, dry_run: bool) -> None:
    """Print a compact model-level plan summary."""
    summary = result.document["summary"]
    verb = "Planned" if dry_run else "Wrote"
    selection_suffix = ""
    if "unselected_config_count" in summary:
        selection_suffix = (
            f"; {summary['unselected_config_count']} unselected"
        )
    print(
        f"{verb} {summary['job_count']} jobs covering "
        f"{summary['planned_config_count']} of "
        f"{summary['config_count']} task configs; "
        f"{summary['skipped_config_count']} skipped"
        f"{selection_suffix}."
    )
    jobs_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in result.document["jobs"]:
        jobs_by_model[job["model_preset"]].append(job)
    for model, jobs in sorted(jobs_by_model.items()):
        total = sum(job["task_estimated_runtime_seconds"] for job in jobs)
        print(
            f"  {model}: {len(jobs)} jobs, "
            f"{_format_walltime(total)} estimated task time"
        )
    if not result.document["execution_ready"]:
        print(
            "WARNING: analysis-only plan includes erasure prompts without "
            "registered benchmarks."
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Build and optionally write a SLURM plan."""
    arguments = parse_args(argv)
    output_dir = resolve_output_dir(arguments.config_root, arguments.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    policy = PlanningPolicy(
        target_job_seconds=_hours_to_seconds(
            arguments.target_job_hours,
            name="target_job_hours",
        ),
        maximum_walltime_seconds=_hours_to_seconds(
            arguments.maximum_walltime_hours,
            name="maximum_walltime_hours",
        ),
        maximum_tasks_per_job=arguments.maximum_tasks_per_job,
        model_load_seconds=arguments.model_load_seconds,
        safety_factor=arguments.safety_factor,
        walltime_round_seconds=arguments.walltime_round_minutes * 60,
        minimum_walltime_seconds=arguments.minimum_walltime_minutes * 60,
    )
    result = build_job_plan(
        arguments.config_root,
        output_dir,
        policy=policy,
        config_list=arguments.config_list,
        conda_env=arguments.conda_env,
        allow_unsupported_erasure_prompts=(
            arguments.allow_unsupported_erasure_prompts
        ),
    )
    if not arguments.dry_run:
        write_job_plan(output_dir, result)
    _print_summary(result, dry_run=arguments.dry_run)


if __name__ == "__main__":
    main()
