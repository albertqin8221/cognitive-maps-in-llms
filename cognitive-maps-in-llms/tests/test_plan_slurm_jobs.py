"""Tests for deterministic SLURM job planning and rendering."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import pytest
import toml

from tools import plan_slurm_jobs as planner


pytestmark = pytest.mark.codex


def _write_config(
    path: Path,
    *,
    output_folder: Path,
    runtime: int,
    kind: str = "mosaic_experiment",
    model: str = "gemma-27b",
    prompt: str = "ABC_family_tree_who_is",
    profile: str = "1xgpu80",
    dependencies: dict[str, Path] | None = None,
    scheduling: dict[str, Any] | None = None,
) -> Path:
    document = {
        "output_folder": str(output_folder.resolve()),
        "metadata": {},
        "inputs": {
            "kind": kind,
            "model_preset": model,
            "prompt_preset": prompt,
            "experiment": "exp_1_behavior",
            "settings": {"n_trials": 1},
        },
        "dependencies": {
            name: str(dependency.resolve())
            for name, dependency in (dependencies or {}).items()
        },
        "scheduling": scheduling
        if scheduling is not None
        else {
            "estimated_runtime_seconds": runtime,
            "gpu_profile": profile,
        },
    }
    if kind != "mosaic_experiment":
        document["inputs"].pop("experiment")
    if kind == "erasure":
        document["inputs"]["settings"]["start_seed"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml.dumps(document), encoding="utf-8")
    return path.resolve()


def _small_policy(**overrides: Any) -> planner.PlanningPolicy:
    values = {
        "target_job_seconds": 350,
        "maximum_walltime_seconds": 3600,
        "maximum_tasks_per_job": 10,
        "model_load_seconds": 0,
        "safety_factor": 1.0,
        "walltime_round_seconds": 1,
        "minimum_walltime_seconds": 0,
    }
    values.update(overrides)
    return planner.PlanningPolicy(**values)


def test_output_dir_defaults_next_to_config_root(tmp_path: Path) -> None:
    config_root = tmp_path / "example_run" / "task_configs"

    arguments = planner.parse_args([str(config_root)])

    assert arguments.output_dir is None
    assert planner.resolve_output_dir(
        arguments.config_root,
        arguments.output_dir,
    ) == tmp_path / "example_run" / "job_plan"


def test_explicit_output_dir_overrides_default(tmp_path: Path) -> None:
    config_root = tmp_path / "example_run" / "task_configs"
    output_dir = tmp_path / "custom_plan"

    arguments = planner.parse_args(
        [str(config_root), "--output-dir", str(output_dir)]
    )

    assert planner.resolve_output_dir(
        arguments.config_root,
        arguments.output_dir,
    ) == output_dir


def test_dependencies_stay_together_and_every_config_appears_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task_configs"
    recording = _write_config(
        root / "recording_config.toml",
        output_folder=tmp_path / "recording_output",
        runtime=100,
        kind="recording",
    )
    erasure = _write_config(
        root / "erasure_config.toml",
        output_folder=tmp_path / "erasure_output",
        runtime=200,
        kind="erasure",
        dependencies={"recordings": recording},
    )
    independent = _write_config(
        root / "experiment_config.toml",
        output_folder=tmp_path / "experiment_output",
        runtime=300,
    )

    result = planner.build_job_plan(
        root,
        tmp_path / "plan",
        policy=_small_policy(),
    )

    assert result.document["summary"] == {
        "config_count": 3,
        "planned_config_count": 3,
        "skipped_config_count": 0,
        "dependency_bundle_count": 2,
        "job_count": 2,
        "task_estimated_runtime_seconds": 600,
        "skipped_task_estimated_runtime_seconds": 0,
    }
    jobs = result.document["jobs"]
    all_configs = [Path(path) for job in jobs for path in job["configs"]]
    assert len(all_configs) == len(set(all_configs)) == 3
    assert set(all_configs) == {recording, erasure, independent}

    dependency_job = next(job for job in jobs if str(erasure) in job["configs"])
    assert dependency_job["configs"].index(str(recording)) < (
        dependency_job["configs"].index(str(erasure))
    )
    assert str(recording) in dependency_job["configs"]
    assert dependency_job["internal_dependencies"] == {
        str(erasure): [str(recording)]
    }


def test_config_list_plans_only_selected_and_requires_completed_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task_configs"
    recording_output = tmp_path / "recording_output"
    recording = _write_config(
        root / "recording_config.toml",
        output_folder=recording_output,
        runtime=100,
        kind="recording",
    )
    consumer = _write_config(
        root / "consumer_config.toml",
        output_folder=tmp_path / "consumer_output",
        runtime=200,
        dependencies={"recordings": recording},
    )
    independent = _write_config(
        root / "independent_config.toml",
        output_folder=tmp_path / "independent_output",
        runtime=300,
    )
    config_list = tmp_path / "configs.txt"
    config_list.write_text(f"{consumer}\n{independent}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unselected dependency is incomplete"):
        planner.build_job_plan(
            root,
            tmp_path / "incomplete_plan",
            policy=_small_policy(),
            config_list=config_list,
        )

    recording_output.mkdir()
    (recording_output / planner.COMPLETION_FILENAME).write_text(
        "{}\n", encoding="utf-8"
    )
    result = planner.build_job_plan(
        root,
        tmp_path / "plan",
        policy=_small_policy(),
        config_list=config_list,
    )

    planned = {
        Path(path)
        for job in result.document["jobs"]
        for path in job["configs"]
    }
    assert planned == {consumer, independent}
    assert recording not in planned
    assert result.document["config_list"] == str(config_list.resolve())
    assert result.document["summary"]["config_count"] == 3
    assert result.document["summary"]["planned_config_count"] == 2
    assert result.document["summary"]["unselected_config_count"] == 1
    assert all(not job["internal_dependencies"] for job in result.document["jobs"])


def test_config_list_rejects_duplicates_and_skipped_configs(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    skipped = _write_config(
        root / "skipped_config.toml",
        output_folder=tmp_path / "skipped_output",
        runtime=100,
        scheduling={
            "estimated_runtime_seconds": 100,
            "gpu_profile": "1xgpu80",
            "skip": True,
        },
    )
    duplicate_list = tmp_path / "duplicates.txt"
    duplicate_list.write_text(f"{skipped}\n{skipped}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate config path"):
        planner.build_job_plan(
            root,
            tmp_path / "duplicate_plan",
            config_list=duplicate_list,
        )

    selected_list = tmp_path / "skipped.txt"
    selected_list.write_text(f"{skipped}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selects skipped configs"):
        planner.build_job_plan(
            root,
            tmp_path / "skipped_plan",
            config_list=selected_list,
        )

    outside = _write_config(
        tmp_path / "outside" / "outside_config.toml",
        output_folder=tmp_path / "outside_output",
        runtime=100,
    )
    outside_list = tmp_path / "outside.txt"
    outside_list.write_text(f"{outside}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the config tree"):
        planner.build_job_plan(
            root,
            tmp_path / "outside_plan",
            config_list=outside_list,
        )


def test_rendered_script_has_resources_and_explicit_paths(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    config = _write_config(
        root / "experiment_config.toml",
        output_folder=tmp_path / "output",
        runtime=100,
    )

    result = planner.build_job_plan(
        root,
        tmp_path / "plan",
        policy=_small_policy(
            model_load_seconds=20,
            safety_factor=1.5,
            walltime_round_seconds=60,
        ),
    )

    script = next(iter(result.scripts.values()))
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --constraint=gpu80" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=00:03:00" in script
    assert "conda activate cognitive-maps" in script
    assert str(config) in script
    assert "*_config.toml" not in script
    assert not any(
        line.lstrip().lower().startswith("sbatch ")
        for line in script.splitlines()
    )


@pytest.mark.parametrize("conda_env", [None, "transformers", "/tmp/env with spaces;$HOME"])
def test_cli_selects_conda_environment(tmp_path: Path, conda_env: str | None) -> None:
    root = tmp_path / "task_configs"
    _write_config(
        root / "experiment_config.toml",
        output_folder=tmp_path / "output",
        runtime=100,
    )
    output_dir = tmp_path / "plan"
    argv = [str(root), "--output-dir", str(output_dir)]
    if conda_env is not None:
        argv.extend(["--conda-env", conda_env])

    planner.main(argv)

    script = next((output_dir / "slurm_scripts").glob("*.slurm")).read_text()
    activation = next(line for line in script.splitlines() if line.startswith("conda activate "))
    assert shlex.split(activation) == ["conda", "activate", conda_env or "cognitive-maps"]


def test_empty_conda_environment_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        planner.build_job_plan(tmp_path / "configs", tmp_path / "plan", conda_env=" ")


def test_write_job_plan_and_dry_build_do_not_submit(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    _write_config(
        root / "experiment_config.toml",
        output_folder=tmp_path / "output",
        runtime=100,
    )
    output_dir = tmp_path / "plan"
    result = planner.build_job_plan(root, output_dir, policy=_small_policy())
    assert not output_dir.exists()

    planner.write_job_plan(output_dir, result)

    document = json.loads((output_dir / planner.PLAN_FILENAME).read_text())
    assert document == result.document
    assert (output_dir / "slurm_logs").is_dir()
    scripts = list((output_dir / "slurm_scripts").glob("*.slurm"))
    assert len(scripts) == 1
    with pytest.raises(FileExistsError, match="must be new"):
        planner.write_job_plan(output_dir, result)


def test_incomplete_scheduling_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    _write_config(
        root / "experiment_config.toml",
        output_folder=tmp_path / "output",
        runtime=100,
        scheduling={},
    )

    with pytest.raises(ValueError, match="scheduling must contain"):
        planner.build_job_plan(root, tmp_path / "plan", policy=_small_policy())


def test_skipped_configs_are_reported_but_not_planned(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    active = _write_config(
        root / "active_config.toml",
        output_folder=tmp_path / "active_output",
        runtime=100,
    )
    skipped = _write_config(
        root / "skipped_config.toml",
        output_folder=tmp_path / "skipped_output",
        runtime=200,
        scheduling={
            "estimated_runtime_seconds": 200,
            "gpu_profile": "1xgpu80",
            "skip": True,
        },
    )

    result = planner.build_job_plan(
        root,
        tmp_path / "plan",
        policy=_small_policy(),
    )

    assert result.document["skipped_configs"] == [str(skipped)]
    assert result.document["summary"] == {
        "config_count": 2,
        "planned_config_count": 1,
        "skipped_config_count": 1,
        "dependency_bundle_count": 1,
        "job_count": 1,
        "task_estimated_runtime_seconds": 100,
        "skipped_task_estimated_runtime_seconds": 200,
    }
    assert result.document["jobs"][0]["configs"] == [str(active)]


def test_active_config_cannot_depend_on_skipped_config(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    skipped = _write_config(
        root / "recording_config.toml",
        output_folder=tmp_path / "recording_output",
        runtime=100,
        kind="recording",
        scheduling={
            "estimated_runtime_seconds": 100,
            "gpu_profile": "1xgpu80",
            "skip": True,
        },
    )
    _write_config(
        root / "consumer_config.toml",
        output_folder=tmp_path / "consumer_output",
        runtime=100,
        dependencies={"recordings": skipped},
    )

    with pytest.raises(ValueError, match="active config depends on skipped"):
        planner.build_job_plan(root, tmp_path / "plan", policy=_small_policy())


def test_skip_must_be_boolean(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    _write_config(
        root / "experiment_config.toml",
        output_folder=tmp_path / "output",
        runtime=100,
        scheduling={
            "estimated_runtime_seconds": 100,
            "gpu_profile": "1xgpu80",
            "skip": "yes",
        },
    )

    with pytest.raises(ValueError, match="scheduling.skip must be Boolean"):
        planner.build_job_plan(root, tmp_path / "plan", policy=_small_policy())


def test_skipped_unsupported_erasure_does_not_block_plan(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    recording = _write_config(
        root / "recording_config.toml",
        output_folder=tmp_path / "recording_output",
        runtime=100,
        kind="recording",
        prompt="ABC_east_west",
        scheduling={
            "estimated_runtime_seconds": 100,
            "gpu_profile": "1xgpu80",
            "skip": True,
        },
    )
    erasure = _write_config(
        root / "erasure_config.toml",
        output_folder=tmp_path / "erasure_output",
        runtime=200,
        kind="erasure",
        prompt="ABC_east_west",
        dependencies={"recordings": recording},
        scheduling={
            "estimated_runtime_seconds": 200,
            "gpu_profile": "1xgpu80",
            "skip": True,
        },
    )
    active = _write_config(
        root / "experiment_config.toml",
        output_folder=tmp_path / "experiment_output",
        runtime=300,
        prompt="ABC_east_west",
    )

    result = planner.build_job_plan(
        root,
        tmp_path / "plan",
        policy=_small_policy(),
    )

    assert result.document["execution_ready"] is True
    assert result.document["skipped_configs"] == [str(erasure), str(recording)]
    assert result.document["jobs"][0]["configs"] == [str(active)]


def test_registered_non_family_erasure_is_execution_ready(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    _write_config(
        root / "erasure_config.toml",
        output_folder=tmp_path / "output",
        runtime=100,
        kind="erasure",
        prompt="ABC_east_west",
    )

    result = planner.build_job_plan(
        root,
        tmp_path / "plan",
        policy=_small_policy(),
    )

    assert result.document["execution_ready"] is True
    assert result.document["unsupported_erasure_configs"] == []


def test_unregistered_erasure_requires_explicit_override(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    _write_config(
        root / "erasure_config.toml",
        output_folder=tmp_path / "output",
        runtime=100,
        kind="erasure",
        prompt="ABC_not_registered",
    )

    with pytest.raises(ValueError, match="no usable benchmark"):
        planner.build_job_plan(root, tmp_path / "strict_plan", policy=_small_policy())

    result = planner.build_job_plan(
        root,
        tmp_path / "analysis_plan",
        policy=_small_policy(),
        allow_unsupported_erasure_prompts=True,
    )
    assert result.document["execution_ready"] is False


def test_oversized_dependency_bundle_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "task_configs"
    producer = _write_config(
        root / "producer_config.toml",
        output_folder=tmp_path / "producer_output",
        runtime=200,
        kind="recording",
    )
    _write_config(
        root / "consumer_config.toml",
        output_folder=tmp_path / "consumer_output",
        runtime=200,
        dependencies={"recordings": producer},
    )

    with pytest.raises(ValueError, match="exceeds maximum wall time"):
        planner.build_job_plan(
            root,
            tmp_path / "plan",
            policy=_small_policy(maximum_walltime_seconds=350),
        )
