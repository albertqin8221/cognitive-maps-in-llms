"""Populate scheduling estimates in generated task configuration files.

Usage:
    python tools/populate_scheduling.py PATH/TO/task_configs --dry-run
    python tools/populate_scheduling.py PATH/TO/task_configs
    python tools/populate_scheduling.py PATH/TO/task_configs \
        --apply-full-run-skip-policy

The estimator is intentionally small and specific to the current paper run.
It validates every configuration before changing any file. Existing matching
runtime annotations are left unchanged; conflicting runtime annotations are
never overwritten. Skip annotations change only when the policy flag is used.
"""

from __future__ import annotations

import argparse
import math
import os
import stat
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class ModelScheduling:
    """Scheduling calibration for one model on one GPU profile."""

    gpu_profile: str
    seconds_per_forward_pass: float


@dataclass(frozen=True)
class PromptProfile:
    """Information needed to count the queries in one query batch."""

    query_mode: str
    n_query_relations: int

    def query_count(self, n_entities: int) -> int:
        """Return the query count for a concrete entity pool."""
        if self.query_mode == "who_is":
            return n_entities * self.n_query_relations
        if self.query_mode == "yes_no":
            return n_entities * (n_entities - 1) * self.n_query_relations
        raise ValueError(f"Unsupported query mode: {self.query_mode!r}")


@dataclass(frozen=True)
class ExperimentCost:
    """Forward-pass coefficients for one mosaic experiment."""

    fixed_passes: int
    setup_passes_per_seed: int
    query_batches_per_seed: int
    n_query_entities: int


@dataclass(frozen=True)
class PlannedAnnotation:
    """A validated annotation and optional replacement file contents."""

    path: Path
    model_preset: str
    estimated_runtime_seconds: int
    gpu_profile: str
    skip: bool
    replacement: str | None


# These measurements were extracted from the legacy orchestration workbooks.
# A timing is coupled to its GPU profile; changing the profile requires a new
# calibration rather than silently reusing the old timing.
MODEL_SCHEDULING = {
    "gemma-4b": ModelScheduling("1xgpu40", 0.090680),
    "gemma-12b": ModelScheduling("1xgpu40", 0.143133),
    "gemma-27b": ModelScheduling("1xgpu80", 0.151161),
    "llama-8b": ModelScheduling("1xgpu40", 0.075238),
    "llama-70b": ModelScheduling("2xgpu80", 0.231273),
    "qwen-0p6b": ModelScheduling("mig", 0.054285),
    "qwen-1p7b": ModelScheduling("mig", 0.072878),
    "qwen-4b": ModelScheduling("1xgpu40", 0.060287),
    "qwen-8b": ModelScheduling("1xgpu40", 0.077483),
    "qwen-14b": ModelScheduling("1xgpu40", 0.105551),
    "qwen-32b": ModelScheduling("1xgpu80", 0.118788),
    "qwen-30b-a3b": ModelScheduling("1xgpu80", 0.197931),
}


FAMILY_WHO_IS = PromptProfile("who_is", 5)
FAMILY_YES_NO = PromptProfile("yes_no", 5)
STANDARD_YES_NO = PromptProfile("yes_no", 2)
VARIABLE_EQUIVALENCE = PromptProfile("yes_no", 1)
UNIQUE_LOCATION_WHO_IS = PromptProfile("who_is", 7)
UNIQUE_LOCATION_YES_NO = PromptProfile("yes_no", 7)


# Only presets selected for the current final-run manifests are included. The
# excluded AB/CD schemas contain four context entities and are not compatible
# with the simple three-entity experiment cost model below.
PROMPT_PROFILES = {
    "ABC_family_tree_who_is": FAMILY_WHO_IS,
    "ABC_alien_family_tree_who_is": FAMILY_WHO_IS,
    "ABC_family_tree_yes_no": FAMILY_YES_NO,
    "ABC_alien_family_tree_yes_no": FAMILY_YES_NO,
    "ABC_gene_regulatory_network": STANDARD_YES_NO,
    "ABC_neural_circuit": STANDARD_YES_NO,
    "ABC_older_younger": STANDARD_YES_NO,
    "ABC_taller_shorter": STANDARD_YES_NO,
    "ABC_faster_slower": STANDARD_YES_NO,
    "ABC_alien_older_younger": STANDARD_YES_NO,
    "ABC_alien_taller_shorter": STANDARD_YES_NO,
    "ABC_alien_faster_slower": STANDARD_YES_NO,
    "ABC_company_structure": STANDARD_YES_NO,
    "ABC_alien_company_structure": STANDARD_YES_NO,
    "ABC_proposition_equivalence": STANDARD_YES_NO,
    "ABC_variable_equivalence": VARIABLE_EQUIVALENCE,
    "ABC_east_west": STANDARD_YES_NO,
    "ABC_north_south": STANDARD_YES_NO,
    "ABC_ahead_behind": STANDARD_YES_NO,
    "ABC_family_trees_descendant_ancestor": STANDARD_YES_NO,
    "ABC_older_younger_unique_locations_who_is": UNIQUE_LOCATION_WHO_IS,
    "ABC_older_younger_unique_locations_yes_no": UNIQUE_LOCATION_YES_NO,
}


ALIEN_PROMPT_PRESETS = {
    "ABC_alien_family_tree_who_is",
    "ABC_alien_family_tree_yes_no",
    "ABC_alien_older_younger",
    "ABC_alien_taller_shorter",
    "ABC_alien_faster_slower",
    "ABC_alien_company_structure",
}
NEW_NON_ALIEN_PROMPT_PRESETS = {
    "ABC_older_younger_unique_locations_who_is",
}
NEW_NON_ALIEN_SKIPPED_EXPERIMENTS = {
    "exp_5a_bottleneck",
    "exp_5b_sparse_mosaic",
    "exp_5c_editing",
}


# For each seed, setup passes create recordings or caches and each query batch
# evaluates every query generated for the specified entity-pool size. Fixed
# passes are constructor-time shape or patch-mask probes.
EXPERIMENT_COSTS = {
    "exp_1_behavior": ExperimentCost(0, 0, 4, 3),
    "exp_2_separability": ExperimentCost(1, 10, 5, 6),
    "exp_3_additivity": ExperimentCost(1, 6, 2, 3),
    "exp_4_retrospective_prospective": ExperimentCost(1, 32, 16, 3),
    "exp_5a_bottleneck": ExperimentCost(1, 4, 4, 3),
    "exp_5b_sparse_mosaic": ExperimentCost(1, 6, 6, 3),
    "exp_5c_editing": ExperimentCost(0, 28, 14, 6),
    "exp_s1_positions_rope": ExperimentCost(1, 2, 2, 3),
    "exp_s2_positions_indirect_swap": ExperimentCost(1, 4, 2, 3),
}


# Recording was empirically about 25% slower than counting its four context
# passes per seed as vanilla forwards. Erasure is calibrated from the mini run:
# gemma-27b family-tree who-is with two trials took an estimated 840 seconds,
# equivalent to this many ordinary passes per query and erasure trial.
RECORDING_PASS_MULTIPLIER = 1.25
ERASURE_EQUIVALENT_PASSES_PER_QUERY_PER_TRIAL = 185.23296352895258
SCHEDULING_TODO = "# TODO: Estimate scheduling resources for this task."


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a task configuration."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _required_table(
    document: dict[str, Any], name: str, *, path: Path
) -> dict[str, Any]:
    """Return a required TOML table with a contextual error."""
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {name} must be a table")
    return value


def _required_name(table: dict[str, Any], name: str, *, path: Path) -> str:
    """Return a required nonempty string field."""
    value = table.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: {name} must be a nonempty string")
    return value


def _positive_n_trials(settings: dict[str, Any], *, path: Path) -> int:
    """Read and validate the trial count used by every current task."""
    value = settings.get("n_trials")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path}: inputs.settings.n_trials must be positive")
    return value


def estimate_forward_passes(
    *,
    kind: str,
    prompt_profile: PromptProfile,
    settings: dict[str, Any],
    experiment: str | None,
    path: Path,
) -> float:
    """Estimate equivalent vanilla forward passes for one task."""
    n_trials = _positive_n_trials(settings, path=path)

    if kind == "recording":
        return 4 * n_trials * RECORDING_PASS_MULTIPLIER

    if kind == "erasure":
        queries = prompt_profile.query_count(n_entities=3)
        return (
            n_trials
            * queries
            * ERASURE_EQUIVALENT_PASSES_PER_QUERY_PER_TRIAL
        )

    if kind != "mosaic_experiment":
        raise ValueError(f"{path}: unsupported task kind {kind!r}")
    if experiment is None:
        raise ValueError(f"{path}: mosaic_experiment has no experiment name")
    try:
        cost = EXPERIMENT_COSTS[experiment]
    except KeyError as exc:
        raise ValueError(f"{path}: unsupported experiment {experiment!r}") from exc

    queries = prompt_profile.query_count(cost.n_query_entities)
    return cost.fixed_passes + n_trials * (
        cost.setup_passes_per_seed
        + cost.query_batches_per_seed * queries
    )


def estimate_runtime_seconds(document: dict[str, Any], *, path: Path) -> tuple[int, str]:
    """Return estimated runtime and GPU profile for one task config."""
    inputs = _required_table(document, "inputs", path=path)
    settings = _required_table(inputs, "settings", path=path)
    kind = _required_name(inputs, "kind", path=path)
    model_preset = _required_name(inputs, "model_preset", path=path)
    prompt_preset = _required_name(inputs, "prompt_preset", path=path)

    try:
        model = MODEL_SCHEDULING[model_preset]
    except KeyError as exc:
        raise ValueError(f"{path}: unsupported model preset {model_preset!r}") from exc
    try:
        prompt = PROMPT_PROFILES[prompt_preset]
    except KeyError as exc:
        raise ValueError(f"{path}: unsupported prompt preset {prompt_preset!r}") from exc

    experiment_value = inputs.get("experiment")
    experiment = experiment_value if isinstance(experiment_value, str) else None
    equivalent_passes = estimate_forward_passes(
        kind=kind,
        prompt_profile=prompt,
        settings=settings,
        experiment=experiment,
        path=path,
    )
    runtime = max(1, math.ceil(equivalent_passes * model.seconds_per_forward_pass))
    return runtime, model.gpu_profile


def should_skip_for_full_run(document: dict[str, Any], *, path: Path) -> bool:
    """Return whether the documented final-run policy excludes one config."""
    inputs = _required_table(document, "inputs", path=path)
    kind = _required_name(inputs, "kind", path=path)
    prompt_preset = _required_name(inputs, "prompt_preset", path=path)
    experiment_value = inputs.get("experiment")
    experiment = experiment_value if isinstance(experiment_value, str) else None

    if prompt_preset in ALIEN_PROMPT_PRESETS:
        is_behavior = kind == "mosaic_experiment" and experiment == "exp_1_behavior"
        return not is_behavior

    if prompt_preset in NEW_NON_ALIEN_PROMPT_PRESETS:
        return kind == "erasure" or (
            kind == "mosaic_experiment"
            and experiment in NEW_NON_ALIEN_SKIPPED_EXPERIMENTS
        )

    return False


def _replace_scheduling_section(
    source: str,
    *,
    estimated_runtime_seconds: int,
    gpu_profile: str,
    skip: bool = False,
) -> str:
    """Replace one existing scheduling table while preserving other text."""
    lines = [
        line
        for line in source.splitlines(keepends=True)
        if line.strip() != SCHEDULING_TODO
    ]
    header_indices = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "[scheduling]"
    ]
    if len(header_indices) != 1:
        raise ValueError(
            f"Expected exactly one [scheduling] table, found {len(header_indices)}"
        )

    start = header_indices[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = index
            break

    replacement = [
        "[scheduling]\n",
        f"estimated_runtime_seconds = {estimated_runtime_seconds}\n",
        f'gpu_profile = "{gpu_profile}"\n',
    ]
    if skip:
        replacement.append("skip = true\n")
    if end < len(lines) and replacement[-1].endswith("\n"):
        replacement.append("\n")
    return "".join(lines[:start] + replacement + lines[end:])


def plan_annotations(
    config_paths: Sequence[Path],
    *,
    apply_full_run_skip_policy: bool = False,
) -> list[PlannedAnnotation]:
    """Validate configs and prepare every annotation without writing files."""
    planned: list[PlannedAnnotation] = []
    errors: list[str] = []

    for path in config_paths:
        try:
            document = _read_toml(path)
            scheduling = _required_table(document, "scheduling", path=path)
            inputs = _required_table(document, "inputs", path=path)
            model_preset = _required_name(
                inputs,
                "model_preset",
                path=path,
            )
            runtime, gpu_profile = estimate_runtime_seconds(document, path=path)
            expected = {
                "estimated_runtime_seconds": runtime,
                "gpu_profile": gpu_profile,
            }
            allowed_fields = set(expected) | {"skip"}
            if set(scheduling) - allowed_fields:
                raise ValueError(
                    f"{path}: conflicting scheduling annotation: "
                    f"found {scheduling!r}, expected {expected!r} with "
                    "optional Boolean skip"
                )
            skip = scheduling.get("skip", False)
            if not isinstance(skip, bool):
                raise ValueError(f"{path}: scheduling.skip must be Boolean")
            existing_estimate = {
                key: scheduling[key]
                for key in expected
                if key in scheduling
            }
            if scheduling and existing_estimate != expected:
                raise ValueError(
                    f"{path}: conflicting scheduling annotation: "
                    f"found {scheduling!r}, expected {expected!r}"
                )

            desired_skip = (
                should_skip_for_full_run(document, path=path)
                if apply_full_run_skip_policy
                else skip
            )
            replacement = None
            if not scheduling or (
                apply_full_run_skip_policy and skip != desired_skip
            ):
                replacement = _replace_scheduling_section(
                    path.read_text(encoding="utf-8"),
                    estimated_runtime_seconds=runtime,
                    gpu_profile=gpu_profile,
                    skip=desired_skip,
                )
                parsed_replacement = tomllib.loads(replacement)
                serialized_expected = dict(expected)
                if desired_skip:
                    serialized_expected["skip"] = True
                if parsed_replacement["scheduling"] != serialized_expected:
                    raise ValueError(f"{path}: failed to serialize scheduling data")

            planned.append(
                PlannedAnnotation(
                    path=path,
                    model_preset=model_preset,
                    estimated_runtime_seconds=runtime,
                    gpu_profile=gpu_profile,
                    skip=desired_skip,
                    replacement=replacement,
                )
            )
        except Exception as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError("Invalid task configs:\n- " + "\n- ".join(errors))
    return planned


def _write_annotations(planned: Sequence[PlannedAnnotation]) -> None:
    """Stage every changed file, then atomically replace the originals."""
    staged: list[tuple[Path, Path]] = []
    try:
        for annotation in planned:
            if annotation.replacement is None:
                continue
            mode = stat.S_IMODE(annotation.path.stat().st_mode)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=annotation.path.parent,
                prefix=f".{annotation.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(annotation.replacement)
                temporary = Path(handle.name)
            os.chmod(temporary, mode)
            staged.append((temporary, annotation.path))

        for temporary, destination in staged:
            temporary.replace(destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def populate_scheduling(
    config_root: Path,
    *,
    dry_run: bool = False,
    apply_full_run_skip_policy: bool = False,
) -> list[PlannedAnnotation]:
    """Annotate every generated task config below ``config_root``."""
    config_root = config_root.resolve()
    if not config_root.is_dir():
        raise ValueError(f"Config root is not a directory: {config_root}")
    paths = sorted(config_root.rglob("*_config.toml"))
    if not paths:
        raise ValueError(f"No *_config.toml files found below {config_root}")
    if any(path.is_symlink() for path in paths):
        raise ValueError("Refusing to rewrite task configurations through symlinks")

    planned = plan_annotations(
        paths,
        apply_full_run_skip_policy=apply_full_run_skip_policy,
    )
    if not dry_run:
        _write_annotations(planned)
    return planned


def _format_duration(seconds: int) -> str:
    """Format an aggregate duration for the command-line summary."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _print_summary(
    planned: Sequence[PlannedAnnotation],
    *,
    dry_run: bool,
) -> None:
    """Print compact aggregate results rather than thousands of paths."""
    changed = sum(item.replacement is not None for item in planned)
    totals: dict[str, int] = defaultdict(int)
    for item in planned:
        totals[item.model_preset] += item.estimated_runtime_seconds

    verb = "Would annotate" if dry_run else "Annotated"
    print(
        f"{verb} {changed} of {len(planned)} task configs; "
        f"{len(planned) - changed} already matched."
    )
    skipped = sum(item.skip for item in planned)
    if skipped:
        print(f"  {skipped} task configs marked to skip.")
    for model, seconds in sorted(totals.items()):
        print(f"  {model}: {_format_duration(seconds)} estimated task time")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_root",
        type=Path,
        help="Root containing generated *_config.toml files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report annotations without changing files.",
    )
    parser.add_argument(
        "--apply-full-run-skip-policy",
        action="store_true",
        help=(
            "Apply the documented final-run exclusions: behavior only for alien "
            "prompts, and no erasure or Experiment 5 tasks for "
            "ABC_older_younger_unique_locations_who_is."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Populate scheduling annotations from the command line."""
    arguments = parse_args(argv)
    planned = populate_scheduling(
        arguments.config_root,
        dry_run=arguments.dry_run,
        apply_full_run_skip_policy=arguments.apply_full_run_skip_policy,
    )
    _print_summary(planned, dry_run=arguments.dry_run)


if __name__ == "__main__":
    main()
