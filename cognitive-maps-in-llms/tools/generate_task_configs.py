"""Generate concrete task configs from a manifest.

Usage:
    python generate_task_configs.py manifest.toml --dry-run
    python generate_task_configs.py manifest.toml --config-root /absolute/new/root

The default config root is <manifest-directory>/task_configs. Each config is stored
under <config-root>/<model>/<prompt>/<task_name>_config.toml. The config root must
not already exist.

The numbered steps in generate_task_configs() describe the complete workflow.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any

import toml
from toml.decoder import InlineTableDict

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


OUTPUT_ROOT_BY_KIND = {
    "mosaic_experiment": "results",
    "recording": "artifacts",
    "erasure": "results",
}


class InlineTable(dict, InlineTableDict):
    """Tell the TOML writer to keep experiment settings in an inline table."""


def read_toml(path: Path) -> dict[str, Any]:
    """Read a manifest or task configuration.

    Args:
        path: File to read.

    Returns:
        The parsed TOML document.
    """
    with path.open("rb") as handle:
        return tomllib.load(handle)


def validate_name(value: Any, label: str) -> None:
    """Require a nonempty name that can safely be used as one path component.

    Args:
        value: Run ID, task name, model preset, or prompt preset.
        label: Field description used in error messages.

    Raises:
        ValueError: The value is empty, is not a string, or contains path syntax.
    """
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or any(character in value for character in ("/", "\\", "\0", "\n", "\r"))
    ):
        raise ValueError(f"{label} must be a nonempty, single path component: {value!r}")


def absolute_path(value: Any, label: str) -> Path:
    """Validate an absolute path from a manifest or producer config.

    Args:
        value: Path string stored in the TOML document.
        label: Field description used in error messages.

    Returns:
        The path, without requiring its target to exist.

    Raises:
        ValueError: The value is not an absolute path string.
    """
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path: {value!r}")
    return Path(value)


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the manifest and order tasks with producers before consumers.

    Args:
        manifest: Parsed manifest with shared presets and task definitions.

    Returns:
        Tasks in dependency order. Existing producer configs are not part of
        this graph; exact matches are selected during dependency resolution.

    Raises:
        ValueError: Names, task kinds, settings, or dependencies are invalid.
        KeyError: A required manifest field is missing.
        graphlib.CycleError: Current-run tasks contain a dependency cycle.
    """
    validate_name(manifest["run_id"], "run_id")
    for field in ("model_presets", "prompt_presets"):
        values = manifest[field]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field} must be a nonempty list")
        for value in values:
            validate_name(value, field)
        if len(set(values)) != len(values):
            raise ValueError(f"{field} contains duplicate names")

    tasks = manifest["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty array of tables")
    by_name = {}
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Each task must be a table")
        name = task["name"]
        validate_name(name, "task name")
        if name in by_name:
            raise ValueError(f"Duplicate task name: {name}")
        kind = task["kind"]
        if kind not in OUTPUT_ROOT_BY_KIND:
            raise ValueError(f"Task {name}: unknown kind {kind!r}")
        root_field = OUTPUT_ROOT_BY_KIND[kind]
        absolute_path(manifest[root_field], root_field)
        if kind == "mosaic_experiment":
            experiment = task.get("experiment")
            if not isinstance(experiment, str) or not experiment.strip():
                raise ValueError(f"Task {name}: an experiment name is required")
            # Registry lookup and constructor validation happen in run_configs,
            # keeping manifest expansion independent of experiment imports.
        for field in ("settings", "dependencies"):
            if not isinstance(task.get(field, {}), dict):
                raise ValueError(f"Task {name}: {field} must be a table")
        overlap = task.get("settings", {}).keys() & task.get("dependencies", {}).keys()
        if overlap:
            raise ValueError(f"Task {name}: settings and dependencies overlap: {overlap}")
        overrides = task.get("settings_by_model", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Task {name}: settings_by_model must be a table")
        for model, settings in overrides.items():
            if model not in manifest["model_presets"]:
                raise ValueError(f"Task {name}: settings_by_model has unknown model {model!r}")
            if not isinstance(settings, dict):
                raise ValueError(f"Task {name}: settings_by_model.{model} must be a table")
            overlap = settings.keys() & task.get("dependencies", {}).keys()
            if overlap:
                raise ValueError(f"Task {name}: settings and dependencies overlap: {overlap}")
        by_name[name] = task

    graph = {}
    for name, task in by_name.items():
        graph[name] = []
        for reference in task.get("dependencies", {}).values():
            if isinstance(reference, str):
                if reference not in by_name:
                    raise ValueError(f"Task {name}: unknown dependency {reference!r}")
                graph[name].append(reference)
            elif isinstance(reference, dict) and set(reference) == {"config"}:
                absolute_path(reference["config"], "existing dependency config")
            elif isinstance(reference, dict) and set(reference) == {"configs"}:
                paths = reference["configs"]
                if not isinstance(paths, list) or not paths:
                    raise ValueError(
                        f"Task {name}: existing dependency configs must be a "
                        "nonempty list"
                    )
                normalized_paths = [
                    absolute_path(path, "existing dependency config")
                    for path in paths
                ]
                if len(set(normalized_paths)) != len(normalized_paths):
                    raise ValueError(
                        f"Task {name}: existing dependency configs contain "
                        "duplicate paths"
                    )
            else:
                raise ValueError(
                    f"Task {name}: a dependency must be a task name or "
                    "{ config = ... } or { configs = [...] }"
                )
    return [by_name[name] for name in TopologicalSorter(graph).static_order()]


def config_path(root: Path, model: str, prompt: str, task: str) -> Path:
    """Build the agreed config filename for one concrete task.

    Args:
        root: Absolute config root for one manifest invocation.
        model: Model preset name.
        prompt: Prompt preset name.
        task: Task name from the manifest.

    Returns:
        The producer or consumer config path.
    """
    return root / model / prompt / f"{task}_config.toml"


def select_existing_dependency(
    reference: dict[str, Any],
    model: str,
    prompt: str,
) -> Path:
    """Select the unique existing config for one model and prompt.

    Args:
        reference: A validated ``config`` or ``configs`` dependency reference.
        model: Consumer model preset.
        prompt: Consumer prompt preset.

    Returns:
        The unique producer config declaring the requested presets.

    Raises:
        OSError: If a referenced config cannot be read.
        ValueError: If zero or multiple configs declare the requested presets.
    """
    values = (
        [reference["config"]] if "config" in reference else reference["configs"]
    )
    paths = [absolute_path(value, "existing dependency config") for value in values]
    matches = []
    for path in paths:
        document = read_toml(path)
        inputs = document.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError(f"Existing dependency config has no inputs table: {path}")
        if (
            inputs.get("model_preset") == model
            and inputs.get("prompt_preset") == prompt
        ):
            matches.append(path)

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one existing dependency config for "
            f"model={model!r}, prompt={prompt!r}; found {len(matches)}"
        )
    return matches[0]


def resolve_dependencies(
    task: dict[str, Any], config_root: Path, model: str, prompt: str
) -> dict[str, str]:
    """Resolve task references to absolute producer-config paths.

    Args:
        task: Validated task definition.
        config_root: Root where this invocation's configs will be written.
        model: Consumer model preset, also used for its producers.
        prompt: Consumer prompt preset, also used for its producers.

    Returns:
        Dependency argument names mapped to producer-config path strings.
        Newly planned producers need not exist yet. Existing dependencies are
        selected from the exact config paths supplied by the manifest.
    """
    dependencies = {}
    for argument, reference in task.get("dependencies", {}).items():
        if isinstance(reference, str):
            path = config_path(config_root, model, prompt, reference)
        else:
            path = select_existing_dependency(reference, model, prompt)
        dependencies[argument] = str(path)
    return dependencies


def generate_configs(
    manifest_path: Path, config_root: Path | None = None, *, dry_run: bool = False
) -> list[Path]:
    """Validate, expand, and write one config per task/model/prompt combination.

    Args:
        manifest_path: Manifest file; a relative CLI path is made absolute.
        config_root: Destination root, defaulting to the manifest's sibling
            directory named ``task_configs``. Relative CLI paths are made absolute.
        dry_run: If True, validate and return planned paths without writing.

    Returns:
        Config paths in producer-before-consumer order for each model/prompt.

    Raises:
        FileExistsError: The config root already exists, including for dry runs.
        ValueError: The manifest or an older dependency is invalid.
        OSError: Input reading or output writing fails. A write failure may leave
            a partial config root; inspect it before choosing a fresh destination.
    """
    # 1. Read and validate the plan, including current-run dependency cycles.
    manifest_path = manifest_path.resolve()
    manifest = read_toml(manifest_path)
    tasks = validate_manifest(manifest)
    if config_root is None:
        config_root = manifest_path.parent / "task_configs"
    config_root = config_root.absolute()
    if config_root.exists() or config_root.is_symlink():
        raise FileExistsError(f"Config root must be new: {config_root}")

    # 2. Resolve every dependency and serialize every config before writing any.
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    planned = {}
    for model in manifest["model_presets"]:
        for prompt in manifest["prompt_presets"]:
            for task in tasks:
                root_field = OUTPUT_ROOT_BY_KIND[task["kind"]]
                output_folder = (
                    Path(manifest[root_field])
                    / manifest["run_id"]
                    / model
                    / prompt
                    / task["name"]
                )
                inputs = {
                    "kind": task["kind"],
                    "model_preset": model,
                    "prompt_preset": prompt,
                }
                if "experiment" in task:
                    inputs["experiment"] = task["experiment"]
                settings = dict(task.get("settings", {}))
                settings.update(task.get("settings_by_model", {}).get(model, {}))
                inputs["settings"] = InlineTable(settings)
                config = {
                    "output_folder": str(output_folder),
                    "metadata": {
                        "source_manifest": str(manifest_path),
                        "run_id": manifest["run_id"],
                        "created_at": created_at,
                    },
                    "inputs": inputs,
                    "dependencies": resolve_dependencies(task, config_root, model, prompt),
                    "scheduling": {},
                }
                content = toml.dumps(config, encoder=toml.TomlPreserveInlineDictEncoder())
                content += (
                    "# Scheduling is consumed by the job-planning tools and "
                    "ignored by the task runner.\n"
                )
                content += "# TODO: Estimate scheduling resources for this task.\n"
                tomllib.loads(content)
                path = config_path(config_root, model, prompt, task["name"])
                planned[path] = content

    # 3. Reserve a new root, then create each file exclusively.
    if not dry_run:
        config_root.mkdir(parents=True, exist_ok=False)
        for path, content in planned.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
    return list(planned)


def main() -> None:
    """Parse CLI options, generate configs, and report their paths."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="Manifest TOML file.")
    parser.add_argument(
        "--config-root",
        type=Path,
        help="New config root; defaults to manifest-directory/task_configs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without writing.")
    args = parser.parse_args()
    paths = generate_configs(args.manifest, args.config_root, dry_run=args.dry_run)
    for path in paths:
        print(path)
    verb = "Planned" if args.dry_run else "Wrote"
    print(f"{verb} {len(paths)} task configs.")


if __name__ == "__main__":
    main()
