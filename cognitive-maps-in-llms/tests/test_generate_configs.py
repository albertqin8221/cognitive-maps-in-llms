"""Small filesystem checks for manifest expansion and dependency safety."""

from copy import deepcopy
from graphlib import CycleError
from pathlib import Path

import pytest
import toml

from tools.generate_task_configs import config_path, generate_configs, read_toml


pytestmark = pytest.mark.codex


@pytest.fixture
def manifest(tmp_path):
    """Return a plan with consumers deliberately listed before producers."""
    return {
        "run_id": "example_run",
        "model_presets": ["model_a", "model_b"],
        "prompt_presets": ["prompt_a", "prompt_b"],
        "results": str(tmp_path / "results"),
        "artifacts": str(tmp_path / "artifacts"),
        "tasks": [
            {
                "name": "consumer",
                "kind": "mosaic_experiment",
                "experiment": "placeholder",
                "settings": {"n_trials": 50, "label": 'A "quote" and a \\ slash'},
                "dependencies": {"recordings": "producer"},
            },
            {
                "name": "producer",
                "kind": "recording",
                "settings": {"n_trials": 50, "start_seed": 10000},
            },
        ],
    }


def write_manifest(tmp_path, manifest):
    """Write a temporary TOML manifest and return its path."""
    path = tmp_path / "manifest.toml"
    path.write_text(toml.dumps(manifest), encoding="utf-8")
    return path


def test_expansion_paths_settings_and_writable_files(tmp_path, manifest):
    """Generated configs preserve settings and connect each matching producer."""
    source = write_manifest(tmp_path, manifest)
    paths = generate_configs(source)
    root = tmp_path / "task_configs"
    assert len(paths) == 8
    for model in manifest["model_presets"]:
        for prompt in manifest["prompt_presets"]:
            producer = config_path(root, model, prompt, "producer")
            consumer = config_path(root, model, prompt, "consumer")
            assert paths.index(producer) < paths.index(consumer)
            config = read_toml(consumer)
            assert config["dependencies"] == {"recordings": str(producer)}
            assert config["inputs"]["settings"] == manifest["tasks"][0]["settings"]
            assert "output_folder" not in config["inputs"]
            assert config["output_folder"] == str(
                tmp_path / "results" / "example_run" / model / prompt / "consumer"
            )
            assert read_toml(producer)["output_folder"] == str(
                tmp_path / "artifacts" / "example_run" / model / prompt / "producer"
            )
            assert config["metadata"]["source_manifest"] == str(source)
            assert config["metadata"]["created_at"].tzinfo is not None
            assert config["scheduling"] == {}
            assert "ignored by the task runner" in consumer.read_text()
            assert "# TODO: Estimate scheduling resources" in consumer.read_text()
    assert all(path.stat().st_mode & 0o200 for path in paths)
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "artifacts").exists()


def test_erasure_outputs_and_consumer_dependencies_use_results(tmp_path, manifest):
    manifest["tasks"].append({
        "name": "iterative_erasure",
        "kind": "erasure",
        "dependencies": {"recordings": "producer"},
    })
    manifest["tasks"][0]["dependencies"]["erasure_trajectory"] = "iterative_erasure"
    generate_configs(write_manifest(tmp_path, manifest))
    root = tmp_path / "task_configs"
    for model in manifest["model_presets"]:
        for prompt in manifest["prompt_presets"]:
            consumer = read_toml(config_path(root, model, prompt, "consumer"))
            erasure = read_toml(Path(consumer["dependencies"]["erasure_trajectory"]))
            assert erasure["output_folder"] == str(
                tmp_path / "results" / "example_run" / model / prompt / "iterative_erasure"
            )
            recording = read_toml(Path(erasure["dependencies"]["recordings"]))
            assert recording["output_folder"] == str(
                tmp_path / "artifacts" / "example_run" / model / prompt / "producer"
            )


def test_dry_run_and_existing_root(tmp_path, manifest):
    """Preview writes nothing, and an existing root is never reused."""
    source = write_manifest(tmp_path, manifest)
    root = tmp_path / "explicit_root"
    planned = generate_configs(source, root, dry_run=True)
    assert not root.exists()
    written = generate_configs(source, root)
    assert written == planned
    before = {path: path.read_bytes() for path in written}
    with pytest.raises(FileExistsError, match="must be new"):
        generate_configs(source, root)
    assert before == {path: path.read_bytes() for path in written}


@pytest.mark.parametrize("problem", ["cycle", "unknown", "duplicate", "traversal"])
def test_invalid_graph_or_names_write_nothing(tmp_path, manifest, problem):
    """Invalid plans are rejected before creating the config root."""
    expected = ValueError
    if problem == "cycle":
        manifest["tasks"][1]["dependencies"] = {"consumer": "consumer"}
        expected = CycleError
    elif problem == "unknown":
        manifest["tasks"][0]["dependencies"]["recordings"] = "missing"
    elif problem == "duplicate":
        manifest["tasks"].append(deepcopy(manifest["tasks"][0]))
    else:
        manifest["run_id"] = "../escape"
    source = write_manifest(tmp_path, manifest)
    root = tmp_path / "new_configs"
    with pytest.raises(expected):
        generate_configs(source, root)
    assert not root.exists()


def _write_existing_config(
    path: Path,
    *,
    model: str,
    prompt: str,
) -> Path:
    """Write the fields used to select an existing dependency config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        toml.dumps(
            {
                "output_folder": str(path.parent / "output"),
                "inputs": {
                    "kind": "recording",
                    "model_preset": model,
                    "prompt_preset": prompt,
                    "settings": {},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_existing_dependency_uses_exact_config_without_completion(
    tmp_path,
    manifest,
):
    """Generation preserves an exact config path without requiring outputs."""
    manifest["model_presets"] = ["model_a"]
    manifest["prompt_presets"] = ["prompt_a"]
    producer = _write_existing_config(
        tmp_path / "arbitrary" / "producer.toml",
        model="model_a",
        prompt="prompt_a",
    )
    manifest["tasks"][0]["dependencies"]["recordings"] = {
        "config": str(producer)
    }
    source = write_manifest(tmp_path, manifest)
    root = tmp_path / "new_configs"
    generate_configs(source, root)
    consumer = read_toml(config_path(root, "model_a", "prompt_a", "consumer"))
    assert consumer["dependencies"]["recordings"] == str(producer)


def test_existing_dependency_collection_matches_declared_presets(
    tmp_path,
    manifest,
):
    """A config collection is selected by declared model and prompt fields."""
    producers = [
        _write_existing_config(
            tmp_path / "arbitrary" / f"producer_{model}_{prompt}.toml",
            model=model,
            prompt=prompt,
        )
        for model in manifest["model_presets"]
        for prompt in manifest["prompt_presets"]
    ]
    manifest["tasks"][0]["dependencies"]["recordings"] = {
        "configs": [str(path) for path in reversed(producers)]
    }
    source = write_manifest(tmp_path, manifest)
    root = tmp_path / "new_configs"
    generate_configs(source, root)

    expected = {
        (
            read_toml(path)["inputs"]["model_preset"],
            read_toml(path)["inputs"]["prompt_preset"],
        ): path
        for path in producers
    }
    for model in manifest["model_presets"]:
        for prompt in manifest["prompt_presets"]:
            consumer = read_toml(config_path(root, model, prompt, "consumer"))
            assert consumer["dependencies"]["recordings"] == str(
                expected[(model, prompt)]
            )


@pytest.mark.parametrize("state", ["missing", "duplicate", "wrong_presets"])
def test_invalid_existing_dependency_collection_writes_nothing(
    tmp_path,
    manifest,
    state,
):
    """Existing dependency selection requires one match for every consumer."""
    manifest["model_presets"] = ["model_a"]
    manifest["prompt_presets"] = ["prompt_a"]
    producer = tmp_path / "existing" / "producer.toml"
    if state != "missing":
        _write_existing_config(
            producer,
            model="wrong" if state == "wrong_presets" else "model_a",
            prompt="prompt_a",
        )
    paths = [str(producer)]
    if state == "duplicate":
        duplicate = _write_existing_config(
            tmp_path / "elsewhere" / "duplicate.toml",
            model="model_a",
            prompt="prompt_a",
        )
        paths.append(str(duplicate))
    manifest["tasks"][0]["dependencies"]["recordings"] = {
        "configs": paths
    }
    source = write_manifest(tmp_path, manifest)
    root = tmp_path / "new_configs"

    with pytest.raises((OSError, ValueError)):
        generate_configs(source, root)
    assert not root.exists()
