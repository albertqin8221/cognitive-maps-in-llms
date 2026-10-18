import pytest
import toml

from tools.generate_task_configs import generate_configs, read_toml


pytestmark = pytest.mark.codex


def _manifest(tmp_path, overrides):
    source = tmp_path / "manifest.toml"
    source.write_text(toml.dumps({
        "run_id": "example",
        "model_presets": ["gemma-27b", "qwen-32b"],
        "prompt_presets": ["ABC_family_tree_yes_no"],
        "results": str(tmp_path / "results"),
        "tasks": [{
            "name": "bottleneck",
            "kind": "mosaic_experiment",
            "experiment": "exp_5a_bottleneck",
            "settings": {"n_trials": 50, "dividing_layer_fraction": 0.5},
            "settings_by_model": overrides,
        }],
    }))
    return source


def test_model_override_preserves_other_models_and_shared_settings(tmp_path):
    source = _manifest(tmp_path, {
        "qwen-32b": {"dividing_layer_fraction": 2 / 3},
    })
    configs = [read_toml(path) for path in generate_configs(source)]
    assert {c["inputs"]["model_preset"]: c["inputs"]["settings"] for c in configs} == {
        "gemma-27b": {"n_trials": 50, "dividing_layer_fraction": 0.5},
        "qwen-32b": {"n_trials": 50, "dividing_layer_fraction": 2 / 3},
    }


@pytest.mark.parametrize("overrides", [
    [],
    {"qwen-typo": {"dividing_layer_fraction": 2 / 3}},
    {"qwen-32b": 0.5},
])
def test_invalid_model_override_writes_nothing(tmp_path, overrides):
    source = _manifest(tmp_path, overrides)
    with pytest.raises(ValueError, match="settings_by_model"):
        generate_configs(source)
    assert not (tmp_path / "task_configs").exists()


def test_model_override_cannot_shadow_dependency(tmp_path):
    source = _manifest(tmp_path, {"qwen-32b": {"recordings": "wrong"}})
    manifest = read_toml(source)
    manifest["artifacts"] = str(tmp_path / "artifacts")
    manifest["tasks"][0]["dependencies"] = {"recordings": "recording"}
    manifest["tasks"].append({"name": "recording", "kind": "recording"})
    source.write_text(toml.dumps(manifest))
    with pytest.raises(ValueError, match="settings and dependencies overlap"):
        generate_configs(source)
    assert not (tmp_path / "task_configs").exists()
