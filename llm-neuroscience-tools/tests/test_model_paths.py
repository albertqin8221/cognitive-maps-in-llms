from pathlib import Path

import pytest

from llm_neuroscience_tools.model_paths import (
    default_local_name,
    get_model_path,
    get_models_dir,
)
from llm_neuroscience_tools.presets import get_preset_model_names
from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS


pytestmark = pytest.mark.codex


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_root_has_setup_instructions(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("LNT_MODELS_DIR", raising=False)
    else:
        monkeypatch.setenv("LNT_MODELS_DIR", value)
    assert "qwen-0p6b" in get_preset_model_names()
    with pytest.raises(ValueError, match="export LNT_MODELS_DIR="):
        get_models_dir()


@pytest.mark.parametrize("repo_id", ["", "qwen-0p6b", "../model", "owner/..", "/tmp/model", "owner/repo/subdir"])
def test_invalid_repo_ids_cannot_form_model_paths(repo_id):
    with pytest.raises(ValueError, match="owner/repo"):
        default_local_name(repo_id)


def test_model_config_reads_environment_on_each_access(monkeypatch, tmp_path):
    config = MODEL_CONFIGS["qwen-0p6b"]
    for root in (tmp_path / "first", tmp_path / "second"):
        expected = root / "Qwen__Qwen3-0.6B"
        expected.mkdir(parents=True)
        monkeypatch.setenv("LNT_MODELS_DIR", str(root))
        assert config.model_path == str(expected)
        assert config.tokenizer_path == str(expected)


def test_missing_model_can_be_download_target_but_not_loaded(monkeypatch, tmp_path):
    root = tmp_path / "new-models"
    monkeypatch.setenv("LNT_MODELS_DIR", str(root))
    expected = root / "Qwen__Qwen3-0.6B"
    assert get_model_path("Qwen/Qwen3-0.6B", must_exist=False) == expected
    assert not root.exists()
    with pytest.raises(FileNotFoundError, match="download_model.py --model Qwen/Qwen3-0.6B"):
        _ = MODEL_CONFIGS["qwen-0p6b"].model_path


def test_root_must_not_be_a_file(monkeypatch, tmp_path):
    root = tmp_path / "file"
    root.touch()
    monkeypatch.setenv("LNT_MODELS_DIR", str(root))
    with pytest.raises(NotADirectoryError):
        get_models_dir()


def test_explicit_root_overrides_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LNT_MODELS_DIR", str(tmp_path / "default"))
    assert get_models_dir(tmp_path / "override") == tmp_path / "override"


def test_user_path_expansion():
    assert get_models_dir("~/lnt-models") == Path.home() / "lnt-models"


def test_canonical_symlink_can_point_to_existing_model(monkeypatch, tmp_path):
    existing = tmp_path / "Llama-3.1-70b-Instruct"
    existing.mkdir()
    canonical = tmp_path / "meta-llama__Llama-3.1-70B-Instruct"
    canonical.symlink_to(existing, target_is_directory=True)
    monkeypatch.setenv("LNT_MODELS_DIR", str(tmp_path))
    assert Path(MODEL_CONFIGS["llama-70b"].model_path).samefile(existing)
