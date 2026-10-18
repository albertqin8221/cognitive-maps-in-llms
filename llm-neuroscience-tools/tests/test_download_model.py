import importlib.util
from pathlib import Path

import pytest

from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS


pytestmark = pytest.mark.codex


@pytest.fixture
def downloader(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "tools" / "download_model.py"
    spec = importlib.util.spec_from_file_location("download_model", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbid_download(**kwargs):
        pytest.fail("Unexpected Hub call: downloads must be mocked in this test")

    monkeypatch.setattr(module, "snapshot_download", forbid_download)
    return module


@pytest.mark.parametrize("flag", ["--model-preset", "-m", "--preset"])
def test_preset_download_is_found_by_runtime_config(downloader, monkeypatch, tmp_path, flag):
    root = tmp_path / "models"
    monkeypatch.setenv("LNT_MODELS_DIR", str(root))
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True)
        return str(target)

    monkeypatch.setattr(downloader, "snapshot_download", download)
    assert downloader.main([flag, "qwen-0p6b", "--revision", "test-revision"]) == 0
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "Qwen/Qwen3-0.6B"
    assert calls[0]["revision"] == "test-revision"
    assert calls[0]["local_dir"] == str(root / "Qwen__Qwen3-0.6B")
    assert MODEL_CONFIGS["qwen-0p6b"].model_path == calls[0]["local_dir"]
    assert MODEL_CONFIGS["qwen-0p6b"].tokenizer_path == calls[0]["local_dir"]


@pytest.mark.parametrize("env_root", [None, "another-root"])
def test_explicit_destination_and_custom_name_still_work(
    downloader, monkeypatch, tmp_path, env_root,
):
    if env_root is None:
        monkeypatch.delenv("LNT_MODELS_DIR", raising=False)
    else:
        monkeypatch.setenv("LNT_MODELS_DIR", str(tmp_path / env_root))
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr(downloader, "snapshot_download", download)
    output = tmp_path / "explicit"
    assert downloader.main([
        "--model", "Qwen/Qwen3-0.6B", "--output-dir", str(output),
        "--local-name", "custom-name",
    ]) == 0
    assert calls[0]["local_dir"] == str(output / "custom-name")


@pytest.mark.parametrize("detailed", [False, True])
def test_dry_run_does_not_create_directories(downloader, monkeypatch, tmp_path, detailed):
    root = tmp_path / "not-created"
    monkeypatch.setenv("LNT_MODELS_DIR", str(root))
    monkeypatch.setattr(downloader, "_SNAPSHOT_DOWNLOAD_PARAMS", {"dry_run"} if detailed else set())
    calls = []

    def preview(**kwargs):
        calls.append(kwargs)
        return []

    if detailed:
        monkeypatch.setattr(downloader, "snapshot_download", preview)
    assert downloader.main(["--preset", "qwen-0p6b", "--dry-run"]) == 0
    assert not root.exists()
    if detailed:
        assert calls[0]["dry_run"] is True
        assert calls[0]["local_dir"] == str(root / "Qwen__Qwen3-0.6B")


@pytest.mark.parametrize(
    "args, expected_error",
    [
        (["--preset", "qwen-0p6b"], "export LNT_MODELS_DIR="),
        (["--preset", "unknown"], "Unknown model preset"),
        (["--model", "../model"], "owner/repo"),
    ],
)
def test_setup_errors_do_not_download(downloader, monkeypatch, capsys, args, expected_error):
    monkeypatch.delenv("LNT_MODELS_DIR", raising=False)
    assert downloader.main(args) == 1
    assert expected_error in capsys.readouterr().err


def test_model_and_preset_are_mutually_exclusive(downloader):
    with pytest.raises(SystemExit) as error:
        downloader.parse_args(["--model", "Qwen/Qwen3-0.6B", "--preset", "qwen-0p6b"])
    assert error.value.code == 2
