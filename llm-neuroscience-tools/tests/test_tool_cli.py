"""CLI checks without model files, downloads, or GPU inference."""

import builtins
import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.codex

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
BUILDERS = [
    "build_name_pools/build_female_name_pool.py",
    "build_name_pools/build_alien_name_pool.py",
    "build_name_pools/build_places_name_pool.py",
]
SCRIPTS = [
    "download_model.py",
    "list_presets.py",
    "chatbot.py",
    "tokenization_playground.py",
    "benchmark_forward_passes.py",
    *BUILDERS,
]


@pytest.fixture
def load_tool(monkeypatch):
    from transformers import AutoTokenizer
    from llm_neuroscience_tools.backends.base import HFDecoderBackend

    def forbid_loading(*args, **kwargs):
        pytest.fail("CLI tests must not load models/tokenizers or download files")

    monkeypatch.delenv("LNT_MODELS_DIR", raising=False)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbid_loading)
    monkeypatch.setattr(HFDecoderBackend, "from_model_path", forbid_loading)

    def load(relative_path):
        path = TOOLS_DIR / relative_path
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "snapshot_download"):
            monkeypatch.setattr(module, "snapshot_download", forbid_loading)
        return module

    return load


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_exits_without_loading(load_tool, script, capsys):
    tool = load_tool(script)
    with pytest.raises(SystemExit) as error:
        tool.main(["--help"])
    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [
    ["qwen-0p6b"], ["--max-new-tokens", "0"], ["--temperature", "-1"],
    ["--temperature", "nan"], ["--temperature", "inf"], ["--top-p", "0"],
    ["--top-p", "1.1"],
])
def test_chatbot_rejects_invalid_or_conflicting_arguments(load_tool, extra):
    with pytest.raises(SystemExit) as error:
        load_tool("chatbot.py").main(["--model-preset", "qwen-0p6b", *extra])
    assert error.value.code == 2


@pytest.mark.parametrize("extra", [
    ["--num-trials", "0"], ["--num-trials", "-1"], ["--prompt-preset", "unknown"],
])
def test_benchmark_rejects_invalid_arguments(load_tool, extra):
    with pytest.raises(SystemExit) as error:
        load_tool("benchmark_forward_passes.py").main(["-m", "qwen-0p6b", *extra])
    assert error.value.code == 2


def builder_args(script, output):
    args = ["--output-path", str(output)]
    if "places" not in script:
        args += ["--num-names", "2"]
    return args


@pytest.mark.parametrize("script", BUILDERS)
def test_builder_tokenizer_source_is_exclusive(load_tool, tmp_path, script):
    tool = load_tool(script)
    argv = ["--tokenizer-path", "/local/tokenizer", *builder_args(script, tmp_path / "names.txt")]
    assert tool.parse_args(argv).tokenizer_path == "/local/tokenizer"
    with pytest.raises(SystemExit) as error:
        tool.parse_args([*argv, "--model-preset", "qwen-0p6b"])
    assert error.value.code == 2


@pytest.mark.parametrize("script", BUILDERS)
@pytest.mark.parametrize("output", ["names.csv", "missing/names.txt"])
def test_builder_rejects_invalid_output_before_loading(load_tool, tmp_path, script, output):
    with pytest.raises(SystemExit) as error:
        load_tool(script).main(["-m", "qwen-0p6b", *builder_args(script, tmp_path / output)])
    assert error.value.code == 2


@pytest.mark.parametrize("script", BUILDERS[:2])
@pytest.mark.parametrize("count", ["0", "-1"])
def test_builder_requires_positive_count(load_tool, tmp_path, script, count):
    with pytest.raises(SystemExit) as error:
        load_tool(script).main(["-m", "qwen-0p6b", "-n", count, "-o", str(tmp_path / "names.txt")])
    assert error.value.code == 2


@pytest.mark.parametrize("script", BUILDERS)
def test_builder_writes_selected_names_in_order(load_tool, monkeypatch, tmp_path, script):
    tool = load_tool(script)
    names = ["first", "second"]
    helper = {
        BUILDERS[0]: "get_first_n_compatible_female_names_for_preset",
        BUILDERS[1]: "get_first_n_low_probability_alien_names_for_preset",
        BUILDERS[2]: "get_compatible_places_for_preset",
    }[script]
    calls = []

    def select(model_name, n=None):
        calls.append((model_name, n))
        return names

    monkeypatch.setattr(tool, helper, select)
    output = tmp_path / "names.txt"
    tool.main(["-m", "qwen-0p6b", *builder_args(script, output)])
    assert calls == [("qwen-0p6b", None if "places" in script else 2)]
    assert output.read_text() == "first\nsecond\n"


def test_alien_help_without_wordfreq(load_tool, monkeypatch):
    original_import = builtins.__import__

    def import_without_wordfreq(name, *args, **kwargs):
        if name == "wordfreq":
            raise ModuleNotFoundError("wordfreq not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_wordfreq)
    tool = load_tool(BUILDERS[1])
    with pytest.raises(SystemExit) as error:
        tool.main(["--help"])
    assert error.value.code == 0
    with pytest.raises(ImportError, match="python -m pip install wordfreq"):
        tool.compute_scores(tool.DEFAULT_LANGUAGES)
