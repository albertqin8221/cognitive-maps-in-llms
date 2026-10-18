"""Runtime bundle types, model presets, and builders for experiment runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any
from transformers import AutoTokenizer

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.backends.dummy import DummyBackend
from llm_neuroscience_tools.formatting import OutputFormatter
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime

from .model_configs import MODEL_CONFIGS, ModelConfig
from .prompt_configs import PROMPT_CONFIGS, PromptConfig


def load_names(name_pool_name: str, tokenizer_family_folder: str) -> list[str]:
    """Load a packaged name pool from the data/name_pools directory."""
    return (
        files("llm_neuroscience_tools")
        .joinpath(f"data/name_pools/{tokenizer_family_folder}/{name_pool_name}.txt")
        .read_text()
        .splitlines()
    )


def _load_tokenizer(tokenizer_path: str):
    """Load a tokenizer from a local model path."""
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )


# ------------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """Loaded model components that can be shared across prompt presets."""

    backend: LMBackend
    tokenizer: Any
    model_config: ModelConfig


@dataclass
class RuntimeBundle:
    """Fully initialized runtime components for running an experiment."""

    backend: LMBackend
    prompt_runtime: RelationalPromptRuntime
    formatter: OutputFormatter
    model_config: ModelConfig
    prompt_config: PromptConfig


def load_preset_model(
    model_name: str,
    use_dummy: bool = False,
) -> ModelBundle:
    """Load a model backend and tokenizer without selecting a prompt preset.

    Args:
        model_name: Name of a preset in ``MODEL_CONFIGS``.
        use_dummy: If true, build a ``DummyBackend`` using the preset's
            configured shapes. Its tokenizer is still loaded from local files.
    """
    try:
        model_config = MODEL_CONFIGS[model_name]
    except KeyError as exc:
        raise ValueError(f"Model name {model_name!r} not recognized.") from exc

    if use_dummy:
        tokenizer = _load_tokenizer(model_config.tokenizer_path)
        backend = DummyBackend(
            tokenizer=tokenizer,
            D_model=model_config.shapes["D_model"],
            N_layers=model_config.shapes["N_layers"],
        )
    else:
        backend = model_config.backend_cls.from_model_path(
            model_config.model_path,
            **model_config.backend_kwargs,
        )
        tokenizer = backend.tokenizer

    return ModelBundle(
        backend=backend,
        tokenizer=tokenizer,
        model_config=model_config,
    )


def build_prompt_runtime(
    model: ModelBundle,
    prompt_name: str,
) -> RuntimeBundle:
    """Build a fresh prompt runtime and formatter around an already-loaded model.

    The returned bundle shares the backend, tokenizer, and model configuration
    with ``model``. The new prompt runtime compiles its own token landmarks.
    No model weights or tokenizer files are loaded here.
    """
    try:
        prompt_config = PROMPT_CONFIGS[prompt_name]
    except KeyError as exc:
        raise ValueError(f"Prompt name {prompt_name!r} not recognized.") from exc

    name_pool = load_names(
        prompt_config.name_pool_name,
        model.model_config.tokenizer_family_folder,
    )
    prompt_runtime = RelationalPromptRuntime(
        schema=prompt_config.prompt_schema,
        tokenizer=model.tokenizer,
        name_pool=name_pool,
        relation_pool=prompt_config.relation_pool,
    )

    return RuntimeBundle(
        backend=model.backend,
        prompt_runtime=prompt_runtime,
        formatter=OutputFormatter(tokenizer=model.tokenizer),
        model_config=model.model_config,
        prompt_config=prompt_config,
    )


def build_preset_runtime(
    model_name: str,
    prompt_name: str,
    use_dummy: bool = False,
) -> RuntimeBundle:
    """Load a preset model and build its prompt runtime in one call.

    For several prompt presets, call ``load_preset_model`` once and then
    ``build_prompt_runtime`` for each prompt to avoid reloading model weights.
    ``use_dummy=True`` loads a real local tokenizer but no model weights.
    Unknown model or prompt names raise ``ValueError`` before loading the model.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model name {model_name!r} not recognized.")
    if prompt_name not in PROMPT_CONFIGS:
        raise ValueError(f"Prompt name {prompt_name!r} not recognized.")

    model = load_preset_model(model_name, use_dummy=use_dummy)
    return build_prompt_runtime(model, prompt_name)


def get_preset_model_names():
    return list(MODEL_CONFIGS.keys())

def get_preset_prompt_names():
    return list(PROMPT_CONFIGS.keys())
