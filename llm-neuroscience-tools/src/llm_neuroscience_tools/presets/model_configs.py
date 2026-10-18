"""Model preset definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.backends.gemma import GemmaBackend
from llm_neuroscience_tools.backends.llama import LlamaBackend
from llm_neuroscience_tools.backends.qwen import QwenBackend
from llm_neuroscience_tools.model_paths import get_model_path


@dataclass(frozen=True)
class ModelConfig:
    """Model/backend configuration used to build runtimes."""

    name: str
    repo_id: str
    backend_cls: type[LMBackend]
    tokenizer_family_folder: str
    n_parameters: int
    shapes: dict[str, int]
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def model_path(self) -> str:
        """Resolve this preset's local model directory when it is requested."""
        return str(get_model_path(self.repo_id))

    @property
    def tokenizer_path(self) -> str:
        """Resolve the tokenizer directory from the same model snapshot."""
        return self.model_path


# -------------------- Gemma models --------------------


GEMMA_4B_CONFIG = ModelConfig(
    name = "gemma-4b",
    repo_id = "google/gemma-3-4b-it",
    backend_cls = GemmaBackend,
    tokenizer_family_folder = "gemma",
    n_parameters = 4_300_079_472,
    shapes = {"D_model": 2560, "N_layers": 34, "N_heads": 4, "D_head": 256},
)

GEMMA_12B_CONFIG = ModelConfig(
    name = "gemma-12b",
    repo_id = "google/gemma-3-12b-it",
    backend_cls = GemmaBackend,
    tokenizer_family_folder = "gemma",
    n_parameters = 12_187_325_040,
    shapes = {"D_model": 3840, "N_layers": 48, "N_heads": 8, "D_head": 256},
)

GEMMA_27B_CONFIG = ModelConfig(
    name = "gemma-27b",
    repo_id = "google/gemma-3-27b-it",
    backend_cls = GemmaBackend,
    tokenizer_family_folder = "gemma",
    n_parameters = 27_432_406_640,
    shapes = {"D_model": 5376, "N_layers": 62, "N_heads": 16, "D_head": 128},
)


# -------------------- Llama 3 models --------------------


LLAMA_8B_CONFIG = ModelConfig(
    name = "llama-8b",
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    backend_cls = LlamaBackend,
    tokenizer_family_folder = "llama",
    n_parameters = 8_030_261_248,
    shapes = {"D_model": 4096, "N_layers": 32, "N_heads": 8, "D_head": 128},
)

LLAMA_70B_CONFIG = ModelConfig(
    name = "llama-70b",
    repo_id = "meta-llama/Llama-3.1-70B-Instruct",
    backend_cls = LlamaBackend,
    tokenizer_family_folder = "llama",
    n_parameters = 70_553_706_496,
    shapes = {"D_model": 8192, "N_layers": 80, "N_heads": 8, "D_head": 128},
)

# -------------------- Qwen 3 models --------------------


QWEN_0p6B_CONFIG = ModelConfig(
    name = "qwen-0p6b",
    repo_id = "Qwen/Qwen3-0.6B",
    backend_cls = QwenBackend,
    tokenizer_family_folder = "qwen",
    n_parameters = 751_632_384,
    shapes = {"D_model": 1024, "N_layers": 28, "N_heads": 8, "D_head": 128},
)

QWEN_1p7B_CONFIG = ModelConfig(
    name = "qwen-1p7b",
    repo_id = "Qwen/Qwen3-1.7B",
    backend_cls = QwenBackend,
    tokenizer_family_folder = "qwen",
    n_parameters = 2_031_739_904,
    shapes = {"D_model": 2048, "N_layers": 28, "N_heads": 8, "D_head": 128},
)

QWEN_4B_CONFIG = ModelConfig(
    name = "qwen-4b",
    repo_id = "Qwen/Qwen3-4B",
    backend_cls = QwenBackend,
    tokenizer_family_folder = "qwen",
    n_parameters = 4_022_468_096,
    shapes = {"D_model": 2560, "N_layers": 36, "N_heads": 8, "D_head": 128},
)

QWEN_8B_CONFIG = ModelConfig(
    name = "qwen-8b",
    repo_id = "Qwen/Qwen3-8B",
    backend_cls = QwenBackend,
    tokenizer_family_folder = "qwen",
    n_parameters = 8_190_735_360,
    shapes = {"D_model": 4096, "N_layers": 36, "N_heads": 8, "D_head": 128},
)

QWEN_14B_CONFIG = ModelConfig(
    name = "qwen-14b",
    repo_id = "Qwen/Qwen3-14B",
    backend_cls = QwenBackend,
    tokenizer_family_folder = "qwen",
    n_parameters = 14_768_307_200,
    shapes = {"D_model": 5120, "N_layers": 40, "N_heads": 8, "D_head": 128},
)

QWEN_32B_CONFIG = ModelConfig(
    name = "qwen-32b",
    repo_id = "Qwen/Qwen3-32B",
    backend_cls = QwenBackend,
    tokenizer_family_folder = "qwen",
    n_parameters = 32_762_123_264,
    shapes = {"D_model": 5120, "N_layers": 64, "N_heads": 8, "D_head": 128},
)

QWEN_30B_A3B_CONFIG = ModelConfig(
    name = "qwen-30b-a3b",
    repo_id = "Qwen/Qwen3-30B-A3B",
    backend_cls = QwenBackend,
    tokenizer_family_folder = "qwen",
    n_parameters = 30_532_122_624,
    shapes = {"D_model": 2048, "N_layers": 48, "N_heads": 4, "D_head": 128},
)

# ------------------------------------------------------------------------------


MODEL_CONFIGS = {
    config.name: config 
    for config in 
    [
        GEMMA_4B_CONFIG,
        GEMMA_12B_CONFIG,
        GEMMA_27B_CONFIG,
        LLAMA_8B_CONFIG,
        LLAMA_70B_CONFIG,
        QWEN_0p6B_CONFIG,
        QWEN_1p7B_CONFIG,
        QWEN_4B_CONFIG,
        QWEN_8B_CONFIG,
        QWEN_14B_CONFIG,
        QWEN_32B_CONFIG,
        QWEN_30B_A3B_CONFIG,
    ]
}
