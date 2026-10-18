# Shared fixtures file. 
# Contains a single fixture that builds a runtime with each model preset
# along with a basic family tree prompt


import pytest
import torch
from _pytest.fixtures import SubRequest

from llm_neuroscience_tools.presets import (
    RuntimeBundle,
    build_preset_runtime,
    get_preset_model_names,
)


PROMPT_PRESET_NAME = "ABC_family_tree_who_is"
MODEL_PRESET_NAMES = get_preset_model_names()


@pytest.fixture(scope="module", params=MODEL_PRESET_NAMES)
def runtime_bundle(request: SubRequest) -> RuntimeBundle:
    if not torch.cuda.is_available():
        pytest.skip("GPU required!")
    return build_preset_runtime(
        model_name=request.param,
        prompt_name=PROMPT_PRESET_NAME,
        use_dummy=False,
    )
