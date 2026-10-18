import pytest

from llm_neuroscience_tools.presets.build_preset_runtimes import load_names
from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS
from llm_neuroscience_tools.presets.prompt_configs import PROMPT_CONFIGS


pytestmark = pytest.mark.codex

REQUIRED_NAME_POOLS = sorted({
    (model.tokenizer_family_folder, prompt.name_pool_name)
    for model in MODEL_CONFIGS.values()
    for prompt in PROMPT_CONFIGS.values()
})


@pytest.mark.parametrize("tokenizer_family, name_pool", REQUIRED_NAME_POOLS)
def test_required_name_pool_is_packaged(tokenizer_family, name_pool):
    names = load_names(name_pool, tokenizer_family)
    assert names, f"Empty name pool: {tokenizer_family}/{name_pool}"
