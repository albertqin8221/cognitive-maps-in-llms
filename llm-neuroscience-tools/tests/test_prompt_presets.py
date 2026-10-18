import re

import pytest
from _pytest.fixtures import SubRequest

from llm_neuroscience_tools.presets import get_preset_prompt_names, build_preset_runtime
from llm_neuroscience_tools.presets.build_preset_runtimes import RuntimeBundle
from llm_neuroscience_tools.presets.prompt_configs import PROMPT_CONFIGS
from model_preset_test_helpers import build_test_input


# Builds a runtime bundle for a given prompt preset
# The model preset is used only for its tokenizer - we test things with 
# the Llama tokenizer
PROMPT_PRESET_NAMES: list[str] = get_preset_prompt_names()
MODEL_PRESET_NAME: str = "llama-70b"


@pytest.mark.codex
def test_alien_name_variants_are_registered() -> None:
    expected_variants = {
        "ABC_alien_older_younger": "ABC_older_younger",
        "ABC_alien_taller_shorter": "ABC_taller_shorter",
        "ABC_alien_faster_slower": "ABC_faster_slower",
        "ABC_alien_company_structure": "ABC_company_structure",
    }

    for alien_name, human_name in expected_variants.items():
        alien_config = PROMPT_CONFIGS[alien_name]
        human_config = PROMPT_CONFIGS[human_name]
        assert alien_config.name_pool_name == "alien_names"
        assert alien_config.prompt_schema is human_config.prompt_schema
        assert alien_config.relation_pool is human_config.relation_pool


@pytest.fixture(scope='module', params=PROMPT_PRESET_NAMES)
def runtime_bundle(request: SubRequest) -> RuntimeBundle:
    return build_preset_runtime(
        model_name  = MODEL_PRESET_NAME,
        prompt_name = request.param,
        use_dummy   = True
    ) 


# Smoke test that builds a concrete prompt and prints for inspection
def test_render_prompt_smoke(runtime_bundle: RuntimeBundle) -> None:

    my_prompt_runtime = runtime_bundle.prompt_runtime
    my_formatter      = runtime_bundle.formatter

    prompt = my_prompt_runtime.render_prompt_as_tokens(
        **build_test_input(my_prompt_runtime)
    )
    print()
    print(f'Smoke test for prompt family `{runtime_bundle.prompt_config.name}`',
          f'using `{runtime_bundle.model_config.name}` tokenizer.')
    print('Rendered prompt:')
    print(my_formatter.get_pretty_prompt(prompt, my_prompt_runtime.landmarks))


@pytest.mark.codex
def test_build_query_specs_are_complete_and_renderable(
    runtime_bundle: RuntimeBundle,
) -> None:
    prompt_runtime = runtime_bundle.prompt_runtime
    context_entities = prompt_runtime.sample_entities(seed=0)
    query_specs = prompt_runtime.build_query_specs(context_entities)

    n_entities = len(context_entities)
    n_relations = len(prompt_runtime.relation_pool.query_relations)
    if prompt_runtime.query_mode == "yes_no":
        assert len(query_specs) == n_entities * (n_entities - 1) * n_relations
        expected_answer_tokens = {"Y": "Yes", "N": "No", "U": "Unknown"}
    else:
        assert prompt_runtime.query_mode == "who_is"
        assert len(query_specs) == n_entities * n_relations
        expected_answer_tokens = {**context_entities, "U": "Unknown"}

    context_relations = {
        symbol: "FWD"
        for symbol in prompt_runtime.schema.context_relation_symbols
    }
    for query_spec in query_specs.values():
        assert query_spec["answer_tokens"] == expected_answer_tokens
        prompt_runtime.render_prompt_as_tokens(
            context_entities=context_entities,
            context_relations=context_relations,
            query_entities=query_spec["query_entities"],
            query_relations=query_spec["query_relations"],
        )


@pytest.mark.codex
def test_render_prompt_with_empty_context(runtime_bundle: RuntimeBundle) -> None:
    prompt_runtime = runtime_bundle.prompt_runtime
    test_input = build_test_input(prompt_runtime)

    prompt = prompt_runtime.render_prompt_as_tokens(
        context_entities=None,
        context_relations=None,
        query_entities=test_input["query_entities"],
        query_relations=test_input["query_relations"],
    )

    base_runtime = prompt_runtime.prompt_runtime
    context_symbols = (
        prompt_runtime.schema.context_entity_symbols
        + prompt_runtime.schema.context_relation_symbols
    )
    for symbol in context_symbols:
        for begin, end in base_runtime.slot_metadata[symbol]["locations"]:
            assert prompt[begin:end] == [
                base_runtime.filler_token_id
            ] * (end - begin)


@pytest.mark.codex
def test_render_prompt_rejects_incomplete_or_fully_empty_inputs(
    runtime_bundle: RuntimeBundle,
) -> None:
    prompt_runtime = runtime_bundle.prompt_runtime
    test_input = build_test_input(prompt_runtime)

    with pytest.raises(ValueError, match="context_entities and context_relations"):
        prompt_runtime.render_prompt_as_tokens(
            context_entities=test_input["context_entities"],
            context_relations=None,
            query_entities=test_input["query_entities"],
            query_relations=test_input["query_relations"],
        )

    with pytest.raises(ValueError, match="query_entities and query_relations"):
        prompt_runtime.render_prompt_as_tokens(
            context_entities=test_input["context_entities"],
            context_relations=test_input["context_relations"],
            query_entities=test_input["query_entities"],
            query_relations=None,
        )

    with pytest.raises(ValueError, match="At least one"):
        prompt_runtime.render_prompt_as_tokens()


# Test that the landmark positions for each substitution key matches what we expect
def test_landmark_positions(runtime_bundle: RuntimeBundle) -> None:

    # Build a prompt
    my_prompt_runtime = runtime_bundle.prompt_runtime
    base_prompt_runtime = my_prompt_runtime.prompt_runtime
    test_input = build_test_input(my_prompt_runtime)
    context_entities  = test_input['context_entities']
    context_relations = test_input['context_relations']
    query_entities    = test_input['query_entities']
    query_relations   = test_input['query_relations']
    prompt = my_prompt_runtime.render_prompt_as_tokens(**test_input)

    # Build the substitutions dictionary
    substitutions = context_entities | query_entities
    substitutions |= {
        symbol: my_prompt_runtime._resolve_relation(relation)
        for symbol, relation in context_relations.items()
    }
    substitutions |= {
        symbol: my_prompt_runtime._resolve_relation(relation)
        for symbol, relation in query_relations.items()
    }

    # Check that the prompt is the right length
    assert my_prompt_runtime.landmarks["PROMPT_END"] == len(prompt) - 1

    # Check that each primitive landmark points to the correct token from the
    # corresponding rendered substitution value.
    for landmark_name, position in my_prompt_runtime.landmarks.items():
        stem = landmark_name
        offset = 0
        is_indexed_landmark = "." in landmark_name
        if is_indexed_landmark:
            stem, offset_text = landmark_name.rsplit(".", 1)
            offset = int(offset_text)

        base_key = re.sub(r"\d+$", "", stem)
        if base_key not in my_prompt_runtime.schema.substitution_keys:
            continue

        encoded_substitution = my_prompt_runtime.tokenizer.encode(
            " " + substitutions[base_key],
            add_special_tokens=False,
        )
        if is_indexed_landmark:
            if offset < len(encoded_substitution):
                expected_token_id = encoded_substitution[offset]
            else:
                expected_token_id = base_prompt_runtime.filler_token_id
        else:
            expected_token_id = encoded_substitution[0]

        assert prompt[position] == expected_token_id, (
            f"{landmark_name}: expected token id {expected_token_id}, "
            f"got {prompt[position]}"
        )
