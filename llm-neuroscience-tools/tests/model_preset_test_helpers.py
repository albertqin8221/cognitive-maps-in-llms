# This file contains three functions that build "default" prompts to test out 
# prompt and model presets.
# The default prompt has the structure:
# <A> has daughter <B>. <B> has daughter <C>. Who is <B>'s mother? 


from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime


def build_test_input_from_entities(
    prompt_runtime: RelationalPromptRuntime,
    entities: dict[str, str],
) -> dict[str, dict]:
    entities_list = list(entities.values())

    context_entities = entities
    context_relations = {
        R: "FWD" for R in prompt_runtime.schema.context_relation_symbols
    }
    query_entities = {
        E: entities_list[(1 - i) % len(entities_list)]
        for i, E in enumerate(prompt_runtime.schema.query_entity_symbols)
    }
    query_relation_options = prompt_runtime.relation_pool.query_relations
    assert len(query_relation_options)>0, (
        "Query_relations must contain at least one item."
    )
    default_query_relation = (
        query_relation_options[1]
        if len(query_relation_options) > 1
        else query_relation_options[0]
    )
    query_relations = {
        symbol: default_query_relation
        for symbol in prompt_runtime.schema.query_relation_symbols
    }

    return {
        "context_entities": context_entities,
        "context_relations": context_relations,
        "query_entities": query_entities,
        "query_relations": query_relations,
    }


def build_test_input(
    prompt_runtime: RelationalPromptRuntime,
) -> dict[str, dict]:
    return build_test_input_from_entities(
        prompt_runtime,
        entities=prompt_runtime.sample_entities(0),
    )


def build_nonoverlapping_test_inputs(
    prompt_runtime: RelationalPromptRuntime,
) -> tuple[dict, dict]:
    entities_A, entities_B = prompt_runtime.sample_nonoverlapping_entities(1)
    return (
        build_test_input_from_entities(prompt_runtime, entities_A),
        build_test_input_from_entities(prompt_runtime, entities_B),
    )
