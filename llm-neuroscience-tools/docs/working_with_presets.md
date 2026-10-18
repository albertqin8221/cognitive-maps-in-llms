# Working With Presets

[README](../README.md)

## Prerequisites

Install the package as described in the [README](../README.md#installation).
Set `LNT_MODELS_DIR` to the parent directory containing the downloaded models.

## Runtime bundles

`llm_neuroscience_tools` provides a collection of presets for running experiments with different models and prompt families. The main entry point is:

```python
from llm_neuroscience_tools.presets import build_preset_runtime
```

A runtime is built from a model preset and a prompt preset:

```python
runtime = build_preset_runtime(
    model_name="gemma-27b",
    prompt_name="ABC_family_tree_who_is"
)
```

This returns a `RuntimeBundle` with five fields:

- `runtime.backend`
- `runtime.prompt_runtime`
- `runtime.formatter`
- `runtime.model_config`
- `runtime.prompt_config`

---

## Minimal workflow

A typical pattern is:

```python
from llm_neuroscience_tools.presets import build_preset_runtime

runtime = build_preset_runtime(
    model_name="gemma-27b",
    prompt_name="ABC_family_tree_who_is"
)

backend = runtime.backend
world = runtime.prompt_runtime
formatter = runtime.formatter
entities = world.sample_entities(seed=0)

prompt_inputs = {
    "context_entities": entities,
    "context_relations": {"R_AB": "FWD", "R_BC": "FWD"},
    "query_entities": {"QE": entities["B"]},
    "query_relations": {"QR": "REV"},
}

prompt_ids = world.render_prompt_as_tokens(**prompt_inputs)
logits = backend.forward(prompt_ids)

print(formatter.get_prompt_text(prompt_ids))
print(formatter.get_pretty_top_k(logits, k=10))
```

In this example:

- Entity names are drawn from the preset's tokenizer-specific name pool.
- `FWD` means “use the preset’s canonical forward relation” (`daughter` in this case)
- `REV` means “use the preset’s canonical backward relation” (`mother` in this case)

---

## Dummy mode

To test prompt construction or experiment plumbing without loading a real model onto the GPU, use:

```python
runtime = build_preset_runtime(
    model_name="gemma-27b",
    prompt_name="ABC_family_tree_who_is",
    use_dummy=True,
)
```

In this mode, the prompts are still rendered normally using the appropriate tokenizer,
but the backend returns random shape-correct logits and activations. 

---

## Available presets

`build_preset_runtime(...)` pulls together two config registries:

- [`model_configs.py`](../src/llm_neuroscience_tools/presets/model_configs.py):
  model repository IDs, resolved paths, backend classes, and model dimensions.
- [`prompt_configs.py`](../src/llm_neuroscience_tools/presets/prompt_configs.py):
  prompt schemas and name/relation pools. Templates live in
  [`prompt_definitions/`](../src/llm_neuroscience_tools/presets/prompt_definitions/).

Current available model presets:

- `gemma-4b`
- `gemma-12b`
- `gemma-27b`
- `llama-8b`
- `llama-70b`
- `qwen-0p6b`
- `qwen-1p7b`
- `qwen-4b`
- `qwen-8b`
- `qwen-14b`
- `qwen-32b`
- `qwen-30b-a3b`

Current available prompt presets:

- `ABC_family_tree_who_is`
- `ABC_family_tree_yes_no`
- `ABC_alien_family_tree_who_is`
- `ABC_alien_family_tree_yes_no`
- `ABC_gene_regulatory_network`
- `ABC_neural_circuit`
- `ABC_older_younger`
- `ABC_taller_shorter`
- `ABC_faster_slower`
- `ABC_alien_older_younger`
- `ABC_alien_taller_shorter`
- `ABC_alien_faster_slower`
- `ABC_company_structure`
- `ABC_alien_company_structure`
- `ABC_proposition_equivalence`
- `ABC_alien_proposition_equivalence`
- `ABC_variable_equivalence`
- `ABC_alien_variable_equivalence`
- `ABC_east_west`
- `ABC_alien_east_west`
- `ABC_north_south`
- `ABC_alien_north_south`
- `ABC_ahead_behind`
- `ABC_alien_ahead_behind`
- `AB_CD_family_trees_who_is`
- `AB_CD_family_trees_yes_no`
- `ABC_family_trees_descendant_ancestor`
- `ABC_alien_family_trees_descendant_ancestor`
- `ABC_older_younger_unique_locations_who_is`
- `ABC_older_younger_unique_locations_yes_no`
- `ABC_alien_older_younger_unique_locations_yes_no`

To inspect the registries programmatically without loading a tokenizer or model:

```python
from llm_neuroscience_tools.presets import (
    get_preset_model_names,
    get_preset_prompt_names,
)

print(get_preset_model_names())
print(get_preset_prompt_names())
```
