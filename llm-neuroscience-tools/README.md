## Description

Python tools needed to run the experiments described in the paper. This package provides:

- **Prompt runtimes:** Tools to generate structured prompts.
- **Model backends:** Small wrappers for Gemma, Llama, and Qwen models that support 
interventions on activations and KV caches.
- **Base experiment:** A simple interface for running systematic experiments. 
- **Presets:** A ready-to-use set of prompt families and models. 

Paper-specific experiments, analyses, and figures live in the sibling
[`cognitive-maps-in-llms` package](../cognitive-maps-in-llms/README.md).

## Installation

Python 3.10 or newer is required. Follow the
[installation instructions](../README.md#installation) to install both packages.
Then, from the combined repository root, enter the tools package directory:

```bash
cd llm-neuroscience-tools
```

The Python import name is `llm_neuroscience_tools`. 

### Downloading models

To use the model presets, the model parameters must be downloaded separately.
First, set the environment variable `LNT_MODELS_DIR` to the directory where models should 
be stored:

```bash
export LNT_MODELS_DIR=/path/to/models
```

To persist this setting across sessions, set this variable in `~/.bashrc`. 
Note that when submitting jobs to a cluster, this variable should be set near 
the top of each job script. 

To download each model, run from the tools package directory:

```bash
python tools/download_model.py --model-preset qwen-0p6b --dry-run
python tools/download_model.py --model-preset qwen-0p6b
```

The `--revision` option selects a particular Hugging Face revision. See 
[model revisions](docs/model_revisions.md) for the exact revisions used to 
generate the figures in the paper.

Model subfolders use the Hugging Face repository ID with `/` replaced by `__`:

```text
models/
    Qwen__Qwen3-0.6B/
    google__gemma-3-27b-it/
    meta-llama__Llama-3.1-70B-Instruct/
```

Existing downloads can be used directly as long as the subdirectory name follows
this convention. 

## Quick start

The following example loads the gemma-27b preset, generates a single family tree 
prompt, and runs the model on it.

```python
from llm_neuroscience_tools.presets import build_preset_runtime

bundle = build_preset_runtime(
    model_name="gemma-27b",
    prompt_name="ABC_family_tree_who_is"
)

world = bundle.prompt_runtime
entities = world.sample_entities(seed=0)
prompt_ids = world.render_prompt_as_tokens(
    context_entities=entities,
    context_relations={"R_AB": "FWD", "R_BC": "FWD"},
    query_entities={"QE": entities["B"]},
    query_relations={"QR": "REV"},
)

print(bundle.formatter.get_pretty_prompt(prompt_ids, landmarks=world.landmarks))

logits = bundle.backend.forward(prompt_ids)
print(bundle.formatter.get_pretty_top_k(logits, k=5))
```

## Further documentation

- [Working with presets](docs/working_with_presets.md)
- [Core runtime interfaces](docs/core_runtime_interfaces.md)
- [Defining an experiment](docs/defining_an_experiment.md)
- [Package tools and tests](docs/tools_and_tests.md)
- [Model revisions](docs/model_revisions.md)

## Package layout

```text
src/llm_neuroscience_tools/
    base_experiment.py   Basic trial structure and NDJSON output
    formatting.py        Prompt / prediction inspection
    model_paths.py       Shared model-directory and folder-name rules
    backends/            Model wrappers; hooks for recording and patching
    prompts/             Prompt schemas and runtimes
    presets/             Model and prompt configurations
    data/name_pools/     Entity name pools
docs/                    Usage and API documentation
tests/                   Automated tests
tools/                   Model downloads, preset inspection, and other utilities
```
