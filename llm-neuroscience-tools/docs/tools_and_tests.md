# Package tools and tests

[README](../README.md)

The [`tools/`](../tools/) directory contains scripts for downloading models,
inspecting tokenizers and models, benchmarking, and building name pools.
Run the examples below from the `llm-neuroscience-tools/` package directory, with
the package installed in your active Python environment.

For tools that load a model or tokenizer by preset, first set:

```bash
export LNT_MODELS_DIR=/path/to/models
```

Preset directories use the canonical Hugging Face repository name, with `/`
replaced by `__` (for example, `Qwen__Qwen3-0.6B`). Except for the downloader,
these tools expect the required files to be available locally.

## `list_presets.py`

Prints the counts and sorted names from `MODEL_CONFIGS` and `PROMPT_CONFIGS`:

```bash
python tools/list_presets.py
```

This does not load models or require `LNT_MODELS_DIR`; a listed preset is not 
proof that its model has been downloaded.

## `download_model.py`

Downloads a model snapshot, including its tokenizer and configuration files,
using Hugging Face Hub. Select exactly one source: a package preset or a Hugging
Face repository ID.

```bash
python tools/download_model.py --model-preset qwen-0p6b --dry-run
python tools/download_model.py --model-preset qwen-0p6b
```

Alternatively, select the repository and destination explicitly:

```bash
python tools/download_model.py --model Qwen/Qwen3-0.6B --output-dir /path/to/models
```

- `--model-preset NAME` / `-m`: resolve a model preset to its Hugging Face repository ID.
- `--model OWNER/REPOSITORY`: download a repository directly. This does not
  register a new package preset.
- `--output-dir PATH` / `-o`: override `LNT_MODELS_DIR` for this download only. To load
  the result through presets later, set `LNT_MODELS_DIR` to this same parent path.
- `--revision REF`: select a branch, tag, or commit. For reproducibility, use a
  full commit hash; see [model revisions](model_revisions.md).
- `--local-name NAME`: override the final directory name. Preset loading only
  discovers canonical `owner__repository` names, so avoid this override for
  ordinary preset downloads.
- `--dry-run`: preview without downloading model files. With a Hub version that
  supports detailed previews, this may still contact the Hub for file metadata;
  otherwise it prints only the resolved target and revision.

The script creates the destination root if needed.

## `tokenization_playground.py`

Loads the local tokenizers for `gemma-27b`, `llama-70b`, and `qwen-32b`, then
repeatedly prompts for a string:

```bash
python tools/tokenization_playground.py
```

For each input, it prints token IDs from each tokenizer.

## `chatbot.py`

Loads one model preset and streams an interactive conversation:

```bash
python tools/chatbot.py --model-preset qwen-0p6b --max-new-tokens 256 --temperature 0
```

Select the model with `--model-preset` / `-m`. Generation options are:

- `--max-new-tokens`: positive response-token limit; default `1280`.
- `--temperature`: default `0.7`; `0` selects greedy decoding, while positive
  values enable sampling.
- `--top-p`: sampling threshold in `(0, 1]`; default `0.9`. 

Use `/clear` to forget the conversation without reloading the model, and `/quit`
or `/exit` to stop. 

## `benchmark_forward_passes.py`

Measures runtime initialization and the mean duration of three backend methods:
`forward`, `forward_with_recording`, and `forward_with_patching`. In my experience,
patching and recording does not significantly change the time for a forward pass.

```bash
python tools/benchmark_forward_passes.py \
    --model-preset qwen-0p6b \
    --prompt-preset ABC_family_tree_who_is \
    --num-trials 10
```

Options:

- `--model-preset` / `-m`: required model preset.
- `--prompt-preset` / `-p`: defaults to `ABC_family_tree_who_is`.
- `--num-trials` / `-n`: positive number of timed trials per variant; default `10`.

The script prepares donor recordings and runs a warm-up pass, then times the
requested number of trials for each of the three variants. Results and progress
are printed.

## `build_name_pools/`

These utilities build entity lists. Normal users do not need to run them: 
the package already includes name pools under `src/llm_neuroscience_tools/data/name_pools/`.

All three builders require a local tokenizer and keep only strings for which
both `name` and `" " + name` encode as exactly one token, with special tokens
disabled.

Shared CLI options:

- Select exactly one of `--model-preset` / `-m` and `--tokenizer-path` / `-t`.
- Supply `--output-path` / `-o`, ending in `.txt`. The parent directory must
  already exist.

Each writes one name per line and overwrites an existing output file without
confirmation. Counts, preset names, output suffixes, and parent directories are
checked before loading a tokenizer. Use a new output path to preserve an
existing pool.


### `build_female_name_pool.py`

Reads the bundled `SSA_female_names_2000.txt` in source order and selects the first
requested number of tokenizer-compatible names:

```bash
python tools/build_name_pools/build_female_name_pool.py \
    --model-preset qwen-0p6b --num-names 20 --output-path /tmp/new_female_names.txt
```

`--num-names` / `-n` is required; use a positive count. If too few compatible
names are found, the CLI raises an error before writing.

`SSA_female_names_2000.txt` is the candidate data file, with comma-separated
`name,sex,count` rows, not a generated one-name-per-line pool. The loader uses
the name column and preserves row order; it does not filter the sex column or
sort by count. Its path is relative to the script, not the working directory.
There is no CLI option for replacing this source file, although the Python
helper functions accept a `source_path` argument.

### `build_alien_name_pool.py`

Ranks all lowercase three-letter strings by increasing mean `wordfreq`
probability across ten languages (`en`, `es`, `fr`, `de`, `pt`, `it`, `nl`, `ru`,
`tr`, `pl`), breaking ties alphabetically. It then takes the first requested
number that pass the tokenizer check. Low probability is a selection heuristic,
not a guarantee that a string has no meaning in any language.

This tool additionally requires `wordfreq`, which is not a package dependency:

```bash
python -m pip install wordfreq
python tools/build_name_pools/build_alien_name_pool.py \
    --model-preset qwen-0p6b --num-names 20 --output-path /tmp/new_alien_names.txt
```

`--num-names` / `-n` is required; use a positive count. Too few compatible strings
causes an error before writing. Language selection is defined in the script,
not exposed as a CLI option. Record the tokenizer revision and `wordfreq`
version if reproducing a generated pool.

`wordfreq` is loaded only when computing scores, so `--help` works without it.

### `build_places_name_pool.py`

Filters the script's fixed `PLACES` list, retaining its order:

```bash
python tools/build_name_pools/build_places_name_pool.py \
    --model-preset qwen-0p6b --output-path /tmp/new_place_names.txt
```

There is no `--num-names` option: all compatible places are written. The
candidate list lives in the Python script; there is no external input-file
option.

## Tests

Run the small tests that need neither downloaded model files nor a GPU:

```bash
python -m pytest tests/test_model_paths.py tests/test_download_model.py tests/test_tool_cli.py tests/test_prompt_runtime.py tests/test_cache_data.py
```

The prompt-preset tests use a dummy backend with the local `llama-70b` tokenizer:

```bash
python -m pytest tests/test_prompt_presets.py
```

Real-model integration tests are marked `gpu_required` and use the model presets
in the shared fixture. On appropriately provisioned GPU hardware, select a model
explicitly, for example:

```bash
python -m pytest -m gpu_required -k 'qwen-0p6b' tests/test_model_presets.py tests/test_model_presets_KV_patching.py
```

An unfiltered test run can attempt to load every configured model on a GPU machine.
On Della, run model tests within a suitable allocation.

Tests written by Codex or another AI coding agent are marked `codex`:

```bash
# Run only agent-written tests
python -m pytest -m codex

# Exclude agent-written tests (remaining tests may need local models or a GPU)
python -m pytest -m "not codex"
```
