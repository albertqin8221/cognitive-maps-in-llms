# Core Runtime Interfaces

[README](../README.md)

The three runtime objects used throughout `llm_neuroscience_tools` are:

- `RelationalPromptRuntime`
Builds prompts from a prompt schema. Its main method, `render_prompt_as_tokens(...)`, returns a list of token IDs that can be passed directly to a backend.
- `LMBackend`
Wraps the language model, and exposes methods for recording and patching internal activations and KV caches.
- `OutputFormatter`
Provides helper methods for printing prompts and model outputs in a readable format.

---

## RelationalPromptRuntime

A `RelationalPromptRuntime` combines a `RelationalPromptSchema`, a tokenizer, a name pool, and a `RelationPool`. It is responsible for rendering prompts from a pre-defined prompt template, and also contains convenience methods for sampling entities and defining token landmarks. 

---

### Accessing a `RelationalPromptRuntime`

When using preset runtimes, the RelationalPromptRuntime is available from the returned RuntimeBundle:

```python
from llm_neuroscience_tools.presets import build_preset_runtime

bundle = build_preset_runtime(
    model_name="llama-70b",
    prompt_name="ABC_family_tree_who_is",
)

my_runtime = bundle.prompt_runtime
```

Preset runtimes are built from predefined `RelationalPromptSchema` objects. To define a new  prompt family, create a custom schema:

---

### Specifying a Prompt Template

A custom prompt template can be defined by creating a `RelationalPromptSchema` (available from the public `llm_neuroscience_tools.prompts` module).

A `RelationalPromptSchema` has a string `template` plus six required schema fields:

- `context_entity_symbols`
- `context_relation_symbols`
- `query_entity_symbols`
- `query_relation_symbols`
- `query_mode`
- `derived_landmarks`

The `template` marks substitution slots using the form `<KEYocc:width>`, where:

- `KEY` is the substitution key,
- `occ` is an optional occurrence index,
- `width` is an optional token width.

Each slot’s base key must be declared in exactly one of the four symbol groups. If the same key appears more than once in the template, each occurrence must use an explicit index, such as `<B1>` and `<B2>`. Slots must also be preceded by a space so that placeholders are compiled correctly.

The `derived_landmarks` field defines additional named token positions in terms of slot positions, such as `{"CTXT_START": "A1", "CTXT_END": "C1+1"}`. Pass an empty dictionary if none are needed. These landmarks are compiled by the runtime and are useful for later analyses.

Slot widths default to one token. Substitutions are encoded with a leading space;
values longer than the declared width raise an error, and shorter values are
padded with space tokens. Choose names and relations that fit the selected
tokenizer and slot widths. The tokenizer must encode a single space as one token.

The following example defines a simple prompt template:

```python
from llm_neuroscience_tools.prompts import RelationalPromptSchema

ABC_family_tree_who_is_schema = RelationalPromptSchema(
    template="""
Instructions:
	1) Read the context describing a small family.
	2) At the end, answer the question with exactly one word. 
Context: <A1> has <R_AB> <B1>. <B2> has <R_BC> <C1>.
Question: Who is <QE>'s <QR>?
Answer: 
""",
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE",),
    query_relation_symbols=("QR",),
    query_mode="who_is",
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
)
```

---

### Using the `RelationalPromptRuntime`

---

#### Sampling Entities

```text
sample_entities(seed: int | None = None) -> dict[str, str]
```

Samples one context by assigning concrete names to the schema’s context entity symbols. The returned dictionary maps each context entity symbol to a sampled name from the runtime’s name pool:

```text
sample_nonoverlapping_entities(
    seed: int | None = None,
) -> tuple[dict[str, str], dict[str, str]]
```

Samples two contexts whose entity names do not overlap. This is useful for experiments that compare two prompts, or for donor/acceptor setups in activation patching.

---

#### Building Queries

```text
build_query_specs(
    entity_symbols_to_names: Mapping[str, str],
) -> dict[str, dict[str, Any]]
```

Builds every query for the runtime's configured `query_mode` and query
relations. Yes/no mode produces every ordered pair of distinct entities;
who-is mode produces a query for every entity. The returned mapping uses
schematic labels and contains the query substitutions and answer tokens needed
by an experiment.

---

#### Rendering Prompts

```text
render_prompt_as_tokens(
    context_entities: dict[str, str] | None = None,
    context_relations: dict[str, str] | None = None,
    query_entities: dict[str, str] | None = None,
    query_relations: dict[str, str] | None = None,
) -> list[int]
```

Instantiates the template and returns the rendered prompt as token IDs. This is usually the main rendering method used in experiments, since its output can be passed directly to an `LMBackend`.

Each substitution dictionary must provide exactly the symbols declared in the corresponding part of the schema. For example, `context_entities` must contain the schema’s context entity symbols, while `context_relations` must contain the schema’s context relation symbols.

The context arguments must either both be provided or both be omitted, and the
same rule applies to the query arguments. If a group is omitted, the runtime
fills all of its substitution slots directly with the configured filler token.
At least one group must be provided. An empty context is useful for constructing
a query prompt whose context activations will subsequently be replaced by a
mosaic.

Relation substitutions may use two shorthand markers:

- `"FWD"`: use the relation pool’s canonical forward relation
- `"REV"`: use the relation pool’s canonical backward relation

For example, in the family-tree presets, `"FWD"` resolves to `"daughter"` and `"REV"` resolves to `"mother"`.

```text
render_prompt(
    context_entities: dict[str, str] | None = None,
    context_relations: dict[str, str] | None = None,
    query_entities: dict[str, str] | None = None,
    query_relations: dict[str, str] | None = None,
) -> str
```

Instantiates the template and returns the rendered prompt as a string. Internally, this calls `render_prompt_as_tokens(...)` and decodes the resulting token IDs.

---

#### Token Landmarks

```python
runtime.landmarks
```

`landmarks` is a dictionary mapping landmark names to token positions in the compiled template. Primitive landmarks come from substitution slots, such as `"A1"` or `"B2"`. Multi-token slots expose one landmark per token position, such as `"R_AB.0"` and `"R_AB.1"`. The runtime also compiles and includes any derived landmarks declared by the schema.

---

#### Query Mode

```python
runtime.query_mode
```

`query_mode` is the required query convention inherited from the
`RelationalPromptSchema`. It is either `"yes_no"` or `"who_is"`.

---

#### Runtime Flags

```python
runtime.flags
```

`flags` contains any optional schema metadata. `query_mode` is a dedicated
schema field and must not be repeated in `flags`.

---

### Example

The following example defines a new relational prompt family from scratch. It creates a `RelationalPromptSchema`, pairs it with a `RelationPool`, samples concrete entities, and renders a prompt. Replace the tokenizer path with a local tokenizer directory and check that the example names and relations fit its one-token slots.

```python
from transformers import AutoTokenizer

from llm_neuroscience_tools.prompts import RelationalPromptSchema
from llm_neuroscience_tools.prompts.relational import (
    RelationPool,
    RelationalPromptRuntime,
)

schema = RelationalPromptSchema(
    template="""
Instructions:
	1) Read the context describing a set of comparisons between people.
	2) Answer the question with Yes, No, or Unknown.
Context: <A1> is <R_AB> than <B1>. <B2> is <R_BC> than <C1>.
Question: Is <QE> <QR> than <QCAND>?
Answer: 
""",
    context_entity_symbols   = ("A", "B", "C"),
    context_relation_symbols = ("R_AB", "R_BC"),
    query_entity_symbols     = ("QE", "QCAND"),
    query_relation_symbols   = ("QR",),
    query_mode="yes_no",
    derived_landmarks={
        "CTXT_START": "A1",
        "CTXT_END": "C1+1",
    },
)

relation_pool = RelationPool(
    forward_relation="taller",
    backward_relation="shorter",
    query_relations=("taller", "shorter"),
)

tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/local/model-or-tokenizer",
    local_files_only=True,
)

name_pool = ["Alice", "Beth", "Carol"]

my_runtime = RelationalPromptRuntime(
    schema=schema,
    tokenizer=tokenizer,
    name_pool=name_pool,
    relation_pool=relation_pool,
)

context_entities = my_runtime.sample_entities(seed=0)

prompt_text = my_runtime.render_prompt(
    context_entities=context_entities,
    context_relations={
        "R_AB": "FWD",
        "R_BC": "FWD",
    },
    query_entities={
        "QE": context_entities["A"],
        "QCAND": context_entities["C"],
    },
    query_relations={
        "QR": "FWD",
    },
)

print(prompt_text)
```

Here, `"FWD"` resolves to the relation pool’s forward_relation, so both context relations become `"taller"`. The rendered question asks whether the first sampled entity is taller than the third sampled entity.

---

## LMBackend

`LMBackend` is the common interface for running language models in `llm_neuroscience_tools`. Concrete backends wrap specific model families, such as `Llama`, `Gemma`, or `Qwen`, while exposing the methods for forward passes, activation recording/patching, and KV-cache recording/patching.

---

### Instantiating

Most users will access an `LMBackend` through a preset runtime bundle:

```python
from llm_neuroscience_tools.presets import build_preset_runtime

bundle = build_preset_runtime(
    model_name="llama-70b",
    prompt_name="ABC_family_tree_who_is",
)

my_backend   = bundle.backend
my_runtime   = bundle.prompt_runtime
my_formatter = bundle.formatter
```

The preset builder also constructs the matching tokenizer, prompt runtime, formatter, model config, and prompt config. For forward-pass plumbing checks, pass `use_dummy=True` instead of loading model weights. Dummy mode still requires a local tokenizer; it does not simulate intervention effects or implement KV operations.

Backends can also be constructed directly from a local model path:

```python
from llm_neuroscience_tools.backends.llama import LlamaBackend

my_backend = LlamaBackend.from_model_path("/path/to/local/model")
```

---

### Vanilla Forward Pass

```text
forward(prompt_ids: list[int]) -> np.ndarray
```

Runs an ordinary next-token forward pass on a single tokenized prompt. The input should be a list of token IDs, usually produced by `RelationalPromptRuntime.render_prompt_as_tokens(...)`.

Returns the final-token logits as a NumPy array with shape `[vocab_size]`.

The following example renders a prompt, then runs a forward pass.

```python
context_entities = my_runtime.sample_entities(seed=0)

prompt_ids = my_runtime.render_prompt_as_tokens(
    context_entities=context_entities,
    context_relations={"R_AB": "FWD", "R_BC": "FWD"},
    query_entities={"QE": context_entities["B"]},
    query_relations={"QR": "REV"},
)

logits = my_backend.forward(prompt_ids)

print(my_formatter.get_prompt_text(prompt_ids))
print(my_formatter.get_pretty_top_k(logits, k=10))
```

---

### Activation Patching and Recording

```text
forward_with_recording(
    prompt_ids: list[int],
) -> tuple[torch.Tensor, np.ndarray]
```

Runs a forward pass while recording residual-stream activations at the inputs to
the transformer blocks. Returns `Z_tla` and `logits`, where `Z_tla` has shape
`[T_context, N_layers, D_model]`; here `T_context` is the full supplied prompt
length, including any query tokens.

```text
forward_with_patching(
    prompt_ids: list[int],
    Z_tla: torch.Tensor,
    mask_tla: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray]
```

Runs a forward pass while patching residual-stream activations. `Z_tla` provides the activation values to patch in, and `mask_tla` is a boolean tensor with the same shape. Positions where `mask_tla` is `True` are overwritten during the forward pass.

The following example records activations from a donor prompt, then patches the donor’s context-token activations into an acceptor prompt. Token landmarks from `my_runtime.landmarks` define the patch region.

Both prompts keep the donor query entity so that the question is held fixed while
the context names change. The KV example below uses the same convention.

```python
import torch

donor_entities, acceptor_entities = my_runtime.sample_nonoverlapping_entities(seed=0)

donor_prompt_ids = my_runtime.render_prompt_as_tokens(
    context_entities=donor_entities,
    context_relations={"R_AB": "FWD", "R_BC": "FWD"},
    query_entities={"QE": donor_entities["B"]},
    query_relations={"QR": "REV"},
)

acceptor_prompt_ids = my_runtime.render_prompt_as_tokens(
    context_entities=acceptor_entities,
    context_relations={"R_AB": "FWD", "R_BC": "FWD"},
    query_entities={"QE": donor_entities["B"]},
    query_relations={"QR": "REV"},
)

donor_Z_tla, donor_logits = my_backend.forward_with_recording(donor_prompt_ids)

mask_tla = torch.zeros_like(donor_Z_tla, dtype=torch.bool)

ctxt_start = my_runtime.landmarks["CTXT_START"]
ctxt_end = my_runtime.landmarks["CTXT_END"]

mask_tla[ctxt_start:ctxt_end + 1, :, :] = True

patched_Z_tla, patched_logits = my_backend.forward_with_patching(
    prompt_ids=acceptor_prompt_ids,
    Z_tla=donor_Z_tla,
    mask_tla=mask_tla,
)
```

---

### KV Patching and Recording

```text
forward_with_KV_recording(
    prompt_ids: list[int],
) -> tuple[CacheData, np.ndarray]
```

Runs a forward pass while recording the model’s key/value cache. Returns a `CacheData` object plus the final-token logits.

`CacheData` is available from `llm_neuroscience_tools.backends.KV_patching`.
Its `K_tlia` and `V_tlia` tensors use axes `[token, layer, KV head, head dimension]`.

```text
forward_with_KV_patching(
    prompt_ids: list[int],
    cache_data: CacheData,
    patch_end: int,
) -> tuple[CacheData, np.ndarray]
```

Runs a forward pass using cached keys and values from `cache_data`. The cache is patched through the inclusive token index `patch_end`; the backend then runs the remaining prompt tokens using that patched prefix cache.

The supplied cache must contain tokens through `patch_end`, and the prompt must
have at least one token after the patched prefix.

The following example records the donor prompt’s KV cache, then uses the donor cache through the context region while running the acceptor prompt. As with activation patching, landmarks such as `my_runtime.landmarks["CTXT_END"]` are often used to choose the patch boundary.

```python
donor_entities, acceptor_entities = my_runtime.sample_nonoverlapping_entities(seed=0)

donor_prompt_ids = my_runtime.render_prompt_as_tokens(
    context_entities=donor_entities,
    context_relations={"R_AB": "FWD", "R_BC": "FWD"},
    query_entities={"QE": donor_entities["B"]},
    query_relations={"QR": "REV"},
)

acceptor_prompt_ids = my_runtime.render_prompt_as_tokens(
    context_entities=acceptor_entities,
    context_relations={"R_AB": "FWD", "R_BC": "FWD"},
    query_entities={"QE": donor_entities["B"]},
    query_relations={"QR": "REV"},
)

donor_cache_data, donor_logits = my_backend.forward_with_KV_recording(
    donor_prompt_ids
)

patch_end = my_runtime.landmarks["CTXT_END"]

patched_cache_data, patched_logits = my_backend.forward_with_KV_patching(
    prompt_ids=acceptor_prompt_ids,
    cache_data=donor_cache_data,
    patch_end=patch_end,
)
```

---

## OutputFormatter

`OutputFormatter` provides utilities for interpreting model outputs and inspecting rendered prompts. It is usually accessed through a preset runtime bundle:

```python
from llm_neuroscience_tools.presets import build_preset_runtime

bundle = build_preset_runtime(
    model_name="llama-70b",
    prompt_name="ABC_family_tree_who_is",
)

my_formatter = bundle.formatter
```

However, it can also be instantiated manually by passing a tokenizer:

```python
from llm_neuroscience_tools.formatting import OutputFormatter

my_formatter = OutputFormatter(tokenizer=my_tokenizer)
```

A tokenizer is required. `OutputFormatter` does not load a default tokenizer;
use the tokenizer associated with the model whose outputs you are inspecting.

---

### Main Methods

```text
get_logit(token: str, logits: np.ndarray) -> float
```

Returns the logit assigned to a single token string.

`token` must be a token string from the tokenizer's vocabulary, including any
tokenizer-specific whitespace marker. This method uses `convert_tokens_to_ids`;
it does not tokenize arbitrary words or score multi-token strings. The same
restriction applies to `get_prob`.

```text
get_prob(token: str, logits: np.ndarray) -> float
```

Returns the softmax probability assigned to a single token string.

```text
get_pretty_top_k(logits: np.ndarray, k: int = 10) -> str
```

Formats the top `k` next-token predictions as a readable table, including token strings and logits.

```text
get_prompt_text(prompt_ids: list[int]) -> str
```

Decodes a tokenized prompt back into text.

```text
get_pretty_prompt(
    prompt_ids: list[int],
    landmarks: dict | None = None,
) -> str
```

Formats a tokenized prompt as a readable table with token indices, token IDs, decoded tokens, and optional landmark tags. If `landmarks` is provided, token positions with matching landmark names are labeled in the output.
