from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from tqdm import tqdm

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime


ACTIVATION_FILENAME = "Z_gstla.npy"
KEY_CACHE_FILENAME = "K_gstlia.npy"
VALUE_CACHE_FILENAME = "V_gstlia.npy"
METADATA_FILENAME = "recordings_metadata.json"

RelationPattern: TypeAlias = tuple[str, dict[str, str]]


@dataclass(frozen=True)
class _ContextRecordings:
    """Context-cropped arrays recorded during one model forward pass."""

    activations: np.ndarray
    keys: np.ndarray
    values: np.ndarray


def _get_relation_patterns(
    prompt_runtime: RelationalPromptRuntime,
) -> tuple[RelationPattern, ...]:
    """Return the four graph structures supported by a prompt runtime.

    Args:
        prompt_runtime: Runtime whose context schema determines whether the
            context is a three-entity chain or two disconnected pairs.

    Returns:
        Ordered graph labels and their context-relation substitutions.

    Raises:
        ValueError: If the prompt does not use a supported context schema.
    """
    entity_symbols = prompt_runtime.schema.context_entity_symbols
    relation_symbols = prompt_runtime.schema.context_relation_symbols
    if relation_symbols != ("R_AB", "R_BC"):
        raise ValueError(
            "Recording requires context relation symbols ('R_AB', 'R_BC'); "
            f"got {relation_symbols!r}."
        )

    relation_maps = (
        {"R_AB": "FWD", "R_BC": "FWD"},
        {"R_AB": "REV", "R_BC": "FWD"},
        {"R_AB": "FWD", "R_BC": "REV"},
        {"R_AB": "REV", "R_BC": "REV"},
    )
    if entity_symbols == ("A", "B", "C"):
        labels = (
            "A --fwd--> B --fwd--> C",
            "A --rev--> B --fwd--> C",
            "A --fwd--> B --rev--> C",
            "A --rev--> B --rev--> C",
        )
    elif entity_symbols == ("A", "B", "C", "D"):
        labels = (
            "A --fwd--> B; C --fwd--> D",
            "A --rev--> B; C --fwd--> D",
            "A --fwd--> B; C --rev--> D",
            "A --rev--> B; C --rev--> D",
        )
    else:
        raise ValueError(
            "Recording supports context entities ('A', 'B', 'C') or "
            f"('A', 'B', 'C', 'D'); got {entity_symbols!r}."
        )

    return tuple(zip(labels, relation_maps))


def _get_context_bounds(
    prompt_runtime: RelationalPromptRuntime,
) -> tuple[int, int]:
    """Return the inclusive context-token bounds.

    Args:
        prompt_runtime: Runtime containing compiled token landmarks.

    Returns:
        The inclusive ``(start, end)`` token positions of the context.

    Raises:
        ValueError: If the context landmarks are reversed.
    """
    context_start = prompt_runtime.landmarks["CTXT_START"]
    context_end = prompt_runtime.landmarks["CTXT_END"]
    if context_end < context_start:
        raise ValueError(
            f"Invalid context bounds: start={context_start}, end={context_end}."
        )
    return context_start, context_end


def _record_one_context(
    backend: LMBackend,
    prompt_runtime: RelationalPromptRuntime,
    context_entities: dict[str, str],
    context_relations: dict[str, str],
    context_start: int,
    context_end: int,
) -> _ContextRecordings:
    """Record and crop activations, keys, and values for one context.

    Args:
        backend: Model backend used for the forward pass.
        prompt_runtime: Runtime used to render the context-only prompt.
        context_entities: Concrete entity-name substitutions.
        context_relations: Forward/reverse relation substitutions.
        context_start: Inclusive first context-token position.
        context_end: Inclusive final context-token position.

    Returns:
        Float32 NumPy arrays cropped to the context token span.
    """
    prompt_ids = prompt_runtime.render_prompt_as_tokens(
        context_entities=context_entities,
        context_relations=context_relations,
    )
    activations, cache, _ = backend.forward_with_activation_and_KV_recording(
        prompt_ids
    )

    # Only context positions are reused by downstream mosaic and erasure tasks.
    context_slice = slice(context_start, context_end + 1)
    return _ContextRecordings(
        activations=(
            activations[context_slice, ...].to(torch.float32).cpu().numpy()
        ),
        keys=cache.K_tlia[context_slice, ...].to(torch.float32).cpu().numpy(),
        values=cache.V_tlia[context_slice, ...].to(torch.float32).cpu().numpy(),
    )


def _validate_shapes(
    recordings: _ContextRecordings,
    activation_shape: tuple[int, ...],
    kv_shape: tuple[int, ...],
) -> None:
    """Check that one recording matches the allocated artifact shapes.

    Args:
        recordings: Arrays produced for one prompt.
        activation_shape: Expected context-cropped activation shape.
        kv_shape: Expected context-cropped key and value shape.

    Raises:
        ValueError: If any recorded array has an unexpected shape.
    """
    if recordings.activations.shape != activation_shape:
        raise ValueError(
            "Activation shape mismatch: expected "
            f"{activation_shape}, got {recordings.activations.shape}."
        )
    if recordings.keys.shape != kv_shape or recordings.values.shape != kv_shape:
        raise ValueError(
            f"KV shape mismatch: expected {kv_shape}, got "
            f"K={recordings.keys.shape}, V={recordings.values.shape}."
        )


def _context_landmarks(
    prompt_runtime: RelationalPromptRuntime,
    context_start: int,
    context_end: int,
) -> dict[str, int]:
    """Express context landmarks relative to the cropped arrays.

    Args:
        prompt_runtime: Runtime containing compiled token landmarks.
        context_start: Inclusive first context-token position.
        context_end: Inclusive final context-token position.

    Returns:
        Context landmark names mapped to indices in the stored arrays.
    """
    return {
        name: position - context_start
        for name, position in prompt_runtime.landmarks.items()
        if context_start <= position <= context_end
    }


def _build_metadata(
    prompt_runtime: RelationalPromptRuntime,
    relation_patterns: tuple[RelationPattern, ...],
    start_seed: int,
    n_trials: int,
    context_start: int,
    context_end: int,
) -> dict[str, object]:
    """Build the minimal metadata needed to interpret recording arrays.

    Args:
        prompt_runtime: Runtime used to produce the recordings.
        relation_patterns: Ordered graph labels and relation substitutions.
        start_seed: First entity-sampling seed.
        n_trials: Number of consecutive seeds recorded.
        context_start: Inclusive original context start position.
        context_end: Inclusive original context end position.

    Returns:
        JSON-serializable recording metadata.
    """
    return {
        "start_seed": start_seed,
        "n_trials": n_trials,
        "relation_patterns": [
            {"name": name, "context_relations": relations}
            for name, relations in relation_patterns
        ],
        "activation_axes": [
            "graph_structure",
            "seed",
            "token",
            "layer",
            "activation",
        ],
        "kv_axes": [
            "graph_structure",
            "seed",
            "token",
            "layer",
            "kv_head",
            "head_dim",
        ],
        "context_landmarks": _context_landmarks(
            prompt_runtime,
            context_start,
            context_end,
        ),
    }


def record_activations_and_kv(
    backend: LMBackend,
    prompt_runtime: RelationalPromptRuntime,
    output_dir: Path,
    *,
    n_trials: int,
    start_seed: int,
) -> None:
    """Record context activations and KV caches across four graph structures.

    The same sampled entity names are reused for all four structures at each
    seed. Arrays are cropped to the context span and written incrementally as
    float32 ``.npy`` files to avoid holding the full artifacts in memory.

    Args:
        backend: Initialized model backend supporting simultaneous activation
            and KV-cache recording.
        prompt_runtime: Initialized relational prompt runtime.
        output_dir: Directory in which recording artifacts will be written.
        n_trials: Number of consecutive entity-sampling seeds to record.
        start_seed: First entity-sampling seed.

    Raises:
        ValueError: If ``n_trials`` is not positive, the prompt schema is not
            supported, or recorded tensor shapes change between prompts.
        FileExistsError: If any output artifact already exists.
        NotImplementedError: If the backend does not support combined
            activation and KV-cache recording.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be positive, got {n_trials}.")

    relation_patterns = _get_relation_patterns(prompt_runtime)
    context_start, context_end = _get_context_bounds(prompt_runtime)

    # Prepare output files
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_path = output_dir / ACTIVATION_FILENAME
    key_cache_path = output_dir / KEY_CACHE_FILENAME
    value_cache_path = output_dir / VALUE_CACHE_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    artifact_paths = (
        activation_path,
        key_cache_path,
        value_cache_path,
        metadata_path,
    )
    for path in artifact_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    # Write the index-to-structure mapping before recording so that partial
    # arrays remain interpretable if a later model pass fails. Successful task
    # completion is tracked separately by the orchestration layer.
    metadata = _build_metadata(
        prompt_runtime=prompt_runtime,
        relation_patterns=relation_patterns,
        start_seed=start_seed,
        n_trials=n_trials,
        context_start=context_start,
        context_end=context_end,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Use the first real trial to determine array shapes, avoiding a redundant
    # shape-inference forward pass.
    first_entities = prompt_runtime.sample_entities(seed=start_seed)
    first_recording = _record_one_context(
        backend=backend,
        prompt_runtime=prompt_runtime,
        context_entities=first_entities,
        context_relations=relation_patterns[0][1],
        context_start=context_start,
        context_end=context_end,
    )
    activation_shape = first_recording.activations.shape
    kv_shape = first_recording.keys.shape
    _validate_shapes(first_recording, activation_shape, kv_shape)

    full_activation_shape = (
        len(relation_patterns),
        n_trials,
        *activation_shape,
    )
    full_kv_shape = (len(relation_patterns), n_trials, *kv_shape)
    activation_array = np.lib.format.open_memmap(
        activation_path,
        mode="w+",
        dtype=np.float32,
        shape=full_activation_shape,
    )
    key_array = np.lib.format.open_memmap(
        key_cache_path,
        mode="w+",
        dtype=np.float32,
        shape=full_kv_shape,
    )
    value_array = np.lib.format.open_memmap(
        value_cache_path,
        mode="w+",
        dtype=np.float32,
        shape=full_kv_shape,
    )

    completed_passes = 0
    total_passes = len(relation_patterns) * n_trials
    try:
        with tqdm(total=total_passes, desc="recordings", disable=None) as progress:
            for seed_index in range(n_trials):
                entities = (
                    first_entities
                    if seed_index == 0
                    else prompt_runtime.sample_entities(seed=start_seed + seed_index)
                )
                for graph_index, (_, relations) in enumerate(relation_patterns):
                    if seed_index == 0 and graph_index == 0:
                        recordings = first_recording
                    else:
                        recordings = _record_one_context(
                            backend=backend,
                            prompt_runtime=prompt_runtime,
                            context_entities=entities,
                            context_relations=relations,
                            context_start=context_start,
                            context_end=context_end,
                        )

                    _validate_shapes(recordings, activation_shape, kv_shape)
                    activation_array[graph_index, seed_index, ...] = (
                        recordings.activations
                    )
                    key_array[graph_index, seed_index, ...] = recordings.keys
                    value_array[graph_index, seed_index, ...] = recordings.values

                    completed_passes += 1
                    progress.update(1)
                    if completed_passes % 10 == 0:
                        activation_array.flush()
                        key_array.flush()
                        value_array.flush()
    finally:
        # Preserve completed slices if a later model pass fails. The tools layer
        # will omit its completion marker for an incomplete task.
        activation_array.flush()
        key_array.flush()
        value_array.flush()
