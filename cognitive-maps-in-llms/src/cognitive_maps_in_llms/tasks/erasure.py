from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime

from cognitive_maps_in_llms.benchmarks import load_benchmark
from cognitive_maps_in_llms.tasks.key_value_erasure_tools import (
    STRUCTURE,
    STRUCTURE_NAME,
    ErasureState,
    build_all_objectives_for_one_context,
    evaluate_mean_objective_score,
    load_average_cache,
    take_gradient_erasure_step,
)
from cognitive_maps_in_llms.tasks.layer_split import resolve_dividing_layer


ERASURE_STATE_FILENAME = "erasure_state.json"
METADATA_FILENAME = "erasure_metadata.json"
MASK_AXES = ["context_token", "layer", "kv_head"]


def _write_json(path: Path, data: object) -> None:
    """Atomically write one JSON artifact."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _remaining_sites(erasure_state: ErasureState) -> int:
    """Return the total number of retained key and value sites."""
    key_mask, value_mask = erasure_state.masks_at_iteration()
    return int(key_mask.sum() + value_mask.sum())


def _normalized_fidelity(
    score: float,
    initial_score: float,
    fully_erased_score: float,
) -> float:
    """Place a score on the initial-to-fully-erased scale."""
    return (score - fully_erased_score) / (initial_score - fully_erased_score)


def run_erasure(
    backend: LMBackend,
    prompt_runtime: RelationalPromptRuntime,
    output_dir: Path,
    *,
    benchmark_name: str,
    recordings: Path,
    start_seed: int,
    n_trials: int,
    coarse_fraction: float = 0.10,
    fidelity_threshold: float = 0.90,
    rewind_multiplier: float = 2.0,
    dividing_layer_fraction: float = 0.5,
) -> None:
    """Compute an adaptive gradient-guided KV-erasure trajectory.

    The trajectory begins from the "keys-first" mosaic: keys are retained
    before the configured layer division and values from that division onward.
    Erasure is coarse until performance falls below the requested fidelity,
    then rewinds and continues one site at a time.

    Args:
        backend: Initialized model backend used for recording and patching.
        prompt_runtime: Runtime used to sample contexts and render queries.
        output_dir: Directory in which task artifacts will be written.
        benchmark_name: Name of a packaged TOML benchmark.
        recordings: Output directory from the recording task.
        start_seed: First entity-sampling seed used for the objectives.
        n_trials: Number of consecutive sampled contexts.
        coarse_fraction: Fraction of currently retained sites erased during
            each coarse update.
        fidelity_threshold: Normalized score at which coarse erasure stops.
        rewind_multiplier: Factor applied to the remaining-site count when
            choosing the state from which fine erasure begins.
        dividing_layer_fraction: Fractional model depth separating retained
            keys from retained values. The resolved layer uses floor rounding.

    Raises:
        ValueError: If a setting is invalid or the initial and fully erased
            objective scores are equal.
        FileExistsError: If either output artifact already exists.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be positive, got {n_trials}.")
    if not 0 < coarse_fraction <= 1:
        raise ValueError("coarse_fraction must fall within (0, 1].")
    if not 0 <= fidelity_threshold <= 1:
        raise ValueError("fidelity_threshold must fall within [0, 1].")
    if rewind_multiplier < 1:
        raise ValueError("rewind_multiplier must be at least 1.")

    # Set up output folder
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / ERASURE_STATE_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    for path in (state_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    # Construct the list of objectives for each seed x query in the benchmark
    cases = load_benchmark(benchmark_name)
    average_cache = load_average_cache(
        recordings,
        backend,
        prompt_runtime,
    )
    objectives = []
    for seed in range(start_seed, start_seed + n_trials):
        objectives.extend(
            build_all_objectives_for_one_context(
                backend,
                prompt_runtime,
                average_cache,
                cases,
                seed,
            )
        )

    context_start = prompt_runtime.landmarks["CTXT_START"]
    context_end = prompt_runtime.landmarks["CTXT_END"]
    mask_shape = tuple(
        average_cache.K_tlia[context_start : context_end + 1].shape[:-1]
    )
    if len(mask_shape) != 3:
        raise ValueError(f"Expected a three-dimensional mask, got {mask_shape}.")

    # Retain K before the layer division and V from the division onward.
    dividing_layer = resolve_dividing_layer(
        mask_shape[1],
        dividing_layer_fraction,
    )
    key_mask = np.zeros(mask_shape, dtype=bool)
    value_mask = np.zeros(mask_shape, dtype=bool)
    key_mask[:, :dividing_layer, :] = True
    value_mask[:, dividing_layer:, :] = True
    erasure_state = ErasureState.from_masks(key_mask, value_mask)
    initial_remaining = _remaining_sites(erasure_state)

    # Compute initial and final scores so that we can go from score -> fidelity
    initial_score = evaluate_mean_objective_score(
        objectives,
        key_mask,
        value_mask,
    )
    fully_erased_score = evaluate_mean_objective_score(
        objectives,
        np.zeros(mask_shape, dtype=bool),
        np.zeros(mask_shape, dtype=bool),
    )
    if initial_score == fully_erased_score:
        raise ValueError(
            "Initial and fully erased scores are equal; normalized fidelity "
            "is undefined."
        )

    # Construct metadata file
    metadata = {
        "start_seed": start_seed,
        "n_trials": n_trials,
        "benchmark_name": benchmark_name,
        "structure": {
            "name": STRUCTURE_NAME,
            "context_relations": STRUCTURE,
        },
        "mask_axes": MASK_AXES,
        "initial_mosaic": {
            "name": "key_then_value",
            "dividing_layer_fraction": float(dividing_layer_fraction),
            "dividing_layer": dividing_layer,
        },
        "schedule": {
            "coarse_fraction": coarse_fraction,
            "fidelity_threshold": fidelity_threshold,
            "rewind_multiplier": rewind_multiplier,
        },
        "initial_score": initial_score,
        "fully_erased_score": fully_erased_score,
    }
    _write_json(metadata_path, metadata)
    _write_json(state_path, erasure_state.to_serializable())

    # Coarse updates become naturally smaller as the trajectory approaches
    # full erasure. Checkpoint after every completed gradient evaluation.
    crossed_threshold = False
    with tqdm(
        total=initial_remaining,
        desc="coarse erasure",
        unit="site",
        disable=None,
    ) as progress:
        while (n_remaining := _remaining_sites(erasure_state)) > 0:
            n_erasures = max(1, math.ceil(coarse_fraction * n_remaining))
            score = take_gradient_erasure_step(
                objectives,
                erasure_state,
                n_erasures=n_erasures,
            )
            _write_json(state_path, erasure_state.to_serializable())

            fidelity = _normalized_fidelity(
                score,
                initial_score,
                fully_erased_score,
            )
            progress.update(n_erasures)
            progress.set_postfix(fidelity=f"{fidelity:.3f}")
            if fidelity < fidelity_threshold:
                crossed_threshold = True
                remaining_after_step = _remaining_sites(erasure_state)
                rewind_target = min(
                    initial_remaining,
                    math.ceil(rewind_multiplier * remaining_after_step),
                )
                erasure_state = erasure_state.rewound_to_at_least(rewind_target)
                _write_json(state_path, erasure_state.to_serializable())
                break

    if crossed_threshold:
        fine_remaining = _remaining_sites(erasure_state)
        with tqdm(
            total=fine_remaining,
            desc="fine erasure",
            unit="site",
            disable=None,
        ) as progress:
            while _remaining_sites(erasure_state) > 0:
                take_gradient_erasure_step(objectives, erasure_state)
                _write_json(state_path, erasure_state.to_serializable())
                progress.update(1)
