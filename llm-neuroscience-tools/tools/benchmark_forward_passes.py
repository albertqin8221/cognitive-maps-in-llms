"""Benchmark preset backend forward-pass variants."""

from __future__ import annotations

import argparse
import time
from typing import Sequence

import torch
from tqdm import tqdm

from llm_neuroscience_tools.presets import (
    build_preset_runtime,
    get_preset_model_names,
    get_preset_prompt_names,
)
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime


DEFAULT_PROMPT_PRESET = "ABC_family_tree_who_is"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark vanilla, recording, and patching forward passes.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model-preset",
        "--model_preset",
        "-m",
        required=True,
        help=(
            "Preset model name to benchmark. "
            "Run `python tools/list_presets.py` "
            "to list valid presets."
        ),
    )
    parser.add_argument(
        "--prompt-preset",
        "--prompt_preset",
        "-p",
        default=DEFAULT_PROMPT_PRESET,
        help=(
            "Preset prompt family used to generate trial prompts. "
            f"Defaults to {DEFAULT_PROMPT_PRESET}. "
            "Run `python tools/list_presets.py` "
            "to list valid presets."
        ),
    )
    parser.add_argument(
        "--num-trials",
        "--n_trials",
        "-n",
        dest="n_trials",
        type=int,
        default=10,
        help="Number of timed trials per forward-pass variant. Defaults to 10.",
    )
    args = parser.parse_args(argv)
    if args.n_trials <= 0:
        parser.error("--num-trials must be positive.")
    try:
        validate_presets(args.model_preset, args.prompt_preset)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_presets(model_preset: str, prompt_preset: str) -> None:
    """Validate preset names and raise readable CLI errors if needed."""
    valid_model_names = set(get_preset_model_names())
    valid_prompt_names = set(get_preset_prompt_names())

    if model_preset not in valid_model_names:
        raise ValueError(
            f"Unrecognized model preset {model_preset!r}. "
            "Run `python tools/list_presets.py` "
            "to list valid presets."
        )
    if prompt_preset not in valid_prompt_names:
        raise ValueError(
            f"Unrecognized prompt preset {prompt_preset!r}. "
            "Run `python tools/list_presets.py` "
            "to list valid presets."
        )


def build_test_input_from_seed(
    prompt_runtime: RelationalPromptRuntime,
    seed: int,
) -> dict[str, dict[str, str]]:
    """Build a deterministic preset-compatible test input from a seed."""
    entities = prompt_runtime.sample_entities(seed)
    entities_list = list(entities.values())

    context_relations = {
        symbol: "FWD"
        for symbol in prompt_runtime.schema.context_relation_symbols
    }
    query_entities = {
        symbol: entities_list[(1 - idx) % len(entities_list)]
        for idx, symbol in enumerate(prompt_runtime.schema.query_entity_symbols)
    }
    query_relation_options = prompt_runtime.relation_pool.query_relations
    if not query_relation_options:
        raise ValueError("Prompt preset has no query relations.")
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
        "context_entities": entities,
        "context_relations": context_relations,
        "query_entities": query_entities,
        "query_relations": query_relations,
    }


def render_prompt_ids(
    prompt_runtime: RelationalPromptRuntime,
    seed: int,
) -> list[int]:
    """Render one deterministic prompt to token ids."""
    return prompt_runtime.render_prompt_as_tokens(
        **build_test_input_from_seed(prompt_runtime, seed)
    )


def maybe_synchronize() -> None:
    """Wait for queued CUDA work so timings reflect completed execution."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(fn) -> float:
    """Measure wall-clock seconds for one callable invocation."""
    maybe_synchronize()
    start = time.perf_counter()
    fn()
    maybe_synchronize()
    return time.perf_counter() - start


def format_summary(
    runtime_load_seconds: float,
    vanilla_seconds: list[float],
    recording_seconds: list[float],
    patching_seconds: list[float],
) -> str:
    """Render the benchmark result as plain text."""
    vanilla_mean = sum(vanilla_seconds) / len(vanilla_seconds)
    recording_mean = sum(recording_seconds) / len(recording_seconds)
    patching_mean = sum(patching_seconds) / len(patching_seconds)
    labels_and_values = [
        ("Runtime initialization time:", runtime_load_seconds),
        ("Mean time for backend.forward(...):", vanilla_mean),
        ("Mean time for backend.forward_with_recording(...):", recording_mean),
        ("Mean time for backend.forward_with_patching(...):", patching_mean),
    ]
    label_width = max(len(label) for label, _ in labels_and_values)
    value_width = 12

    result_lines = [
        f"{label.ljust(label_width)} {value:>{value_width}.6f} s"
        for label, value in labels_and_values
    ]

    return "\n".join(
        [
            "",
            "Results:",
            *result_lines,
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the forward-pass benchmark CLI."""
    args = parse_args(argv)

    load_start = time.perf_counter()
    runtime_bundle = build_preset_runtime(
        model_name=args.model_preset,
        prompt_name=args.prompt_preset,
        use_dummy=False,
    )
    maybe_synchronize()
    runtime_load_seconds = time.perf_counter() - load_start
    backend = runtime_bundle.backend
    prompt_runtime = runtime_bundle.prompt_runtime

    warmup_seed = 10_000
    warmup_prompt_ids = render_prompt_ids(prompt_runtime, warmup_seed)
    prompt_ids_by_trial = [
        render_prompt_ids(prompt_runtime, seed)
        for seed in range(args.n_trials)
    ]

    donor_data_by_trial: list[tuple[torch.Tensor, torch.Tensor]] = []
    for prompt_ids in prompt_ids_by_trial:
        recorded_activations, _ = backend.forward_with_recording(prompt_ids)
        donor_data_by_trial.append(
            (recorded_activations, torch.ones_like(recorded_activations, dtype=torch.bool))
        )

    backend.forward(warmup_prompt_ids)

    vanilla_seconds: list[float] = []
    recording_seconds: list[float] = []
    patching_seconds: list[float] = []

    with tqdm(total=3 * args.n_trials, desc="Benchmarking", unit="pass") as progress:
        for prompt_ids in prompt_ids_by_trial:
            vanilla_seconds.append(time_call(lambda prompt_ids=prompt_ids: backend.forward(prompt_ids)))
            progress.update(1)

        for prompt_ids in prompt_ids_by_trial:
            recording_seconds.append(
                time_call(
                    lambda prompt_ids=prompt_ids: backend.forward_with_recording(prompt_ids)
                )
            )
            progress.update(1)

        for prompt_ids, (donor_activations, patch_mask) in zip(
            prompt_ids_by_trial,
            donor_data_by_trial,
            strict=True,
        ):
            patching_seconds.append(
                time_call(
                    lambda prompt_ids=prompt_ids,
                    donor_activations=donor_activations,
                    patch_mask=patch_mask: backend.forward_with_patching(
                        prompt_ids=prompt_ids,
                        Z_tla=donor_activations,
                        mask_tla=patch_mask,
                    )
                )
            )
            progress.update(1)

    summary = format_summary(
        runtime_load_seconds=runtime_load_seconds,
        vanilla_seconds=vanilla_seconds,
        recording_seconds=recording_seconds,
        patching_seconds=patching_seconds,
    )

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
