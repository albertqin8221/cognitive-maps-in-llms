"""Tools for constructing differentiable key-value erasure objectives."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from operator import index as integer_index
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch

from llm_neuroscience_tools.backends import LMBackend
from llm_neuroscience_tools.backends.KV_patching import CacheData
from llm_neuroscience_tools.prompts import RelationalPromptRuntime

from cognitive_maps_in_llms.benchmarks import BenchmarkCase
from cognitive_maps_in_llms.tasks.recording import (
    KEY_CACHE_FILENAME,
    METADATA_FILENAME,
    VALUE_CACHE_FILENAME,
)


STRUCTURE = {"R_AB": "FWD", "R_BC": "FWD"}
STRUCTURE_NAME = "A --fwd--> B --fwd--> C"
KV_AXES = [
    "graph_structure",
    "seed",
    "token",
    "layer",
    "kv_head",
    "head_dim",
]


# -----------------------------------------------------------------------------
# Data loading


def _load_average_context_kv(
    recording_directory: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and average the canonical structure's context KV recordings.

    Args:
        recording_directory: Directory produced by the recording task.

    Returns:
        Averaged key and value arrays with shape
        ``[context_token, layer, kv_head, head_dim]``.

    Raises:
        ValueError: If the arrays or recording metadata are inconsistent.
        FileNotFoundError: If a required recording artifact is absent.
    """
    metadata_path = recording_directory / METADATA_FILENAME
    key_path = recording_directory / KEY_CACHE_FILENAME
    value_path = recording_directory / VALUE_CACHE_FILENAME

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata.get("kv_axes") != KV_AXES:
        raise ValueError(
            f"Expected KV axes {KV_AXES!r}, got {metadata.get('kv_axes')!r}."
        )

    keys = np.load(key_path, mmap_mode="r", allow_pickle=False)
    values = np.load(value_path, mmap_mode="r", allow_pickle=False)
    if keys.shape != values.shape:
        raise ValueError(
            f"Recorded key/value shapes differ: K={keys.shape}, V={values.shape}."
        )
    if keys.ndim != len(KV_AXES):
        raise ValueError(
            f"Expected {len(KV_AXES)}-dimensional KV arrays, got shape {keys.shape}."
        )

    relation_patterns = metadata.get("relation_patterns")
    if not isinstance(relation_patterns, list):
        raise ValueError("Recording metadata must contain relation_patterns.")
    if len(relation_patterns) != keys.shape[0]:
        raise ValueError(
            "The number of relation patterns does not match the graph axis: "
            f"metadata={len(relation_patterns)}, array={keys.shape[0]}."
        )

    graph_index = next(
        (
            index
            for index, pattern in enumerate(relation_patterns)
            if isinstance(pattern, dict)
            and pattern.get("context_relations") == STRUCTURE
        ),
        None,
    )
    if graph_index is None:
        raise ValueError(
            f"No recorded graph structure matches {STRUCTURE_NAME!r}."
        )

    n_trials = metadata.get("n_trials")
    if n_trials != keys.shape[1]:
        raise ValueError(
            "The recorded seed axis does not match n_trials: "
            f"metadata={n_trials!r}, array={keys.shape[1]}."
        )

    # The graph structure is fixed; only different sampled contexts are averaged.
    average_keys = np.mean(keys[graph_index], axis=0)
    average_values = np.mean(values[graph_index], axis=0)
    return average_keys, average_values


def load_average_cache(
    recording_directory: Path | str,
    backend: LMBackend,
    prompt_runtime: RelationalPromptRuntime,
) -> CacheData:
    """Construct a full cache containing the recorded average context KV.

    One context-only model pass supplies the cache shape, dtype, devices, and
    per-layer device map. Its activations are discarded; the returned cache is
    zero outside the context and contains the recording average inside it.

    Args:
        recording_directory: Directory produced by the recording task.
        backend: Model backend used to create a correctly configured cache.
        prompt_runtime: Runtime used to render the shape-inference prompt and
            locate the context token span.

    Returns:
        A full ``CacheData`` whose context positions contain averaged keys and
        values.

    Raises:
        ValueError: If the recorded context shape does not match the live model
            and prompt runtime.
        FileNotFoundError: If a required recording artifact is absent.
    """
    average_keys, average_values = _load_average_context_kv(
        Path(recording_directory)
    )
    context_start = prompt_runtime.landmarks["CTXT_START"]
    context_end = prompt_runtime.landmarks["CTXT_END"]

    # Any valid context prompt can provide the cache's runtime-specific shell.
    entities = prompt_runtime.sample_entities(seed=0)
    prompt_ids = prompt_runtime.render_prompt_as_tokens(
        context_entities=entities,
        context_relations=STRUCTURE,
        query_entities=None,
        query_relations=None,
    )
    with torch.no_grad():
        template_cache, _ = backend.forward_with_KV_recording(prompt_ids)

    context_slice = slice(context_start, context_end + 1)
    live_key_shape = tuple(template_cache.K_tlia[context_slice].shape)
    live_value_shape = tuple(template_cache.V_tlia[context_slice].shape)
    if average_keys.shape != live_key_shape:
        raise ValueError(
            "Averaged keys do not match the live context cache: "
            f"recording={average_keys.shape}, live={live_key_shape}."
        )
    if average_values.shape != live_value_shape:
        raise ValueError(
            "Averaged values do not match the live context cache: "
            f"recording={average_values.shape}, live={live_value_shape}."
        )

    average_cache = template_cache.empty_like()
    average_cache.K_tlia[context_slice] = torch.as_tensor(
        average_keys,
        device=average_cache.K_tlia.device,
        dtype=average_cache.K_tlia.dtype,
    )
    average_cache.V_tlia[context_slice] = torch.as_tensor(
        average_values,
        device=average_cache.V_tlia.device,
        dtype=average_cache.V_tlia.dtype,
    )
    return average_cache


# -----------------------------------------------------------------------------
# Objective generation


LogitScore: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
KVObjective: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MaskIndex: TypeAlias = tuple[int, int, int]


def build_correct_answer_margin_score(
    tokenizer: Any,
    *,
    answer_tokens: Mapping[str, str],
    correct_answer: str,
    candidate_answers: Sequence[str],
) -> LogitScore:
    """Build a differentiable correct-answer logit-margin scorer.

    The score is the correct-answer logit minus the mean logit of the other
    candidates. Tokens not named in ``candidate_answers`` are ignored.

    Args:
        tokenizer: Tokenizer used by the model and prompt runtime.
        answer_tokens: Symbolic answers mapped to their concrete token strings.
        correct_answer: Symbolic answer that should receive the highest logit.
        candidate_answers: Symbolic answers included in the score.

    Returns:
        A function mapping batched sequence logits to a scalar tensor.

    Raises:
        ValueError: If the candidates are invalid, missing, or do not each
            tokenize to one distinct token.
    """
    candidates = tuple(candidate_answers)
    if len(candidates) < 2:
        raise ValueError("At least two candidate answers are required.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidate answers must be unique.")
    if correct_answer not in candidates:
        raise ValueError(
            f"Correct answer {correct_answer!r} is not among the candidates."
        )

    missing_answers = [
        answer for answer in candidates if answer not in answer_tokens
    ]
    if missing_answers:
        raise ValueError(
            f"Answer tokens are missing candidates: {missing_answers}."
        )

    answer_ids: dict[str, int] = {}
    for answer in candidates:
        token = answer_tokens[answer]
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Expected answer token {token!r} to tokenize once, got {token_ids}."
            )
        answer_ids[answer] = token_ids[0]

    if len(set(answer_ids.values())) != len(answer_ids):
        raise ValueError("Candidate answers must have distinct token IDs.")

    correct_id = answer_ids[correct_answer]
    incorrect_ids = [
        token_id
        for answer, token_id in answer_ids.items()
        if answer != correct_answer
    ]

    def score(logits_btv: torch.Tensor) -> torch.Tensor:
        """Return the final-token correct-answer margin."""
        return (
            logits_btv[0, -1, correct_id]
            - logits_btv[0, -1, incorrect_ids].mean()
        )

    return score


def build_interpolated_cache(
    control_cache: CacheData,
    average_cache: CacheData,
    context_start: int,
    context_end: int,
    key_mask: torch.Tensor,
    value_mask: torch.Tensor,
) -> CacheData:
    """Interpolate the context portion of two key-value caches.

    A mask value of one retains the control activation; a value of zero uses
    the averaged activation. Positions outside the context are copied from the
    control cache without modification.

    Args:
        control_cache: Cache recorded from the sampled control context.
        average_cache: Shape-compatible cache containing averaged context KV.
        context_start: Inclusive first context-token position.
        context_end: Inclusive final context-token position.
        key_mask: Key weights with shape ``[context_token, layer, kv_head]``.
        value_mask: Value weights with the same shape as ``key_mask``.

    Returns:
        A new cache containing the interpolated context activations.

    Raises:
        ValueError: If the caches, context bounds, or masks have incompatible
            shapes.
    """
    if control_cache.K_tlia.shape != average_cache.K_tlia.shape:
        raise ValueError("Control and average key caches must have the same shape.")
    if control_cache.V_tlia.shape != average_cache.V_tlia.shape:
        raise ValueError("Control and average value caches must have the same shape.")
    if control_cache.K_tlia.shape != control_cache.V_tlia.shape:
        raise ValueError("Key and value cache tensors must have the same shape.")

    n_tokens, n_layers, n_heads, _ = control_cache.K_tlia.shape
    if not 0 <= context_start <= context_end < n_tokens:
        raise ValueError(
            "Context bounds must fall within the cache token axis; got "
            f"start={context_start}, end={context_end}, n_tokens={n_tokens}."
        )

    expected_mask_shape = (
        context_end - context_start + 1,
        n_layers,
        n_heads,
    )
    if tuple(key_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"Expected key_mask shape {expected_mask_shape}, "
            f"got {tuple(key_mask.shape)}."
        )
    if tuple(value_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"Expected value_mask shape {expected_mask_shape}, "
            f"got {tuple(value_mask.shape)}."
        )

    context_slice = slice(context_start, context_end + 1)
    key_weights = key_mask.to(control_cache.K_tlia.device).unsqueeze(-1)
    value_weights = value_mask.to(control_cache.V_tlia.device).unsqueeze(-1)

    interpolated_cache = control_cache.clone()
    interpolated_cache.K_tlia[context_slice] = (
        (1 - key_weights) * average_cache.K_tlia[context_slice]
        + key_weights * control_cache.K_tlia[context_slice]
    )
    interpolated_cache.V_tlia[context_slice] = (
        (1 - value_weights) * average_cache.V_tlia[context_slice]
        + value_weights * control_cache.V_tlia[context_slice]
    )
    return interpolated_cache


def _build_objective(
    *,
    backend: LMBackend,
    prompt_ids: list[int],
    control_cache: CacheData,
    average_cache: CacheData,
    context_start: int,
    context_end: int,
    scoring_function: LogitScore,
) -> KVObjective:
    """Build one query-specific differentiable KV objective."""

    def objective(
        key_mask: torch.Tensor,
        value_mask: torch.Tensor,
    ) -> torch.Tensor:
        interpolated_cache = build_interpolated_cache(
            control_cache=control_cache,
            average_cache=average_cache,
            context_start=context_start,
            context_end=context_end,
            key_mask=key_mask,
            value_mask=value_mask,
        )
        _, logits = backend.forward_with_differentiable_KV_patching(
            prompt_ids=prompt_ids,
            cache_data=interpolated_cache,
            patch_end=context_end,
        )
        return scoring_function(logits)

    return objective


def _candidate_answers(query_mode: str) -> tuple[str, ...]:
    """Return the answer symbols included in an erasure score."""
    if query_mode == "yes_no":
        return ("Y", "N")
    if query_mode == "who_is":
        return ("A", "B", "C")
    raise ValueError(f"Unsupported query mode: {query_mode!r}.")


def build_all_objectives_for_one_context(
    backend: LMBackend,
    prompt_runtime: RelationalPromptRuntime,
    average_cache: CacheData,
    cases: Sequence[BenchmarkCase],
    seed: int,
) -> list[KVObjective]:
    """Build all benchmark objectives for one sampled control context.

    The function records the control cache once. Each returned closure reuses
    that cache and the supplied average cache, interpolates their context
    portions, and evaluates one benchmark query.

    Args:
        backend: Model backend used for KV recording and differentiable patching.
        prompt_runtime: Runtime used to sample entities and render prompts.
        average_cache: Full cache whose context positions contain averaged KV.
        cases: Benchmark cases. Only the canonical forward-forward structure is
            used by this factory.
        seed: Seed used to sample the concrete entity names.

    Returns:
        One differentiable objective function per matching benchmark case.

    Raises:
        ValueError: If no cases match the supported structure or a benchmark
            query is unavailable from the prompt runtime.
    """
    context_start = prompt_runtime.landmarks["CTXT_START"]
    context_end = prompt_runtime.landmarks["CTXT_END"]
    entities = prompt_runtime.sample_entities(seed)

    context_prompt = prompt_runtime.render_prompt_as_tokens(
        context_entities=entities,
        context_relations=STRUCTURE,
        query_entities=None,
        query_relations=None,
    )
    with torch.no_grad():
        control_cache, _ = backend.forward_with_KV_recording(context_prompt)

    matching_cases = [
        case for case in cases if case["structure"] == STRUCTURE_NAME
    ]
    if not matching_cases:
        raise ValueError(f"No benchmark cases match structure {STRUCTURE_NAME!r}.")

    query_specs = prompt_runtime.build_query_specs(entities)
    candidates = _candidate_answers(prompt_runtime.query_mode)
    objectives: list[KVObjective] = []

    # The masks are the optimization variables; model weights remain fixed.
    model = getattr(backend, "model", None)
    if model is not None:
        model.requires_grad_(False)

    for case in matching_cases:
        try:
            query_spec = query_specs[case["query"]]
        except KeyError as exc:
            raise ValueError(
                "Benchmark query is not supported by the prompt runtime: "
                f"{case['query']!r}."
            ) from exc

        query_prompt = prompt_runtime.render_prompt_as_tokens(
            context_entities=None,
            context_relations=None,
            query_entities=query_spec["query_entities"],
            query_relations=query_spec["query_relations"],
        )
        scoring_function = build_correct_answer_margin_score(
            prompt_runtime.tokenizer,
            answer_tokens=query_spec["answer_tokens"],
            correct_answer=case["answer"],
            candidate_answers=candidates,
        )
        objectives.append(
            _build_objective(
                backend=backend,
                prompt_ids=query_prompt,
                control_cache=control_cache,
                average_cache=average_cache,
                context_start=context_start,
                context_end=context_end,
                scoring_function=scoring_function,
            )
        )

    return objectives


# -----------------------------------------------------------------------------
# A container that bundles all the data in an "erasure trajectory"


@dataclass
class ErasureState:
    """Data needed to reconstruct an iterative KV-erasure trajectory.

    Attributes:
        key_erasure_round_tli: Iteration in which each key site was erased.
            The axes are ``[context_token, layer, kv_head]``.
        value_erasure_round_tli: Iteration in which each value site was erased,
            with the same axes as ``key_erasure_round_tli``.
        score_by_iteration: Mean objective score before each erasure round.

    Notes:
        An erasure round of ``-1`` denotes a site that has not been erased.
        Round ``0`` denotes a site erased in the initial mosaic. Positive
        rounds group all sites selected by one gradient evaluation.
    """

    key_erasure_round_tli: np.ndarray
    value_erasure_round_tli: np.ndarray
    score_by_iteration: list[float]

    def _validate_mask_index(
        self,
        erasure_round_tli: np.ndarray,
        mask_index: MaskIndex,
        *,
        cache_kind: str,
    ) -> MaskIndex:
        """Validate and normalize one mask index without changing state."""
        if erasure_round_tli.ndim != 3:
            raise ValueError(
                f"The {cache_kind} erasure array must have three dimensions; "
                f"got shape {erasure_round_tli.shape}."
            )
        if not np.issubdtype(erasure_round_tli.dtype, np.integer):
            raise ValueError(
                f"The {cache_kind} erasure array must be integer-valued."
            )
        if len(mask_index) != 3:
            raise ValueError(
                "A mask index must contain context-token, layer, and KV-head "
                f"coordinates; got {mask_index!r}."
            )

        coordinates: list[int] = []
        for axis_name, coordinate, axis_size in zip(
            ("context token", "layer", "KV head"),
            mask_index,
            erasure_round_tli.shape,
        ):
            if isinstance(coordinate, bool):
                raise ValueError(f"The {axis_name} coordinate must be an integer.")
            try:
                coordinate = integer_index(coordinate)
            except TypeError as exc:
                raise ValueError(
                    f"The {axis_name} coordinate must be an integer."
                ) from exc
            if not 0 <= coordinate < axis_size:
                raise IndexError(
                    f"The {axis_name} coordinate {coordinate} is outside "
                    f"[0, {axis_size})."
                )
            coordinates.append(coordinate)

        normalized_index: MaskIndex = (
            coordinates[0],
            coordinates[1],
            coordinates[2],
        )
        previous_iteration = erasure_round_tli[normalized_index]
        if previous_iteration != -1:
            raise ValueError(
                f"The {cache_kind} site {normalized_index} was already erased "
                f"at iteration {int(previous_iteration)}."
            )
        return normalized_index

    def record_erasure_step(
        self,
        key_indices: Sequence[MaskIndex],
        value_indices: Sequence[MaskIndex],
        score: float,
    ) -> None:
        """Record one score and the batch of erasures selected from it.

        All supplied key and value sites receive the same erasure round. The
        entire update is validated before the state is changed.

        Args:
            key_indices: Key-mask sites selected during this gradient step.
            value_indices: Value-mask sites selected during this gradient step.
            score: Mean objective score before the sites are erased.

        Raises:
            ValueError: If the score, history, or any site is invalid.
            IndexError: If an index coordinate lies outside its mask.
        """
        key_indices = tuple(key_indices)
        value_indices = tuple(value_indices)
        if not key_indices and not value_indices:
            raise ValueError("At least one key or value site must be erased.")
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError("score must be a finite number.")
        if not np.isfinite(score):
            raise ValueError("score must be a finite number.")

        normalized_key_indices = [
            self._validate_mask_index(
                self.key_erasure_round_tli,
                mask_index,
                cache_kind="key",
            )
            for mask_index in key_indices
        ]
        normalized_value_indices = [
            self._validate_mask_index(
                self.value_erasure_round_tli,
                mask_index,
                cache_kind="value",
            )
            for mask_index in value_indices
        ]
        if len(set(normalized_key_indices)) != len(normalized_key_indices):
            raise ValueError("A key site cannot be erased twice in one step.")
        if len(set(normalized_value_indices)) != len(normalized_value_indices):
            raise ValueError("A value site cannot be erased twice in one step.")

        latest_round = max(
            0,
            int(self.key_erasure_round_tli.max()),
            int(self.value_erasure_round_tli.max()),
        )
        if len(self.score_by_iteration) != latest_round:
            raise ValueError(
                "Score history does not align with the recorded erasure rounds: "
                f"scores={len(self.score_by_iteration)}, latest_round={latest_round}."
            )

        next_round = latest_round + 1
        for mask_index in normalized_key_indices:
            self.key_erasure_round_tli[mask_index] = next_round
        for mask_index in normalized_value_indices:
            self.value_erasure_round_tli[mask_index] = next_round
        self.score_by_iteration.append(float(score))

    @classmethod
    def from_masks(
        cls,
        key_mask: np.ndarray,
        value_mask: np.ndarray,
    ) -> ErasureState:
        """Create a state from arbitrary initial key and value masks.

        Retained sites are represented by ``True`` and receive an erasure
        round of ``-1``. Sites that begin erased are represented by ``False``
        and receive erasure round ``0``.

        Args:
            key_mask: Initial key mask with axes
                ``[context_token, layer, kv_head]``.
            value_mask: Initial value mask with the same shape as ``key_mask``.

        Returns:
            A state representing the supplied initial mosaic and an empty
            score history.

        Raises:
            ValueError: If the masks are not same-shaped, three-dimensional,
                nonempty Boolean arrays.
        """
        if key_mask.shape != value_mask.shape:
            raise ValueError(
                "Initial key and value masks must have the same shape; got "
                f"key={key_mask.shape}, value={value_mask.shape}."
            )
        if key_mask.ndim != 3 or any(axis_size < 1 for axis_size in key_mask.shape):
            raise ValueError(
                "Initial masks must have three positive dimensions; got "
                f"shape {key_mask.shape}."
            )
        if key_mask.dtype != np.bool_ or value_mask.dtype != np.bool_:
            raise ValueError("Initial key and value masks must be Boolean arrays.")

        return cls(
            key_erasure_round_tli=np.where(key_mask, -1, 0),
            value_erasure_round_tli=np.where(value_mask, -1, 0),
            score_by_iteration=[],
        )

    @classmethod
    def initialize(
        cls,
        mask_shape: tuple[int, int, int],
    ) -> ErasureState:
        """Create an unerased state for a given mask shape.

        Args:
            mask_shape: Number of context tokens, layers, and KV heads.

        Returns:
            A state with both erasure-round arrays initialized to ``-1`` and
            an empty score history.

        Raises:
            ValueError: If ``mask_shape`` does not contain three positive
                dimensions.
        """
        if len(mask_shape) != 3 or any(axis_size < 1 for axis_size in mask_shape):
            raise ValueError(
                "mask_shape must contain three positive dimensions; got "
                f"{mask_shape!r}."
            )

        retained_mask = np.ones(mask_shape, dtype=bool)
        return cls.from_masks(retained_mask, retained_mask)

    def masks_at_iteration(
        self,
        iteration: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct key and value masks at one iteration.

        Args:
            iteration: Historical iteration to reconstruct. If ``None``, use
                the latest completed iteration.

        Returns:
            Boolean key and value masks, where ``True`` means retain the
            corresponding control-cache activation.

        Raises:
            ValueError: If ``iteration`` is not a valid historical iteration.
        """
        latest_iteration = max(
            0,
            int(self.key_erasure_round_tli.max()),
            int(self.value_erasure_round_tli.max()),
        )
        if iteration is None:
            iteration = latest_iteration
        else:
            if isinstance(iteration, bool):
                raise ValueError("iteration must be a nonnegative integer.")
            try:
                iteration = integer_index(iteration)
            except TypeError as exc:
                raise ValueError(
                    "iteration must be a nonnegative integer."
                ) from exc
            if not 0 <= iteration <= latest_iteration:
                raise ValueError(
                    f"iteration must fall within [0, {latest_iteration}]; "
                    f"got {iteration}."
                )

        key_mask = (
            (self.key_erasure_round_tli == -1)
            | (self.key_erasure_round_tli > iteration)
        )
        value_mask = (
            (self.value_erasure_round_tli == -1)
            | (self.value_erasure_round_tli > iteration)
        )
        return key_mask, value_mask

    def rewound_to_at_least(self, n_remaining: int) -> ErasureState:
        """Return a copy rewound to at least a requested number of sites.

        The latest historical iteration satisfying the request is used, so the
        method rewinds no farther than necessary. The current state is not
        modified.

        Args:
            n_remaining: Minimum total number of retained key and value sites.

        Returns:
            An independent state at the selected historical iteration.

        Raises:
            ValueError: If ``n_remaining`` is invalid or exceeds the number of
                sites retained by the initial masks.
        """
        if isinstance(n_remaining, bool):
            raise ValueError("n_remaining must be a nonnegative integer.")
        try:
            n_remaining = integer_index(n_remaining)
        except TypeError as exc:
            raise ValueError(
                "n_remaining must be a nonnegative integer."
            ) from exc
        if n_remaining < 0:
            raise ValueError("n_remaining must be a nonnegative integer.")

        initial_key_mask, initial_value_mask = self.masks_at_iteration(0)
        initial_remaining = int(initial_key_mask.sum() + initial_value_mask.sum())
        if n_remaining > initial_remaining:
            raise ValueError(
                f"Requested {n_remaining} retained sites, but the initial "
                f"masks retain only {initial_remaining}."
            )

        latest_iteration = max(
            0,
            int(self.key_erasure_round_tli.max()),
            int(self.value_erasure_round_tli.max()),
        )
        selected_iteration = 0
        for iteration in range(latest_iteration, -1, -1):
            key_mask, value_mask = self.masks_at_iteration(iteration)
            if int(key_mask.sum() + value_mask.sum()) >= n_remaining:
                selected_iteration = iteration
                break

        key_rounds = self.key_erasure_round_tli.copy()
        value_rounds = self.value_erasure_round_tli.copy()
        key_rounds[key_rounds > selected_iteration] = -1
        value_rounds[value_rounds > selected_iteration] = -1
        return ErasureState(
            key_erasure_round_tli=key_rounds,
            value_erasure_round_tli=value_rounds,
            score_by_iteration=self.score_by_iteration[:selected_iteration],
        )

    def to_serializable(self) -> dict[str, object]:
        """Convert the state into a JSON-compatible representation.

        Returns:
            A mapping containing the key and value erasure rounds and score
            history as JSON-compatible lists.

        Raises:
            ValueError: If the state contains inconsistent arrays, invalid
                erasure rounds, or non-finite scores.
        """
        self._validate_serialized_values(
            self.key_erasure_round_tli,
            self.value_erasure_round_tli,
            self.score_by_iteration,
        )
        return {
            "key_erasure_round_tli": self.key_erasure_round_tli.tolist(),
            "value_erasure_round_tli": self.value_erasure_round_tli.tolist(),
            "score_by_iteration": [
                float(score) for score in self.score_by_iteration
            ],
        }

    @classmethod
    def from_serializable(
        cls,
        data: Mapping[str, object],
    ) -> ErasureState:
        """Reconstruct an erasure state from serialized data.

        Args:
            data: Mapping produced by ``to_serializable``.

        Returns:
            The reconstructed and validated state.

        Raises:
            ValueError: If the mapping has missing or additional fields, or
                contains inconsistent arrays, invalid erasure rounds, or
                non-finite scores.
        """
        expected_fields = {
            "key_erasure_round_tli",
            "value_erasure_round_tli",
            "score_by_iteration",
        }
        if set(data) != expected_fields:
            raise ValueError(
                f"Serialized state must contain exactly {sorted(expected_fields)}."
            )

        try:
            key_rounds = np.asarray(data["key_erasure_round_tli"])
            value_rounds = np.asarray(data["value_erasure_round_tli"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Erasure rounds must be rectangular numeric arrays."
            ) from exc

        raw_scores = data["score_by_iteration"]
        if not isinstance(raw_scores, list):
            raise ValueError("score_by_iteration must be a list.")
        scores = list(raw_scores)

        cls._validate_serialized_values(key_rounds, value_rounds, scores)
        return cls(
            key_erasure_round_tli=key_rounds.astype(np.int64, copy=False),
            value_erasure_round_tli=value_rounds.astype(np.int64, copy=False),
            score_by_iteration=[float(score) for score in scores],
        )

    @staticmethod
    def _validate_serialized_values(
        key_rounds: np.ndarray,
        value_rounds: np.ndarray,
        scores: Sequence[float],
    ) -> None:
        """Validate values stored in or loaded from an erasure-state JSON."""
        if key_rounds.shape != value_rounds.shape:
            raise ValueError(
                "Key and value erasure arrays must have the same shape; got "
                f"key={key_rounds.shape}, value={value_rounds.shape}."
            )
        if key_rounds.ndim != 3 or any(size < 1 for size in key_rounds.shape):
            raise ValueError(
                "Erasure arrays must have three positive dimensions; got "
                f"shape {key_rounds.shape}."
            )
        if not np.issubdtype(key_rounds.dtype, np.integer):
            raise ValueError("Key erasure rounds must be integers.")
        if not np.issubdtype(value_rounds.dtype, np.integer):
            raise ValueError("Value erasure rounds must be integers.")
        if np.any(key_rounds < -1) or np.any(value_rounds < -1):
            raise ValueError("Erasure rounds must be greater than or equal to -1.")

        for score in scores:
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError("Scores must be finite numbers.")
            if not np.isfinite(score):
                raise ValueError("Scores must be finite numbers.")


# -----------------------------------------------------------------------------
# Gradient erasure


def evaluate_mean_objective_score(
    objectives: KVObjective | Sequence[KVObjective],
    key_mask: np.ndarray,
    value_mask: np.ndarray,
) -> float:
    """Evaluate the mean objective score without constructing autograd graphs.

    Args:
        objectives: One objective or a nonempty sequence of objectives.
        key_mask: Boolean key mask with axes
            ``[context_token, layer, kv_head]``.
        value_mask: Boolean value mask with the same shape as ``key_mask``.

    Returns:
        Mean scalar objective score.

    Raises:
        ValueError: If no objectives are supplied or the masks are invalid.
    """
    if callable(objectives):
        objective_sequence = (objectives,)
    else:
        objective_sequence = tuple(objectives)
    if not objective_sequence:
        raise ValueError("At least one objective is required.")
    if key_mask.shape != value_mask.shape or key_mask.ndim != 3:
        raise ValueError("Key and value masks must be same-shaped 3D arrays.")
    if key_mask.dtype != np.bool_ or value_mask.dtype != np.bool_:
        raise ValueError("Key and value masks must be Boolean arrays.")

    key_tensor = torch.as_tensor(key_mask, dtype=torch.bfloat16)
    value_tensor = torch.as_tensor(value_mask, dtype=torch.bfloat16)
    total = 0.0
    with torch.no_grad():
        for objective in objective_sequence:
            score = objective(key_tensor, value_tensor)
            total += score.detach().float().cpu().item()
    return total / len(objective_sequence)


def take_gradient_erasure_step(
    objectives: KVObjective | Sequence[KVObjective],
    erasure_state: ErasureState,
    n_erasures: int = 1,
) -> float:
    """Erase the sites with the smallest averaged absolute gradients.

    Objective functions are evaluated one at a time. Each backward pass frees
    its own autograd graph while accumulating the gradient of the mean score on
    the shared key and value masks.

    Args:
        objectives: One objective or a nonempty sequence of objectives.
        erasure_state: State containing the current masks and erasure history.
        n_erasures: Number of individual key or value sites to erase.

    Returns:
        Mean objective score before the selected sites are erased.

    Raises:
        ValueError: If no objectives are supplied, ``n_erasures`` is invalid,
            or too few eligible sites remain.
        RuntimeError: If an objective does not produce gradients for both
            masks.
    """
    if callable(objectives):
        objective_sequence = (objectives,)
    else:
        objective_sequence = tuple(objectives)
    if not objective_sequence:
        raise ValueError("At least one objective is required.")

    if isinstance(n_erasures, bool):
        raise ValueError("n_erasures must be a positive integer.")
    try:
        n_erasures = integer_index(n_erasures)
    except TypeError as exc:
        raise ValueError("n_erasures must be a positive integer.") from exc
    if n_erasures < 1:
        raise ValueError("n_erasures must be a positive integer.")

    # Upgrade masks to pytorch tensors that require gradients
    key_mask_array, value_mask_array = erasure_state.masks_at_iteration()
    key_mask = torch.tensor(
        key_mask_array,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    value_mask = torch.tensor(
        value_mask_array,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    # Accumulate the gradient of the mean without retaining more than one
    # objective's DAG at a time
    mean_score = 0.0
    n_objectives = len(objective_sequence)
    for objective in objective_sequence:
        score = objective(key_mask, value_mask)
        (score / n_objectives).backward()
        mean_score += score.detach().float().cpu().item() / n_objectives
        del score

    if key_mask.grad is None or value_mask.grad is None:
        raise RuntimeError("Objectives must produce gradients for both masks.")

    key_effects = key_mask.grad.detach().float().cpu().numpy().ravel()
    value_effects = value_mask.grad.detach().float().cpu().numpy().ravel()
    effects = np.abs(np.concatenate((key_effects, value_effects)))
    eligible = np.concatenate(
        (key_mask_array.ravel(), value_mask_array.ravel())
    )
    n_eligible = int(eligible.sum())
    if n_erasures > n_eligible:
        raise ValueError(
            f"Requested {n_erasures} erasures, but only "
            f"{n_eligible} eligible sites remain."
        )

    effects[~eligible] = np.inf
    selected_indices = np.argsort(effects, kind="stable")[:n_erasures]
    sites_per_cache = key_mask_array.size

    selected_key_indices: list[MaskIndex] = []
    selected_value_indices: list[MaskIndex] = []
    for flat_index in selected_indices:
        if flat_index < sites_per_cache:
            coordinates = np.unravel_index(flat_index, key_mask_array.shape)
            mask_index: MaskIndex = (
                int(coordinates[0]),
                int(coordinates[1]),
                int(coordinates[2]),
            )
            selected_key_indices.append(mask_index)
        else:
            value_flat_index = flat_index - sites_per_cache
            coordinates = np.unravel_index(
                value_flat_index,
                value_mask_array.shape,
            )
            mask_index = (
                int(coordinates[0]),
                int(coordinates[1]),
                int(coordinates[2]),
            )
            selected_value_indices.append(mask_index)

    erasure_state.record_erasure_step(
        selected_key_indices,
        selected_value_indices,
        mean_score,
    )
    return mean_score
