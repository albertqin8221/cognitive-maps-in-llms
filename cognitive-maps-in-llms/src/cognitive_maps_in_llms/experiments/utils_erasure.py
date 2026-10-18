from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Sequence

import numpy as np

from cognitive_maps_in_llms.tasks.erasure import (
    ERASURE_STATE_FILENAME,
    METADATA_FILENAME,
)
from cognitive_maps_in_llms.tasks.key_value_erasure_tools import ErasureState


SUPPORTED_ERASURE_CONDITIONS = (
    "REPLACE_ALL",
    "REPLACE_NONE",
    "FIDELITY_50",
    "FIDELITY_80",
    "FIDELITY_95",
    "FIDELITY_95_SITES_X2",
)

_FIDELITY_BY_CONDITION = {
    "FIDELITY_50": 0.50,
    "FIDELITY_80": 0.80,
    "FIDELITY_95": 0.95,
}
_BUFFER_BASE_CONDITION = "FIDELITY_95"
_BUFFER_MULTIPLIER = 2.0


@dataclass(frozen=True)
class ResolvedErasureCondition:
    """Masks and trajectory iteration associated with one named condition.

    Attributes:
        iteration: Erasure iteration used to reconstruct the masks. This is
            ``None`` for the synthetic ``REPLACE_ALL`` and ``REPLACE_NONE``
            endpoints.
        key_mask: Boolean key mask with axes
            ``[context_token, layer, kv_head]``.
        value_mask: Boolean value mask with the same shape as ``key_mask``.
        n_keys: Number of retained key sites in ``key_mask``.
        n_values: Number of retained value sites in ``value_mask``.
    """

    iteration: int | None
    key_mask: np.ndarray
    value_mask: np.ndarray
    n_keys: int
    n_values: int


@dataclass(frozen=True)
class _ErasureTrajectory:
    state: ErasureState
    initial_score: float
    fully_erased_score: float


def load_erasure_conditions(
    erasure_dir: Path | str,
    condition_names: Sequence[str],
) -> dict[str, ResolvedErasureCondition]:
    """Load and resolve named conditions from an erasure task output.

    The endpoint conditions ``REPLACE_ALL`` and ``REPLACE_NONE`` return
    all-true and all-false masks, respectively. Fidelity conditions select
    the last trajectory state before normalized fidelity first falls below
    the requested threshold. ``FIDELITY_95_SITES_X2`` selects the latest
    state retaining at least twice as many sites as the resolved
    ``FIDELITY_95`` state, capped at the number retained initially.

    Args:
        erasure_dir: Directory containing ``erasure_state.json`` and
            ``erasure_metadata.json`` from the erasure task.
        condition_names: Unique condition names to resolve. Supported names
            are listed in ``SUPPORTED_ERASURE_CONDITIONS``.

    Returns:
        A dictionary mapping each requested name to its resolved key and
        value masks, retained key and value counts, and, for
        trajectory-derived conditions, its iteration.

    Raises:
        FileNotFoundError: If either erasure artifact does not exist.
        TypeError: If a condition name is not a string.
        ValueError: If no conditions are requested, a name is duplicated or
            unsupported, or the erasure artifacts cannot resolve a requested
            fidelity condition.
    """

    if isinstance(condition_names, str):
        raise TypeError(
            "condition_names must be a sequence of names, not a string."
        )
    names = tuple(condition_names)
    if not names:
        raise ValueError("At least one erasure condition must be requested.")
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"Erasure condition names must be strings, got {name!r}."
            )
    if len(set(names)) != len(names):
        raise ValueError("Erasure condition names must not contain duplicates.")

    unsupported = sorted(set(names) - set(SUPPORTED_ERASURE_CONDITIONS))
    if unsupported:
        raise ValueError(
            f"Unsupported erasure condition(s): {unsupported}. Supported "
            f"conditions are {list(SUPPORTED_ERASURE_CONDITIONS)}."
        )

    trajectory = _load_erasure_trajectory(Path(erasure_dir))
    mask_shape = trajectory.state.key_erasure_round_tli.shape
    resolved: dict[str, ResolvedErasureCondition] = {}

    for name in names:
        if name == "REPLACE_ALL":
            resolved[name] = _make_resolved_condition(
                iteration=None,
                key_mask=np.ones(mask_shape, dtype=bool),
                value_mask=np.ones(mask_shape, dtype=bool),
            )
        elif name == "REPLACE_NONE":
            resolved[name] = _make_resolved_condition(
                iteration=None,
                key_mask=np.zeros(mask_shape, dtype=bool),
                value_mask=np.zeros(mask_shape, dtype=bool),
            )
        elif name in _FIDELITY_BY_CONDITION:
            iteration = _resolve_fidelity_iteration(
                trajectory,
                _FIDELITY_BY_CONDITION[name],
            )
            key_mask, value_mask = trajectory.state.masks_at_iteration(iteration)
            resolved[name] = _make_resolved_condition(
                iteration=iteration,
                key_mask=key_mask,
                value_mask=value_mask,
            )
        else:
            iteration = _resolve_buffered_iteration(trajectory)
            key_mask, value_mask = trajectory.state.masks_at_iteration(iteration)
            resolved[name] = _make_resolved_condition(
                iteration=iteration,
                key_mask=key_mask,
                value_mask=value_mask,
            )

    return resolved


def _make_resolved_condition(
    *,
    iteration: int | None,
    key_mask: np.ndarray,
    value_mask: np.ndarray,
) -> ResolvedErasureCondition:
    return ResolvedErasureCondition(
        iteration=iteration,
        key_mask=key_mask,
        value_mask=value_mask,
        n_keys=int(key_mask.sum()),
        n_values=int(value_mask.sum()),
    )


def _load_erasure_trajectory(erasure_dir: Path) -> _ErasureTrajectory:
    state_path = erasure_dir / ERASURE_STATE_FILENAME
    metadata_path = erasure_dir / METADATA_FILENAME

    with state_path.open(encoding="utf-8") as file:
        state_data = json.load(file)
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(state_data, dict):
        raise ValueError(f"Expected a JSON object in {state_path}.")
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}.")

    state = ErasureState.from_serializable(state_data)
    initial_score = _metadata_score(metadata, "initial_score", metadata_path)
    fully_erased_score = _metadata_score(
        metadata,
        "fully_erased_score",
        metadata_path,
    )
    if initial_score == fully_erased_score:
        raise ValueError(
            "Initial and fully erased scores are equal; normalized fidelity "
            "is undefined."
        )

    return _ErasureTrajectory(
        state=state,
        initial_score=initial_score,
        fully_erased_score=fully_erased_score,
    )


def _metadata_score(metadata: dict[str, object], name: str, path: Path) -> float:
    score = metadata.get(name)
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError(f"{name} in {path} must be a finite number.")
    score = float(score)
    if not math.isfinite(score):
        raise ValueError(f"{name} in {path} must be a finite number.")
    return score


def _resolve_fidelity_iteration(
    trajectory: _ErasureTrajectory,
    target_fidelity: float,
) -> int:
    scores = trajectory.state.score_by_iteration
    if not scores:
        raise ValueError("The erasure trajectory does not contain any scores.")

    denominator = trajectory.initial_score - trajectory.fully_erased_score
    fidelities = [
        (score - trajectory.fully_erased_score) / denominator
        for score in scores
    ]
    for iteration, fidelity in enumerate(fidelities):
        if fidelity < target_fidelity:
            return max(0, iteration - 1)

    key_mask, value_mask = trajectory.state.masks_at_iteration()
    if np.any(key_mask) or np.any(value_mask):
        raise ValueError(
            f"Normalized fidelity never fell below {target_fidelity:.0%} "
            "in the recorded trajectory."
        )

    # The terminal fully erased state has fidelity zero but no subsequent
    # gradient evaluation, so its score is not present in score_by_iteration.
    return max(0, len(scores) - 1)


def _resolve_buffered_iteration(trajectory: _ErasureTrajectory) -> int:
    base_iteration = _resolve_fidelity_iteration(
        trajectory,
        _FIDELITY_BY_CONDITION[_BUFFER_BASE_CONDITION],
    )
    base_key_mask, base_value_mask = trajectory.state.masks_at_iteration(
        base_iteration
    )
    base_remaining = int(base_key_mask.sum() + base_value_mask.sum())

    initial_key_mask, initial_value_mask = trajectory.state.masks_at_iteration(0)
    initial_remaining = int(initial_key_mask.sum() + initial_value_mask.sum())
    target_remaining = min(
        initial_remaining,
        math.ceil(_BUFFER_MULTIPLIER * base_remaining),
    )

    latest_iteration = len(trajectory.state.score_by_iteration)
    selected_iteration = 0
    for iteration in range(latest_iteration + 1):
        key_mask, value_mask = trajectory.state.masks_at_iteration(iteration)
        n_remaining = int(key_mask.sum() + value_mask.sum())
        if n_remaining >= target_remaining:
            selected_iteration = iteration
    return selected_iteration
