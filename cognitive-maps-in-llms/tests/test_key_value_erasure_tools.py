from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from llm_neuroscience_tools.backends.KV_patching import CacheData

from cognitive_maps_in_llms.benchmarks import BenchmarkCase
from cognitive_maps_in_llms.tasks.key_value_erasure_tools import (
    ErasureState,
    build_all_objectives_for_one_context,
    build_correct_answer_margin_score,
    build_interpolated_cache,
    evaluate_mean_objective_score,
    load_average_cache,
    take_gradient_erasure_step,
)
from cognitive_maps_in_llms.tasks.recording import (
    KEY_CACHE_FILENAME,
    METADATA_FILENAME,
    VALUE_CACHE_FILENAME,
)


pytestmark = pytest.mark.codex


class FakeTokenizer:
    token_ids = {
        "Yes": 1,
        "No": 2,
        "Unknown": 3,
        "Alice": 4,
        "Bob": 5,
        "Carol": 6,
    }

    def encode(self, token: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [self.token_ids[token]]


def _cache(keys: torch.Tensor, values: torch.Tensor) -> CacheData:
    device_map = [(torch.device("cpu"), torch.device("cpu"))]
    return CacheData(keys, values, device_map)


def test_correct_answer_margin_ignores_answers_outside_candidate_set() -> None:
    scorer = build_correct_answer_margin_score(
        FakeTokenizer(),
        answer_tokens={
            "A": "Alice",
            "B": "Bob",
            "C": "Carol",
            "U": "Unknown",
        },
        correct_answer="A",
        candidate_answers=("A", "B", "C"),
    )
    logits = torch.zeros((1, 1, 7))
    logits[0, 0, 3] = 100.0
    logits[0, 0, 4:7] = torch.tensor([9.0, 3.0, 0.0])

    assert scorer(logits).item() == pytest.approx(7.5)


def test_interpolation_changes_only_the_context() -> None:
    control = _cache(
        torch.full((4, 1, 1, 2), 10.0),
        torch.full((4, 1, 1, 2), 20.0),
    )
    average = _cache(
        torch.full((4, 1, 1, 2), -2.0),
        torch.full((4, 1, 1, 2), -4.0),
    )
    key_mask = torch.tensor([[[0.0]], [[1.0]]], requires_grad=True)
    value_mask = torch.tensor([[[0.25]], [[0.5]]], requires_grad=True)

    result = build_interpolated_cache(
        control,
        average,
        context_start=1,
        context_end=2,
        key_mask=key_mask,
        value_mask=value_mask,
    )

    assert torch.equal(result.K_tlia[0], control.K_tlia[0])
    assert torch.equal(result.K_tlia[3], control.K_tlia[3])
    assert torch.equal(result.K_tlia[1, 0, 0], torch.tensor([-2.0, -2.0]))
    assert torch.equal(result.K_tlia[2, 0, 0], torch.tensor([10.0, 10.0]))
    assert torch.equal(result.V_tlia[1, 0, 0], torch.tensor([2.0, 2.0]))
    assert torch.equal(result.V_tlia[2, 0, 0], torch.tensor([8.0, 8.0]))


class FakePromptRuntime:
    query_mode = "yes_no"
    landmarks = {"CTXT_START": 1, "CTXT_END": 2}
    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.render_calls: list[dict[str, Any]] = []

    def sample_entities(self, seed: int) -> dict[str, str]:
        return {"A": "Alice", "B": "Bob", "C": "Carol"}

    def build_query_specs(
        self, entities: dict[str, str]
    ) -> dict[str, dict[str, Any]]:
        return {
            "A --daughter--> B?": {
                "query_entities": {"QE": entities["A"], "QCAND": entities["B"]},
                "query_relations": {"QR": "daughter"},
                "answer_tokens": {"Y": "Yes", "N": "No", "U": "Unknown"},
            }
        }

    def render_prompt_as_tokens(self, **substitutions: Any) -> list[int]:
        self.render_calls.append(substitutions)
        return [10, 11, 12, 13]


class FakeBackend:
    def __init__(self) -> None:
        self.recording_calls = 0
        self.patch_ends: list[int] = []

    def forward_with_KV_recording(
        self, prompt_ids: list[int]
    ) -> tuple[CacheData, np.ndarray]:
        self.recording_calls += 1
        return (
            _cache(
                torch.full((len(prompt_ids), 1, 1, 1), 4.0),
                torch.full((len(prompt_ids), 1, 1, 1), 6.0),
            ),
            np.zeros(7),
        )

    def forward_with_differentiable_KV_patching(
        self,
        prompt_ids: list[int],
        cache_data: CacheData,
        patch_end: int,
    ) -> tuple[CacheData, torch.Tensor]:
        self.patch_ends.append(patch_end)
        context_score = (
            cache_data.K_tlia[1:3].mean() + cache_data.V_tlia[1:3].mean()
        )
        logits = torch.zeros((1, 1, 7))
        logits[0, 0, 1] = context_score
        return cache_data, logits


def _write_recordings(recording_directory: Path) -> None:
    """Write a small recording-task output for an average-cache test."""
    recording_directory.mkdir()
    keys = np.zeros((2, 3, 2, 1, 1, 1), dtype=np.float32)
    values = np.zeros_like(keys)
    for seed_index, (key_value, value_value) in enumerate(
        ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    ):
        keys[0, seed_index] = key_value
        values[0, seed_index] = value_value
    keys[1] = 100.0
    values[1] = 200.0

    np.save(recording_directory / KEY_CACHE_FILENAME, keys)
    np.save(recording_directory / VALUE_CACHE_FILENAME, values)
    metadata = {
        "n_trials": 3,
        "kv_axes": [
            "graph_structure",
            "seed",
            "token",
            "layer",
            "kv_head",
            "head_dim",
        ],
        "relation_patterns": [
            {
                "name": "A --fwd--> B --fwd--> C",
                "context_relations": {"R_AB": "FWD", "R_BC": "FWD"},
            },
            {
                "name": "A --rev--> B --fwd--> C",
                "context_relations": {"R_AB": "REV", "R_BC": "FWD"},
            },
        ],
    }
    (recording_directory / METADATA_FILENAME).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_load_average_cache_averages_seeds_for_canonical_structure(
    tmp_path: Path,
) -> None:
    recording_directory = tmp_path / "recordings"
    _write_recordings(recording_directory)
    backend = FakeBackend()
    runtime = FakePromptRuntime()

    average_cache = load_average_cache(
        recording_directory,
        backend,  # type: ignore[arg-type]
        runtime,  # type: ignore[arg-type]
    )

    assert backend.recording_calls == 1
    assert torch.all(average_cache.K_tlia[[0, 3]] == 0)
    assert torch.all(average_cache.V_tlia[[0, 3]] == 0)
    assert torch.all(average_cache.K_tlia[1:3] == 3.0)
    assert torch.all(average_cache.V_tlia[1:3] == 4.0)
    assert runtime.render_calls[0]["context_relations"] == {
        "R_AB": "FWD",
        "R_BC": "FWD",
    }


def test_factory_records_once_and_builds_differentiable_objectives() -> None:
    runtime = FakePromptRuntime()
    backend = FakeBackend()
    average_cache = _cache(
        torch.full((4, 1, 1, 1), -100.0),
        torch.full((4, 1, 1, 1), -100.0),
    )
    average_cache.K_tlia[1:3] = 0.0
    average_cache.V_tlia[1:3] = 0.0
    cases: list[BenchmarkCase] = [
        {
            "structure": "A --fwd--> B --fwd--> C",
            "query": "A --daughter--> B?",
            "answer": "Y",
            "kind": "one-hop",
        },
        {
            "structure": "A --rev--> B --fwd--> C",
            "query": "A --daughter--> B?",
            "answer": "N",
            "kind": "one-hop",
        },
    ]

    objectives = build_all_objectives_for_one_context(
        backend, runtime, average_cache, cases, seed=7  # type: ignore[arg-type]
    )
    key_mask = torch.ones((2, 1, 1), requires_grad=True)
    value_mask = torch.ones((2, 1, 1), requires_grad=True)
    score = objectives[0](key_mask, value_mask)
    score.backward()

    assert len(objectives) == 1
    assert backend.recording_calls == 1
    assert backend.patch_ends == [2]
    assert score.item() == pytest.approx(10.0)
    assert key_mask.grad is not None
    assert value_mask.grad is not None
    assert runtime.render_calls[1]["context_entities"] is None
    assert runtime.render_calls[1]["query_entities"] == {
        "QE": "Alice",
        "QCAND": "Bob",
    }


def test_erasure_state_records_one_key_value_batch_atomically() -> None:
    state = ErasureState(
        key_erasure_round_tli=np.full((2, 3, 4), -1, dtype=np.int64),
        value_erasure_round_tli=np.full((2, 3, 4), -1, dtype=np.int64),
        score_by_iteration=[],
    )

    state.record_erasure_step(
        key_indices=[(1, 2, 3)],
        value_indices=[(0, 1, 2)],
        score=2.5,
    )

    assert state.key_erasure_round_tli[1, 2, 3] == 1
    assert state.value_erasure_round_tli[0, 1, 2] == 1
    assert state.key_erasure_round_tli[0, 1, 2] == -1
    assert state.value_erasure_round_tli[1, 2, 3] == -1
    assert state.score_by_iteration == [2.5]


def test_erasure_state_initializes_from_arbitrary_masks() -> None:
    key_mask = np.ones((2, 4, 3), dtype=bool)
    value_mask = np.ones((2, 4, 3), dtype=bool)
    key_mask[:, 2:, :] = False
    value_mask[:, :2, :] = False

    state = ErasureState.from_masks(key_mask, value_mask)

    assert np.all(state.key_erasure_round_tli[:, :2, :] == -1)
    assert np.all(state.key_erasure_round_tli[:, 2:, :] == 0)
    assert np.all(state.value_erasure_round_tli[:, :2, :] == 0)
    assert np.all(state.value_erasure_round_tli[:, 2:, :] == -1)
    assert state.score_by_iteration == []


def test_erasure_state_initialize_retains_every_site() -> None:
    state = ErasureState.initialize((2, 3, 4))

    assert np.all(state.key_erasure_round_tli == -1)
    assert np.all(state.value_erasure_round_tli == -1)


def test_erasure_state_reconstructs_historical_masks() -> None:
    key_mask = np.array([[[False, True], [True, True]]])
    value_mask = np.array([[[True, False], [True, True]]])
    state = ErasureState.from_masks(key_mask, value_mask)
    state.record_erasure_step([(0, 0, 1)], [], score=4.0)
    state.record_erasure_step([], [(0, 1, 0)], score=3.5)

    key_at_zero, value_at_zero = state.masks_at_iteration(0)
    key_at_one, value_at_one = state.masks_at_iteration(1)
    key_latest, value_latest = state.masks_at_iteration()

    assert np.array_equal(key_at_zero, key_mask)
    assert np.array_equal(value_at_zero, value_mask)
    assert not key_at_one[0, 0, 1]
    assert value_at_one[0, 1, 0]
    assert not key_latest[0, 0, 1]
    assert not value_latest[0, 1, 0]


def test_erasure_state_rewinds_without_changing_original() -> None:
    state = ErasureState.initialize((1, 1, 4))
    state.record_erasure_step([(0, 0, 0)], [], score=8.0)
    state.record_erasure_step([], [(0, 0, 0), (0, 0, 1)], score=7.0)
    state.record_erasure_step([(0, 0, 1)], [], score=5.0)

    rewound = state.rewound_to_at_least(6)

    assert state.score_by_iteration == [8.0, 7.0, 5.0]
    assert rewound.score_by_iteration == [8.0]
    rewound_key, rewound_value = rewound.masks_at_iteration()
    assert int(rewound_key.sum() + rewound_value.sum()) == 7
    assert rewound.key_erasure_round_tli[0, 0, 1] == -1
    assert rewound.value_erasure_round_tli[0, 0, 0] == -1


@pytest.mark.parametrize("iteration", [-1, 2, 1.5, True])
def test_erasure_state_rejects_invalid_mask_iteration(
    iteration: object,
) -> None:
    state = ErasureState.initialize((1, 1, 1))

    with pytest.raises(ValueError):
        state.masks_at_iteration(iteration)  # type: ignore[arg-type]


def test_erasure_state_json_values_round_trip() -> None:
    key_mask = np.array([[[True, False], [True, True]]])
    value_mask = np.array([[[False, True], [True, False]]])
    state = ErasureState.from_masks(key_mask, value_mask)
    state.record_erasure_step([(0, 1, 0)], [], score=2.5)
    state.record_erasure_step([], [(0, 1, 0)], score=2.25)

    serialized = state.to_serializable()
    restored = ErasureState.from_serializable(
        json.loads(json.dumps(serialized))
    )

    assert set(serialized) == {
        "key_erasure_round_tli",
        "value_erasure_round_tli",
        "score_by_iteration",
    }
    assert np.array_equal(
        restored.key_erasure_round_tli,
        state.key_erasure_round_tli,
    )
    assert np.array_equal(
        restored.value_erasure_round_tli,
        state.value_erasure_round_tli,
    )
    assert restored.score_by_iteration == [2.5, 2.25]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("key_erasure_round_tli", [[[-2]]]),
        ("value_erasure_round_tli", [[[0.5]]]),
        ("score_by_iteration", [float("inf")]),
    ],
)
def test_erasure_state_rejects_invalid_serialized_values(
    field: str,
    invalid_value: object,
) -> None:
    serialized: dict[str, object] = {
        "key_erasure_round_tli": [[[-1]]],
        "value_erasure_round_tli": [[[-1]]],
        "score_by_iteration": [1.0],
    }
    serialized[field] = invalid_value

    with pytest.raises(ValueError):
        ErasureState.from_serializable(serialized)


def test_erasure_state_refuses_to_overwrite_an_erasure() -> None:
    state = ErasureState(
        key_erasure_round_tli=np.full((2, 3, 4), -1, dtype=np.int64),
        value_erasure_round_tli=np.full((2, 3, 4), -1, dtype=np.int64),
        score_by_iteration=[],
    )
    state.record_erasure_step([(1, 2, 3)], [], score=2.5)

    with pytest.raises(ValueError, match="already erased"):
        state.record_erasure_step([(1, 2, 3)], [], score=2.0)


@pytest.mark.parametrize(
    ("mask_index", "error"),
    [
        ((-1, 0, 0), IndexError),
        ((2, 0, 0), IndexError),
    ],
)
def test_erasure_state_validates_erasure_coordinates(
    mask_index: tuple[int, int, int],
    error: type[Exception],
) -> None:
    state = ErasureState(
        key_erasure_round_tli=np.full((2, 3, 4), -1, dtype=np.int64),
        value_erasure_round_tli=np.full((2, 3, 4), -1, dtype=np.int64),
        score_by_iteration=[],
    )

    with pytest.raises(error):
        state.record_erasure_step([], [mask_index], score=2.5)


def test_erasure_state_rejects_entire_invalid_batch() -> None:
    state = ErasureState.initialize((1, 1, 2))

    with pytest.raises(IndexError):
        state.record_erasure_step(
            key_indices=[(0, 0, 0), (0, 0, 2)],
            value_indices=[],
            score=2.5,
        )

    assert np.all(state.key_erasure_round_tli == -1)
    assert state.score_by_iteration == []


def test_gradient_erasure_step_streams_and_averages_objectives() -> None:
    state = ErasureState.initialize((1, 1, 2))

    def first_objective(
        key_mask: torch.Tensor,
        value_mask: torch.Tensor,
    ) -> torch.Tensor:
        return (
            4 * key_mask[0, 0, 0]
            + 2 * key_mask[0, 0, 1]
            + 6 * value_mask[0, 0, 0]
            + value_mask[0, 0, 1]
        )

    def second_objective(
        key_mask: torch.Tensor,
        value_mask: torch.Tensor,
    ) -> torch.Tensor:
        return (
            2 * key_mask[0, 0, 0]
            + 4 * value_mask[0, 0, 0]
            - 0.5 * value_mask[0, 0, 1]
        )

    score = take_gradient_erasure_step(
        [first_objective, second_objective],
        state,
        n_erasures=2,
    )

    assert score == pytest.approx(9.25)
    assert state.value_erasure_round_tli[0, 0, 1] == 1
    assert state.key_erasure_round_tli[0, 0, 1] == 1
    assert state.key_erasure_round_tli[0, 0, 0] == -1
    assert state.value_erasure_round_tli[0, 0, 0] == -1
    assert state.score_by_iteration == [pytest.approx(9.25)]


def test_gradient_erasure_step_accepts_one_objective() -> None:
    state = ErasureState.initialize((1, 1, 1))

    def objective(
        key_mask: torch.Tensor,
        value_mask: torch.Tensor,
    ) -> torch.Tensor:
        return key_mask.sum() + 2 * value_mask.sum()

    score = take_gradient_erasure_step(objective, state)

    assert score == pytest.approx(3.0)
    assert state.key_erasure_round_tli[0, 0, 0] == 1
    assert state.value_erasure_round_tli[0, 0, 0] == -1


def test_mean_objective_score_does_not_build_gradients() -> None:
    observed_grad_modes: list[bool] = []

    def objective(
        key_mask: torch.Tensor,
        value_mask: torch.Tensor,
    ) -> torch.Tensor:
        observed_grad_modes.append(torch.is_grad_enabled())
        return key_mask.sum() + value_mask.sum()

    key_mask = np.ones((1, 1, 2), dtype=bool)
    value_mask = np.zeros((1, 1, 2), dtype=bool)
    score = evaluate_mean_objective_score(
        [objective, objective],
        key_mask,
        value_mask,
    )

    assert score == pytest.approx(2.0)
    assert observed_grad_modes == [False, False]
