from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cognitive_maps_in_llms.experiments.utils_erasure import (
    load_erasure_conditions,
)
from cognitive_maps_in_llms.tasks.erasure import (
    ERASURE_STATE_FILENAME,
    METADATA_FILENAME,
)
from cognitive_maps_in_llms.tasks.key_value_erasure_tools import ErasureState


pytestmark = pytest.mark.codex


def _write_erasure_trajectory(output_dir: Path) -> ErasureState:
    key_mask = np.ones((1, 3, 1), dtype=bool)
    value_mask = np.array([[[True], [True], [False]]])
    state = ErasureState.from_masks(key_mask, value_mask)

    erasures = [
        ([(0, 0, 0)], [], 100.0),
        ([], [(0, 0, 0)], 96.0),
        ([(0, 1, 0)], [], 94.0),
        ([], [(0, 1, 0)], 79.0),
        ([(0, 2, 0)], [], 49.0),
    ]
    for key_indices, value_indices, score in erasures:
        state.record_erasure_step(key_indices, value_indices, score)

    output_dir.mkdir()
    (output_dir / ERASURE_STATE_FILENAME).write_text(
        json.dumps(state.to_serializable()),
        encoding="utf-8",
    )
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps({"initial_score": 100.0, "fully_erased_score": 0.0}),
        encoding="utf-8",
    )
    return state


def test_load_erasure_conditions_resolves_endpoints_and_fidelities(
    tmp_path: Path,
) -> None:
    state = _write_erasure_trajectory(tmp_path / "erasure")

    conditions = load_erasure_conditions(
        tmp_path / "erasure",
        [
            "REPLACE_ALL",
            "REPLACE_NONE",
            "FIDELITY_50",
            "FIDELITY_80",
            "FIDELITY_95",
            "FIDELITY_95_SITES_X2",
        ],
    )

    assert conditions["REPLACE_ALL"].iteration is None
    assert np.all(conditions["REPLACE_ALL"].key_mask)
    assert np.all(conditions["REPLACE_ALL"].value_mask)
    assert conditions["REPLACE_ALL"].n_keys == 3
    assert conditions["REPLACE_ALL"].n_values == 3
    assert conditions["REPLACE_NONE"].iteration is None
    assert not np.any(conditions["REPLACE_NONE"].key_mask)
    assert not np.any(conditions["REPLACE_NONE"].value_mask)
    assert conditions["REPLACE_NONE"].n_keys == 0
    assert conditions["REPLACE_NONE"].n_values == 0

    expected_iterations = {
        "FIDELITY_50": 3,
        "FIDELITY_80": 2,
        "FIDELITY_95": 1,
        "FIDELITY_95_SITES_X2": 0,
    }
    for name, expected_iteration in expected_iterations.items():
        condition = conditions[name]
        assert condition.iteration == expected_iteration
        expected_key_mask, expected_value_mask = state.masks_at_iteration(
            expected_iteration
        )
        np.testing.assert_array_equal(condition.key_mask, expected_key_mask)
        np.testing.assert_array_equal(condition.value_mask, expected_value_mask)
        assert condition.n_keys == int(expected_key_mask.sum())
        assert condition.n_values == int(expected_value_mask.sum())


def test_buffer_condition_uses_combined_key_and_value_site_count(
    tmp_path: Path,
) -> None:
    _write_erasure_trajectory(tmp_path / "erasure")

    conditions = load_erasure_conditions(
        tmp_path / "erasure",
        ["FIDELITY_95", "FIDELITY_95_SITES_X2"],
    )

    base = conditions["FIDELITY_95"]
    buffered = conditions["FIDELITY_95_SITES_X2"]
    base_remaining = int(base.key_mask.sum() + base.value_mask.sum())
    buffered_remaining = int(
        buffered.key_mask.sum() + buffered.value_mask.sum()
    )
    assert base_remaining == 4
    assert buffered_remaining == 5
    assert buffered_remaining == min(5, 2 * base_remaining)


@pytest.mark.parametrize("condition_names", [[], ["unknown"], ["REPLACE_ALL"] * 2])
def test_load_erasure_conditions_rejects_invalid_names(
    tmp_path: Path,
    condition_names: list[str],
) -> None:
    _write_erasure_trajectory(tmp_path / "erasure")

    with pytest.raises(ValueError):
        load_erasure_conditions(tmp_path / "erasure", condition_names)


def test_load_erasure_conditions_rejects_one_bare_string(tmp_path: Path) -> None:
    _write_erasure_trajectory(tmp_path / "erasure")

    with pytest.raises(TypeError, match="not a string"):
        load_erasure_conditions(
            tmp_path / "erasure",
            "REPLACE_ALL",  # type: ignore[arg-type]
        )


def test_fidelity_condition_requires_a_score_trajectory(tmp_path: Path) -> None:
    output_dir = tmp_path / "erasure"
    output_dir.mkdir()
    state = ErasureState.initialize((1, 1, 1))
    (output_dir / ERASURE_STATE_FILENAME).write_text(
        json.dumps(state.to_serializable()),
        encoding="utf-8",
    )
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps({"initial_score": 1.0, "fully_erased_score": 0.0}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not contain any scores"):
        load_erasure_conditions(output_dir, ["FIDELITY_80"])
