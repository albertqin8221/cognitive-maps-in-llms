from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from llm_neuroscience_tools.backends.KV_patching import CacheData

from cognitive_maps_in_llms.tasks import erasure as erasure_module
from cognitive_maps_in_llms.tasks.key_value_erasure_tools import ErasureState
from cognitive_maps_in_llms.tasks.layer_split import resolve_dividing_layer


pytestmark = pytest.mark.codex


def _cache() -> CacheData:
    keys = torch.zeros((2, 4, 1, 1))
    values = torch.zeros_like(keys)
    device_map = [(torch.device("cpu"), torch.device("cpu"))] * 4
    return CacheData(keys, values, device_map)


class _Runtime:
    landmarks = {"CTXT_START": 0, "CTXT_END": 1}


def test_run_erasure_uses_named_benchmark_and_writes_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_benchmarks: list[str] = []
    objective_seeds: list[int] = []

    def load_benchmark(name: str) -> list[dict[str, str]]:
        loaded_benchmarks.append(name)
        return []

    def build_objectives(
        backend: Any,
        prompt_runtime: Any,
        average_cache: CacheData,
        cases: list[dict[str, str]],
        seed: int,
    ) -> list[Any]:
        objective_seeds.append(seed)

        def objective(
            key_mask: torch.Tensor,
            value_mask: torch.Tensor,
        ) -> torch.Tensor:
            return key_mask.sum() + value_mask.sum()

        return [objective]

    monkeypatch.setattr(erasure_module, "load_benchmark", load_benchmark)
    monkeypatch.setattr(
        erasure_module,
        "load_average_cache",
        lambda *args: _cache(),
    )
    monkeypatch.setattr(
        erasure_module,
        "build_all_objectives_for_one_context",
        build_objectives,
    )

    output_dir = tmp_path / "erasure"
    erasure_module.run_erasure(
        object(),  # type: ignore[arg-type]
        _Runtime(),  # type: ignore[arg-type]
        output_dir,
        benchmark_name="spatial_relationships_yes_no",
        recordings=tmp_path / "recordings",
        start_seed=20,
        n_trials=2,
        coarse_fraction=0.5,
        fidelity_threshold=0.9,
        rewind_multiplier=2.0,
        dividing_layer_fraction=0.75,
    )

    metadata = json.loads(
        (output_dir / erasure_module.METADATA_FILENAME).read_text()
    )
    state = ErasureState.from_serializable(
        json.loads(
            (output_dir / erasure_module.ERASURE_STATE_FILENAME).read_text()
        )
    )

    assert loaded_benchmarks == ["spatial_relationships_yes_no"]
    assert objective_seeds == [20, 21]
    assert metadata["benchmark_name"] == "spatial_relationships_yes_no"
    assert metadata["mask_axes"] == ["context_token", "layer", "kv_head"]
    assert metadata["initial_mosaic"] == {
        "name": "key_then_value",
        "dividing_layer_fraction": 0.75,
        "dividing_layer": 3,
    }
    assert metadata["initial_score"] == pytest.approx(8.0)
    assert metadata["fully_erased_score"] == pytest.approx(0.0)
    key_mask, value_mask = state.masks_at_iteration()
    assert not np.any(key_mask)
    assert not np.any(value_mask)
    assert len(state.score_by_iteration) == 5

    initial_keys, initial_values = state.masks_at_iteration(0)
    assert np.all(initial_keys[:, :3])
    assert not np.any(initial_keys[:, 3:])
    assert not np.any(initial_values[:, :3])
    assert np.all(initial_values[:, 3:])


@pytest.mark.parametrize(
    ("n_layers", "fraction", "expected"),
    [(4, 0.5, 2), (5, 2 / 3, 3), (7, 2 / 3, 4)],
)
def test_resolve_dividing_layer_uses_floor(
    n_layers: int,
    fraction: float,
    expected: int,
) -> None:
    assert resolve_dividing_layer(n_layers, fraction) == expected


@pytest.mark.parametrize("fraction", [True, 0, 1, -0.1, 1.1, "half"])
def test_resolve_dividing_layer_rejects_invalid_fraction(
    fraction: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        resolve_dividing_layer(4, fraction)  # type: ignore[arg-type]
