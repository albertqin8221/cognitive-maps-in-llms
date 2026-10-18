"""Focused tests for configurable Figure 5 mosaics and editing controls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from llm_neuroscience_tools.backends.KV_patching import CacheData

from cognitive_maps_in_llms.experiments import exp_5a_bottleneck as exp_5a
from cognitive_maps_in_llms.experiments.exp_5c_editing import (
    Experiment_5C_Editing,
)
from cognitive_maps_in_llms.experiments.utils_erasure import (
    ResolvedErasureCondition,
)


pytestmark = pytest.mark.codex


def _cache(fill: float, *, n_layers: int = 5) -> CacheData:
    keys = torch.full((6, n_layers, 1, 1), fill)
    values = torch.full_like(keys, fill)
    device_map = [(torch.device("cpu"), torch.device("cpu"))] * n_layers
    return CacheData(keys, values, device_map)


def test_exp_5a_resolves_configured_dividing_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exp_5a, "load_average_cache", lambda *args: _cache(0))
    runtime = SimpleNamespace(
        tokenizer=object(),
        landmarks={"CTXT_START": 1, "CTXT_END": 4},
    )

    experiment = exp_5a.Experiment_5A_Relational_Bottleneck(
        object(),  # type: ignore[arg-type]
        runtime,  # type: ignore[arg-type]
        n_trials=1,
        recordings="recordings",
        dividing_layer_fraction=2 / 3,
    )

    assert experiment.DIVIDING_LAYER == 3


class _EditingRuntime:
    tokenizer = object()
    landmarks = {
        "CTXT_START": 1,
        "CTXT_END": 4,
        "A1": 1,
        "B1": 2,
        "B2": 3,
        "C1": 4,
    }

    def sample_nonoverlapping_entities(
        self,
        seed: int,
    ) -> tuple[dict[str, str], dict[str, str]]:
        return (
            {"A": "donor-a", "B": "donor-b", "C": "donor-c"},
            {"A": "control-a", "B": "control-b", "C": "control-c"},
        )

    def render_prompt_as_tokens(self, **kwargs: Any) -> list[int]:
        return [0]


class _EditingBackend:
    def __init__(self, donor: CacheData, control: CacheData) -> None:
        self._caches = iter((donor, control))

    def forward_with_KV_recording(
        self,
        prompt: list[int],
    ) -> tuple[CacheData, None]:
        return next(self._caches), None


def test_exp_5c_none_is_unedited_control() -> None:
    donor = _cache(1, n_layers=2)
    control = _cache(2, n_layers=2)
    runtime = _EditingRuntime()
    condition = ResolvedErasureCondition(
        iteration=3,
        key_mask=np.ones((4, 2, 1), dtype=bool),
        value_mask=np.ones((4, 2, 1), dtype=bool),
        n_keys=8,
        n_values=8,
    )
    experiment = Experiment_5C_Editing.__new__(Experiment_5C_Editing)
    experiment.backend = _EditingBackend(donor, control)
    experiment.prompt_runtime = runtime
    experiment.CTXT_START = 1
    experiment.CTXT_END = 4
    experiment.CONTEXT = slice(1, 5)
    experiment.masks_by_condition = {"FIDELITY_95": condition}

    state = experiment.build_trial_state(
        {
            "seed": 0,
            "condition": "FIDELITY_95",
            "editing_locs": "NONE",
        }
    )

    assert torch.equal(state.tensor_data.K_tlia, control.K_tlia)
    assert torch.equal(state.tensor_data.V_tlia, control.V_tlia)
    assert state.trial_info["n_keys"] == 0
    assert state.trial_info["n_values"] == 0
