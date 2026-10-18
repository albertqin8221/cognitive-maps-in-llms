from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cognitive_maps_in_llms.tasks.recording import (
    ACTIVATION_FILENAME,
    KEY_CACHE_FILENAME,
    METADATA_FILENAME,
    VALUE_CACHE_FILENAME,
    record_activations_and_kv,
)


pytestmark = pytest.mark.codex


class FakePromptRuntime:
    def __init__(self) -> None:
        self.schema = SimpleNamespace(
            context_entity_symbols=("A", "B", "C"),
            context_relation_symbols=("R_AB", "R_BC"),
        )
        self.landmarks = {
            "CTXT_START": 1,
            "A1": 1,
            "B1": 2,
            "C1": 3,
            "CTXT_END": 3,
            "QE": 4,
        }
        self.sampled_seeds: list[int] = []
        self.rendered_contexts: list[
            tuple[dict[str, str], dict[str, str]]
        ] = []

    def sample_entities(self, seed: int) -> dict[str, str]:
        self.sampled_seeds.append(seed)
        return {symbol: f"{symbol}{seed}" for symbol in ("A", "B", "C")}

    def render_prompt_as_tokens(
        self,
        context_entities: dict[str, str],
        context_relations: dict[str, str],
    ) -> list[int]:
        self.rendered_contexts.append(
            (dict(context_entities), dict(context_relations))
        )
        return [0, 1, 2, 3, 4]


class FakeRecordingBackend:
    def __init__(self) -> None:
        self.n_calls = 0

    def forward_with_activation_and_KV_recording(self, prompt_ids: list[int]):
        self.n_calls += 1
        value = float(self.n_calls)
        activations = torch.full((len(prompt_ids), 2, 3), value)
        cache = SimpleNamespace(
            K_tlia=torch.full((len(prompt_ids), 2, 1, 2), value),
            V_tlia=torch.full((len(prompt_ids), 2, 1, 2), -value),
        )
        return activations, cache, np.zeros(5)


class FailingRecordingBackend:
    def forward_with_activation_and_KV_recording(self, prompt_ids: list[int]):
        raise RuntimeError("model pass failed")


def test_record_activations_and_kv_writes_expected_artifacts(tmp_path) -> None:
    backend = FakeRecordingBackend()
    prompt_runtime = FakePromptRuntime()
    output_dir = tmp_path / "recordings"

    record_activations_and_kv(
        backend,
        prompt_runtime,
        output_dir,
        n_trials=2,
        start_seed=10,
    )

    activations = np.load(output_dir / ACTIVATION_FILENAME)
    keys = np.load(output_dir / KEY_CACHE_FILENAME)
    values = np.load(output_dir / VALUE_CACHE_FILENAME)
    metadata = json.loads((output_dir / METADATA_FILENAME).read_text())

    assert activations.shape == (4, 2, 3, 2, 3)
    assert keys.shape == values.shape == (4, 2, 3, 2, 1, 2)
    assert activations.dtype == keys.dtype == values.dtype == np.float32
    assert backend.n_calls == 8
    assert prompt_runtime.sampled_seeds == [10, 11]
    assert all(
        context == prompt_runtime.rendered_contexts[0][0]
        for context, _ in prompt_runtime.rendered_contexts[:4]
    )
    assert all(
        context == prompt_runtime.rendered_contexts[4][0]
        for context, _ in prompt_runtime.rendered_contexts[4:]
    )
    assert metadata["start_seed"] == 10
    assert metadata["n_trials"] == 2
    assert metadata["context_landmarks"]["CTXT_START"] == 0
    assert metadata["context_landmarks"]["CTXT_END"] == 2


def test_recording_metadata_is_written_before_first_model_pass(tmp_path) -> None:
    output_dir = tmp_path / "recordings"

    with pytest.raises(RuntimeError, match="model pass failed"):
        record_activations_and_kv(
            FailingRecordingBackend(),
            FakePromptRuntime(),
            output_dir,
            n_trials=1,
            start_seed=10,
        )

    metadata = json.loads((output_dir / METADATA_FILENAME).read_text())
    assert metadata["start_seed"] == 10
    assert len(metadata["relation_patterns"]) == 4
    assert not (output_dir / ACTIVATION_FILENAME).exists()


def test_record_activations_and_kv_refuses_to_overwrite(tmp_path) -> None:
    output_dir = tmp_path / "recordings"
    output_dir.mkdir()
    (output_dir / ACTIVATION_FILENAME).touch()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        record_activations_and_kv(
            FakeRecordingBackend(),
            FakePromptRuntime(),
            output_dir,
            n_trials=1,
            start_seed=0,
        )
