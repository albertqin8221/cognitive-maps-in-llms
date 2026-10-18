from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest

from cognitive_maps_in_llms.analysis.utils import get_fwd_fwd_context_tokens


pytestmark = pytest.mark.codex


def test_context_landmarks_use_single_guillemets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePromptRuntime:
        landmarks = {
            "CTXT_START": 0,
            "CTXT_END": 4,
            "A1": 0,
            "B1": 1,
            "B2": 2,
            "C1": 3,
            "D1": 4,
        }

        def sample_entities(self, *, seed: int) -> dict[str, str]:
            assert seed == 0
            return {}

        def render_prompt_as_tokens(self, **_: object) -> list[int]:
            return [0, 1, 2, 3, 4]

    tokenizer = SimpleNamespace(decode=lambda token_ids: f" token{token_ids[0]}")
    runtime = SimpleNamespace(
        prompt_runtime=FakePromptRuntime(),
        backend=SimpleNamespace(tokenizer=tokenizer),
    )
    presets = ModuleType("llm_neuroscience_tools.presets")
    presets.build_preset_runtime = lambda **_: runtime  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llm_neuroscience_tools.presets", presets)

    labels = get_fwd_fwd_context_tokens("model", "prompt")

    assert labels == [" ‹A›", " ‹B›", " ‹B›", " ‹C›", " ‹D›"]
