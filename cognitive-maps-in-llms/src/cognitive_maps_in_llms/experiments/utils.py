"""General helpers shared by experiment implementations."""

from __future__ import annotations

from typing import Mapping, Any

import numpy as np
import torch

from llm_neuroscience_tools.backends.base import LMBackend
from llm_neuroscience_tools.prompts.relational import RelationalPromptRuntime


def build_context_patching_mask(
    backend: LMBackend,
    prompt_runtime: RelationalPromptRuntime,
) -> torch.Tensor:
    """Build a mask selecting every activation in the context token span."""

    # Record activity on a dummy prompt
    entities = prompt_runtime.sample_entities(seed=0)
    prompt = prompt_runtime.render_prompt_as_tokens(
        context_entities=entities,
        context_relations={"R_AB": "FWD", "R_BC": "FWD"},
    )
    activations, _ = backend.forward_with_recording(prompt)

    context_start = prompt_runtime.landmarks["CTXT_START"]
    context_end = prompt_runtime.landmarks["CTXT_END"]
    mask = torch.zeros_like(activations, dtype=torch.bool)
    mask[context_start:context_end + 1, ...] = True
    return mask


def get_single_token_logit(token: str, logits: np.ndarray, tokenizer) -> float:
    """Return the logit for `token`.

    Args:
        token: The string whose token logit should be retrieved.
        logits: The model logits for the next-token distribution.
        tokenizer: The tokenizer used to map `token` to token ids.

    Returns:
        The scalar logit corresponding to `token`.

    Raises:
        ValueError: If `token` tokenizes to zero or multiple token ids.
    """
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Expected {token!r} to tokenize to exactly 1 token, got {len(token_ids)}: {token_ids}"
        )
    return float(logits[token_ids[0]])


def get_answer_token_logits_and_probs(
    answer_tokens: Mapping[str, str],
    logits: np.ndarray,
    tokenizer: Any,
) -> dict[str, dict[str, float]]:
    """Return full-vocabulary logits and probabilities for answer tokens.

    Args:
        answer_tokens: Mapping from answer symbols to their token strings.
        logits: Model logits for the next-token distribution.
        tokenizer: Tokenizer used to map each answer string to one token ID.

    Returns:
        A dictionary containing symbol-keyed ``logits`` and ``probs`` mappings.
        Probabilities are normalized across the model's full vocabulary.
    """
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return {
        "logits": {
            symbol: get_single_token_logit(token, logits, tokenizer)
            for symbol, token in answer_tokens.items()
        },
        "probs": {
            symbol: get_single_token_logit(token, probs, tokenizer)
            for symbol, token in answer_tokens.items()
        },
    }
