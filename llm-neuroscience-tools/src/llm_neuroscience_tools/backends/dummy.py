"""Dummy backend implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import LMBackend


class DummyBackend(LMBackend):
    """Dummy backend for running a language model with interventions.

    Each method returns random tensors of the correct shape.
    """

    @classmethod
    def from_model_path(cls, model_path: str, **kwargs: Any) -> "DummyBackend":
        """Dummy constructor matching the shared backend interface.

        This backend does not load a real model from ``model_path``. Callers
        must therefore provide the tokenizer and shape information explicitly.
        """
        try:
            tokenizer  = kwargs["tokenizer"]
            D_model    = kwargs["D_model"]
            N_layers   = kwargs["N_layers"]
            vocab_size = kwargs["N_vocab"]
        except KeyError as exc:
            raise ValueError(
                "DummyBackend.from_model_path requires tokenizer, D_model, and N_layers."
            ) from exc

        vocab_size = kwargs.get("vocab_size")
        return cls(
            tokenizer=tokenizer,
            D_model=D_model,
            N_layers=N_layers,
            vocab_size=vocab_size,
        )

    def __init__(
        self,
        tokenizer,
        D_model: int,
        N_layers: int,
        vocab_size: int | None = None,
    ) -> None:
        self.D_model = D_model
        self.N_layers = N_layers
        self.device = "cpu"
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size if vocab_size is None else vocab_size

    def forward(self, prompt_ids: list[int]) -> np.ndarray:
        """Return random final-token logits."""
        rng = np.random.default_rng(seed=sum(prompt_ids))
        return rng.uniform(0, 100, size=self.vocab_size)

    def forward_with_recording(
            self, 
            prompt_ids: list[int]
        ) -> tuple[torch.Tensor, np.ndarray]:
        """Return random activations and final-token logits."""
        Z_tla_dummy = self._get_random_activations(
            T_context = len(prompt_ids),
            seed = sum(prompt_ids)
        )
        logits = self.forward(prompt_ids)
        return Z_tla_dummy, logits

    def forward_with_patching(
        self,
        prompt_ids: list[int],
        Z_tla: torch.Tensor,
        mask_tla: torch.Tensor,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Return random activations and final-token logits."""
        Z_tla_dummy = self._get_random_activations(
            T_context = len(prompt_ids),
            seed = sum(prompt_ids)
        )
        logits = self.forward(prompt_ids)
        return Z_tla_dummy, logits

    def _get_random_activations(self, T_context, seed) -> torch.Tensor:
        """Return random activations with shape ``[T, L, A]``."""
        rng = np.random.default_rng(seed=seed)
        Z_tla = rng.uniform(-10, 10, size=(T_context, self.N_layers, self.D_model))
        return torch.tensor(Z_tla, dtype=torch.bfloat16, device=self.device)

