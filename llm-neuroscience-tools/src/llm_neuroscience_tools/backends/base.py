"""Base backend interfaces and shared backend abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, cast

import numpy as np
import torch

from .hooks import Recorder, Patcher
from .KV_patching import CacheData

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache


class LMBackend(ABC):
    """Abstract interface for running an LM with optional interventions.

    Concrete backends are expected to expose a small shared surface:
    - load a backend from a model path,
    - run an ordinary next-token forward pass,
    - optionally, record and patch activations,
    - optionally, record and patch the KV cache.
    """

    @classmethod
    @abstractmethod
    def from_model_path(cls, model_path: str, **kwargs: Any) -> "LMBackend":
        """Construct a backend from a local model path."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, prompt_ids: list[int]) -> np.ndarray:
        """Run an unperturbed forward pass on a prompt.

        Args:
            prompt_ids: Token ids for a single prompt.

        Returns:
            Final-token logits as a numpy array with shape ``[vocab_size]``.
        """
        raise NotImplementedError

    def forward_with_recording(
            self,                    
            prompt_ids: list[int]
        ) -> tuple[torch.Tensor, np.ndarray]:
        """Run a forward pass while recording residual activations.

        Args:
            prompt_ids: Token ids for a single prompt.

        Returns:
            Tuple ``(recorded_activations_tla, final_token_logits)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__ } does not support activation recording."
        )

    def forward_with_patching(
            self,
            prompt_ids: list[int],
            Z_tla: torch.Tensor,
            mask_tla: torch.Tensor,
        ) -> tuple[torch.Tensor, np.ndarray]:
        """Run a forward pass while patching residual activations.

        Args:
            prompt_ids: Token ids for a single prompt.
            Z_tla: Patch values with shape ``[T_context, N_layers, D_model]``.
            mask_tla: Boolean patch mask with the same shape as ``Z_tla``.

        Returns:
            Tuple ``(recorded_activations_tla, final_token_logits)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__ } does not support activation patching."
        )

    def forward_with_KV_recording(
            self, 
            prompt_ids: list[int]
        ) -> tuple[CacheData, np.ndarray]:
        """Run a forward pass while recording the KV cache.

        Args:
            prompt_ids: Token ids for a single prompt.

        Returns:
            Tuple ``(recorded_cache_data, final_token_logits)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__ } does not support KV cache recording."
        )
    
    def forward_with_KV_patching(
            self,
            prompt_ids: list[int],
            cache_data: CacheData,
            patch_end: int
        ) -> tuple[CacheData, np.ndarray]:
        """Run a forward pass while patching cached keys and values. Keys and 
        values are patched for token positions in the range ``[0, patch_end]``.

        Args:
            prompt_ids: Token ids for a single prompt.
            cache_data: Source cache containing patched key/value activations.
                Must contain activations for at least ``patch_end`` tokens.
            patch_end: Inclusive upper bound on token positions to patch.

        Returns:
            Tuple ``(recorded_cache_data, final_token_logits)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__ } does not support KV cache patching."
        )

    def forward_with_differentiable_KV_patching(
        self,
        prompt_ids: list[int],
        cache_data: CacheData,
        patch_end: int
    ) -> tuple[CacheData, torch.Tensor]:
        """Runs a forward pass using a KV cache built from ``cache_data``.
        In contrast with the standard method for KV patching, this method 
        preserves the autograd graph.

        Args:
            prompt_ids: Token ids for a single prompt.
            cache_data: Source cache containing key/value activations for the
                prompt prefix.
            patch_end: Inclusive upper bound on token positions to include
                when building the initial KV cache.

        Returns:
            Tuple ``(recorded_cache_data, final_token_logits)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__ } does not support simultaneous activation and KV recording."
        )

    def forward_with_activation_and_KV_recording(self, 
                                               prompt_ids: list[int]
                                ) -> tuple[torch.Tensor, CacheData, np.ndarray]:
        """Run a forward pass while recording the KV cache *and* the residual 
        stream. 

        Args:
            prompt_ids: Token ids for a single prompt.

        Returns:
            Tuple ``(recorded_activations_tla, recorded_cache_data, 
                     final_token_logits)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__ } does not support simultaneous activation and KV recording."
        )


class HFDecoderBackend(LMBackend):

    # -------- Model instantiation --------

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict | None = "auto",
        **kwargs: Any
    ) -> HFDecoderBackend:
        """Alternate constructor that loads the model and tokenizer from a set 
        of files on disk.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only = True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map       = device_map,
            dtype            = torch_dtype,
            local_files_only = True,
        )
        return cls(model, tokenizer)
    

    # -------- Some small helpers --------

    def _get_layers(self) -> Sequence[Any]:
        """Returns the ordered sequence of transformer blocks to hook. 

        Subclasses should implement this to expose the modules whose input 
        residual stream will be recorded and/or patched.
        """
        raise NotImplementedError
    
    def _get_storage_device(self) -> torch.device:
        """Returns the device that should store the recorded activations. 
        
        By default, the recorded activaitons are stored on the input device; 
        however, subclasses can override this method for more sophisticated 
        behavior. 
        """
        return self.model.get_input_embeddings().weight.device

    def _list_to_tensor(self, prompt_ids: list[int]) -> torch.Tensor:
        """Convert prompt token ids into a batch-size-1 input tensor.

        The returned tensor has shape ``[1, T_context]`` and is placed on the
        device of the model's input embedding layer.
        """
        device = self.model.get_input_embeddings().weight.device
        return torch.tensor(prompt_ids, device=device)[None, :]
    
    def _get_logits(self, output: CausalLMOutputWithPast) -> torch.Tensor:
        """Extract the logits tensor from a Hugging Face causal-LM output.
        This helper is mostly to satisfy the typechecker. 
        """
        logits = output.logits
        if logits is None:
            raise RuntimeError("Model returned no logits.")
        return cast(torch.Tensor, logits)
    

    # -------- Forward pass --------

    def _forward_verbose(self, 
                         prompt_ids: list[int]
                         ) -> CausalLMOutputWithPast:
        """Run a forward pass and return the full HF output object.

        Args:
            prompt_ids: Prompt token ids for one sequence.

        Returns:
            ``CausalLMOutputWithPast`` from the model call.
        """
        prompt_ids_tensor = self._list_to_tensor(prompt_ids)
        with torch.no_grad():
            output = self.model(prompt_ids_tensor)
        return output
    
    def forward(self, 
                prompt_ids: list[int]
                ) -> np.ndarray:
        """
        Run a forward pass and returns logits at the final position.

        Args:
            prompt_ids: Prompt token ids.

        Returns:
            Final-token logits as numpy array `[vocab_size]`, `float32`.
        """
        output = self._forward_verbose(prompt_ids)
        logits = self._get_logits(output)
        return (
            logits
            .to(torch.float32)
            .cpu()
            .numpy()[0, -1, :]
        )
    

    # -------- Vanilla activation patching --------

    def forward_with_recording(self, 
                               prompt_ids: list[int]
                               ) -> tuple[torch.Tensor, np.ndarray]:
        """Run a forward pass while recording residual activations.

        Args:
            prompt_ids: Prompt token ids.

        Returns:
            Tuple ``(recorded_activations_tla, final_token_logits)``.
        """

        # Run a forward pass with a recorder
        with Recorder(
            layers         = self._get_layers(),
            storage_device = self._get_storage_device()
        ) as rec:
            with torch.no_grad():
                logits = self.forward(prompt_ids)
                Z_btla = rec.get_activations()

        # Retrieve the activations
        return Z_btla[0, ...], logits

    def forward_with_patching(self, 
                              prompt_ids: list[int], 
                              Z_tla: torch.Tensor,
                              mask_tla: torch.Tensor
                              ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Run a patched forward pass.

        On the forward pass, patches in the source tensor `Z_tla` at 
        positions/layers/residual dimensions where `mask_tla` is True.

        Args:
            prompt_ids: Prompt token ids.
            Z_tla: Patching values with shape `[T_context, N_layers, D_model]`.
            mask_tla: Patch mask with same shape as `Z_tla`.

        Returns:
            Tuple `(recorded_Z_tla, output)`.

        Raises:
            AssertionError: If `Z_tla`/`mask_tla` do not match expected shape.
        """

        # Sanity checks for the input format
        assert Z_tla.shape == mask_tla.shape, (
            'Z_tla and mask_tla must have the same shape; got:'
            f'Z_tla shape: {Z_tla.shape}'
            f'mask_tla shape: {mask_tla.shape}'
        )
        assert mask_tla.dtype == torch.bool, (
            f'mask_tla must be a boolean array; got: {mask_tla.dtype}'
        )

        with Patcher(
            layers               = self._get_layers(),
            patching_values_btla = Z_tla[None, ...],
            patching_mask_btla   = mask_tla[None, ...]
        ):
            return self.forward_with_recording(prompt_ids)
        

    # -------- KV patching --------

    def forward_with_KV_recording(self, 
                                  prompt_ids: list[int]
                                  ) -> tuple[CacheData, np.ndarray]:
        """Run a forward pass while recording the KV cache.

        Args:
            prompt_ids: Token ids for a single prompt.

        Returns:
            Tuple ``(recorded_cache_data, final_token_logits)``.
        """
        cache = DynamicCache()
        prompt_ids_tensor = self._list_to_tensor(prompt_ids)
        with torch.no_grad():
            output = self.model(prompt_ids_tensor, past_key_values=cache)
            logits = (self._get_logits(output).to(torch.float32)
                      .cpu()
                      .numpy()[0, -1, :])
            cache_data = CacheData.from_dynamic_cache(
                dynamic_cache  = cache, 
                storage_device = self._get_storage_device()
            )
        return cache_data, logits
    
    def forward_with_KV_patching(
        self,
        prompt_ids: list[int],
        cache_data: CacheData,
        patch_end: int
    ) -> tuple[CacheData, np.ndarray]:
        """Run a forward pass using a KV cache built from ``cache_data``.

        Args:
            prompt_ids: Token ids for a single prompt.
            cache_data: Source cache containing key/value activations for the
                prompt prefix.
            patch_end: Inclusive upper bound on token positions to include
                when building the initial KV cache.

        Returns:
            Tuple ``(recorded_cache_data, final_token_logits)``.
        """

        # We build a DynamicCache in the token range [0, patch_end], then 
        # truncate the prompt to (patch_end, prompt_end]
        cache = cache_data.to_dynamic_cache(max_token_idx=patch_end)
        truncated_prompt_ids = prompt_ids[patch_end+1:]
        truncated_prompt_ids_tensor = self._list_to_tensor(truncated_prompt_ids)

        # Now, run a forward pass on the remaining tokens
        with torch.no_grad():
            output = self.model(truncated_prompt_ids_tensor, 
                                past_key_values=cache)
            logits = (self._get_logits(output).to(torch.float32)
                      .cpu()
                      .numpy()[0, -1, :])
            cache_data = CacheData.from_dynamic_cache(
                dynamic_cache  = cache, 
                storage_device = self._get_storage_device()
            )
        return cache_data, logits

    def forward_with_differentiable_KV_patching(
        self,
        prompt_ids: list[int],
        cache_data: CacheData,
        patch_end: int
    ) -> tuple[CacheData, torch.Tensor]:
        """Runs a forward pass using a KV cache built from ``cache_data``.
        In contrast with the standard method for KV patching, this method 
        preserves the autograd graph.

        Args:
            prompt_ids: Token ids for a single prompt.
            cache_data: Source cache containing key/value activations for the
                prompt prefix.
            patch_end: Inclusive upper bound on token positions to include
                when building the initial KV cache.

        Returns:
            Tuple ``(recorded_cache_data, final_token_logits)``.
        """

        # We build a DynamicCache in the token range [0, patch_end], then 
        # truncate the prompt to (patch_end, prompt_end]
        cache = cache_data.to_dynamic_cache(max_token_idx=patch_end)
        truncated_prompt_ids = prompt_ids[patch_end+1:]
        truncated_prompt_ids_tensor = self._list_to_tensor(truncated_prompt_ids)

        # Now, run a forward pass on the remaining tokens
        output = self.model(truncated_prompt_ids_tensor, 
                            past_key_values=cache)
        logits = self._get_logits(output)
        cache_data = CacheData.from_dynamic_cache(
            dynamic_cache  = cache, 
            storage_device = self._get_storage_device()
        )
        return cache_data, logits

    def forward_with_activation_and_KV_recording(self, 
                                               prompt_ids: list[int]
                                ) -> tuple[torch.Tensor, CacheData, np.ndarray]:
        """Run a forward pass while recording the KV cache *and* the residual 
        stream. 

        Args:
            prompt_ids: Token ids for a single prompt.

        Returns:
            Tuple ``(recorded_activations_tla, recorded_cache_data, 
                     final_token_logits)``.
        """
        cache = DynamicCache()
        prompt_ids_tensor = self._list_to_tensor(prompt_ids)
        with Recorder(
            layers         = self._get_layers(),
            storage_device = self._get_storage_device()
        ) as rec:
            with torch.no_grad():
                output = self.model(prompt_ids_tensor, past_key_values=cache)
                logits = (self._get_logits(output).to(torch.float32)
                        .cpu()
                        .numpy()[0, -1, :])
                cache_data = CacheData.from_dynamic_cache(
                    dynamic_cache  = cache, 
                    storage_device = self._get_storage_device()
                )
                Z_btla = rec.get_activations()
        return Z_btla[0, ...], cache_data, logits
