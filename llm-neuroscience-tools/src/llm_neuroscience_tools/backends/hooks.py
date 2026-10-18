"""Shared hook utilities for recording and patching model activations."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch


# Copied over the Recorder class from:
# llm-neuroscience-tools/src/llm_neuroscience_tools/backends/hooks.py
# We can play around with getting this to work with multiple GPUs

from typing import Sequence, Callable, Any


class Recorder:
    """A context manager that records the contents of the residual stream
    at each layer.

    On entry, hooks are attached to the provided sequence of layers. These 
    hooks write activations into a preallocated tensor. On exit, all hooks are
    removed.
    """

    def __init__(
        self,
        layers: Sequence[Any],
        storage_device: torch.device | str
    ) -> None:
        """Store the layer sequence and output tensor specification.

        Args:
            layers: Sequence of model layers to hook.
            storage_device: Device that holds the recorded activations.
        """
        self.layers           = layers
        self.storage_device   = storage_device
        self.activations_btla: torch.Tensor
        self.handles: list[Any]

    def __enter__(self) -> "Recorder":
        """Initialize empty activation tensor and hook the model.
        """
        self.handles = []
        self.handles.append(self.layers[0].register_forward_pre_hook(
            self._get_preallocation_hook()
        ))
        for l, layer in enumerate(self.layers):
            handle = layer.register_forward_pre_hook(
                self._get_layer_hook(l)
            )
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Remove all hooks.
        """
        for handle in self.handles:
            handle.remove()

    def get_activations(self) -> torch.Tensor:
        """Return recorded activations as as a torch Tensor with shape:
          ``[batch_size, T_context, N_layers, D_model]``.
        """
        return self.activations_btla

    def _get_layer_hook(self, layer_index: int) -> Callable[..., Any]:
        """Returns a hook that dumps the inputs of a module into the lth slice 
        of the recorded activation tensor Z_tla.
        """
        def hook(module, inp):
            self.activations_btla[:, :, layer_index, :] = inp[0].detach().to(
                device = self.storage_device, 
                dtype  = self.activations_btla.dtype
            )
        return hook

    def _get_preallocation_hook(self) -> Callable[..., Any]:
        """Returns a hook that reads the shape and dtype of the input, 
        then allocates a matching tensor to hold the activations.
        """
        def hook(module, inp):
            batch_size = inp[0].shape[0]
            t_context  = inp[0].shape[1]
            d_model    = inp[0].shape[2]
            self.activations_btla = torch.zeros(
                size   = (batch_size, t_context, len(self.layers), d_model),
                dtype  = inp[0].dtype,
                device = self.storage_device,
            )
        return hook


class Patcher:
    """A context manager that writes into the residual stream at each layer. 

    On entry, hooks are attached to the provided layer sequence. During the
    forward pass, entries selected by ``patching_mask_btla`` are replaced by the
    corresponding values in ``patching_values_btla``. On exit, all hooks are
    removed.
    """

    def __init__(
        self,
        layers: Sequence[Any],
        patching_values_btla: torch.Tensor,
        patching_mask_btla:   torch.Tensor,
    ) -> None:
        """Store the layer sequence and patch tensors.

        Args:
            layers: Sequence of model layers to hook.
            patching_values_btla: Replacement activations with shape ``[B, T, L, A]``.
            patching_mask_btla: Boolean mask selecting which activations to replace.
            device: Device on which patch tensors should live.
            dtype: Optional dtype used for the patch values tensor.
        """
        self.layers = layers
        self.patching_values_btla = patching_values_btla
        self.patching_mask_btla   = patching_mask_btla
        self.handles: list[Any]

    def __enter__(self) -> "Patcher":
        """Hook the model.
        """
        self.handles = []
        for layer_index, layer in enumerate(self.layers):
            handle = layer.register_forward_pre_hook(
                self._get_layer_hook(layer_index)
            )
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Remove all hooks.
        """
        for handle in self.handles:
            handle.remove()

    def _get_layer_hook(self, layer_index: int) -> Callable[..., Any]:
        """Returns a hook that dumps the inputs of a module into the lth slice 
        of the recorded activation tensor Z_tla.
        """
        def hook(module, inp):
            z_bta = inp[0]
            z_bta = torch.where(
                self.patching_mask_btla[:, :, layer_index, :].to(z_bta.device),
                self.patching_values_btla[:, :, layer_index, :].to(z_bta.device),
                z_bta,
            )
            return z_bta

        return hook
