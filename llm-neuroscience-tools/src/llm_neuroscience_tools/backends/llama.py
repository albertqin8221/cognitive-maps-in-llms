"""Llama backend implementation."""

from .base import HFDecoderBackend


class LlamaBackend(HFDecoderBackend):
    def _get_layers(self):
        return self.model.model.layers

