"""Qwen backend implementation."""

from .base import HFDecoderBackend


class QwenBackend(HFDecoderBackend):
    def _get_layers(self):
        return self.model.model.layers

