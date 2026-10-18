"""Gemma backend implementation."""

from .base import HFDecoderBackend


class GemmaBackend(HFDecoderBackend):
    def _get_layers(self):
        return self.model.model.language_model.layers

