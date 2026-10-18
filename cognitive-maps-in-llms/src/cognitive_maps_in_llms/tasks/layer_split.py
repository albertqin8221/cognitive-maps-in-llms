"""Utilities for resolving fractional layer divisions."""

from __future__ import annotations

from numbers import Real


def resolve_dividing_layer(n_layers: int, fraction: float) -> int:
    """Resolve a fractional depth to an interior layer boundary using floor."""
    if n_layers < 2:
        raise ValueError(f"n_layers must be at least 2, got {n_layers}.")
    if isinstance(fraction, bool) or not isinstance(fraction, Real):
        raise TypeError("dividing_layer_fraction must be a number.")
    if not 0 < fraction < 1:
        raise ValueError("dividing_layer_fraction must fall within (0, 1).")

    dividing_layer = int(n_layers * float(fraction))
    if not 0 < dividing_layer < n_layers:
        raise ValueError(
            "dividing_layer_fraction resolves outside the interior layer "
            f"boundaries: n_layers={n_layers}, fraction={fraction}."
        )
    return dividing_layer
