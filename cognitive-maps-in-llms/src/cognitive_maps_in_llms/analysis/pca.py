"""CPU PCA and trajectory plots for context activation recordings."""
from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from .style import EXPORT_RC_PARAMS, PAPER_MOSAIC_STYLE


RELATION_PATTERN_LABELS = {
    "A_fwd_B_fwd_C": "A -> B -> C",
    "A_rev_B_fwd_C": "A <- B -> C",
    "A_fwd_B_rev_C": "A -> B <- C",
    "A_rev_B_rev_C": "A <- B <- C",
    "A_fwd_B__C_fwd_D": "A -> B; C -> D",
    "A_rev_B__C_fwd_D": "A <- B; C -> D",
    "A_fwd_B__C_rev_D": "A -> B; C <- D",
    "A_rev_B__C_rev_D": "A <- B; C <- D",
}
RELATION_PATTERN_COLORS = {
    "A_fwd_B_fwd_C": "tab:red",
    "A_rev_B_fwd_C": "tab:orange",
    "A_fwd_B_rev_C": "tab:green",
    "A_rev_B_rev_C": "tab:blue",
    "A_fwd_B__C_fwd_D": "tab:red",
    "A_rev_B__C_fwd_D": "tab:orange",
    "A_fwd_B__C_rev_D": "tab:green",
    "A_rev_B__C_rev_D": "tab:blue",
}

@dataclass(frozen=True)
class ActivationRecordings:
    """Read-only context activations with axes graph, seed, token, layer, feature."""

    Z_gstla: np.ndarray
    metadata: Mapping[str, Any]
    directory: Path

    @property
    def context_landmarks(self) -> dict[str, int]:
        return dict(self.metadata["context_landmarks"])


def load_activation_recordings(directory: Path | str) -> ActivationRecordings:
    """Memory-map a completed recording task without loading model weights."""
    directory = Path(directory)
    # Recording metadata is written before the arrays; it alone cannot certify
    # that every graph/seed slice was recorded successfully.
    with (directory / "completed.json").open(encoding="utf-8") as handle:
        if not isinstance(json.load(handle), dict):
            raise ValueError("Recording completion marker must be a JSON object.")
    with (directory / "recordings_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_axes = ["graph_structure", "seed", "token", "layer", "activation"]
    if (
        not isinstance(metadata, dict)
        or metadata.get("activation_axes") != expected_axes
    ):
        raise ValueError(f"Expected activation_axes={expected_axes}.")
    activations = np.load(directory / "Z_gstla.npy", mmap_mode="r", allow_pickle=False)
    if activations.ndim != 5 or any(size == 0 for size in activations.shape):
        raise ValueError("Activations must have five nonempty axes.")
    if not np.issubdtype(activations.dtype, np.floating):
        raise ValueError("Activations must have a floating-point dtype.")
    patterns = metadata.get("relation_patterns", [])
    if len(patterns) != activations.shape[0] or any(
        not isinstance(pattern, dict) or not isinstance(pattern.get("name"), str)
        for pattern in patterns
    ):
        raise ValueError("relation_patterns must label every graph structure.")
    if metadata.get("n_trials") != activations.shape[1]:
        raise ValueError("n_trials does not match the recorded seed axis.")
    landmarks = metadata.get("context_landmarks")
    if not isinstance(landmarks, dict):
        raise ValueError("Recordings must provide context_landmarks.")
    for name, index in landmarks.items():
        _index(index, activations.shape[2], f"landmark {name!r}")
    return ActivationRecordings(activations, metadata, directory)


@dataclass(frozen=True)
class PCAProjection:
    """Fixed-basis projection, centered separately at each recorded layer."""

    Y_gslq: np.ndarray
    Y_glq: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    means_la: np.ndarray
    metadata: Mapping[str, Any]
    token: str | int
    token_idx: int
    reference_layer: int
    random_state: int
    average_over_seeds: bool = False


def _index(value: int, size: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= value < size
    ):
        raise ValueError(f"{name} must be an integer in [0, {size}); got {value!r}.")
    return int(value)


def compute_pca_projection(
    recordings: ActivationRecordings,
    *,
    token: str | int,
    reference_layer: int,
    n_components: int = 3,
    random_state: int = 0,
    average_over_seeds: bool = False,
) -> PCAProjection:
    """Fit graph/seed PCA at one token/layer, then project that token at all layers.

    Uses the legacy sklearn ``auto`` solver and default random state of zero.
    Each projected layer is centered by its own graph/seed mean, not the mean
    at the reference layer. Only one layer's activation slice is copied at once.
    With ``average_over_seeds=True``, fit the basis to graph-structure means
    over seeds; individual seeds are still projected at every layer. This mode
    permits at most ``min(n_graphs - 1, n_features)`` components.
    No artifacts are written.
    """
    from sklearn.decomposition import PCA

    Z = recordings.Z_gstla
    g, s, t, l, a = Z.shape
    if isinstance(token, str):
        if token not in recordings.context_landmarks:
            raise ValueError(
                f"Unknown token landmark {token!r}; available: {tuple(recordings.context_landmarks)}"
            )
        token_idx = recordings.context_landmarks[token]
    else:
        token_idx = token
    token_idx = _index(token_idx, t, "token")
    reference_layer = _index(reference_layer, l, "reference_layer")
    if not isinstance(average_over_seeds, bool):
        raise ValueError("average_over_seeds must be a bool.")
    n_samples = g if average_over_seeds else g * s
    max_components = min(g - 1, a) if average_over_seeds else min(g * s, a)
    n_components = _index(n_components, max_components + 1, "n_components")
    if n_components == 0 or n_samples < 2:
        raise ValueError("PCA requires positive n_components and at least two samples.")
    random_state = _index(random_state, 2**32, "random_state")
    X = np.array(Z[:, :, token_idx, reference_layer, :], dtype=np.float32).reshape(
        g * s, a
    )
    if not np.isfinite(X).all():
        raise ValueError("Reference-layer activations contain non-finite values.")
    if average_over_seeds:
        X = X.reshape(g, s, a).mean(axis=1, dtype=np.float64).astype(np.float32)
    if not np.any(X.var(axis=0, dtype=np.float64) > 0):
        raise ValueError(
            "PCA is undefined: reference-layer fitting samples have zero variance."
        )
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=random_state)
    pca.fit(X)
    del X
    means = np.empty((l, a), dtype=np.float32)
    Y = np.empty((g, s, l, n_components), dtype=np.float32)
    for layer in range(l):
        X = np.array(Z[:, :, token_idx, layer, :], dtype=np.float32).reshape(g * s, a)
        if not np.isfinite(X).all():
            raise ValueError(f"Activations contain non-finite values at layer {layer}.")
        means[layer] = X.mean(axis=0, dtype=np.float64)
        X -= means[layer]
        Y[:, :, layer, :] = (X @ pca.components_.T).reshape(g, s, n_components)
        del X
    return PCAProjection(
        Y,
        Y.mean(axis=1),
        pca.components_,
        pca.explained_variance_,
        pca.explained_variance_ratio_,
        means,
        recordings.metadata,
        token,
        token_idx,
        reference_layer,
        random_state,
        average_over_seeds,
    )


class _SavePlot:
    fig: Figure

    def save(
        self,
        path: Path | str,
        *,
        format: str | None = None,
        dpi: int = 300,
        bbox_inches: str | None = "tight",
        transparent: bool = False,
    ) -> None:
        with mpl.rc_context(EXPORT_RC_PARAMS):
            self.fig.savefig(
                path,
                format=format,
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
            )


@dataclass(frozen=True)
class PCAActivitySpacePlot(_SavePlot):
    fig: Figure
    ax: Axes
    projection: PCAProjection
    artists: dict[str, Any]


@dataclass(frozen=True)
class PCALayerTrajectoriesPlot(_SavePlot):
    fig: Figure
    axes: np.ndarray
    projection: PCAProjection
    artists: dict[str, Any]


def _plot_selection(
    projection: PCAProjection,
    components: Sequence[int],
    layer_range: tuple[int, int] | None,
) -> tuple[list[int], slice, np.ndarray]:
    components = tuple(components)
    if not components or len(set(components)) != len(components):
        raise ValueError("components must be nonempty and distinct.")
    indices = [
        _index(pc, projection.Y_gslq.shape[-1] + 1, "component") - 1
        for pc in components
    ]
    if min(indices) < 0:
        raise ValueError("Component numbers start at 1.")
    n_layers = projection.Y_gslq.shape[2]
    start, stop = (0, n_layers) if layer_range is None else layer_range
    start = _index(start, n_layers, "layer_range start")
    stop = _index(stop, n_layers + 1, "layer_range stop")
    if stop <= start:
        raise ValueError("layer_range must have start < stop.")
    return indices, slice(start, stop), np.arange(start, stop)


def _groups(
    projection: PCAProjection,
    colors: Mapping[str, Any] | Sequence[Any] | None,
):
    names = [p["name"] for p in projection.metadata["relation_patterns"]]
    if (
        colors is not None
        and not isinstance(colors, Mapping)
        and len(colors) != len(names)
    ):
        raise ValueError("colors must contain one color per graph structure.")
    for i, name in enumerate(names):
        # Current recordings use arrow labels; legacy recordings used identifiers.
        label = (
            relation_pattern_display_name(name)
            .replace("--fwd-->", "→")
            .replace("--rev-->", "←")
        )
        color = (
            relation_pattern_color(name, i)
            if colors is None
            else (colors[name] if isinstance(colors, Mapping) else colors[i])
        )
        yield i, label, color


def plot_pca_activity_space(
    projection: PCAProjection,
    *,
    components: Sequence[int] = (1, 2),
    show_seed_trajectories: bool = True,
    layer_range: tuple[int, int] | None = None,
    marker: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (PAPER_MOSAIC_STYLE.figsize[0],) * 2,
    legend: bool = False,
    colors: Mapping[str, Any] | Sequence[Any] | None = None,
) -> PCAActivitySpacePlot:
    """Plot layer trajectories in two or three selected PCs (numbered from 1)."""
    with mpl.rc_context(PAPER_MOSAIC_STYLE.rc_params()):
        indices, layers, _ = _plot_selection(projection, components, layer_range)
        ndim = len(indices)
        if ndim not in (2, 3):
            raise ValueError("Activity-space plots require two or three components.")
        if ax is not None and (getattr(ax, "name", None) == "3d") != (ndim == 3):
            raise ValueError("The supplied axes dimensionality must match components.")
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d" if ndim == 3 else None)
        else:
            fig = ax.figure
        Y = projection.Y_gslq[:, :, layers, :][..., indices]
        means = projection.Y_glq[:, layers, :][..., indices]
        artists: dict[str, Any] = {"seed_trajectories": [], "mean_trajectories": []}
        for g, label, color in _groups(projection, colors):
            if show_seed_trajectories:
                for seed in Y[g]:
                    artists["seed_trajectories"].extend(
                        ax.plot(*seed.T, color=color, alpha=0.04, linewidth=0.8)
                    )
            artists["mean_trajectories"].extend(
                ax.plot(
                    *means[g].T,
                    color=color,
                    label=label,
                    alpha=0.95,
                    linewidth=1.25,
                    marker=marker,
                )
            )
        points = means.reshape(-1, ndim)
        limits = (
            square_limits_for_xy(points) if ndim == 2 else square_limits_for_xyz(points)
        )
        for j, axis in enumerate("xyz"[:ndim]):
            getattr(ax, f"set_{axis}label")(f"PC{indices[j] + 1}")
            getattr(ax, f"set_{axis}lim")(*limits[j])
        ax.set_box_aspect(1 if ndim == 2 else (1, 1, 1))
        if title is not None:
            ax.set_title(title, fontsize=PAPER_MOSAIC_STYLE.font_size)
        artists["legend"] = (
            ax.legend(
                fontsize=PAPER_MOSAIC_STYLE.legend_font_size, frameon=False
            )
            if legend else None
        )
        if ax is fig.axes[0] and len(fig.axes) == 1:
            fig.tight_layout()
        return PCAActivitySpacePlot(fig, ax, projection, artists)


def plot_pca_layer_trajectories(
    projection: PCAProjection,
    *,
    components: Sequence[int] | None = None,
    show_seed_trajectories: bool = True,
    layer_range: tuple[int, int] | None = None,
    title: str | None = None,
    axes: Sequence[Axes] | None = None,
    figsize: tuple[float, float] | None = None,
    legend: bool = False,
    colors: Mapping[str, Any] | Sequence[Any] | None = None,
) -> PCALayerTrajectoriesPlot:
    """Plot selected PC coordinates against layer, one panel per component."""
    with mpl.rc_context(PAPER_MOSAIC_STYLE.rc_params()):
        if components is None:
            components = tuple(range(1, projection.Y_gslq.shape[-1] + 1))
        indices, layers, layer_numbers = _plot_selection(
            projection, components, layer_range
        )
        if axes is None:
            fig, axes_array = plt.subplots(
                len(indices),
                1,
                sharex=True,
                squeeze=False,
                figsize=figsize or (
                    PAPER_MOSAIC_STYLE.figsize[0],
                    1.25 * PAPER_MOSAIC_STYLE.figsize[1] * len(indices),
                ),
            )
            axes_array = axes_array.ravel()
        else:
            axes_array = np.asarray(axes, dtype=object).reshape(-1)
            if len(axes_array) != len(indices) or any(
                getattr(ax, "name", None) == "3d" for ax in axes_array
            ):
                raise ValueError("Supply one 2D axes per selected component.")
            fig = axes_array[0].figure
            if any(ax.figure is not fig for ax in axes_array):
                raise ValueError("All axes must belong to the same figure.")
        artists: dict[str, Any] = {"seed_trajectories": [], "mean_trajectories": []}
        for ax, pc in zip(axes_array, indices):
            for g, label, color in _groups(projection, colors):
                if show_seed_trajectories:
                    for seed in projection.Y_gslq[g, :, layers, pc]:
                        artists["seed_trajectories"].extend(
                            ax.plot(
                                layer_numbers, seed, color=color, alpha=0.04, linewidth=0.8
                            )
                        )
                artists["mean_trajectories"].extend(
                    ax.plot(
                        layer_numbers,
                        projection.Y_glq[g, layers, pc],
                        color=color,
                        label=label,
                        alpha=0.95,
                        linewidth=1.25,
                    )
                )
            ax.set_ylabel(f"PC{pc + 1}")
        axes_array[-1].set_xlabel("Layer")
        artists["legend"] = (
            axes_array[0].legend(
                fontsize=PAPER_MOSAIC_STYLE.legend_font_size, frameon=False
            )
            if legend else None
        )
        if title is not None:
            fig.suptitle(title, fontsize=PAPER_MOSAIC_STYLE.font_size)
        if axes is None:
            fig.tight_layout()
        return PCALayerTrajectoriesPlot(fig, axes_array, projection, artists)


def relation_pattern_display_name(name: str) -> str:
    return RELATION_PATTERN_LABELS.get(name, name)


def relation_pattern_color(name: str, idx: int) -> str:
    if name in RELATION_PATTERN_COLORS:
        return RELATION_PATTERN_COLORS[name]
    fallback_colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    return fallback_colors[idx % len(fallback_colors)]


def square_limits_for_xy(
    points_n2: NDArray[np.floating],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = float(np.nanmin(points_n2[:, 0]))
    x_max = float(np.nanmax(points_n2[:, 0]))
    y_min = float(np.nanmin(points_n2[:, 1]))
    y_max = float(np.nanmax(points_n2[:, 1]))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_width = 0.5 * max(x_max - x_min, y_max - y_min)
    if half_width == 0:
        half_width = 1.0
    return (
        (x_center - half_width, x_center + half_width),
        (y_center - half_width, y_center + half_width),
    )


def square_limits_for_xyz(
    points_n3: NDArray[np.floating],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    x_min = float(np.nanmin(points_n3[:, 0]))
    x_max = float(np.nanmax(points_n3[:, 0]))
    y_min = float(np.nanmin(points_n3[:, 1]))
    y_max = float(np.nanmax(points_n3[:, 1]))
    z_min = float(np.nanmin(points_n3[:, 2]))
    z_max = float(np.nanmax(points_n3[:, 2]))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)
    half_width = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    if half_width == 0:
        half_width = 1.0
    return (
        (x_center - half_width, x_center + half_width),
        (y_center - half_width, y_center + half_width),
        (z_center - half_width, z_center + half_width),
    )
