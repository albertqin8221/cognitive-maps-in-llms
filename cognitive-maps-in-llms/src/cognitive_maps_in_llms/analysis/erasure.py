from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from cognitive_maps_in_llms.analysis.style import (
    EXPORT_RC_PARAMS,
    PAPER_MOSAIC_STYLE,
)

TrajectoryXAxis = Literal["iteration", "remaining_sites"]
TrajectoryYAxis = Literal["score", "fidelity"]
ERASURE_STATE_FILENAME = "erasure_state.json"
METADATA_FILENAME = "erasure_metadata.json"
ERASURE_TRAJECTORY_FIGSIZE = (
    PAPER_MOSAIC_STYLE.figsize[0],
    1.25 * PAPER_MOSAIC_STYLE.figsize[1],
)
REMAINING_KV_FIGSIZE = (
    3 * PAPER_MOSAIC_STYLE.figsize[0],
    1.5 * PAPER_MOSAIC_STYLE.figsize[1],
)


@dataclass(frozen=True)
class ErasureTrajectory:
    """An erasure state and the metadata needed to interpret its scores."""

    key_erasure_round_tli: NDArray[np.int64]
    value_erasure_round_tli: NDArray[np.int64]
    score_by_iteration: NDArray[np.float64]
    initial_score: float
    fully_erased_score: float
    metadata: Mapping[str, Any]
    directory: Path

    @property
    def latest_iteration(self) -> int:
        return max(
            0,
            int(self.key_erasure_round_tli.max()),
            int(self.value_erasure_round_tli.max()),
        )

    def masks_at_iteration(
        self,
        iteration: int | None = None,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        if iteration is None:
            iteration = self.latest_iteration
        if isinstance(iteration, bool) or not isinstance(
            iteration,
            (int, np.integer),
        ):
            raise ValueError("iteration must be a nonnegative integer.")
        iteration = int(iteration)
        if not 0 <= iteration <= self.latest_iteration:
            raise ValueError(
                f"iteration must fall within [0, {self.latest_iteration}]; "
                f"got {iteration}."
            )
        key_mask = (
            (self.key_erasure_round_tli == -1)
            | (self.key_erasure_round_tli > iteration)
        )
        value_mask = (
            (self.value_erasure_round_tli == -1)
            | (self.value_erasure_round_tli > iteration)
        )
        return key_mask, value_mask


@dataclass(frozen=True)
class ErasureTrajectoryPlot:
    fig: Figure
    ax: Axes
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    artists: dict[str, Any]

    def save(
        self,
        path: Path | str,
        *,
        format: str | None = None,
        dpi: int = 300,
        bbox_inches: str = "tight",
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
class RemainingKVPlot:
    fig: Figure
    axes: tuple[Axes, Axes]
    key_counts: NDArray[np.int64]
    value_counts: NDArray[np.int64]
    iteration: int
    artists: dict[str, Any]

    def save(
        self,
        path: Path | str,
        *,
        format: str | None = None,
        dpi: int = 300,
        bbox_inches: str = "tight",
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


def load_erasure_trajectory(erasure_dir: Path | str) -> ErasureTrajectory:
    """Load the two JSON artifacts written by the iterative-erasure task."""
    directory = Path(erasure_dir)
    state_path = directory / ERASURE_STATE_FILENAME
    metadata_path = directory / METADATA_FILENAME

    state_data = _load_json_object(state_path)
    metadata = _load_json_object(metadata_path)
    initial_score = _finite_metadata_score(
        metadata,
        "initial_score",
        metadata_path,
    )
    fully_erased_score = _finite_metadata_score(
        metadata,
        "fully_erased_score",
        metadata_path,
    )
    if initial_score == fully_erased_score:
        raise ValueError(
            "Initial and fully erased scores are equal; normalized fidelity "
            "is undefined."
        )

    key_rounds = _erasure_round_array(state_data, "key_erasure_round_tli")
    value_rounds = _erasure_round_array(state_data, "value_erasure_round_tli")
    if key_rounds.shape != value_rounds.shape:
        raise ValueError(
            "Key and value erasure arrays must have the same shape; got "
            f"key={key_rounds.shape}, value={value_rounds.shape}."
        )
    scores = _score_array(state_data)
    latest_iteration = max(0, int(key_rounds.max()), int(value_rounds.max()))
    if scores.size != latest_iteration:
        raise ValueError(
            "Score history does not align with the recorded erasure rounds: "
            f"scores={scores.size}, latest_round={latest_iteration}."
        )

    return ErasureTrajectory(
        key_erasure_round_tli=key_rounds,
        value_erasure_round_tli=value_rounds,
        score_by_iteration=scores,
        initial_score=initial_score,
        fully_erased_score=fully_erased_score,
        metadata=metadata,
        directory=directory,
    )


def plot_erasure_trajectory(
    trajectory: ErasureTrajectory | Path | str,
    *,
    x_axis: TrajectoryXAxis = "iteration",
    y_axis: TrajectoryYAxis = "score",
    ax: Axes | None = None,
    figsize: tuple[float, float] = ERASURE_TRAJECTORY_FIGSIZE,
    color: str = "tab:blue",
    transition_iteration: int | None = None,
    marked_iterations: Mapping[str, int] | None = None,
) -> ErasureTrajectoryPlot:
    """Plot scores recorded before each step of an erasure trajectory.

    The default reproduces the legacy score-versus-iteration plot. Remaining
    sites and normalized fidelity are available without changing the recorded
    trajectory or evaluating the model again. Transition and marker locations
    are specified as iterations and mapped onto the selected horizontal axis.
    """
    trajectory = _coerce_trajectory(trajectory)
    scores = trajectory.score_by_iteration
    if scores.size == 0:
        raise ValueError("The erasure trajectory does not contain any scores.")

    iterations = np.arange(scores.size, dtype=np.int64)
    if x_axis == "iteration":
        x = iterations.astype(np.float64)
        xlabel = "Iteration"
    elif x_axis == "remaining_sites":
        x = np.asarray(
            [
                _remaining_site_count(trajectory, int(iteration))
                for iteration in iterations
            ],
            dtype=np.float64,
        )
        xlabel = "Remaining key/value sites"
    else:
        raise ValueError(f"Unknown x_axis: {x_axis!r}.")

    if y_axis == "score":
        y = scores
        ylabel = "Score"
    elif y_axis == "fidelity":
        y = (
            (scores - trajectory.fully_erased_score)
            / (trajectory.initial_score - trajectory.fully_erased_score)
        )
        ylabel = "Normalized fidelity"
    else:
        raise ValueError(f"Unknown y_axis: {y_axis!r}.")

    with mpl.rc_context(PAPER_MOSAIC_STYLE.rc_params()):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        (line,) = ax.plot(x, y, color=color)
        transition_line = None
        if transition_iteration is not None:
            validated_transition_iteration = _validate_plotted_iteration(
                transition_iteration,
                n_iterations=scores.size,
            )
            transition_line = ax.axvline(
                x[validated_transition_iteration],
                color="0.4",
                linestyle=":",
                linewidth=1,
                label="Coarse/fine transition",
            )

        marker_artists = {}
        for label, iteration in (marked_iterations or {}).items():
            iteration = _validate_plotted_iteration(
                iteration,
                n_iterations=scores.size,
            )
            marker_artists[str(label)] = ax.scatter(
                [x[iteration]],
                [y[iteration]],
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
                label=str(label),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if x_axis == "remaining_sites":
            ax.invert_xaxis()
        fig.tight_layout()

    return ErasureTrajectoryPlot(
        fig=fig,
        ax=ax,
        x=x,
        y=y,
        artists={
            "line": line,
            "transition": transition_line,
            "markers": marker_artists,
        },
    )


def plot_remaining_key_values(
    trajectory: ErasureTrajectory | Path | str,
    iteration: int,
    *,
    token_labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = REMAINING_KV_FIGSIZE,
    cmap: str = "bone",
) -> RemainingKVPlot:
    """Plot retained key and value head counts by context token and layer."""
    trajectory = _coerce_trajectory(trajectory)
    key_mask, value_mask = trajectory.masks_at_iteration(iteration)
    key_counts = key_mask.sum(axis=2, dtype=np.int64)
    value_counts = value_mask.sum(axis=2, dtype=np.int64)

    n_context_tokens, n_layers, _ = key_mask.shape
    if token_labels is not None and len(token_labels) != n_context_tokens:
        raise ValueError(
            "token_labels must have one label per context token. "
            f"Got {len(token_labels)} labels for {n_context_tokens} tokens."
        )

    layer_ticks = (
        np.arange(n_layers)
        if n_layers <= 12
        else np.arange(0, n_layers, 10)
    )
    max_count = max(1, int(key_counts.max()), int(value_counts.max()))
    with mpl.rc_context(PAPER_MOSAIC_STYLE.rc_params()):
        fig, axes_array = plt.subplots(
            ncols=2,
            figsize=figsize,
            sharey=True,
        )
        axes = (axes_array[0], axes_array[1])
        images = []
        for ax, counts, title in zip(
            axes,
            (key_counts, value_counts),
            ("Remaining keys", "Remaining values"),
        ):
            image = ax.imshow(
                counts,
                aspect="auto",
                origin="upper",
                cmap=cmap,
                vmin=0,
                vmax=max_count,
            )
            ax.set_title(title)
            ax.set_xlabel("Layer")
            ax.set_xticks(layer_ticks)
            ax.set_xticklabels([str(layer) for layer in layer_ticks])
            images.append(image)

        axes[0].set_yticks(np.arange(n_context_tokens))
        if token_labels is not None:
            axes[0].set_yticklabels(token_labels)
        fig.subplots_adjust(
            left=0.12,
            right=0.80,
            bottom=0.24,
            top=0.84,
            wspace=0.12,
        )
        colorbar = fig.colorbar(
            images[0],
            ax=axes,
            fraction=0.05,
            pad=0.04,
        )
        colorbar.set_label("No. of retained keys/values")

        # Allow slight label overlap to improve readability at fixed panel height.
        row_height_points = axes[0].get_position().height * figsize[1] * 72 / n_context_tokens
        font = font_manager.get_font(font_manager.findfont(axes[0].get_yticklabels()[0].get_fontproperties()))
        font_height = (font.ascender - font.descender) / font.units_per_EM
        label_size = min(PAPER_MOSAIC_STYLE.font_size, 1.05 * row_height_points / font_height)
        axes[0].tick_params(axis="y", labelsize=label_size)

    return RemainingKVPlot(
        fig=fig,
        axes=axes,
        key_counts=key_counts,
        value_counts=value_counts,
        iteration=iteration,
        artists={"images": tuple(images), "colorbars": (colorbar,)},
    )


def _coerce_trajectory(
    trajectory: ErasureTrajectory | Path | str,
) -> ErasureTrajectory:
    if isinstance(trajectory, ErasureTrajectory):
        return trajectory
    return load_erasure_trajectory(trajectory)


def _remaining_site_count(
    trajectory: ErasureTrajectory,
    iteration: int,
) -> int:
    key_mask, value_mask = trajectory.masks_at_iteration(iteration)
    return int(key_mask.sum() + value_mask.sum())


def _validate_plotted_iteration(iteration: int, *, n_iterations: int) -> int:
    if isinstance(iteration, bool) or not isinstance(
        iteration,
        (int, np.integer),
    ):
        raise ValueError("Marked iterations must be integers.")
    iteration = int(iteration)
    if not 0 <= iteration < n_iterations:
        raise ValueError(
            f"Marked iterations must fall within [0, {n_iterations - 1}]; "
            f"got {iteration}."
        )
    return iteration


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def _erasure_round_array(
    state_data: Mapping[str, Any],
    name: str,
) -> NDArray[np.int64]:
    if name not in state_data:
        raise ValueError(f"Missing {name!r} in erasure state.")
    array = np.asarray(state_data[name])
    if array.ndim != 3 or any(axis_size < 1 for axis_size in array.shape):
        raise ValueError(f"{name} must have three positive dimensions.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be integer-valued.")
    if np.any(array < -1):
        raise ValueError(f"{name} cannot contain values below -1.")
    return array.astype(np.int64, copy=False)


def _score_array(state_data: Mapping[str, Any]) -> NDArray[np.float64]:
    raw_scores = state_data.get("score_by_iteration")
    if not isinstance(raw_scores, list):
        raise ValueError("score_by_iteration must be a list.")
    try:
        scores = np.asarray(raw_scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("score_by_iteration must contain numbers.") from exc
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("score_by_iteration must contain finite numbers.")
    return scores


def _finite_metadata_score(
    metadata: Mapping[str, Any],
    name: str,
    path: Path,
) -> float:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} in {path} must be a finite number.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} in {path} must be a finite number.")
    return value
