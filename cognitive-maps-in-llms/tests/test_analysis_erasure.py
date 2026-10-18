from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from cognitive_maps_in_llms.analysis import RunResults
from cognitive_maps_in_llms.analysis.erasure import (
    ERASURE_TRAJECTORY_FIGSIZE,
    ERASURE_STATE_FILENAME,
    METADATA_FILENAME,
    REMAINING_KV_FIGSIZE,
    load_erasure_trajectory,
    plot_erasure_trajectory,
    plot_remaining_key_values,
)


matplotlib.use("Agg")
pytestmark = pytest.mark.codex


def _write_trajectory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "key_erasure_round_tli": np.array(
            [
                [[1, 2], [-1, 0], [3, 3]],
                [[1, 2], [-1, 0], [3, 3]],
            ],
            dtype=np.int64,
        ),
        "value_erasure_round_tli": np.array(
            [
                [[0, 0], [1, 2], [3, -1]],
                [[0, 0], [1, 2], [3, -1]],
            ],
            dtype=np.int64,
        ),
        "score_by_iteration": [8.0, 7.0, 5.0],
    }
    (directory / ERASURE_STATE_FILENAME).write_text(
        json.dumps(
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in state.items()
            }
        ),
        encoding="utf-8",
    )
    (directory / METADATA_FILENAME).write_text(
        json.dumps({"initial_score": 8.0, "fully_erased_score": 2.0}),
        encoding="utf-8",
    )


def test_load_and_plot_erasure_trajectory(tmp_path: Path) -> None:
    _write_trajectory(tmp_path)

    trajectory = load_erasure_trajectory(tmp_path)
    plot = plot_erasure_trajectory(
        trajectory,
        x_axis="remaining_sites",
        y_axis="fidelity",
    )

    assert plot.x.tolist() == [18.0, 14.0, 10.0]
    assert plot.y.tolist() == pytest.approx([1.0, 5 / 6, 0.5])
    assert plot.ax.get_xlabel() == "Remaining key/value sites"
    assert plot.ax.xaxis_inverted()
    assert tuple(plot.fig.get_size_inches()) == pytest.approx(
        ERASURE_TRAJECTORY_FIGSIZE
    )
    assert plot.ax.xaxis.label.get_fontsize() == 8


@pytest.mark.parametrize(
    ("x_axis", "expected_x"),
    [("iteration", 1.0), ("remaining_sites", 14.0)],
)
def test_erasure_markers_follow_selected_x_axis(
    tmp_path: Path,
    x_axis: str,
    expected_x: float,
) -> None:
    _write_trajectory(tmp_path)

    plot = plot_erasure_trajectory(
        tmp_path,
        x_axis=x_axis,
        transition_iteration=1,
        marked_iterations={"Example": 1},
    )

    np.testing.assert_allclose(
        plot.artists["transition"].get_xdata(),
        (expected_x, expected_x),
    )
    np.testing.assert_allclose(
        plot.artists["markers"]["Example"].get_offsets(),
        ((expected_x, 7.0),),
    )


def test_run_results_loads_erasure_trajectory(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    directory = results_root / "model" / "prompt" / "iterative_erasure"
    _write_trajectory(directory)
    results = RunResults(results_root=results_root)

    trajectory = results.load_erasure_trajectory(
        model_preset="model",
        prompt_preset="prompt",
    )

    assert trajectory.directory == directory
    assert results.load_erasure_trajectory(
        model_preset="model",
        prompt_preset="prompt",
    ) is trajectory


def test_run_results_requires_root_for_requested_output(tmp_path: Path) -> None:
    results = RunResults(artifacts_root=tmp_path)

    with pytest.raises(ValueError, match="no results_root"):
        results.load_erasure_trajectory(
            model_preset="model",
            prompt_preset="prompt",
        )


def test_plot_remaining_key_values_aggregates_heads(tmp_path: Path) -> None:
    _write_trajectory(tmp_path)

    plot = plot_remaining_key_values(
        tmp_path,
        iteration=1,
        token_labels=["A", "B"],
    )

    assert plot.key_counts.tolist() == [[1, 1, 2], [1, 1, 2]]
    assert plot.value_counts.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert [axis.get_title() for axis in plot.axes] == [
        "Remaining keys",
        "Remaining values",
    ]
    assert plot.axes[0].get_ylabel() == ""
    assert [tick.get_text() for tick in plot.axes[0].get_yticklabels()] == [
        "A",
        "B",
    ]
    assert tuple(plot.fig.get_size_inches()) == pytest.approx(
        REMAINING_KV_FIGSIZE
    )
    assert plot.fig._suptitle is None
    assert len(plot.artists["colorbars"]) == 1
    assert [image.get_clim() for image in plot.artists["images"]] == [
        (0, 2),
        (0, 2),
    ]
    assert (
        plot.artists["colorbars"][0].ax.get_ylabel()
        == "No. of retained keys/values"
    )
    assert plot.axes[0].xaxis.label.get_fontsize() == 8


def test_dense_token_labels_fit_without_changing_panel_height(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _write_trajectory(tmp_path)
    state_path = tmp_path / ERASURE_STATE_FILENAME
    state = json.loads(state_path.read_text())
    for name in ("key_erasure_round_tli", "value_erasure_round_tli"):
        state[name] = state[name] * 15
    state_path.write_text(json.dumps(state))
    labels = [f"token {i}" for i in range(30)]
    plot = plot_remaining_key_values(tmp_path, iteration=1, token_labels=labels)
    try:
        plot.fig.set_dpi(300)  # Resolve small glyphs as in the PDF export preview.
        plot.fig.canvas.draw()
        ticks = plot.axes[0].get_yticklabels()
        assert [tick.get_text() for tick in ticks] == labels
        assert tuple(plot.fig.get_size_inches()) == pytest.approx(REMAINING_KV_FIGSIZE)
        boxes = [tick.get_window_extent(plot.fig.canvas.get_renderer()) for tick in ticks]
        row_spacing = abs(boxes[0].y0 - boxes[1].y0)
        assert all(lower.y1 - upper.y0 <= 0.15 * row_spacing
                   for upper, lower in zip(boxes, boxes[1:]))
        assert plot.key_counts.shape[0] == 30
    finally:
        plt.close(plot.fig)


def test_remaining_key_values_validates_token_labels(tmp_path: Path) -> None:
    _write_trajectory(tmp_path)

    with pytest.raises(ValueError, match="one label per context token"):
        plot_remaining_key_values(
            tmp_path,
            iteration=1,
            token_labels=["only one"],
        )


def test_remaining_key_values_requires_iteration(tmp_path: Path) -> None:
    _write_trajectory(tmp_path)

    with pytest.raises(TypeError, match="iteration"):
        plot_remaining_key_values(tmp_path)
