from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cognitive_maps_in_llms.analysis.mosaic import (
    Series,
    plot_mosaic_series,
)
from cognitive_maps_in_llms.analysis.results import RunResults
from cognitive_maps_in_llms.analysis.style import PAPER_MOSAIC_STYLE


pytestmark = pytest.mark.codex


MODEL = "gemma-27b"
PROMPT = "ABC_family_tree_who_is"
EXPERIMENT = "exp_2_separability"
QUERY = "B --mother--> ?"


def _write_results(root: Path) -> Path:
    path = root / MODEL / PROMPT / EXPERIMENT / "results.ndjson"
    path.parent.mkdir(parents=True)
    records = []
    for seed in (0, 1):
        for condition in ("CTRL", "MOSAIC"):
            shift = seed + (1 if condition == "MOSAIC" else 0)
            records.append(
                {
                    "trial_key": {
                        "seed": seed,
                        "condition": condition,
                        "structure_pair": ["A <- B -> C", "A -> B -> C"],
                    },
                    "results": {
                        QUERY: {
                            "logits": {
                                "A": float(shift),
                                "B": float(shift + 1),
                                "C": float(shift + 2),
                                "U": float(shift + 3),
                            },
                            "probs": {
                                "A": 0.1,
                                "B": 0.2,
                                "C": 0.3,
                                "U": 0.4,
                            },
                        }
                    },
                }
            )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _load_control(root: Path) -> Series:
    return RunResults(results_root=root).load_series(
        model_preset=MODEL,
        prompt_preset=PROMPT,
        experiment=EXPERIMENT,
        query=QUERY,
        condition="CTRL",
        structure_pair=("A <- B -> C", "A -> B -> C"),
    )


def test_run_results_rejects_missing_results_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Results root does not exist"):
        RunResults(results_root=missing_root)


def test_run_results_requires_at_least_one_root() -> None:
    with pytest.raises(ValueError, match="At least one"):
        RunResults()


def test_run_results_rejects_file_as_results_root(tmp_path: Path) -> None:
    file_root = tmp_path / "results.txt"
    file_root.write_text("not a results directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="Results root is not a directory"):
        RunResults(results_root=file_root)


def test_run_results_resolves_and_loads_one_condition(tmp_path: Path) -> None:
    path = _write_results(tmp_path)

    series = _load_control(tmp_path)

    assert series.ndjson_path == path
    assert series.prompt_preset == PROMPT
    assert series.query == QUERY
    assert series.trial_axes == {
        "condition": "CTRL",
        "structure_pair": ("A <- B -> C", "A -> B -> C"),
    }
    assert [record["seed"] for record in series.records] == [0, 1]
    assert all(record["condition"] == "CTRL" for record in series.records)


def test_run_results_lists_mosaic_options(tmp_path: Path) -> None:
    _write_results(tmp_path)

    options = RunResults(results_root=tmp_path).get_mosaic_options(
        model_preset=MODEL,
        prompt_preset=PROMPT,
        experiment=EXPERIMENT,
    )

    assert options == {
        "queries": (QUERY,),
        "trial_axes": {
            "condition": ("CTRL", "MOSAIC"),
            "structure_pair": (["A <- B -> C", "A -> B -> C"],),
        },
    }


@pytest.mark.parametrize(
    "trial_axes",
    [
        {"condition": "CTRL"},
        {
            "condition": "CTRL",
            "structure_pair": ("A <- B -> C", "A -> B -> C"),
            "seed": 0,
        },
    ],
)
def test_run_results_requires_exactly_the_non_seed_axes(
    tmp_path: Path,
    trial_axes: dict[str, object],
) -> None:
    _write_results(tmp_path)
    results = RunResults(results_root=tmp_path)

    with pytest.raises(ValueError, match="every non-seed axis exactly once"):
        results.load_series(
            model_preset=MODEL,
            prompt_preset=PROMPT,
            experiment=EXPERIMENT,
            query=QUERY,
            **trial_axes,
        )


def test_run_results_rejects_missing_query(tmp_path: Path) -> None:
    _write_results(tmp_path)

    with pytest.raises(KeyError, match="missing"):
        RunResults(results_root=tmp_path).load_series(
            model_preset=MODEL,
            prompt_preset=PROMPT,
            experiment=EXPERIMENT,
            query="A --daughter--> ?",
            condition="CTRL",
            structure_pair=("A <- B -> C", "A -> B -> C"),
        )


def test_plot_uses_all_symbols_and_default_entity_labels(tmp_path: Path) -> None:
    _write_results(tmp_path)
    series = _load_control(tmp_path)

    plot = plot_mosaic_series({"Control": series}, value_kind="probs")

    labels = [tick.get_text() for tick in plot.ax.get_xticklabels()]
    assert labels == ["‹A›", "‹B›", "‹C›", "Unk."]
    assert plot.ax.get_ylabel() == "Probability"
    assert plot.ax.get_ylim() == pytest.approx((0, 1))
    plt.close(plot.fig)


def test_plot_formats_yes_no_symbols() -> None:
    series = Series(
        records=[
            {
                "seed": 0,
                "logits": {"Y": 2.0, "N": 1.0, "U": 0.0},
                "probs": {"Y": 0.6, "N": 0.3, "U": 0.1},
            }
        ],
        query="A --mother--> B?",
        trial_axes={},
        ndjson_path=Path("<memory>"),
    )

    plot = plot_mosaic_series({"Control": series})

    assert [tick.get_text() for tick in plot.ax.get_xticklabels()] == [
        "Yes",
        "No",
        "Unknown",
    ]
    assert plot.ax.get_ylabel() == "Probability ratio"
    plt.close(plot.fig)


def test_plot_can_add_compact_legend() -> None:
    series = Series(
        records=[
            {
                "seed": 0,
                "logits": {"Y": 2.0, "N": 1.0, "U": 0.0},
                "probs": {"Y": 0.6, "N": 0.3, "U": 0.1},
            }
        ],
        query="A --mother--> B?",
        trial_axes={},
        ndjson_path=Path("<memory>"),
    )

    plot = plot_mosaic_series({"Control": series}, legend=True)

    legend = plot.artists["legend"]
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Control"]
    plt.close(plot.fig)


def test_plot_legend_parameter_overrides_style() -> None:
    series = Series(
        records=[
            {
                "seed": 0,
                "logits": {"Y": 2.0, "N": 1.0, "U": 0.0},
                "probs": {"Y": 0.6, "N": 0.3, "U": 0.1},
            }
        ],
        query="A --mother--> B?",
        trial_axes={},
        ndjson_path=Path("<memory>"),
    )
    style = replace(PAPER_MOSAIC_STYLE, legend=True)

    plot = plot_mosaic_series({"Control": series}, style=style, legend=False)

    assert plot.artists["legend"] is None
    plt.close(plot.fig)


def test_probability_subset_preserves_recorded_values() -> None:
    series = Series(
        records=[
            {
                "seed": 0,
                "logits": {"A": 0.0, "B": 1.0, "C": 2.0, "U": 3.0},
                "probs": {"A": 0.1, "B": 0.2, "C": 0.3, "U": 0.4},
            }
        ],
        query=QUERY,
        trial_axes={},
        ndjson_path=Path("<memory>"),
    )

    plot = plot_mosaic_series(
        {"Control": series},
        value_kind="probs",
        symbols=("B", "U"),
    )

    plotted = plot.artists["scatter"]["Control"].get_offsets()[:, 1]
    np.testing.assert_allclose(plotted, (0.2, 0.4))
    assert [tick.get_text() for tick in plot.ax.get_xticklabels()] == [
        "‹B›",
        "Unk.",
    ]
    plt.close(plot.fig)


def test_probability_ratio_subset_is_centered_within_subset() -> None:
    series = Series(
        records=[
            {
                "seed": 0,
                "logits": {"A": 0.0, "B": 1.0, "C": 2.0, "U": 3.0},
                "probs": {"A": 0.1, "B": 0.2, "C": 0.3, "U": 0.4},
            }
        ],
        query=QUERY,
        trial_axes={},
        ndjson_path=Path("<memory>"),
    )

    plot = plot_mosaic_series(
        {"Control": series},
        value_kind="probability_ratio",
        symbols=("B", "U"),
    )

    plotted = plot.artists["scatter"]["Control"].get_offsets()[:, 1]
    np.testing.assert_allclose(plotted, (math.exp(-1), math.exp(1)))
    assert math.prod(plotted) == pytest.approx(1)
    assert plot.ax.get_ylim() == pytest.approx((1e-5, 1e5))
    assert plot.ax.get_ylabel() == "Probability ratio"
    plt.close(plot.fig)


def test_probability_ratio_range_can_be_fixed_with_style() -> None:
    series = Series(
        records=[
            {
                "seed": 0,
                "logits": {"Y": 1.0, "N": 0.0, "U": -1.0},
                "probs": {"Y": 0.6, "N": 0.3, "U": 0.1},
            }
        ],
        query="A --mother--> B?",
        trial_axes={},
        ndjson_path=Path("<memory>"),
    )
    style = replace(PAPER_MOSAIC_STYLE, ratio_decades=10)

    plot = plot_mosaic_series(
        {"Control": series},
        value_kind="probability_ratio",
        style=style,
    )

    assert plot.ax.get_ylim() == pytest.approx((1e-10, 1e10))
    np.testing.assert_allclose(plot.ax.get_yticks(), (1e-10, 1, 1e10))
    plt.close(plot.fig)


def test_fixed_probability_ratio_range_cannot_hide_values() -> None:
    series = Series(
        records=[
            {
                "seed": 0,
                "logits": {"Y": 30.0, "N": 0.0, "U": -30.0},
                "probs": {"Y": 1.0, "N": 0.0, "U": 0.0},
            }
        ],
        query="A --mother--> B?",
        trial_axes={},
        ndjson_path=Path("<memory>"),
    )
    style = replace(PAPER_MOSAIC_STYLE, ratio_decades=5)

    with pytest.raises(ValueError, match="would hide plotted values"):
        plot_mosaic_series(
            {"Control": series},
            value_kind="probability_ratio",
            style=style,
        )
    plt.close("all")


def test_dot_plot_save_preserves_svg_text(tmp_path: Path) -> None:
    series = Series(
        records=[
            {
                "seed": 0,
                "logits": {"Y": 1.0, "N": 0.0, "U": -1.0},
                "probs": {"Y": 0.6, "N": 0.3, "U": 0.1},
            }
        ],
        query="A --mother--> B?",
        trial_axes={},
        ndjson_path=Path("<memory>"),
    )
    plot = plot_mosaic_series({"Control": series}, title="Example")
    output_path = tmp_path / "plot.svg"

    plot.save(output_path)

    svg = output_path.read_text(encoding="utf-8")
    assert "<text" in svg
    assert "Example" in svg
    plt.close(plot.fig)
