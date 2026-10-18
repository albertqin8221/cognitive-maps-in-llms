from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
import math
from pathlib import Path
import random
from typing import Any, Literal, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cognitive_maps_in_llms.analysis.results import Series
from cognitive_maps_in_llms.analysis.style import (
    EXPORT_RC_PARAMS,
    MosaicPlotStyle,
    PAPER_MOSAIC_STYLE,
)


ValueKind = Literal[
    "probs",
    "logits",
    "normalized_logits",
    "probability_ratio",
]

@dataclass(frozen=True)
class DotPlot:
    fig: Figure
    ax: Axes
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
        """Save the figure, preserving text in vector output formats."""
        with mpl.rc_context(EXPORT_RC_PARAMS):
            self.fig.savefig(
                path,
                format=format,
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
            )


def plot_mosaic_series(
    series: Mapping[str, Series],
    *,
    value_kind: ValueKind = "probability_ratio",
    symbols: Sequence[str] | None = None,
    symbol_labels: Mapping[str, str] | None = None,
    title: str | None = None,
    xlabel: str = "",
    ylabel: str | None = None,
    colors: Mapping[str, str] | Sequence[str] | None = None,
    jitter: float | None = None,
    point_size: float | None = None,
    alpha: float | None = None,
    ax: Axes | None = None,
    style: MosaicPlotStyle = PAPER_MOSAIC_STYLE,
    legend: bool | None = None,
) -> DotPlot:
    """Plot one or more mosaic series as jittered candidate-symbol dots.

    ``jitter``, ``point_size``, ``alpha``, and ``legend`` override their
    style-level settings when provided. Passing ``legend=True`` uses the
    compact legend configuration in ``style``.
    """
    if value_kind not in {
        "probs",
        "logits",
        "normalized_logits",
        "probability_ratio",
    }:
        raise ValueError(f"Unknown value_kind: {value_kind!r}")
    series_by_label = dict(series)
    if not series_by_label:
        raise ValueError("At least one Series is required.")

    selected_symbols = _resolve_symbols(
        series_by_label,
        value_kind=value_kind,
        symbols=symbols,
    )
    rows = _series_to_rows(
        series_by_label,
        value_kind=value_kind,
        symbols=selected_symbols,
    )
    if not rows:
        raise ValueError("No plottable rows found for the requested value_kind.")

    with mpl.rc_context(style.rc_params()):
        if ax is None:
            fig, ax = plt.subplots(figsize=style.figsize)
        else:
            fig = ax.figure

        symbol_to_x = {
            symbol: symbol_idx
            for symbol_idx, symbol in enumerate(selected_symbols)
        }
        labels = list(series_by_label)
        offsets = _series_offsets(len(labels), offset_width=style.offset_width)
        colors_by_label = _colors_by_label(labels, colors=colors, style=style)
        plot_jitter = style.jitter if jitter is None else jitter
        plot_point_size = style.point_size if point_size is None else point_size
        plot_alpha = style.alpha if alpha is None else alpha
        scatter_artists = {}

        for series_idx, label in enumerate(labels):
            label_rows = [row for row in rows if row["series_label"] == label]
            rng = random.Random(style.jitter_seed + series_idx)
            x = [
                symbol_to_x[row["symbol"]]
                + offsets[series_idx]
                + plot_jitter * rng.uniform(-1, 1)
                for row in label_rows
            ]
            scatter_artists[label] = ax.scatter(
                x,
                [row["value"] for row in label_rows],
                s=plot_point_size,
                alpha=plot_alpha,
                color=colors_by_label[label],
                lw=0,
                label=label,
            )

        tick_labels = _symbol_tick_labels(selected_symbols, symbol_labels)
        ax.set_xticks(range(len(selected_symbols)), labels=tick_labels)
        ax.set_xlim(-0.5, len(selected_symbols) - 0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ylabel is not None else _default_ylabel(value_kind))
        if title is not None:
            ax.set_title(title)

        values = [float(row["value"]) for row in rows]
        _configure_y_axis(ax, values=values, value_kind=value_kind, style=style)

        show_legend = style.legend if legend is None else legend
        legend_artist = None
        if show_legend:
            legend_artist = ax.legend(
                loc=style.legend_loc,
                bbox_to_anchor=style.legend_bbox_to_anchor,
                fontsize=style.legend_font_size,
                frameon=style.legend_frame_on,
            )
            for handle in legend_artist.legend_handles:
                handle.set_alpha(style.legend_marker_alpha)
                if hasattr(handle, "set_sizes"):
                    handle.set_sizes([style.legend_marker_size])

    return DotPlot(
        fig=fig,
        ax=ax,
        artists={
            "scatter": scatter_artists,
            "legend": legend_artist,
        },
    )


def _series_to_rows(
    series: Mapping[str, Series],
    *,
    value_kind: ValueKind,
    symbols: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, one_series in series.items():
        for record_idx, record in enumerate(one_series.records):
            values_by_symbol = _values_for_record(
                record,
                value_kind=value_kind,
                symbols=symbols,
            )
            for symbol, value in values_by_symbol.items():
                rows.append(
                    {
                        "series_label": label,
                        "record_idx": record_idx,
                        "symbol": symbol,
                        "value_kind": value_kind,
                        "value": value,
                    }
                )
    return rows


def _resolve_symbols(
    series: Mapping[str, Series],
    *,
    value_kind: ValueKind,
    symbols: Sequence[str] | None,
) -> tuple[str, ...]:
    requested_symbols: tuple[str, ...] | None = None
    if symbols is not None:
        requested_symbols = tuple(str(symbol) for symbol in symbols)
        if not requested_symbols:
            raise ValueError("symbols must contain at least one symbol.")
        if len(set(requested_symbols)) != len(requested_symbols):
            raise ValueError("symbols must not contain duplicates.")

    first_symbols: tuple[str, ...] | None = None
    first_symbol_set: set[str] | None = None
    for label, one_series in series.items():
        if not one_series.records:
            raise ValueError(f"Series {label!r} contains no records.")
        for record_idx, record in enumerate(one_series.records):
            field_name = "logits" if value_kind in {
                "logits",
                "normalized_logits",
                "probability_ratio",
            } else "probs"
            values = _numeric_symbol_mapping(
                record.get(field_name),
                field_name=field_name,
            )
            available = tuple(values)
            if requested_symbols is not None:
                missing = sorted(set(requested_symbols) - set(available))
                if missing:
                    raise KeyError(
                        f"Record {record_idx} in series {label!r} is missing "
                        f"requested symbol(s): {missing}"
                    )
                continue

            if first_symbols is None:
                first_symbols = available
                first_symbol_set = set(available)
            elif set(available) != first_symbol_set:
                raise ValueError(
                    "All plotted records must have the same symbols when "
                    "symbols is not specified. "
                    f"Record {record_idx} in series {label!r} has "
                    f"{list(available)}, expected {list(first_symbols)}."
                )

    if requested_symbols is not None:
        return requested_symbols
    assert first_symbols is not None
    return first_symbols


def _values_for_record(
    record: Mapping[str, Any],
    *,
    value_kind: ValueKind,
    symbols: Sequence[str],
) -> dict[str, float]:
    if value_kind in {"logits", "normalized_logits", "probability_ratio"}:
        logits = _numeric_symbol_mapping(record.get("logits"), field_name="logits")
        selected_logits = {symbol: logits[symbol] for symbol in symbols}
        if value_kind == "logits":
            return selected_logits

        mean_logit = sum(selected_logits.values()) / len(selected_logits)
        normalized_logits = {
            symbol: value - mean_logit
            for symbol, value in selected_logits.items()
        }
        if value_kind == "normalized_logits":
            return normalized_logits

        ratios: dict[str, float] = {}
        for symbol, value in normalized_logits.items():
            try:
                ratio = math.exp(value)
            except OverflowError as exc:
                raise ValueError(
                    f"Probability ratio for symbol {symbol!r} overflowed."
                ) from exc
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(
                    f"Probability ratio for symbol {symbol!r} must be positive "
                    f"and finite, got {ratio!r}."
                )
            ratios[symbol] = ratio
        return ratios

    probabilities = _numeric_symbol_mapping(record.get("probs"), field_name="probs")
    selected_probabilities = {
        symbol: probabilities[symbol]
        for symbol in symbols
    }
    return selected_probabilities


def _numeric_symbol_mapping(value: Any, *, field_name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name!r} must map symbols to numeric values.")
    values = {
        str(symbol): float(symbol_value)
        for symbol, symbol_value in value.items()
    }
    if not values:
        raise ValueError(f"{field_name!r} must contain at least one symbol.")
    return values


def _symbol_tick_labels(
    symbols: Sequence[str],
    overrides: Mapping[str, str] | None,
) -> list[str]:
    entity_symbols_present = bool(set(symbols) - {"Y", "N", "U"})
    if entity_symbols_present:
        labels = {
            symbol: "Unk." if symbol == "U" else f"‹{symbol}›"
            for symbol in symbols
        }
    else:
        labels = {"Y": "Yes", "N": "No", "U": "Unknown"}

    if overrides is not None:
        unknown = sorted(set(overrides) - set(symbols))
        if unknown:
            raise ValueError(
                f"symbol_labels contains unplotted symbol(s): {unknown}"
            )
        labels.update({str(symbol): str(label) for symbol, label in overrides.items()})
    return [labels[symbol] for symbol in symbols]


def _configure_y_axis(
    ax: Axes,
    *,
    values: Sequence[float],
    value_kind: ValueKind,
    style: MosaicPlotStyle,
) -> None:
    if value_kind == "probs":
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.5, 1))
        return
    if value_kind != "probability_ratio":
        return

    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Probability ratios must all be positive and finite.")
    ax.set_yscale("log")

    largest_magnitude = max(abs(math.log10(value)) for value in values)
    if style.ratio_decades is None:
        needed = max(
            style.ratio_min_decades,
            largest_magnitude + style.ratio_padding_decades,
        )
        decades = int(
            math.ceil(needed / style.ratio_decade_step)
            * style.ratio_decade_step
        )
    else:
        decades = style.ratio_decades
        if largest_magnitude > decades + 1e-12:
            raise ValueError(
                "Fixed probability-ratio range would hide plotted values: "
                f"data require {largest_magnitude:.3g} decades, but style "
                f"allows {decades}."
            )

    try:
        lower = 10.0 ** -decades
        upper = 10.0 ** decades
    except OverflowError as exc:
        raise ValueError(
            f"Probability-ratio range of {decades} decades is too large."
        ) from exc
    if lower == 0 or not math.isfinite(upper):
        raise ValueError(
            f"Probability-ratio range of {decades} decades is too large."
        )
    ax.set_ylim(lower, upper)
    ax.set_yticks((lower, 1.0, upper))


def _series_offsets(n_series: int, *, offset_width: float) -> list[float]:
    if n_series == 1:
        return [0.0]
    step = 2 * offset_width / (n_series - 1)
    return [-offset_width + step * idx for idx in range(n_series)]


def _colors_by_label(
    labels: Sequence[str],
    *,
    colors: Mapping[str, str] | Sequence[str] | None,
    style: MosaicPlotStyle,
) -> dict[str, str]:
    if isinstance(colors, Mapping):
        missing = [label for label in labels if label not in colors]
        if missing:
            raise ValueError(f"Missing color(s) for series label(s): {missing}")
        return {label: colors[label] for label in labels}

    if colors is not None:
        if len(colors) < len(labels):
            raise ValueError(
                f"Need at least {len(labels)} colors, received {len(colors)}."
            )
        return {label: colors[idx] for idx, label in enumerate(labels)}

    if len(labels) <= len(style.color_sets):
        palette = style.color_sets[len(labels) - 1]
        if len(palette) >= len(labels):
            return {label: palette[idx] for idx, label in enumerate(labels)}

    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    return {label: next(color_cycle) for label in labels}


def _default_ylabel(value_kind: ValueKind) -> str:
    return {
        "probs": "Probability",
        "logits": "Logit",
        "normalized_logits": "Normalized logit",
        "probability_ratio": "Probability ratio",
    }[value_kind]
