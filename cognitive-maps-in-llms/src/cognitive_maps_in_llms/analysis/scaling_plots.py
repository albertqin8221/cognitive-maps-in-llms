from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cognitive_maps_in_llms.analysis.results import RunResults
from cognitive_maps_in_llms.analysis.style import EXPORT_RC_PARAMS
from cognitive_maps_in_llms.analysis.utils import (
    load_ndjson_records,
)
from cognitive_maps_in_llms.benchmarks import (
    BenchmarkCase,
    get_default_benchmark_name,
    load_benchmark,
)


DEFAULT_FIGSIZE = (2.15, 1.25)
EXPERIMENT_1_KEY = "exp_1_behavior"
IGNORED_ANSWER_SYMBOLS = frozenset({"U"})

QWEN_MODELS = (
    "qwen-0p6b",
    "qwen-1p7b",
    "qwen-4b",
    "qwen-8b",
    "qwen-14b",
    "qwen-32b",
)
LLAMA_MODELS = (
    "llama-8b",
    "llama-70b",
)
GEMMA_MODELS = (
    "gemma-4b",
    "gemma-12b",
    "gemma-27b",
)

DEFAULT_MODEL_FAMILIES = {
    "Qwen": QWEN_MODELS,
    "Llama": LLAMA_MODELS,
    "Gemma": GEMMA_MODELS,
}
DEFAULT_FAMILY_COLORS = {
    "Qwen": "#c8322d",
    "Llama": "#2ca33b",
    "Gemma": "#2878b5",
}
DEFAULT_FAMILY_LABELS = {
    "Qwen": "Qwen-3",
    "Llama": "Llama-3.1",
    "Gemma": "Gemma-3",
}
DEFAULT_MODEL_SIZES = {
    "qwen-0p6b": 751_632_384,
    "qwen-1p7b": 2_031_739_904,
    "qwen-4b": 4_022_468_096,
    "qwen-8b": 8_190_735_360,
    "qwen-14b": 14_768_307_200,
    "qwen-32b": 32_762_123_264,
    "llama-8b": 8_030_261_248,
    "llama-70b": 70_553_706_496,
    "gemma-4b": 4_300_079_472,
    "gemma-12b": 12_187_325_040,
    "gemma-27b": 27_432_406_640,
}


@dataclass(frozen=True)
class AccuracyScalingPlot:
    fig: Figure
    ax: Axes
    data: list[dict[str, Any]]
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


def get_model_size(model_preset: str) -> float:
    if model_preset in DEFAULT_MODEL_SIZES:
        return float(DEFAULT_MODEL_SIZES[model_preset])

    from llm_neuroscience_tools.presets.model_configs import MODEL_CONFIGS

    return float(MODEL_CONFIGS[model_preset].n_parameters)


def benchmark_accuracy_from_ndjson(
    path: Path | str,
    benchmark_cases: Sequence[BenchmarkCase],
    *,
    benchmark_kind: str | None = None,
    ignored_answer_symbols: Sequence[str] = tuple(IGNORED_ANSWER_SYMBOLS),
) -> float:
    """Score behavior records against benchmark cases.

    Accuracy is the mean over all matched benchmark-case/seed observations.
    The prediction is the answer with the highest probability after removing
    ignored answer symbols such as ``U`` (Unknown).
    """
    records = load_ndjson_records(path)
    if not records:
        raise ValueError(f"No records found in {Path(path)}")
    return benchmark_accuracy(
        records,
        benchmark_cases,
        benchmark_kind=benchmark_kind,
        ignored_answer_symbols=ignored_answer_symbols,
        json_path=Path(path),
    )


def make_accuracy_scaling_plot(
    results: RunResults,
    *,
    prompt_preset: str,
    benchmark: str | Sequence[BenchmarkCase] | None = None,
    benchmark_kind: str | None = "two-hop",
    model_families: Mapping[str, Sequence[str]] = DEFAULT_MODEL_FAMILIES,
    experiment: str = EXPERIMENT_1_KEY,
    ignored_answer_symbols: Sequence[str] = tuple(IGNORED_ANSWER_SYMBOLS),
    ax: Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    colors: Mapping[str, str] | Sequence[str] | None = None,
    marker: str = "o",
    markersize: float = 4.5,
    linewidth: float = 1.25,
    title: str | None = None,
    xlabel: str = "Parameter Count",
    ylabel: str = "Accuracy",
    ylim: tuple[float, float] | None = (0.35, 1.05),
    yticks: Sequence[float] | None = (0.4, 0.6, 0.8, 1.0),
    legend: bool = True,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: tuple[float, float] | None = (1.04, 0.52),
    chance: float | None = None,
) -> AccuracyScalingPlot:
    """Plot benchmark accuracy against model size for one or more families."""
    benchmark_cases = _resolve_benchmark_cases(
        benchmark,
        prompt_preset=prompt_preset,
    )
    if not model_families:
        raise ValueError("At least one model family is required.")

    ax_was_created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    family_names = list(model_families)
    colors_by_family = _colors_by_family(family_names, colors)
    data: list[dict[str, Any]] = []
    line_artists = {}

    for family_name, model_presets in model_families.items():
        if not model_presets:
            raise ValueError(f"Model family {family_name!r} has no models.")

        family_rows = []
        for model_preset in model_presets:
            json_path = results._mosaic_path(
                model_preset=model_preset,
                prompt_preset=prompt_preset,
                experiment=experiment,
            )
            accuracy = benchmark_accuracy(
                results._load_mosaic_records(json_path),
                benchmark_cases,
                benchmark_kind=benchmark_kind,
                ignored_answer_symbols=ignored_answer_symbols,
                json_path=json_path,
            )
            model_size = get_model_size(model_preset)
            row = {
                "family": family_name,
                "model_preset": model_preset,
                "model_size": model_size,
                "accuracy": accuracy,
                "ndjson_path": json_path,
            }
            data.append(row)
            family_rows.append(row)

        family_rows = sorted(family_rows, key=lambda row: row["model_size"])
        (line,) = ax.plot(
            [row["model_size"] for row in family_rows],
            [row["accuracy"] for row in family_rows],
            color=colors_by_family[family_name],
            marker=marker,
            markersize=markersize,
            markeredgewidth=0,
            linewidth=linewidth,
            label=DEFAULT_FAMILY_LABELS.get(family_name, family_name),
        )
        line_artists[family_name] = line

    chance_artist = None
    if chance is not None:
        chance_artist = ax.axhline(y=chance, color="black", linestyle=":")

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=7,
        width=0.8,
        length=3,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        width=0.6,
        length=2,
    )
    if title is not None:
        ax.set_title(title, fontsize=7, pad=4)

    legend_artist = None
    if legend:
        legend_artist = ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            frameon=False,
            fontsize=6,
            handlelength=2.3,
            handletextpad=0.6,
            borderaxespad=0,
            labelspacing=0.55,
        )

    if ax_was_created:
        fig.subplots_adjust(
            left=0.17,
            right=0.76,
            bottom=0.28,
            top=0.82,
        )

    return AccuracyScalingPlot(
        fig=fig,
        ax=ax,
        data=data,
        artists={
            "lines": line_artists,
            "chance": chance_artist,
            "legend": legend_artist,
        },
    )


def benchmark_accuracy(
    records: Sequence[Mapping[str, Any]],
    benchmark_cases: Sequence[BenchmarkCase],
    *,
    benchmark_kind: str | None = None,
    ignored_answer_symbols: Sequence[str] = tuple(IGNORED_ANSWER_SYMBOLS),
    json_path: Path | str = Path("<records>"),
) -> float:
    """Return benchmark accuracy over all matching cases and seeds."""
    selected_cases = [
        case
        for case in benchmark_cases
        if benchmark_kind is None or case["kind"] == benchmark_kind
    ]
    if not selected_cases:
        kind_description = (
            "the benchmark" if benchmark_kind is None
            else f"benchmark_kind={benchmark_kind!r}"
        )
        raise ValueError(f"No cases selected from {kind_description}.")

    cases_by_structure: dict[str, list[BenchmarkCase]] = {}
    for case in selected_cases:
        cases_by_structure.setdefault(case["structure"], []).append(case)

    ignored = {str(symbol) for symbol in ignored_answer_symbols}
    matches_by_case = {
        (case["structure"], case["query"]): 0
        for case in selected_cases
    }
    n_correct = 0
    n_total = 0
    path = Path(json_path)

    for record_idx, record in enumerate(records):
        trial_key = record.get("trial_key")
        if not isinstance(trial_key, Mapping):
            raise ValueError(
                f"Record {record_idx} in {path} has no mapping-valued "
                "'trial_key' field."
            )
        structure = trial_key.get("structure")
        if not isinstance(structure, str):
            raise ValueError(
                f"Record {record_idx} in {path} has no string-valued "
                "'structure' trial axis."
            )
        structure_cases = cases_by_structure.get(structure)
        if structure_cases is None:
            continue

        results = record.get("results")
        if not isinstance(results, Mapping):
            raise ValueError(
                f"Record {record_idx} in {path} has no mapping-valued "
                "'results' field."
            )

        for case in structure_cases:
            query = case["query"]
            target = case["answer"]
            if query not in results:
                raise KeyError(
                    f"Benchmark query {query!r} is missing from record "
                    f"{record_idx} in {path}."
                )
            query_result = results[query]
            if not isinstance(query_result, Mapping):
                raise ValueError(
                    f"Query {query!r} in record {record_idx} of {path} "
                    "does not map to a dictionary-like result."
                )
            probabilities = _numeric_symbol_mapping(
                query_result.get("probs"),
                field_name="probs",
                json_path=path,
                record_idx=record_idx,
                query=query,
            )
            candidate_probabilities = {
                symbol: probability
                for symbol, probability in probabilities.items()
                if symbol not in ignored
            }
            if target not in candidate_probabilities:
                raise KeyError(
                    f"Benchmark answer {target!r} is missing from the scored "
                    f"probabilities for query {query!r} in record {record_idx} "
                    f"of {path}."
                )
            if len(candidate_probabilities) < 2:
                raise ValueError(
                    f"Query {query!r} in record {record_idx} of {path} has "
                    "fewer than two non-ignored answer symbols."
                )

            competing_probabilities = [
                probability
                for symbol, probability in candidate_probabilities.items()
                if symbol != target
            ]
            n_total += 1
            matches_by_case[(structure, query)] += 1
            if candidate_probabilities[target] > max(competing_probabilities):
                n_correct += 1

    missing_cases = [
        identity
        for identity, match_count in matches_by_case.items()
        if match_count == 0
    ]
    if missing_cases:
        raise ValueError(
            f"No result records matched {len(missing_cases)} benchmark case(s) "
            f"in {path}. First missing case: {missing_cases[0]!r}."
        )
    return n_correct / n_total


def _resolve_benchmark_cases(
    benchmark: str | Sequence[BenchmarkCase] | None,
    *,
    prompt_preset: str,
) -> list[BenchmarkCase]:
    if benchmark is None:
        benchmark = get_default_benchmark_name(prompt_preset)
    if isinstance(benchmark, str):
        return load_benchmark(benchmark)
    cases = list(benchmark)
    if not cases:
        raise ValueError("benchmark must contain at least one case.")
    return cases


def _numeric_symbol_mapping(
    value: Any,
    *,
    field_name: str,
    json_path: Path,
    record_idx: int,
    query: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field_name!r} for query {query!r} in record {record_idx} of "
            f"{json_path} must map symbols to numeric values."
        )
    return {
        str(symbol): float(symbol_value)
        for symbol, symbol_value in value.items()
    }


def _colors_by_family(
    family_names: Sequence[str],
    colors: Mapping[str, str] | Sequence[str] | None,
) -> dict[str, str]:
    if isinstance(colors, Mapping):
        missing = [
            family_name
            for family_name in family_names
            if family_name not in colors
        ]
        if missing:
            raise ValueError(f"Missing color(s) for model family/families: {missing}")
        return {
            family_name: colors[family_name]
            for family_name in family_names
        }

    if colors is not None:
        if len(colors) < len(family_names):
            raise ValueError(
                f"Need at least {len(family_names)} colors, received {len(colors)}."
            )
        return {
            family_name: colors[idx]
            for idx, family_name in enumerate(family_names)
        }

    default_colors = dict(DEFAULT_FAMILY_COLORS)
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors_by_family = {}
    for family_name in family_names:
        if family_name in default_colors:
            colors_by_family[family_name] = default_colors[family_name]
        else:
            colors_by_family[family_name] = next(color_cycle)
    return colors_by_family
