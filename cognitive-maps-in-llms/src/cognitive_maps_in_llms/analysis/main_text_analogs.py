"""Render and export main-text analogs for supplementary comparisons."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Literal, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cognitive_maps_in_llms.analysis.mosaic import DotPlot, plot_mosaic_series
from cognitive_maps_in_llms.analysis.results import RunResults
from cognitive_maps_in_llms.analysis.erasure import ErasureTrajectoryPlot, RemainingKVPlot
from cognitive_maps_in_llms.analysis.scaling_plots import AccuracyScalingPlot, make_accuracy_scaling_plot
from cognitive_maps_in_llms.analysis.utils import query_key_to_text
from cognitive_maps_in_llms.analysis.style import MosaicPlotStyle, PAPER_MOSAIC_STYLE


FigureGroup = Literal[
    "behavior",
    "separability",
    "additivity",
    "retrospective_prospective",
    "erasure",
]
ALL_PANEL_GROUPS: tuple[FigureGroup, ...] = (
    "behavior",
    "separability",
    "additivity",
    "retrospective_prospective",
    "erasure",
)
PanelPlot = DotPlot | AccuracyScalingPlot | ErasureTrajectoryPlot | RemainingKVPlot


# Exact queries, ordering, and titles from figure_2_separability.ipynb.
SEPARABILITY_PANELS = (
    ("is_Es_mother_D", "E --mother--> D?", "Is ‹E›'s mother ‹D›?"),
    ("is_Es_mother_A", "E --mother--> A?", "Is ‹E›'s mother ‹A›?"),
    ("is_Es_daughter_F", "E --daughter--> F?", "Is ‹E›'s daughter ‹F›?"),
    ("is_Es_daughter_C", "E --daughter--> C?", "Is ‹E›'s daughter ‹C›?"),
    ("is_As_granddaughter_C", "A --granddaughter--> C?", "Is ‹A›'s granddaughter ‹C›?"),
    ("is_Cs_grandmother_A", "C --grandmother--> A?", "Is ‹C›'s grandmother ‹A›?"),
)
SEPARABILITY_CONDITIONS = {"Control": "CTRL", "Mosaic": "B1"}
COMPARISON_COLORS = {"Control": "tab:red", "Mosaic": "#80558F"}


# Query vocabulary, in the direction of daughter/mother in the original panels.
_DIRECTIONAL_RELATIONS = {
    "ABC_older_younger": ("older", "younger", "older", "younger"),
    "ABC_taller_shorter": ("taller", "shorter", "taller", "shorter"),
    "ABC_faster_slower": ("faster", "slower", "faster", "slower"),
    "ABC_east_west": ("east", "west", "east", "west"),
    "ABC_north_south": ("north", "south", "north", "south"),
    "ABC_ahead_behind": ("ahead", "behind", "ahead", "behind"),
    "ABC_family_trees_descendant_ancestor": (
        "a descendant", "an ancestor", "a descendant", "an ancestor",
    ),
    "ABC_older_younger_unique_locations_yes_no": (
        "one year older than", "one year younger than",
        "two years older than", "two years younger than",
    ),
}


_SPECIAL_RELATIONS = {
    "ABC_gene_regulatory_network": ("activate", "repress"),
    "ABC_neural_circuit": ("excite", "inhibit"),
    "ABC_proposition_equivalence": ("equivalent to",),
    "ABC_variable_equivalence": ("equal",),
}


def separability_panels(prompt_preset: str) -> tuple[tuple[str, str, str], ...]:
    """Return (filename, query, title) tuples for the selected prompt domain."""
    if prompt_preset == "ABC_family_tree_yes_no":
        return SEPARABILITY_PANELS
    if prompt_preset == "ABC_company_structure":
        # FWD means "has boss": A is junior to B, and B is junior to C.
        queries = [
            "E --senior--> D?", "E --senior--> A?",
            "E --junior--> F?", "E --junior--> C?",
            "A --junior--> C?", "C --senior--> A?",
        ]
    elif prompt_preset in _DIRECTIONAL_RELATIONS:
        forward, backward, forward_two, backward_two = _DIRECTIONAL_RELATIONS[prompt_preset]
        if prompt_preset == "ABC_family_trees_descendant_ancestor":
            # "A has daughter B" means A is an ancestor of B.
            forward = forward_two = "an ancestor"
            backward = backward_two = "a descendant"
        queries = [f"{source} --{relation}--> {target}?" for source, relation, target in (
            ("E", backward, "D"), ("E", backward, "A"),
            ("E", forward, "F"), ("E", forward, "C"),
            ("A", forward_two, "C"), ("C", backward_two, "A"),
        )]
    elif prompt_preset in {"ABC_gene_regulatory_network", "ABC_neural_circuit"}:
        relation = _SPECIAL_RELATIONS[prompt_preset][0]
        queries = [f"{source} --{relation}--> {target}?" for source, target in (
            ("D", "E"), ("E", "F"), ("A", "E"), ("E", "C"), ("A", "C"),
        )]
    elif prompt_preset in _SPECIAL_RELATIONS:
        queries = [
            f"{source} --{relation}--> {target}?"
            for source, target in (("E", "D"), ("E", "A"), ("E", "F"),
                                   ("E", "C"), ("A", "C"), ("C", "A"))
            for relation in _SPECIAL_RELATIONS[prompt_preset]
        ]
    else:
        raise ValueError(f"No separability query selection defined for {prompt_preset!r}.")
    panels = []
    for query in queries:
        title = query_key_to_text(query, prompt_preset=prompt_preset)
        title = title.replace("<", "‹").replace(">", "›")
        filename = re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_")
        panels.append((filename, query, title))
    return tuple(panels)


def render_separability(
    results: RunResults,
    *,
    model_preset: str,
    prompt_preset: str,
    style: MosaicPlotStyle = PAPER_MOSAIC_STYLE,
    shared_legend: bool = True,
) -> dict[str, DotPlot]:
    """Return editable notebook-style panels; do not save or close them."""
    panels = separability_panels(prompt_preset)
    # Load all selections first so a missing query does not leave partial figures.
    loaded = {
        name: {
            label: results.load_series(
                model_preset=model_preset, prompt_preset=prompt_preset,
                experiment="exp_2_separability", query=query,
                patch_location=location,
            )
            for label, location in SEPARABILITY_CONDITIONS.items()
        }
        for name, query, title in panels
    }
    return _plot_comparisons(panels, loaded, "separability", style, shared_legend)


def _plot_comparisons(panels, loaded, group, style, shared_legend):
    plots = {}
    figures_before = set(plt.get_fignums())
    try:
        for name, query, title in panels:
            plots[f"{group}/{name}"] = plot_mosaic_series(
                loaded[name], value_kind="probability_ratio",
                title=title, colors=COMPARISON_COLORS,
                legend=False, style=style,
            )
        if shared_legend:
            with mpl.rc_context(style.rc_params()):
                fig, ax = plt.subplots(figsize=(style.figsize[0], 0.25))
                ax.set_axis_off()
                handles = [
                    Line2D([], [], linestyle="none", marker="o", label=label,
                           color=color, markeredgewidth=0,
                           markersize=style.legend_marker_size ** 0.5,
                           alpha=style.legend_marker_alpha)
                    for label, color in COMPARISON_COLORS.items()
                ]
                legend = ax.legend(
                    handles=handles, loc="center", ncol=2,
                    frameon=style.legend_frame_on, fontsize=style.legend_font_size,
                )
                plots[f"{group}/legend"] = DotPlot(fig, ax, {"legend": legend})
    except Exception:
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)
        raise
    return plots




def comparison_queries(prompt_preset: str, group: str) -> tuple[str, ...]:
    """Approved yes/no selections for figures 3 and 4."""
    separability_panels(prompt_preset)  # Enforce the same supported presets.
    if group not in {"additivity", "retrospective_prospective"}:
        raise ValueError(f"Unknown comparison group: {group!r}")
    equivalence = prompt_preset in {"ABC_proposition_equivalence", "ABC_variable_equivalence"}
    biology = prompt_preset in {"ABC_gene_regulatory_network", "ABC_neural_circuit"}
    if group == "additivity":
        if biology:
            relation = _SPECIAL_RELATIONS[prompt_preset][0]
            return tuple(f"{source} --{relation}--> {target}?" for source, target in (
                ("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"),
            ))
        if prompt_preset == "ABC_family_tree_yes_no":
            relations = ("daughter", "mother")
        elif prompt_preset == "ABC_company_structure":
            relations = ("junior", "senior")
        elif biology or equivalence:
            relations = _SPECIAL_RELATIONS[prompt_preset]
        else:
            relations = _DIRECTIONAL_RELATIONS[prompt_preset][:2]
        pairs = (("B", "A"), ("B", "C"))
    else:
        if prompt_preset == "ABC_family_tree_yes_no":
            return (
                "C --sister--> A?", "A --sister--> C?",
                "C --grandmother--> A?", "A --granddaughter--> C?",
            )
        if prompt_preset == "ABC_older_younger_unique_locations_yes_no":
            return (
                "A --equal in age to--> C?", "C --equal in age to--> A?",
                "A --two years older than--> C?", "C --two years younger than--> A?",
            )
        if prompt_preset == "ABC_company_structure":
            relations = ("junior", "senior")
        elif biology or equivalence:
            relations = _SPECIAL_RELATIONS[prompt_preset]
        else:
            relations = _DIRECTIONAL_RELATIONS[prompt_preset][:2]
        pairs = (("A", "C"),) if biology else (("A", "C"), ("C", "A"))
    return tuple(f"{source} --{relation}--> {target}?"
                 for source, target in pairs for relation in relations)


def _render_comparison(results, model_preset, prompt_preset, group, style, shared_legend):
    queries = comparison_queries(prompt_preset, group)
    if not queries:
        return {}
    panels = [
        (re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_"), query,
         query_key_to_text(query, prompt_preset=prompt_preset).replace("<", "‹").replace(">", "›"))
        for query in queries
    ]
    if group == "additivity":
        experiment = "exp_3_additivity"
        condition = "SWAP"
        axes = {}
    else:
        experiment = "exp_4_retrospective_prospective"
        condition = "ENTITY_RELATION_SPAN"
        axes = {"structure_pair": {
            "donor": "A --rev--> B --fwd--> C",
            "control": "A --fwd--> B --fwd--> C",
        }}
    loaded = {
        name: {
            label: results.load_series(
                model_preset=model_preset, prompt_preset=prompt_preset,
                experiment=experiment, query=query, condition=selected, **axes,
            )
            for label, selected in (("Control", "CTRL"), ("Mosaic", condition))
        }
        for name, query, title in panels
    }
    return _plot_comparisons(panels, loaded, group, style, shared_legend)


def render_additivity(
    results: RunResults, *, model_preset: str, prompt_preset: str,
    style: MosaicPlotStyle = PAPER_MOSAIC_STYLE, shared_legend: bool = True,
) -> dict[str, DotPlot]:
    """Figure 3, including equivalence presets for completeness."""
    return _render_comparison(results, model_preset, prompt_preset,
                              "additivity", style, shared_legend)


def render_retrospective_prospective(
    results: RunResults, *, model_preset: str, prompt_preset: str,
    style: MosaicPlotStyle = PAPER_MOSAIC_STYLE, shared_legend: bool = True,
) -> dict[str, DotPlot]:
    """Figure 4 with the notebook's donor/control structure pair."""
    return _render_comparison(results, model_preset, prompt_preset,
                              "retrospective_prospective", style, shared_legend)



def behavior_query(prompt_preset: str) -> str:
    # Reuse the approved reverse two-step query for ordered domains.
    panels = separability_panels(prompt_preset)
    if prompt_preset in _SPECIAL_RELATIONS:
        query = f"A --{_SPECIAL_RELATIONS[prompt_preset][0]}--> C?"
    else:
        query = panels[-1][1]
    return query


def alien_prompt_preset(prompt_preset: str) -> str | None:
    """Return the supported alien yes/no counterpart, if one exists."""
    if prompt_preset in {
        "ABC_family_tree_yes_no", "ABC_company_structure", "ABC_older_younger",
        "ABC_taller_shorter", "ABC_faster_slower",
        "ABC_proposition_equivalence",
        "ABC_variable_equivalence",
        "ABC_east_west",
        "ABC_north_south",
        "ABC_ahead_behind",
        "ABC_family_trees_descendant_ancestor",
        "ABC_older_younger_unique_locations_yes_no",
    }:
        return prompt_preset.replace("ABC_", "ABC_alien_", 1)
    return None


def _fit_scaling_lower_limit(plot: AccuracyScalingPlot) -> None:
    """Keep the original range when it fits; expand only the lower bound."""
    lower, upper = plot.ax.get_ylim()
    minimum = min(float(value) for line in plot.ax.lines for value in line.get_ydata())
    # Leave room for markers at the lowest data point.
    padding = 0.05 * (upper - lower)
    if minimum - padding < lower:
        plot.ax.set_ylim(minimum - padding, upper)


def render_behavior_scaling(
    results: RunResults,
    *,
    prompt_preset: str,
    human_alpha: float = 1.0,
    alien_alpha: float = 1.0,
) -> AccuracyScalingPlot:
    """Overlay supported alien results; missing inputs raise as for ordinary data.

    The returned data and lines describe the ordinary preset. The alien plot,
    including its data and artists, is available as ``artists["alien_plot"]``.
    """
    alien = alien_prompt_preset(prompt_preset)
    figures_before = set(plt.get_fignums())
    try:
        plot = make_accuracy_scaling_plot(
            results, prompt_preset=prompt_preset,
            title="All 2-step Inferences", legend=alien is None,
        )
        for line in plot.artists["lines"].values():
            line.set_alpha(human_alpha)
        if alien is None:
            _fit_scaling_lower_limit(plot)
            return plot
        alien_plot = make_accuracy_scaling_plot(
            results, prompt_preset=alien, ax=plot.ax, legend=False,
        )
        for line in alien_plot.artists["lines"].values():
            line.set_linestyle("--")
            line.set_alpha(alien_alpha)
        handles = [
            Line2D([], [], color=line.get_color(), marker="o", linestyle="None",
                   markersize=4.5, markeredgewidth=0, label=line.get_label())
            for line in plot.artists["lines"].values()
        ]
        handles.extend([
            Line2D([], [], color="black", linestyle="-", linewidth=1.25,
                   alpha=human_alpha, label="Domain-appropriate names"),
            Line2D([], [], color="black", linestyle="--", linewidth=1.25,
                   alpha=alien_alpha, label="Synthetic names"),
        ])
        plot.artists["legend"] = plot.ax.legend(
            handles=handles, loc="center left", bbox_to_anchor=(1.04, 0.52),
            frameon=False, fontsize=6, handlelength=2.3,
            handletextpad=0.6, borderaxespad=0, labelspacing=0.55,
        )
        plot.artists["alien_plot"] = alien_plot
        _fit_scaling_lower_limit(plot)
        return plot
    except Exception:
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)
        raise


def render_behavior(
    results: RunResults,
    *,
    model_preset: str,
    prompt_preset: str,
    include_scaling: bool = True,
    human_alpha: float = 1.0,
    alien_alpha: float = 1.0,
) -> dict[str, DotPlot | AccuracyScalingPlot]:
    """Return the example, a separate dual-series legend, and optional scaling."""
    query = behavior_query(prompt_preset)
    title = query_key_to_text(query, prompt_preset=prompt_preset)
    title = title.replace("<", "‹").replace(">", "›")
    selection = dict(
        model_preset=model_preset, experiment="exp_1_behavior",
        query=query, structure="A --fwd--> B --fwd--> C",
    )
    series = {
        "Domain-appropriate names": results.load_series(
            prompt_preset=prompt_preset, **selection,
        ),
    }
    alien = alien_prompt_preset(prompt_preset)
    if alien is not None:
        alien_path = results.results_root / model_preset / alien / "exp_1_behavior" / "results.ndjson"
        if alien_path.is_file():
            series["Synthetic names"] = results.load_series(
                prompt_preset=alien, **selection,
            )
    plots = {}
    figures_before = set(plt.get_fignums())
    try:
        style = replace(PAPER_MOSAIC_STYLE, legend_font_size=5, legend_marker_size=14)
        plots["behavior/example"] = plot_mosaic_series(
            series, colors=["tab:green", "tab:orange"],
            value_kind="probability_ratio", title=title, legend=False,
            style=style,
        )
        if len(series) > 1:
            with mpl.rc_context(style.rc_params()):
                fig, ax = plt.subplots(figsize=(1.2, 0.35))
                ax.set_axis_off()
                handles = [
                    Line2D([], [], linestyle="none", marker="o", label=label,
                           color=color, markeredgewidth=0,
                           markersize=style.legend_marker_size ** 0.5,
                           alpha=style.legend_marker_alpha)
                    for label, color in zip(series, ("tab:green", "tab:orange"))
                ]
                legend = ax.legend(
                    handles=handles, loc="center", ncol=1,
                    frameon=style.legend_frame_on, fontsize=style.legend_font_size,
                )
                plots["behavior/example_legend"] = DotPlot(fig, ax, {"legend": legend})
        if include_scaling:
            plots["behavior/scaling"] = render_behavior_scaling(
                results, prompt_preset=prompt_preset,
                human_alpha=human_alpha, alien_alpha=alien_alpha,
            )
    except Exception:
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)
        raise
    return plots


def render_all_main_text_panels(
    results: RunResults,
    *,
    model_preset: str,
    prompt_preset: str,
    groups: Sequence[FigureGroup] = ALL_PANEL_GROUPS,
    human_alpha: float = 1.0,
    alien_alpha: float = 1.0,
) -> dict[str, PanelPlot]:
    """Render selected groups; alpha options control the behavior scaling lines."""
    if isinstance(groups, str):
        raise TypeError("groups must be a sequence of figure-group names.")
    selected_groups = tuple(groups)
    unknown = sorted(set(selected_groups) - set(ALL_PANEL_GROUPS))
    if unknown:
        raise ValueError(
            f"Unknown figure group(s): {unknown}. "
            f"Expected a selection from {list(ALL_PANEL_GROUPS)}."
        )
    if len(set(selected_groups)) != len(selected_groups):
        raise ValueError("groups must not contain duplicates.")

    # Import locally because erasure_figures imports query-selection helpers
    # from this module.
    from cognitive_maps_in_llms.analysis.erasure_figures import render_erasure

    renderers = {
        "behavior": render_behavior,
        "separability": render_separability,
        "additivity": render_additivity,
        "retrospective_prospective": render_retrospective_prospective,
        "erasure": render_erasure,
    }
    common = {
        "model_preset": model_preset,
        "prompt_preset": prompt_preset,
    }
    plots: dict[str, PanelPlot] = {}
    figures_before = set(plt.get_fignums())
    try:
        for group in selected_groups:
            options = (
                {"human_alpha": human_alpha, "alien_alpha": alien_alpha}
                if group == "behavior" else {}
            )
            rendered = renderers[group](results, **common, **options)
            duplicate_names = sorted(set(plots) & set(rendered))
            if duplicate_names:
                raise ValueError(
                    f"Panel names are not unique across groups: {duplicate_names}."
                )
            plots.update(rendered)
    except Exception:
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)
        raise
    return plots


def save_main_text_analogs(
    plots: Mapping[str, DotPlot | AccuracyScalingPlot | ErasureTrajectoryPlot | RemainingKVPlot],
    output_dir: Path | str,
    *,
    formats: Sequence[str] = ("svg", "pdf"),
    overwrite: bool = False,
) -> list[Path]:
    """Export named panels; replace existing destinations only when requested."""
    if not formats or len(set(formats)) != len(formats) or set(formats) - {"svg", "pdf"}:
        raise ValueError("formats must be a nonempty, unique selection of svg/pdf.")
    root = Path(output_dir).resolve()
    targets = []
    for name, plot in plots.items():
        for fmt in formats:
            path = (root / f"{name}.{fmt}").resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Output name escapes output directory: {name!r}")
            targets.append((path, plot, fmt))
    if len({path for path, _, _ in targets}) != len(targets):
        raise ValueError("Plot names resolve to duplicate destinations.")
    for path, _, _ in targets:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
    for path, plot, fmt in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb" if overwrite else "xb") as file, mpl.rc_context(PAPER_MOSAIC_STYLE.rc_params()):
            plot.fig.savefig(file, format=fmt, bbox_inches="tight", dpi=300)
    return [path for path, _, _ in targets]
