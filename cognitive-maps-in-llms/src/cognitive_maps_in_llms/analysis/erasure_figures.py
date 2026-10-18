"""Figure 5 panels from existing experiment results and erasure artifacts."""

from __future__ import annotations

from dataclasses import replace
import re
import warnings
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

from cognitive_maps_in_llms.analysis.erasure import (
    ErasureTrajectoryPlot, RemainingKVPlot,
    plot_erasure_trajectory, plot_remaining_key_values,
)
from cognitive_maps_in_llms.analysis.main_text_analogs import (
    _DIRECTIONAL_RELATIONS, behavior_query,
)
from cognitive_maps_in_llms.analysis.mosaic import DotPlot, plot_mosaic_series
from cognitive_maps_in_llms.analysis.results import RunResults
from cognitive_maps_in_llms.analysis.style import PAPER_MOSAIC_STYLE
from cognitive_maps_in_llms.analysis.utils import (
    get_fwd_fwd_context_tokens, load_ndjson_records, query_key_to_text,
)


SPARSE_CONDITION = "FIDELITY_95_SITES_X2"


def editing_queries(prompt_preset: str) -> tuple[str, str]:
    behavior_query(prompt_preset)  # Validate supported presets.
    if prompt_preset == "ABC_gene_regulatory_network":
        return "D --activate--> B?", "D --repress--> B?"
    if prompt_preset == "ABC_neural_circuit":
        return "D --excite--> B?", "D --inhibit--> B?"
    if prompt_preset == "ABC_family_trees_descendant_ancestor":
        return "D --an ancestor--> B?", "B --a descendant--> D?"
    if prompt_preset == "ABC_family_tree_yes_no":
        forward, backward = "daughter", "mother"
    elif prompt_preset == "ABC_company_structure":
        forward, backward = "junior", "senior"
    elif prompt_preset == "ABC_proposition_equivalence":
        forward = backward = "equivalent to"
    elif prompt_preset == "ABC_variable_equivalence":
        forward = backward = "equal"
    else:
        forward, backward = _DIRECTIONAL_RELATIONS[prompt_preset][:2]
    return f"B --{backward}--> D?", f"D --{forward}--> B?"


def render_erasure(
    results: RunResults, *, model_preset: str, prompt_preset: str,
    include_artifacts: bool = True,
    token_labels: Sequence[str] | None = None,
) -> dict[str, DotPlot | ErasureTrajectoryPlot | RemainingKVPlot]:
    """Render figure 5, using token numbers if local tokenizer files are unavailable."""
    query = behavior_query(prompt_preset)
    common = dict(model_preset=model_preset, prompt_preset=prompt_preset)
    specifications = [
        ("bottlenecks", "exp_5a_bottleneck", query, {
            "Control": {"condition": "CTRL"},
            "Keys-first": {"condition": "K_THEN_V"},
            "Keys-only": {"condition": "K_ONLY"},
            "Values-first": {"condition": "V_THEN_K"},
        }),
        ("sparse_mosaic", "exp_5b_sparse_mosaic", query, {
            "Control": {"condition": "REPLACE_ALL"},
            "Sparse KV subset": {"condition": SPARSE_CONDITION},
        }),
    ]
    for edited_query in editing_queries(prompt_preset):
        name = re.sub(r"[^A-Za-z0-9]+", "_", edited_query).strip("_")
        specifications.append((f"editing/{name}", "exp_5c_editing", edited_query, {
            "Control": {"condition": SPARSE_CONDITION, "editing_locs": "NONE"},
            "Edited": {"condition": SPARSE_CONDITION, "editing_locs": "A_KEYS"},
        }))
    loaded = {
        name: {label: results.load_series(experiment=experiment, query=q, **common, **axes)
               for label, axes in conditions.items()}
        for name, experiment, q, conditions in specifications
    }
    plots = {}
    before = set(plt.get_fignums())
    try:
        for name, experiment, q, conditions in specifications:
            title = query_key_to_text(q, prompt_preset=prompt_preset).replace("<", "‹").replace(">", "›")
            options = {}
            if name == "bottlenecks":
                options = dict(colors=("tab:red", "tab:green", "tab:blue", "tab:orange"), jitter=0.03)
            plots[f"erasure/{name}"] = plot_mosaic_series(
                loaded[name], title=title, value_kind="probability_ratio", legend=True,
                style=(PAPER_MOSAIC_STYLE if name == "sparse_mosaic" else
                       replace(PAPER_MOSAIC_STYLE, legend_loc="center left",
                               legend_bbox_to_anchor=(1.02, 0.5))), **options,
            )
        if include_artifacts:
            plots.update(_render_artifacts(results, model_preset, prompt_preset, token_labels))
    except Exception:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
        raise
    return plots


def _render_artifacts(results, model_preset, prompt_preset, token_labels):
    trajectory = results.load_erasure_trajectory(model_preset=model_preset, prompt_preset=prompt_preset)
    # Use the exact iteration recorded by the experiments, not a newly resolved mask.
    selections = set()
    iterations = set()
    for experiment in ("exp_5b_sparse_mosaic", "exp_5c_editing"):
        path = results.results_root / model_preset / prompt_preset / experiment / "results.ndjson"
        records = load_ndjson_records(path)
        selected = [r["trial_info"] for r in records
                    if r["trial_key"]["condition"] == SPARSE_CONDITION]
        if not selected:
            raise ValueError(f"No {SPARSE_CONDITION} metadata in {path}")
        iterations.update(info["erasure_iteration"] for info in selected)
        # Editing counts describe only the edited subset, not the full sparse mask.
        if experiment == "exp_5b_sparse_mosaic":
            selections.update((info["erasure_iteration"], info["n_keys"], info["n_values"])
                              for info in selected)
    if len(selections) != 1 or len(iterations) != 1:
        raise ValueError("Sparse-mosaic and editing results disagree on the selected erasure mask.")
    iteration, n_keys, n_values = selections.pop()
    key_mask, value_mask = trajectory.masks_at_iteration(iteration)
    if (int(key_mask.sum()), int(value_mask.sum())) != (n_keys, n_values):
        raise ValueError("Erasure artifact counts do not match the recorded experiment counts.")
    # Match the notebooks: fine erasure begins at the first single-site round.
    transition_iteration = next(
        (
            i
            for i in range(len(trajectory.score_by_iteration))
            if (
                (trajectory.key_erasure_round_tli == i + 1).sum()
                + (trajectory.value_erasure_round_tli == i + 1).sum()
            )
            == 1
        ),
        None,
    )
    if transition_iteration is None:
        raise ValueError("No coarse-to-fine transition found in the recorded trajectory.")
    if token_labels is None:
        try:
            token_labels = get_fwd_fwd_context_tokens(model_preset, prompt_preset)
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and not str(exc).startswith("Set LNT_MODELS_DIR "):
                raise
            warnings.warn(
                f"Could not load local tokenizer labels for {model_preset}/{prompt_preset}; "
                "using zero-based token numbers instead. Set LNT_MODELS_DIR and provide "
                f"the model's tokenizer files to enable text labels. Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    plots = {}
    for view in ("zoomed_out", "zoomed_in"):
        plot = plot_erasure_trajectory(
            trajectory,
            x_axis="remaining_sites",
            transition_iteration=transition_iteration,
            marked_iterations={SPARSE_CONDITION: iteration},
        )
        if view == "zoomed_in":
            center = plot.x[transition_iteration]
            half_width = max(center * 1.05, abs(plot.x[iteration] - center) * 1.1, 1)
            plot.ax.set_xlim(center + half_width, center - half_width)
        plots[f"erasure/trajectory_{view}"] = plot
    mask_plot = plot_remaining_key_values(trajectory, iteration=iteration, token_labels=token_labels)
    plots["erasure/kv_mask"] = mask_plot
    total_sites = key_mask.size + value_mask.size
    counts = [n_keys, n_values, total_sites - n_keys - n_values]
    with mpl.rc_context(PAPER_MOSAIC_STYLE.rc_params()):
        fig, ax = plt.subplots(figsize=(4.3, 1.25), layout="constrained")
        wedges, _ = ax.pie(counts, colors=["tab:green", "tab:orange", "lightgrey"], startangle=90)
        ax.set_xlim(-1.1, 3.5)
        legend = ax.legend(
            wedges, [f"{label} ({count:,})" for label, count in zip(
                ("Retained keys", "Retained values", "Erased sites"), counts)],
            loc="center left", bbox_to_anchor=(0.5, 0.5), frameon=False,
            fontsize=PAPER_MOSAIC_STYLE.legend_font_size,
        )
        plots["erasure/kv_pie"] = DotPlot(fig, ax, {"wedges": wedges, "legend": legend})
    return plots
