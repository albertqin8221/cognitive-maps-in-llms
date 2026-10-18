# Plotting utilities

[README](../README.md)

## Locating outputs

The package uses a `RunResults` object to organize outputs from a single run.
To create one,  provide the root containing mosaic experiment results,
other artifacts, or both:

```python
from pathlib import Path

from cognitive_maps_in_llms.analysis import RunResults


results = RunResults(
    results_root=Path("/absolute/path/to/results/final_run_2026-09-30"),
    artifacts_root=Path("/absolute/path/to/artifacts/final_run_2026-09-30"),
)
```

By default, `RunResults` expects mosaic results at 
`<results_root>/<model_preset>/<prompt_preset>/<experiment>/results.ndjson`, 
erasure trajectories at 
`<results_root>/<model_preset>/<prompt_preset>/iterative_erasure/`, 
and activation recordings at 
`<artifacts_root>/<model_preset>/<prompt_preset>/<activation_and_kv_recording>/`.

## Plotting the results of a mosaic experiment

### 1. Load the conditions to compare

Mosaic experiment results are loaded as `Series` objects. Each series contains
the results for one query and experimental condition across all seeds. A typical
plot compares a control series with a mosaic series.

Use `get_mosaic_options` to inspect the queries and "trial_axes" associated with
a given experiment:

```python
options = results.get_mosaic_options(
    model_preset="gemma-27b",
    prompt_preset="ABC_family_tree_yes_no",
    experiment="exp_2_separability",
)

options["queries"]
options["trial_axes"]
```

Call `load_series` once for each condition. Supply every (non-seed) field in the
experiment's `trial_axes` exactly once:

```python
control = results.load_series(
    model_preset="gemma-27b",
    prompt_preset="ABC_family_tree_yes_no",
    experiment="exp_2_separability",
    query="E --mother--> D?",
    patch_location="CTRL",
)

mosaic = results.load_series(
    model_preset="gemma-27b",
    prompt_preset="ABC_family_tree_yes_no",
    experiment="exp_2_separability",
    query="E --mother--> D?",
    patch_location="B1",
)
```

### 2. Plot the conditions

Pass a mapping from labels to series to `plot_mosaic_series`. The labels are
used in the optional legend.

```python
from cognitive_maps_in_llms.analysis import plot_mosaic_series


plot = plot_mosaic_series(
    {"Control": control, "Mosaic": mosaic},
    title="Is ‹E›'s mother ‹D›?",
    legend=True,
)

plot.save("figure.svg")
```

#### Parameters

- `series` : mapping from `str` to `Series`

  The series to plot. Each key identifies a series and is used as its legend
  label.

- `value_kind` : `{"probs", "logits", "normalized_logits", "probability_ratio"}`, default: `"probability_ratio"`

  The quantity plotted for each answer symbol. Normalized logits are centered
  within the plotted symbols in each record; probability ratios divide each
  selected probability by the geometric mean of the selected subset.

- `symbols` : sequence of `str`, optional

  The answer symbols to include, in plotting order.

- `symbol_labels` : mapping from `str` to `str`, optional

  Replacement tick labels for individual answer symbols.

- `jitter` : `float`, optional

  The maximum random horizontal displacement of points from their series
  position. Overrides the value in `style`.

- `point_size` : `float`, optional

  The marker area in points squared. Overrides the value in `style`.

- `alpha` : `float`, optional

  The marker opacity. Overrides the value in `style`.

- `style` : `MosaicPlotStyle`, optional

  Reusable visual settings for mosaic plots. The paper style is used by
  default.

## Plotting benchmark accuracy across model scale

The function `make_accuracy_scaling_plot` loads the behavioral results for each
model, scores them against a benchmark, and plots accuracy against model size:

```python
from cognitive_maps_in_llms.analysis import make_accuracy_scaling_plot


plot = make_accuracy_scaling_plot(
    results,
    prompt_preset="ABC_family_tree_yes_no",
    title="All 2-step Family-Tree Inferences",
)

plot.save("human_two_hop_accuracy.svg")
```

#### Parameters

- `results` : `RunResults`

  The run containing the behavioral results.

- `prompt_preset` : `str`

  The prompt preset to score. Its associated benchmark is used by default.

- `benchmark` : benchmark name or sequence of benchmark cases, optional

  The benchmark used to score the results. If omitted, it is inferred from
  `prompt_preset`.

- `benchmark_kind` : `str` or `None`, default: `"two-hop"`

  The benchmark cases to include. Pass `"one-hop"` to score one-hop cases or
  `None` to score every case.

- `model_families` : mapping from `str` to sequences of `str`, optional

  The model presets to include, grouped by model family. The keys are used as
  family labels. By default, the plot includes the configured Qwen 3, Llama
  3.1, and Gemma 3 models.

- `experiment` : `str`, default: `"exp_1_behavior"`

  The mosaic experiment containing the behavioral results.

- `ignored_answer_symbols` : sequence of `str`, default: `("U",)`

  Answer symbols excluded when selecting the most probable answer.

- `chance` : `float`, optional

  The accuracy at which to draw a horizontal chance-level reference line.

- `marker` : Matplotlib marker, default: `"o"`

  The marker used for each model.

- `markersize` : `float`, default: `4.5`

  The marker size in points.

- `linewidth` : `float`, default: `1.25`

  The width of the lines connecting models from the same family.

- `ylim` : pair of `float` or `None`, default: `(0.35, 1.05)`

  The vertical axis limits. Pass `None` to use Matplotlib's automatic limits.

- `yticks` : sequence of `float` or `None`, default: `(0.4, 0.6, 0.8, 1.0)`

  The vertical tick locations. Pass `None` to use Matplotlib's automatic ticks.

- `legend_loc` : `str`, default: `"center left"`

  The legend location passed to Matplotlib.

- `legend_bbox_to_anchor` : pair of `float` or `None`, default: `(1.04, 0.52)`

  The legend anchor passed to Matplotlib.

## Plotting PCA trajectories

### 1. Load the activation recordings

Recordings are loaded as a `ActivationRecordings` dataclass:

```python
recordings = results.load_activation_recordings(
    model_preset="llama-70b",
    prompt_preset="ABC_family_tree_who_is",
)
```

The `ActivationRecordings` object contains a memory-mapped array of recorded 
activations and their associated metadata.

### 2. Compute the PCA projection

```python
from cognitive_maps_in_llms.analysis import compute_pca_projection

projection = compute_pca_projection(
    recordings,
    token="C1",
    reference_layer=30,
    n_components=3,
)
```

By default, PCA is fitted across structures and seeds at one token and reference
layer. Activations are then centered and projected onto these PCA axes. 
Pass `average_over_seeds=True` to average name substitutions before fitting PCA.

#### Parameters

- `recordings` : `ActivationRecordings`

  The loaded activation recordings.

- `token` : `str` or `int`

  One landmark (such as `"C1"` or `"R_BC.0"`) or a zero-based index into the
  recorded context.

- `reference_layer` : `int`

  Reference layer at which to run the PC. 

- `n_components` : `int`, default: `3`

  Number of principal components to compute.

- `random_state` : `int`, default: `0`

  Random seed used when scikit-learn's `auto` solver selects randomized SVD.

- `average_over_seeds` : `bool`, default: `False`

  Fit PCA to seed-averaged activations at the reference layer. Note that with 
  four graph structures, at most three components are allowed.

### 3. Plot trajectories in PC space

```python
from cognitive_maps_in_llms.analysis import plot_pca_activity_space

plot = plot_pca_activity_space(
    projection,
    components=(1, 2),
    title="Second entity",
    legend=False,
)

plot.save("pca_second_entity.svg")
```

Bold trajectories show seed averages; faint trajectories show individual seeds.

#### Parameters

- `components` : tuple of two or three `int`, default: `(1, 2)`

  PCs to display, numbered starting at **1**. Use `(1, 2, 3)` for 3D.

- `show_seed_trajectories` : `bool`, default: `True`

  Whether to display individual seed trajectories.

- `marker` : Matplotlib marker or `None`, default: `None`

  Optional markers for layers along the mean trajectories.

## Plot a pruning trajectory

### 1. Load a pruning trajectory

Pruning trajectories are loaded as `ErasureTrajectory` objects that contain
the complete history of scores and pruning masks. Use `load_erasure_trajectory`
to load the trajectory for a model and prompt preset:

```python
from cognitive_maps_in_llms.analysis import (
    plot_erasure_trajectory,
    plot_remaining_key_values,
)

trajectory = results.load_erasure_trajectory(
    model_preset="qwen-0p6b",
    prompt_preset="ABC_family_tree_who_is",
)
```

### 2. Plot performance over the trajectory

Use `plot_erasure_trajectory` to plot either the score or normalized fidelity
against either the pruning iteration or the number of remaining key and value
sites:

```python
plot = plot_erasure_trajectory(
    trajectory,
    x_axis="remaining_sites",
    transition_iteration=40,
    marked_iterations={"Example condition": 75},
)

plot.save("erasure_fidelity.svg")
```

#### Parameters

- `trajectory` : `ErasureTrajectory` or path-like

  The trajectory to plot. A path may be supplied instead of loading the
  trajectory separately.

- `x_axis` : `{"iteration", "remaining_sites"}`, default: `"iteration"`

  The horizontal coordinate. The remaining-sites axis is inverted so that
  erasure proceeds from left to right.

- `y_axis` : `{"score", "fidelity"}`, default: `"score"`

  The recorded objective score or its normalized fidelity.

- `color` : Matplotlib color, default: `"tab:blue"`

  The trajectory line color.

- `transition_iteration` : `int`, optional

  Erasure iteration at which to draw a dotted vertical line.

- `marked_iterations` : mapping from `str` to `int`, optional

  Labeled erasure iterations to mark on the trajectory. Labels are attached to
  the marker artists and can be displayed with `plot.ax.legend()`.

### 3. Plot retained keys and values

`plot_remaining_key_values` plots the number of retained keys and values
for each context token and layer at a specified pruning iteration.

Text labels from `get_fwd_fwd_context_tokens` require `LNT_MODELS_DIR` and local
tokenizer files for the selected model; dummy mode loads no model weights.
To use zero-based token numbers, omit both the label-generation call and the
`token_labels` argument below. The shared erasure renderer used by the preview
notebook and batch script warns and falls back to token numbers automatically
when the model directory is unconfigured or local tokenizer files cannot be loaded.

```python
from cognitive_maps_in_llms.analysis import get_fwd_fwd_context_tokens

token_labels = get_fwd_fwd_context_tokens(
    model_preset="qwen-0p6b",
    prompt_preset="ABC_family_tree_who_is",
)

plot = plot_remaining_key_values(
    trajectory,
    iteration=100,
    token_labels=token_labels,
)

plot.save("remaining_key_values.svg")
```

#### Parameters

- `trajectory` : `ErasureTrajectory` or path-like

  The trajectory to plot. A path may be supplied instead of loading the
  trajectory separately.

- `iteration` : `int`

  The erasure iteration to display.

- `token_labels` : sequence of `str`, optional

  Labels for the context-token axis. The sequence must contain one label per
  context token.

- `cmap` : Matplotlib colormap, default: `"bone"`

  The colormap used for the retained-head counts.

## Rendering main-text analogs across models and settings

The package can reproduce panels analogous to the main-text figures for any
supported model and prompt preset, intended for systematic comparisons across 
models and relational-reasoning settings.

Use `render_all_main_text_panels` to render all five groups for one model and
prompt preset with their standard options:

```python
from cognitive_maps_in_llms.analysis import render_all_main_text_panels

plots = render_all_main_text_panels(
    results,
    model_preset="gemma-27b",
    prompt_preset="ABC_family_tree_yes_no",
)
```

Note that when the `behavior` group is selected, this function includes a 
cross-model scaling plot, which requires `exp_1_behavior` results for all 11 models.

##### Parameters

- `results` : `RunResults`

  Results and artifacts from the run to plot.

- `model_preset` : `str`

  Model preset whose results should be plotted.

- `prompt_preset` : `str`

  Prompt preset whose results should be plotted.

- `groups` : sequence of `{"behavior", "separability", "additivity", "retrospective_prospective", "erasure"}`, optional

  Figure groups to render, in order. By default, all five groups are rendered.

##### Returns

- `dict[str, plot]`

  Plot objects keyed by names such as `behavior/example`,
  `separability/is_Es_mother_D`, and `erasure/kv_mask`.

---

The [preview notebook](../figures/notebooks/supplementary/main_text_analogs_preview.ipynb)
uses this function to generate all panels for a given model and prompt preset.

Alternatively, the [rendering script](../figures/scripts/render_main_text_analogs.py)
can be used to generate panels for multiple models and prompt presets at once.
The cross-model scaling plots should be generated using the flag `--scaling-only`.

To generate the comparisons presented in the supplementary information, run:

```bash
RESULTS_ROOT="results/final_run_2026-09-30"

# Gemma-27B across supported prompt presets
python figures/scripts/render_main_text_analogs.py --all-prompts \
  --results-root "$RESULTS_ROOT" \
  --output-dir figures/generated/all_panels_gemma_x_all_tasks --overwrite

# All models evaluated in the "family-trees" setting
python figures/scripts/render_main_text_analogs.py --all-models \
  --results-root "$RESULTS_ROOT" \
  --output-dir figures/generated/all_panels_models_x_family_trees --overwrite

# Cross-model scaling panels
python figures/scripts/render_main_text_analogs.py --scaling-only --all-prompts \
  --results-root "$RESULTS_ROOT" \
  --output-dir figures/generated/scaling_panel_all_tasks --overwrite
```

#### Options:

- `--results-root`: mosaic-experiment results and erasure trajectories.
- `--artifacts-root`: deprecated and ignored; erasure trajectories now live under `--results-root`.
- `--output-dir`: destination for PDF panels. The batch script saves PDFs directly.
- `--overwrite`: replace generated PDFs and refresh the skip report; retain unrelated files and flag potentially stale skipped panels.
- `--models MODEL [MODEL ...]`: select models.
- `--prompts PROMPT [PROMPT ...]`: select prompt presets.
- `--all-models`: select all 11 models.
- `--all-prompts`: select all 14 supported prompt presets.
- `--group GROUP [GROUP ...]`: render only the specified figure groups.
- `--scaling-only`: render scaling plots only; cannot be combined with
  `--models`, `--all-models`, or `--group`.

`--all-models` and `--all-prompts` may be combined. Supported prompts are the
non-alien ABC yes/no presets; who-is and AB_CD presets are excluded.
Behavior panels and scaling plots include synthetic-name comparisons for all
12 supported presets with alien counterparts, including equivalence, spatial,
descendant/ancestor, and exact-age yes/no tasks. The two biology presets retain
a single series because their original presets already use synthetic names.
Scaling plots require both counterparts across all models; example panels add
the synthetic series when the corresponding result file is available.
Scaling panels retain the original y-axis range `(0.35, 1.05)` when it fits
all plotted series, including marker padding. Otherwise only the lower bound
is extended; the upper bound stays at `1.05`.

## Common plotting options

The plotting functions use the following conventions when these general
presentation options apply:

- `title` : `str`, optional

  The axes title.

- `xlabel`, `ylabel` : `str`, optional

  The axis labels. When `ylabel` is omitted, the plotting function supplies an
  appropriate default.

- `ax` : `matplotlib.axes.Axes`, optional

  An existing axes on which to draw the plot. If omitted, a new figure and axes
  are created.

- `figsize` : `(float, float)`, optional

  The figure dimensions in inches when the plotting function creates a figure.

- `legend` : `bool`, optional

  Whether to display a legend.

- `colors` : mapping or sequence of Matplotlib colors, optional

  The colors assigned to the plotted groups. A mapping associates colors with
  labels; a sequence assigns colors in plotting order.

Each plotting function returns an object that exposes the underlying Matplotlib
figure and axes and provides a `save` method. Close figures with
`plt.close(plot.fig)` when producing many plots in a loop.
