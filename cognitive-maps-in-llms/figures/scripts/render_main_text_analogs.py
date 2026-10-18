"""Batch-export main-text analogs as PDFs, with a CSV report of missing inputs.

Plotting and scientific selections live in the analysis package. This script
selects batches, checks inputs, exports figures, and closes them after each group.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROMPTS = (
    "ABC_family_tree_yes_no",
    "ABC_company_structure",
    "ABC_older_younger",
    "ABC_taller_shorter",
    "ABC_faster_slower",
    "ABC_east_west",
    "ABC_north_south",
    "ABC_ahead_behind",
    "ABC_family_trees_descendant_ancestor",
    "ABC_older_younger_unique_locations_yes_no",
    "ABC_gene_regulatory_network",
    "ABC_neural_circuit",
    "ABC_proposition_equivalence",
    "ABC_variable_equivalence",
)
EXPERIMENTS = {
    "behavior": ("exp_1_behavior",),
    "separability": ("exp_2_separability",),
    "additivity": ("exp_3_additivity",),
    "retrospective_prospective": ("exp_4_retrospective_prospective",),
    "erasure": ("exp_5a_bottleneck", "exp_5b_sparse_mosaic", "exp_5c_editing"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path,
                        help="Deprecated; erasure trajectories now live under --results-root")
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--models", nargs="+")
    model_source.add_argument("--all-models", action="store_true")
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompts", nargs="+")
    prompt_source.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace generated PDFs and refresh the skip report")
    parser.add_argument("--group", nargs="+", choices=tuple(EXPERIMENTS))
    parser.add_argument(
        "--scaling-only",
        action="store_true",
        help="Render only scaling plots across all 11 models",
    )
    args = parser.parse_args()
    if args.scaling_only and (args.models or args.all_models or args.group):
        parser.error(
            "--scaling-only cannot be combined with --models, --all-models, or --group"
        )
    args.groups = (
        ["scaling"]
        if args.scaling_only
        else list(dict.fromkeys(args.group or EXPERIMENTS))
    )
    if not args.results_root.is_dir():
        parser.error(f"Results root does not exist: {args.results_root}")
    return args


def all_models() -> list[str]:
    from cognitive_maps_in_llms.analysis.scaling_plots import DEFAULT_MODEL_FAMILIES

    return [model for family in DEFAULT_MODEL_FAMILIES.values() for model in family]


def required_files(
    args: argparse.Namespace, model: str, prompt: str, group: str
) -> list[Path]:
    """Inputs needed before a group can be rendered."""
    if group == "scaling":
        from cognitive_maps_in_llms.analysis.main_text_analogs import alien_prompt_preset

        alien = alien_prompt_preset(prompt)
        prompts = [prompt, alien] if alien is not None else [prompt]
        return [
            args.results_root / name / preset / "exp_1_behavior" / "results.ndjson"
            for preset in prompts for name in all_models()
        ]
    paths = [
        args.results_root / model / prompt / experiment / "results.ndjson"
        for experiment in EXPERIMENTS[group]
    ]
    if group == "erasure":
        directory = args.results_root / model / prompt / "iterative_erasure"
        paths.extend(
            directory / name
            for name in (
                "erasure_state.json",
                "erasure_metadata.json",
            )
        )
    return paths


def render_group(args: argparse.Namespace, model: str, prompt: str, group: str) -> dict:
    """Delegate plot creation to the analysis package."""
    from cognitive_maps_in_llms.analysis.main_text_analogs import (
        render_behavior,
        render_behavior_scaling,
        render_separability,
        render_additivity,
        render_retrospective_prospective,
    )
    from cognitive_maps_in_llms.analysis.erasure_figures import render_erasure
    from cognitive_maps_in_llms.analysis.results import RunResults

    results = RunResults(results_root=args.results_root)
    if group == "scaling":
        return {
            "scaling": render_behavior_scaling(
                results,
                prompt_preset=prompt,
            )
        }
    renderers = {
        "behavior": render_behavior,
        "separability": render_separability,
        "additivity": render_additivity,
        "retrospective_prospective": render_retrospective_prospective,
        "erasure": render_erasure,
    }
    options = {"include_scaling": False} if group == "behavior" else {}
    return renderers[group](
        results,
        model_preset=model,
        prompt_preset=prompt,
        **options,
    )


def is_missing_selection(error: KeyError | ValueError) -> bool:
    """Recognize loader errors for absent queries/conditions; propagate bugs."""
    # Loaders currently expose these cases through messages, not distinct types.
    if isinstance(error, KeyError):
        message = str(error.args[0]) if error.args else ""
        return (
            message.startswith(("Query ", "Benchmark query "))
            and " is missing from record " in message
        )
    message = str(error)
    return (
        message.startswith("No records in ") and "matched trial axes" in message
    ) or (
        message.startswith("No result records matched ")
        and "benchmark case(s)" in message
    )


def export_group(
    args: argparse.Namespace, model: str, prompt: str, group: str
) -> str | None:
    """Save one group; return a skip reason when required data are missing."""
    import matplotlib.pyplot as plt
    from cognitive_maps_in_llms.analysis.main_text_analogs import save_main_text_analogs

    missing = [
        str(path)
        for path in required_files(args, model, prompt, group)
        if not path.is_file()
    ]
    if missing:
        return "Missing files: " + "; ".join(missing)

    destination = (
        args.output_dir / prompt
        if group == "scaling"
        else args.output_dir / model / prompt
    )
    before = set(plt.get_fignums())
    try:
        try:
            plots = render_group(args, model, prompt, group)
        except (KeyError, ValueError) as error:
            if not is_missing_selection(error):
                raise
            return str(error.args[0]) if isinstance(error, KeyError) else str(error)
        saved = save_main_text_analogs(
            plots, destination, formats=("pdf",), overwrite=args.overwrite,
        )
        print(f"Saved {len(saved)} files: {model}/{prompt}/{group}")
        return None
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)


def main() -> None:
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")

    if args.scaling_only:
        models = ["all-models"]
    else:
        models = all_models() if args.all_models else args.models or ["gemma-27b"]
    prompts = (
        PROMPTS if args.all_prompts else args.prompts or ["ABC_family_tree_yes_no"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "skipped_groups.csv"
    with report_path.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as report:
        writer = csv.writer(report)
        writer.writerow(("model", "prompt", "group", "reason"))
        report.flush()
        for prompt in dict.fromkeys(prompts):
            for model in dict.fromkeys(models):
                for group in args.groups:
                    reason = export_group(args, model, prompt, group)
                    if reason is not None:
                        previous = (args.output_dir / prompt / "scaling.pdf"
                                    if group == "scaling" else
                                    args.output_dir / model / prompt / group)
                        if args.overwrite and previous.exists():
                            reason += "; existing panels were retained and may be stale"
                        writer.writerow((model, prompt, group, reason))
                        report.flush()
                        print(f"Skipped {model}/{prompt}/{group}: {reason}")
    print(f"Skip report: {report_path}")


if __name__ == "__main__":
    main()
