from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cognitive_maps_in_llms.analysis.main_text_analogs import (
    SEPARABILITY_PANELS,
    render_all_main_text_panels, render_behavior, render_separability,
    save_main_text_analogs, separability_panels,
    comparison_queries, render_additivity,
)
from cognitive_maps_in_llms.analysis.mosaic import plot_mosaic_series
from cognitive_maps_in_llms.analysis.results import RunResults

pytestmark = pytest.mark.codex


@pytest.fixture
def results(tmp_path):
    path = tmp_path / "gemma-27b/ABC_family_tree_yes_no/exp_2_separability/results.ndjson"
    path.parent.mkdir(parents=True)
    records = [
        {"trial_key": {"seed": seed, "patch_location": location},
         "results": {query: {"logits": {"Y": seed + shift, "N": -shift, "U": -2}}
                     for name, query, title in SEPARABILITY_PANELS}}
        for seed in (0, 1) for location, shift in (("CTRL", 1), ("B1", 3))
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return RunResults(results_root=tmp_path)


def test_matches_notebook_plot_and_export_guard(results, tmp_path):
    kwargs = dict(model_preset="gemma-27b", prompt_preset="ABC_family_tree_yes_no")
    plots = render_separability(results, **kwargs)
    reference = plot_mosaic_series(
        {label: results.load_series(**kwargs, experiment="exp_2_separability",
                                   query="E --mother--> D?", patch_location=location)
         for label, location in (("Control", "CTRL"), ("Mosaic", "B1"))},
        value_kind="probability_ratio", title="Is ‹E›'s mother ‹D›?", legend=False,
    )
    try:
        actual = plots["separability/is_Es_mother_D"]
        for label in ("Control", "Mosaic"):
            a, b = actual.artists["scatter"][label], reference.artists["scatter"][label]
            np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
            np.testing.assert_array_equal(a.get_facecolors(), b.get_facecolors())
        assert actual.ax.get_ylim() == reference.ax.get_ylim()
        assert actual.ax.get_title() == reference.ax.get_title()
        np.testing.assert_array_equal(actual.fig.get_size_inches(), reference.fig.get_size_inches())
        output = tmp_path / "output"
        collision = output / "separability/legend.pdf"
        collision.parent.mkdir(parents=True)
        collision.write_bytes(b"existing artifact")
        with pytest.raises(FileExistsError):
            save_main_text_analogs(plots, output)
        assert list(output.rglob("*.svg")) == []
        assert collision.read_bytes() == b"existing artifact"
        paths = save_main_text_analogs(plots, tmp_path / "fresh")
        assert len(paths) == 14
        assert all(p.stat().st_size > 0 for p in paths)
    finally:
        plt.close(reference.fig)
        for plot in plots.values():
            plt.close(plot.fig)


def test_missing_query_prevents_rendering(results):
    path = next(results.results_root.rglob("*.ndjson"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    del records[-1]["results"][SEPARABILITY_PANELS[-1][1]]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    before = plt.get_fignums()
    with pytest.raises(KeyError, match="C --grandmother--> A"):
        render_separability(results, model_preset="gemma-27b", prompt_preset="ABC_family_tree_yes_no")
    assert plt.get_fignums() == before


@pytest.mark.parametrize("preset", [
    "unknown_preset", "ABC_alien_family_tree_yes_no", "ABC_alien_older_younger",
    "AB_CD_family_trees_yes_no", "ABC_family_tree_who_is",
    "ABC_older_younger_unique_locations_who_is",
])
def test_unsupported_prompts_raise_clear_error(results, preset):
    with pytest.raises(ValueError, match="No separability query selection"):
        render_separability(results, model_preset="gemma-27b", prompt_preset=preset)


@pytest.mark.parametrize("preset, first_query, first_title, count", [
    ("ABC_company_structure", "E --senior--> D?", "Is ‹E› senior to ‹D›?", 6),
    ("ABC_older_younger", "E --younger--> D?", "Is ‹E› younger than ‹D›?", 6),
    ("ABC_gene_regulatory_network", "D --activate--> E?", "Does ‹D› activate ‹E›?", 5),
    ("ABC_variable_equivalence", "E --equal--> D?", "Is ‹E› equal to ‹D›?", 6),
])
def test_domain_queries_and_titles(preset, first_query, first_title, count):
    panels = separability_panels(preset)
    assert len(panels) == count
    assert panels[0][1:] == (first_query, first_title)
    assert len({name for name, _, _ in panels}) == count


@pytest.mark.parametrize("preset, queries", [
    ("ABC_family_trees_descendant_ancestor", (
        "E --a descendant--> D?", "E --a descendant--> A?",
        "E --an ancestor--> F?", "E --an ancestor--> C?",
        "A --an ancestor--> C?", "C --a descendant--> A?",
    )),
    ("ABC_gene_regulatory_network", (
        "D --activate--> E?", "E --activate--> F?",
        "A --activate--> E?", "E --activate--> C?",
        "A --activate--> C?",
    )),
    ("ABC_neural_circuit", (
        "D --excite--> E?", "E --excite--> F?",
        "A --excite--> E?", "E --excite--> C?",
        "A --excite--> C?",
    )),
    ("ABC_proposition_equivalence", (
        "E --equivalent to--> D?", "E --equivalent to--> A?",
        "E --equivalent to--> F?", "E --equivalent to--> C?",
        "A --equivalent to--> C?", "C --equivalent to--> A?",
    )),
])
def test_agreed_special_domain_queries(preset, queries):
    assert tuple(query for _, query, _ in separability_panels(preset)) == queries


@pytest.mark.parametrize("alien_available", [False, True])
def test_behavior_optional_alien_comparison(tmp_path, alien_available):
    record = {
        "trial_key": {"seed": 0, "structure": "A --fwd--> B --fwd--> C"},
        "results": {"C --grandmother--> A?": {"logits": {"Y": 2, "N": -1, "U": 0}}},
    }
    prompts = ["ABC_family_tree_yes_no"]
    if alien_available:
        prompts.append("ABC_alien_family_tree_yes_no")
    for prompt in prompts:
        path = tmp_path / "gemma-27b" / prompt / "exp_1_behavior/results.ndjson"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record) + "\n")
    plots = render_behavior(
        RunResults(results_root=tmp_path), model_preset="gemma-27b",
        prompt_preset="ABC_family_tree_yes_no", include_scaling=False,
    )
    try:
        expected = ["behavior/example"]
        if alien_available:
            expected.append("behavior/example_legend")
        assert list(plots) == expected
        assert plots["behavior/example"].ax.get_legend() is None
        labels = list(plots["behavior/example"].artists["scatter"])
        assert labels == (["Domain-appropriate names", "Synthetic names"]
                          if alien_available else ["Domain-appropriate names"])
        assert plots["behavior/example"].ax.get_title() == "Is ‹C›'s grandmother ‹A›?"
        if alien_available:
            legend = plots["behavior/example_legend"].artists["legend"]
            assert [text.get_text() for text in legend.get_texts()] == labels
        paths = save_main_text_analogs(plots, tmp_path / "export", formats=("pdf",))
        assert {p.stem for p in paths} == ({"example", "example_legend"}
                                         if alien_available else {"example"})
        assert all(p.suffix == ".pdf" and p.read_bytes().startswith(b"%PDF-") for p in paths)
        assert not list((tmp_path / "export").rglob("*.svg"))
    finally:
        for plot in plots.values():
            plt.close(plot.fig)


@pytest.mark.parametrize("preset, relation", [
    ("ABC_proposition_equivalence", "equivalent to"),
    ("ABC_variable_equivalence", "equal"),
])
def test_additivity_includes_equivalence(preset, relation):
    assert comparison_queries(preset, "additivity") == (
        f"B --{relation}--> A?", f"B --{relation}--> C?",
    )


@pytest.mark.parametrize("preset, relations", [
    ("ABC_gene_regulatory_network", ("activate", "repress")),
    ("ABC_neural_circuit", ("excite", "inhibit")),
])
def test_figure_four_biology_only_uses_forward_causal_queries(preset, relations):
    assert comparison_queries(preset, "retrospective_prospective") == tuple(
        f"A --{relation}--> C?" for relation in relations
    )


def test_figure_three_family_query_order():
    assert comparison_queries("ABC_family_tree_yes_no", "additivity") == (
        "B --daughter--> A?", "B --mother--> A?",
        "B --daughter--> C?", "B --mother--> C?",
    )


def test_render_all_main_text_panels_dispatches_selected_groups(results, monkeypatch):
    from cognitive_maps_in_llms.analysis import main_text_analogs

    calls = []

    def render_behavior_stub(received_results, **kwargs):
        calls.append(("behavior", received_results, kwargs))
        return {"behavior/example": object()}

    def render_additivity_stub(received_results, **kwargs):
        calls.append(("additivity", received_results, kwargs))
        return {"additivity/example": object()}

    monkeypatch.setattr(main_text_analogs, "render_behavior", render_behavior_stub)
    monkeypatch.setattr(main_text_analogs, "render_additivity", render_additivity_stub)

    plots = render_all_main_text_panels(
        results,
        model_preset="gemma-27b",
        prompt_preset="ABC_family_tree_yes_no",
        groups=("behavior", "additivity"),
    )

    assert list(plots) == ["behavior/example", "additivity/example"]
    assert [call[0] for call in calls] == ["behavior", "additivity"]
    assert all(call[1] is results for call in calls)
    assert all(
        call[2]
        == {
            "model_preset": "gemma-27b",
            "prompt_preset": "ABC_family_tree_yes_no",
            **({"human_alpha": 1.0, "alien_alpha": 1.0}
               if call[0] == "behavior" else {}),
        }
        for call in calls
    )


def test_render_all_main_text_panels_closes_completed_groups_after_failure(
    results,
    monkeypatch,
):
    from cognitive_maps_in_llms.analysis import main_text_analogs

    def render_behavior_stub(*args, **kwargs):
        fig, _ = plt.subplots()
        return {"behavior/example": object()}

    def render_separability_stub(*args, **kwargs):
        raise RuntimeError("rendering failed")

    monkeypatch.setattr(main_text_analogs, "render_behavior", render_behavior_stub)
    monkeypatch.setattr(
        main_text_analogs,
        "render_separability",
        render_separability_stub,
    )
    figures_before = set(plt.get_fignums())

    with pytest.raises(RuntimeError, match="rendering failed"):
        render_all_main_text_panels(
            results,
            model_preset="gemma-27b",
            prompt_preset="ABC_family_tree_yes_no",
            groups=("behavior", "separability"),
        )

    assert set(plt.get_fignums()) == figures_before


@pytest.mark.parametrize(
    ("groups", "error", "message"),
    [
        ("behavior", TypeError, "sequence"),
        (("unknown",), ValueError, "Unknown figure group"),
        (("behavior", "behavior"), ValueError, "duplicates"),
    ],
)
def test_render_all_main_text_panels_validates_groups(
    results,
    groups,
    error,
    message,
):
    with pytest.raises(error, match=message):
        render_all_main_text_panels(
            results,
            model_preset="gemma-27b",
            prompt_preset="ABC_family_tree_yes_no",
            groups=groups,
        )


@pytest.mark.parametrize("preset, relation", [
    ("ABC_gene_regulatory_network", "activate"),
    ("ABC_neural_circuit", "excite"),
])
def test_additivity_biology_queries_in_both_directions(preset, relation):
    assert comparison_queries(preset, "additivity") == (
        f"A --{relation}--> B?", f"B --{relation}--> A?",
        f"B --{relation}--> C?", f"C --{relation}--> B?",
    )


@pytest.mark.parametrize("prompt", [
    "ABC_family_tree_yes_no", "ABC_company_structure", "ABC_older_younger",
    "ABC_taller_shorter", "ABC_faster_slower",
    "ABC_proposition_equivalence",
    "ABC_variable_equivalence",
    "ABC_east_west",
    "ABC_north_south",
    "ABC_ahead_behind",
    "ABC_family_trees_descendant_ancestor",
    "ABC_older_younger_unique_locations_yes_no",
    "ABC_gene_regulatory_network", "ABC_neural_circuit",
])
def test_behavior_scaling_overlay_and_missing_inputs(tmp_path, prompt):
    from cognitive_maps_in_llms.analysis.main_text_analogs import (
        alien_prompt_preset, render_behavior_scaling,
    )
    from cognitive_maps_in_llms.analysis.scaling_plots import DEFAULT_MODEL_FAMILIES
    from cognitive_maps_in_llms.benchmarks import get_default_benchmark_name, load_benchmark

    alien = alien_prompt_preset(prompt)
    assert alien == (None if prompt in {"ABC_gene_regulatory_network", "ABC_neural_circuit"}
                     else prompt.replace("ABC_", "ABC_alien_", 1))
    cases = load_benchmark(get_default_benchmark_name(prompt))
    alien_paths = []
    for preset in ([prompt, alien] if alien else [prompt]):
        records = {}
        for case in cases:
            record = records.setdefault(case["structure"], {
                "trial_key": {"structure": case["structure"], "seed": 0},
                "results": {},
            })
            answer = case["answer"]
            if preset == alien:
                answer = "N" if answer == "Y" else "Y"
            record["results"][case["query"]] = {
                "probs": {answer: 0.9, "N" if answer == "Y" else "Y": 0.1},
            }
        for models in DEFAULT_MODEL_FAMILIES.values():
            for model in models:
                path = tmp_path / model / preset / "exp_1_behavior/results.ndjson"
                path.parent.mkdir(parents=True)
                path.write_text("".join(json.dumps(r) + "\n" for r in records.values()))
                if preset == alien:
                    alien_paths.append(path)
    results = RunResults(results_root=tmp_path)
    before = set(plt.get_fignums())
    plot = render_behavior_scaling(
        results, prompt_preset=prompt, human_alpha=0.8, alien_alpha=0.4,
    )
    try:
        assert len(plot.ax.lines) == (6 if alien else 3)
        lower, upper = plot.ax.get_ylim()
        assert upper == pytest.approx(1.05)
        if alien:
            assert lower < 0.0
        else:
            assert lower == pytest.approx(0.35)
        assert all(row["accuracy"] == 1.0 for row in plot.data)
        for family, line in plot.artists["lines"].items():
            assert line.get_linestyle() == "-"
            assert line.get_alpha() == 0.8
            if alien:
                other = plot.artists["alien_plot"].artists["lines"][family]
                assert other.get_linestyle() == "--"
                assert other.get_alpha() == 0.4
                assert line.get_color() == other.get_color()
                np.testing.assert_array_equal(line.get_xdata(), other.get_xdata())
                np.testing.assert_array_equal(other.get_ydata(), 0.0)
        if alien:
            assert [t.get_text() for t in plot.ax.get_legend().get_texts()][-2:] == [
                "Domain-appropriate names", "Synthetic names",
            ]
    finally:
        plt.close(plot.fig)
    if alien:
        alien_paths[-1].unlink()
        with pytest.raises(FileNotFoundError):
            render_behavior_scaling(RunResults(results_root=tmp_path), prompt_preset=prompt)
        assert set(plt.get_fignums()) == before
