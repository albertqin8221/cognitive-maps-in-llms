from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from cognitive_maps_in_llms.analysis.erasure_figures import _render_artifacts, editing_queries
from cognitive_maps_in_llms.analysis.results import RunResults

pytestmark = pytest.mark.codex


@pytest.mark.parametrize("preset, expected", [
    ("ABC_family_trees_descendant_ancestor",
     ("D --an ancestor--> B?", "B --a descendant--> D?")),
    ("ABC_company_structure", ("B --senior--> D?", "D --junior--> B?")),
    ("ABC_gene_regulatory_network", ("D --activate--> B?", "D --repress--> B?")),
    ("ABC_family_tree_yes_no", ("B --mother--> D?", "D --daughter--> B?")),
])
def test_editing_query_directions(preset, expected):
    assert editing_queries(preset) == expected


@pytest.fixture
def artifact_results(tmp_path):
    directory = tmp_path / "model/prompt/iterative_erasure"
    directory.mkdir(parents=True)
    (directory / "erasure_state.json").write_text(json.dumps({
        "key_erasure_round_tli": [[[1, 2, -1]]],
        "value_erasure_round_tli": [[[1, 3, 4]]],
        "score_by_iteration": [9, 8, 7, 6],
    }))
    (directory / "erasure_metadata.json").write_text(json.dumps({
        "initial_score": 9, "fully_erased_score": 0,
    }))
    for experiment in ("exp_5b_sparse_mosaic", "exp_5c_editing"):
        path = tmp_path / "model/prompt" / experiment / "results.ndjson"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "trial_key": {"condition": "FIDELITY_95_SITES_X2"},
            "trial_info": {"erasure_iteration": 1, "n_keys": 2 if experiment == "exp_5b_sparse_mosaic" else 0,
                           "n_values": 2 if experiment == "exp_5b_sparse_mosaic" else 0},
        }) + "\n")
    return RunResults(results_root=tmp_path, artifacts_root=tmp_path)


def test_artifacts_use_recorded_mask_and_center_transition(artifact_results):
    results = artifact_results
    tmp_path = results.results_root
    directory = tmp_path / "model/prompt/iterative_erasure"
    plots = _render_artifacts(results, "model", "prompt", ["token"])
    try:
        assert plots["erasure/kv_mask"].iteration == 1
        left, right = plots["erasure/trajectory_zoomed_in"].ax.get_xlim()
        assert (left + right) / 2 == pytest.approx(4)
        legend = plots["erasure/kv_pie"].artists["legend"]
        assert [t.get_text() for t in legend.get_texts()] == [
            "Retained keys (2)", "Retained values (2)", "Erased sites (2)",
        ]
    finally:
        for plot in plots.values():
            plt.close(plot.fig)
    state_path = directory / "erasure_state.json"
    state = json.loads(state_path.read_text())
    state["key_erasure_round_tli"] = [[[1, 1, -1]]]
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="counts do not match"):
        _render_artifacts(RunResults(results_root=tmp_path, artifacts_root=tmp_path),
                          "model", "prompt", ["token"])


@pytest.mark.parametrize("error", [
    FileNotFoundError("Model directory not found"),
    OSError("Cannot load tokenizer"),
    ValueError("Set LNT_MODELS_DIR to your model directory"),
])
def test_artifacts_fall_back_to_token_numbers(artifact_results, monkeypatch, error):
    def unavailable(*args):
        raise error

    monkeypatch.setattr(
        "cognitive_maps_in_llms.analysis.erasure_figures.get_fwd_fwd_context_tokens",
        unavailable,
    )
    with pytest.warns(RuntimeWarning, match="using zero-based token numbers"):
        plots = _render_artifacts(artifact_results, "model", "prompt", None)
    try:
        mask = plots["erasure/kv_mask"]
        assert [label.get_text() for label in mask.fig.axes[0].get_yticklabels()] == ["0"]
        assert "erasure/trajectory_zoomed_in" in plots
    finally:
        for plot in plots.values():
            plt.close(plot.fig)


@pytest.mark.parametrize("error", [ValueError("Unknown preset"), KeyError("CTXT_START")])
def test_artifacts_do_not_hide_label_errors(artifact_results, monkeypatch, error):
    def invalid(*args):
        raise error

    monkeypatch.setattr(
        "cognitive_maps_in_llms.analysis.erasure_figures.get_fwd_fwd_context_tokens",
        invalid,
    )
    with pytest.raises(type(error), match=str(error)):
        _render_artifacts(artifact_results, "model", "prompt", None)
