from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from cognitive_maps_in_llms.analysis.results import RunResults
from cognitive_maps_in_llms.analysis.scaling_plots import (
    benchmark_accuracy,
    make_accuracy_scaling_plot,
)


pytestmark = pytest.mark.codex


YES_NO_CASES = [
    {
        "structure": "A --fwd--> B --fwd--> C",
        "query": "A --granddaughter--> C?",
        "answer": "Y",
        "kind": "two-hop",
    },
    {
        "structure": "A --fwd--> B --fwd--> C",
        "query": "A --daughter--> B?",
        "answer": "Y",
        "kind": "one-hop",
    },
]


def _record(
    *,
    probabilities_by_query: dict[str, dict[str, float]],
    seed: int = 0,
) -> dict:
    return {
        "trial_key": {
            "seed": seed,
            "structure": "A --fwd--> B --fwd--> C",
        },
        "results": {
            query: {"probs": probabilities}
            for query, probabilities in probabilities_by_query.items()
        },
    }


def test_benchmark_accuracy_ignores_unknown() -> None:
    records = [
        _record(
            probabilities_by_query={
                "A --granddaughter--> C?": {
                    "Y": 0.20,
                    "N": 0.10,
                    "U": 0.70,
                },
                "A --daughter--> B?": {"Y": 0.60, "N": 0.30, "U": 0.10},
            }
        )
    ]

    accuracy = benchmark_accuracy(
        records,
        YES_NO_CASES,
        benchmark_kind="two-hop",
    )

    assert accuracy == 1.0


def test_benchmark_accuracy_compares_all_named_answers() -> None:
    cases = [
        {
            "structure": "A --fwd--> B --fwd--> C",
            "query": "A --granddaughter--> ?",
            "answer": "C",
            "kind": "two-hop",
        }
    ]
    records = [
        _record(
            probabilities_by_query={
                "A --granddaughter--> ?": {
                    "A": 0.10,
                    "B": 0.25,
                    "C": 0.20,
                    "U": 0.45,
                }
            }
        )
    ]

    assert benchmark_accuracy(records, cases) == 0.0


def test_benchmark_accuracy_averages_cases_and_seeds() -> None:
    records = [
        _record(
            seed=0,
            probabilities_by_query={
                "A --granddaughter--> C?": {"Y": 0.8, "N": 0.2, "U": 0.0},
                "A --daughter--> B?": {"Y": 0.4, "N": 0.6, "U": 0.0},
            },
        ),
        _record(
            seed=1,
            probabilities_by_query={
                "A --granddaughter--> C?": {"Y": 0.3, "N": 0.7, "U": 0.0},
                "A --daughter--> B?": {"Y": 0.9, "N": 0.1, "U": 0.0},
            },
        ),
    ]

    assert benchmark_accuracy(records, YES_NO_CASES) == 0.5


def test_benchmark_accuracy_rejects_unmatched_case() -> None:
    records = [
        _record(
            probabilities_by_query={
                "A --granddaughter--> C?": {"Y": 0.8, "N": 0.2, "U": 0.0},
                "A --daughter--> B?": {"Y": 0.8, "N": 0.2, "U": 0.0},
            }
        )
    ]
    cases = [
        {
            "structure": "A --rev--> B --rev--> C",
            "query": "A --grandmother--> C?",
            "answer": "Y",
            "kind": "two-hop",
        }
    ]

    with pytest.raises(ValueError, match="No result records matched"):
        benchmark_accuracy(records, cases)


def test_make_accuracy_scaling_plot_reads_current_results_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "test-model"
    prompt = "ABC_family_tree_yes_no"
    path = tmp_path / model / prompt / "exp_1_behavior" / "results.ndjson"
    path.parent.mkdir(parents=True)
    record = _record(
        probabilities_by_query={
            "A --granddaughter--> C?": {"Y": 0.2, "N": 0.1, "U": 0.7},
            "A --daughter--> B?": {"Y": 0.1, "N": 0.8, "U": 0.1},
        }
    )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "cognitive_maps_in_llms.analysis.scaling_plots.get_model_size",
        lambda model_preset: 1_000_000.0,
    )

    plot = make_accuracy_scaling_plot(
        RunResults(results_root=tmp_path),
        prompt_preset=prompt,
        benchmark=YES_NO_CASES,
        model_families={"Qwen": [model]},
    )

    assert plot.data[0]["accuracy"] == 1.0
    assert plot.data[0]["ndjson_path"] == path
    line = plot.artists["lines"]["Qwen"]
    assert list(line.get_xdata()) == pytest.approx([1_000_000.0])
    assert list(line.get_ydata()) == pytest.approx([1.0])
    plt.close(plot.fig)
