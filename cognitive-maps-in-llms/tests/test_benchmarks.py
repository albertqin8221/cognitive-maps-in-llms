from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_maps_in_llms.benchmarks import (
    get_default_benchmark_name,
    load_benchmark,
)
from cognitive_maps_in_llms.benchmarks.loader import (
    DEFAULT_BENCHMARK_BY_PROMPT_PRESET,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib


pytestmark = pytest.mark.codex


BENCHMARK_CASE_COUNTS = {
    "ahead_behind": 24,
    "company_structure": 24,
    "east_west": 24,
    "family_tree_descendant_ancestor": 48,
    "family_tree_who_is": 18,
    "family_tree_yes_no": 48,
    "faster_slower": 24,
    "gene_regulatory_network": 24,
    "neural_circuit": 24,
    "north_south": 24,
    "older_younger": 24,
    "older_younger_unique_locations_who_is": 28,
    "older_younger_unique_locations_yes_no": 88,
    "proposition_equivalence": 36,
    "taller_shorter": 24,
    "variable_equivalence": 18,
}


def test_load_benchmark_returns_packaged_cases_in_order() -> None:
    cases = load_benchmark("family_tree_yes_no")

    assert len(cases) == 48
    assert cases[0] == {
        "structure": "A --fwd--> B --fwd--> C",
        "query": "A --daughter--> B?",
        "answer": "Y",
        "kind": "one-hop",
    }


def test_load_who_is_benchmark() -> None:
    cases = load_benchmark("family_tree_who_is.toml")

    assert len(cases) == 18
    assert {case["answer"] for case in cases} <= {"A", "B", "C"}


@pytest.mark.parametrize(
    ("prompt_preset", "benchmark_name"),
    [
        ("ABC_family_tree_who_is", "family_tree_who_is"),
        ("ABC_alien_family_tree_who_is", "family_tree_who_is"),
        ("ABC_family_tree_yes_no", "family_tree_yes_no"),
        ("ABC_alien_family_tree_yes_no", "family_tree_yes_no"),
        ("ABC_gene_regulatory_network", "gene_regulatory_network"),
        ("ABC_neural_circuit", "neural_circuit"),
        ("ABC_older_younger", "older_younger"),
        ("ABC_alien_older_younger", "older_younger"),
        ("ABC_taller_shorter", "taller_shorter"),
        ("ABC_alien_taller_shorter", "taller_shorter"),
        ("ABC_faster_slower", "faster_slower"),
        ("ABC_alien_faster_slower", "faster_slower"),
        ("ABC_company_structure", "company_structure"),
        ("ABC_alien_company_structure", "company_structure"),
        ("ABC_proposition_equivalence", "proposition_equivalence"),
        ("ABC_variable_equivalence", "variable_equivalence"),
        ("ABC_east_west", "east_west"),
        ("ABC_north_south", "north_south"),
        ("ABC_ahead_behind", "ahead_behind"),
        (
            "ABC_family_trees_descendant_ancestor",
            "family_tree_descendant_ancestor",
        ),
        (
            "ABC_older_younger_unique_locations_who_is",
            "older_younger_unique_locations_who_is",
        ),
        (
            "ABC_older_younger_unique_locations_yes_no",
            "older_younger_unique_locations_yes_no",
        ),
    ],
)
def test_default_benchmark_names(
    prompt_preset: str,
    benchmark_name: str,
) -> None:
    assert get_default_benchmark_name(prompt_preset) == benchmark_name


def test_default_benchmark_rejects_unregistered_prompt() -> None:
    with pytest.raises(ValueError, match="no registered benchmark"):
        get_default_benchmark_name("AB_CD_family_trees_who_is")


@pytest.mark.parametrize(
    ("benchmark_name", "expected_count"),
    sorted(BENCHMARK_CASE_COUNTS.items()),
)
def test_benchmark_case_counts(
    benchmark_name: str,
    expected_count: int,
) -> None:
    cases = load_benchmark(benchmark_name)

    assert len(cases) == expected_count
    assert {case["kind"] for case in cases} <= {"one-hop", "two-hop"}
    assert {case["answer"] for case in cases} <= {"Y", "N", "A", "B", "C"}


def test_unique_age_yes_no_benchmark_is_balanced() -> None:
    cases = load_benchmark("older_younger_unique_locations_yes_no")

    assert sum(case["answer"] == "Y" for case in cases) == 44
    assert sum(case["answer"] == "N" for case in cases) == 44


@pytest.mark.parametrize(
    ("benchmark_name", "expected_structures"),
    [
        (
            "older_younger",
            {
                "A --fwd--> B --fwd--> C",
                "A --rev--> B --rev--> C",
            },
        ),
        (
            "gene_regulatory_network",
            {
                "A --fwd--> B --fwd--> C",
                "A --fwd--> B --rev--> C",
                "A --rev--> B --fwd--> C",
                "A --rev--> B --rev--> C",
            },
        ),
        (
            "variable_equivalence",
            {
                "A --fwd--> B --fwd--> C",
                "A --fwd--> B --rev--> C",
                "A --rev--> B --fwd--> C",
            },
        ),
        (
            "family_tree_descendant_ancestor",
            {
                "A --fwd--> B --fwd--> C",
                "A --fwd--> B --rev--> C",
                "A --rev--> B --fwd--> C",
                "A --rev--> B --rev--> C",
            },
        ),
    ],
)
def test_benchmark_structure_policies(
    benchmark_name: str,
    expected_structures: set[str],
) -> None:
    cases = load_benchmark(benchmark_name)

    assert {case["structure"] for case in cases} == expected_structures


@pytest.mark.parametrize(
    ("benchmark_name", "structure", "query", "answer", "kind"),
    [
        (
            "company_structure",
            "A --fwd--> B --fwd--> C",
            "A --junior--> C?",
            "Y",
            "two-hop",
        ),
        (
            "gene_regulatory_network",
            "A --fwd--> B --rev--> C",
            "A --repress--> C?",
            "Y",
            "two-hop",
        ),
        (
            "neural_circuit",
            "A --rev--> B --rev--> C",
            "A --excite--> C?",
            "Y",
            "two-hop",
        ),
        (
            "proposition_equivalence",
            "A --fwd--> B --rev--> C",
            "A --compatible with--> C?",
            "N",
            "two-hop",
        ),
        (
            "variable_equivalence",
            "A --fwd--> B --fwd--> C",
            "A --equal--> C?",
            "Y",
            "two-hop",
        ),
        (
            "family_tree_descendant_ancestor",
            "A --fwd--> B --rev--> C",
            "A --an ancestor--> C?",
            "N",
            "two-hop",
        ),
        (
            "older_younger_unique_locations_yes_no",
            "A --fwd--> B --fwd--> C",
            "A --two years older than--> C?",
            "Y",
            "two-hop",
        ),
        (
            "older_younger_unique_locations_yes_no",
            "A --fwd--> B --fwd--> C",
            "A --one year older than--> C?",
            "N",
            "two-hop",
        ),
        (
            "older_younger_unique_locations_yes_no",
            "A --fwd--> B --rev--> C",
            "A --equal in age to--> C?",
            "Y",
            "two-hop",
        ),
    ],
)
def test_representative_benchmark_cases(
    benchmark_name: str,
    structure: str,
    query: str,
    answer: str,
    kind: str,
) -> None:
    cases = load_benchmark(benchmark_name)

    assert {
        "structure": structure,
        "query": query,
        "answer": answer,
        "kind": kind,
    } in cases


def test_unique_age_who_is_omits_nonunique_answers() -> None:
    cases = load_benchmark("older_younger_unique_locations_who_is")
    forward_forward_queries = {
        case["query"]
        for case in cases
        if case["structure"] == "A --fwd--> B --fwd--> C"
    }

    assert "A --two years older than--> ?" in forward_forward_queries
    assert "A --older than--> ?" not in forward_forward_queries


def test_example_manifest_prompts_have_default_benchmarks() -> None:
    manifest_path = (
        Path(__file__).parents[1]
        / "orchestration"
        / "example_run"
        / "example_manifest.toml"
    )
    with manifest_path.open("rb") as file:
        manifest = tomllib.load(file)

    missing = set(manifest["prompt_presets"]) - set(DEFAULT_BENCHMARK_BY_PROMPT_PRESET)
    assert not missing, f"Manifest prompts lack default benchmarks: {sorted(missing)}"
