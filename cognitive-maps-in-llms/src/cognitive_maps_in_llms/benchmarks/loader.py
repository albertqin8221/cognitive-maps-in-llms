"""Load packaged relational-reasoning benchmarks."""

from __future__ import annotations

from importlib import resources
from typing import TypedDict, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib


class BenchmarkCase(TypedDict):
    """One query and its expected answer for a graph structure."""

    structure: str
    query: str
    answer: str
    kind: str


_CASE_FIELDS = {"structure", "query", "answer", "kind"}

DEFAULT_BENCHMARK_BY_PROMPT_PRESET = {
    "ABC_family_tree_who_is": "family_tree_who_is",
    "ABC_alien_family_tree_who_is": "family_tree_who_is",
    "ABC_family_tree_yes_no": "family_tree_yes_no",
    "ABC_alien_family_tree_yes_no": "family_tree_yes_no",
    "ABC_gene_regulatory_network": "gene_regulatory_network",
    "ABC_neural_circuit": "neural_circuit",
    "ABC_older_younger": "older_younger",
    "ABC_alien_older_younger": "older_younger",
    "ABC_taller_shorter": "taller_shorter",
    "ABC_alien_taller_shorter": "taller_shorter",
    "ABC_faster_slower": "faster_slower",
    "ABC_alien_faster_slower": "faster_slower",
    "ABC_company_structure": "company_structure",
    "ABC_alien_company_structure": "company_structure",
    "ABC_proposition_equivalence": "proposition_equivalence",
    "ABC_alien_proposition_equivalence": "proposition_equivalence",
    "ABC_variable_equivalence": "variable_equivalence",
    "ABC_alien_variable_equivalence": "variable_equivalence",
    "ABC_east_west": "east_west",
    "ABC_alien_east_west": "east_west",
    "ABC_north_south": "north_south",
    "ABC_alien_north_south": "north_south",
    "ABC_ahead_behind": "ahead_behind",
    "ABC_alien_ahead_behind": "ahead_behind",
    "ABC_family_trees_descendant_ancestor": (
        "family_tree_descendant_ancestor"
    ),
    "ABC_alien_family_trees_descendant_ancestor": (
        "family_tree_descendant_ancestor"
    ),
    "ABC_older_younger_unique_locations_who_is": (
        "older_younger_unique_locations_who_is"
    ),
    "ABC_older_younger_unique_locations_yes_no": (
        "older_younger_unique_locations_yes_no"
    ),
    "ABC_alien_older_younger_unique_locations_yes_no": (
        "older_younger_unique_locations_yes_no"
    ),
}


def get_default_benchmark_name(prompt_preset: str) -> str:
    """Return the benchmark associated with a prompt preset.

    Args:
        prompt_preset: Name of a prompt preset.

    Returns:
        Name of the packaged benchmark for that preset.

    Raises:
        ValueError: If the prompt preset has no registered benchmark.
    """
    try:
        return DEFAULT_BENCHMARK_BY_PROMPT_PRESET[prompt_preset]
    except KeyError as exc:
        raise ValueError(
            f"Prompt preset {prompt_preset!r} has no registered benchmark."
        ) from exc


def load_benchmark(name: str) -> list[BenchmarkCase]:
    """Load and validate a benchmark bundled with the package.

    Args:
        name: Benchmark filename, with or without the ``.toml`` suffix.

    Returns:
        Benchmark cases in the order in which they appear in the TOML file.

    Raises:
        ValueError: If ``name`` is not a simple filename or the benchmark has
            an invalid structure.
        FileNotFoundError: If the requested benchmark does not exist.
    """
    filename = name if name.endswith(".toml") else f"{name}.toml"
    if "/" in filename or "\\" in filename or filename in {".toml", "..toml"}:
        raise ValueError("Benchmark name must be a simple filename.")

    benchmark_file = resources.files(__package__).joinpath(filename)
    with benchmark_file.open("rb") as file:
        data = tomllib.load(file)

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Benchmark must define a [[cases]] array.")

    cases: list[BenchmarkCase] = []
    seen_cases: set[tuple[str, str]] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict) or set(raw_case) != _CASE_FIELDS:
            raise ValueError(
                f"Case {index} must contain exactly {sorted(_CASE_FIELDS)}."
            )
        if not all(isinstance(raw_case[field], str) for field in _CASE_FIELDS):
            raise ValueError(f"Every field in case {index} must be a string.")

        case = cast(BenchmarkCase, raw_case)
        identity = (case["structure"], case["query"])
        if identity in seen_cases:
            raise ValueError(
                "Benchmark contains a duplicate structure/query pair: "
                f"{identity!r}."
            )
        seen_cases.add(identity)
        cases.append(case)

    return cases
