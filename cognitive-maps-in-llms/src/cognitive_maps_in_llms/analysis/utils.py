from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[3] / "results"


@dataclass(frozen=True)
class QueryKey:
    source_symbol: str
    relation: str
    candidate_symbol: str | None


def load_ndjson_records(path: Path | str) -> list[dict[str, Any]]:
    """Load a line-delimited JSON experiment result file."""
    ndjson_path = Path(path)
    with ndjson_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def find_result_ndjson(
    *,
    model_preset: str,
    prompt_preset: str,
    experiment: str,
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    run: str | None = None,
) -> Path:
    """Find a result NDJSON in the current or legacy results layout."""
    root = Path(results_root)
    prompt_dir = root / model_preset / prompt_preset

    current_path = prompt_dir / experiment / "results.ndjson"
    if run is None and current_path.exists():
        return current_path

    experiment_filename = _experiment_filename(experiment)

    if run is not None:
        legacy_path = prompt_dir / run / experiment_filename
        current_run_path = prompt_dir / run / experiment / "results.ndjson"
        for path in (current_run_path, legacy_path):
            if path.exists():
                return path
        raise FileNotFoundError(
            "No result NDJSON found at either "
            f"{current_run_path} or {legacy_path}"
        )

    if not prompt_dir.exists():
        raise FileNotFoundError(
            f"No result directory found for model_preset={model_preset!r}, "
            f"prompt_preset={prompt_preset!r}: {prompt_dir}"
        )

    candidate_paths = sorted(
        run_dir / experiment_filename
        for run_dir in prompt_dir.iterdir()
        if run_dir.is_dir() and (run_dir / experiment_filename).exists()
    )
    if not candidate_paths:
        raise FileNotFoundError(
            f"No results found for experiment={experiment!r} under {prompt_dir}"
        )
    return candidate_paths[-1]


def query_key_to_text(
    query_key: str,
    *,
    prompt_preset: str,
    bracket_symbols: bool = True,
) -> str:
    """Convert a schematic result key like ``A --junior--> B?`` to text."""
    parsed = _parse_query_key(query_key)
    query_format = _query_format_for_prompt_preset(prompt_preset)
    source = _format_query_symbol(parsed.source_symbol, bracket_symbols=bracket_symbols)

    if query_format == "who_is_possessive":
        _require_no_candidate(parsed, query_format=query_format)
        return f"Who is {source}'s {parsed.relation}?"
    if query_format == "who_is_binary":
        _require_no_candidate(parsed, query_format=query_format)
        return f"Who is {source} {parsed.relation}?"

    candidate = _format_query_symbol(
        _require_candidate(parsed, query_format=query_format),
        bracket_symbols=bracket_symbols,
    )
    if query_format == "yes_no_possessive":
        return f"Is {source}'s {parsed.relation} {candidate}?"
    if query_format == "yes_no_to":
        return f"Is {source} {parsed.relation} to {candidate}?"
    if query_format == "yes_no_than":
        return f"Is {source} {parsed.relation} than {candidate}?"
    if query_format == "yes_no_of":
        return f"Is the {source} {parsed.relation} of the {candidate}?"
    if query_format == "does_binary":
        return f"Does {source} {parsed.relation} {candidate}?"
    if query_format == "is_binary":
        return f"Is {source} {parsed.relation} {candidate}?"
    if query_format == "is_binary_of":
        return f"Is {source} {parsed.relation} of {candidate}?"
    if query_format == "is_binary_to":
        return f"Is {source} {parsed.relation} to {candidate}?"
    raise ValueError(f"Unsupported query format: {query_format!r}")


def query_text_to_key(
    query_text: str,
    *,
    prompt_preset: str,
) -> str:
    """Convert generated natural-language query text back to a result key."""
    query_format = _query_format_for_prompt_preset(prompt_preset)
    pattern = _QUERY_TEXT_PATTERNS[query_format]
    match = re.fullmatch(pattern, query_text.strip())
    if match is None:
        raise ValueError(
            f"Query text {query_text!r} does not match the expected format "
            f"for prompt_preset={prompt_preset!r}."
        )

    source = match.group("source")
    relation = match.group("relation")
    candidate = match.groupdict().get("candidate")
    if candidate is None:
        return f"{source} --{relation}--> ?"
    return f"{source} --{relation}--> {candidate}?"


def get_fwd_fwd_context_tokens(
    model_preset: str,
    prompt_preset: str,
    seed: int = 0,
) -> list[str]:
    """Return tokenizer-level context labels for the canonical FWD/FWD prompt."""
    from llm_neuroscience_tools.presets import build_preset_runtime

    runtime = build_preset_runtime(
        model_name=model_preset,
        prompt_name=prompt_preset,
        use_dummy=True,
    )
    prompt_runtime = runtime.prompt_runtime
    tokenizer = runtime.backend.tokenizer
    context_entities = prompt_runtime.sample_entities(seed=seed)
    prompt_ids = prompt_runtime.render_prompt_as_tokens(
        context_entities=context_entities,
        context_relations={"R_AB": "FWD", "R_BC": "FWD"},
    )
    ctxt_start = prompt_runtime.landmarks["CTXT_START"]
    ctxt_end = prompt_runtime.landmarks["CTXT_END"]
    context_ids = prompt_ids[ctxt_start : ctxt_end + 1]
    context_tokens = [tokenizer.decode([token_id]) for token_id in context_ids]

    landmark_labels = {
        "A1": "‹A›",
        "B1": "‹B›",
        "B2": "‹B›",
        "C1": "‹C›",
        "D1": "‹D›",
    }
    for landmark, label in landmark_labels.items():
        if landmark not in prompt_runtime.landmarks:
            continue
        token_idx = prompt_runtime.landmarks[landmark] - ctxt_start
        if not (0 <= token_idx < len(context_tokens)):
            continue
        prefix = " " if context_tokens[token_idx].startswith(" ") else ""
        context_tokens[token_idx] = f"{prefix}{label}"
    return context_tokens


def _experiment_filename(experiment: str) -> str:
    if experiment.endswith(".ndjson"):
        return experiment
    return f"{experiment}.ndjson"


def _parse_query_key(query_key: str) -> QueryKey:
    match = re.fullmatch(
        r"(?P<source>[A-Za-z]+) --(?P<relation>.+)--> (?:(?P<candidate>[A-Za-z]+)\?|\?)",
        query_key.strip(),
    )
    if match is None:
        raise ValueError(
            f"Query key {query_key!r} does not match the expected format "
            "'A --relation--> B?' or 'A --relation--> ?'."
        )

    candidate = match.group("candidate")
    return QueryKey(
        source_symbol=match.group("source"),
        relation=match.group("relation"),
        candidate_symbol=None if candidate == "?" else candidate,
    )


_PROMPT_PRESET_TO_QUERY_FORMAT = {
    "ABC_family_tree_who_is": "who_is_possessive",
    "ABC_alien_family_tree_who_is": "who_is_possessive",
    "ABC_family_tree_yes_no": "yes_no_possessive",
    "ABC_alien_family_tree_yes_no": "yes_no_possessive",
    "ABC_company_structure": "yes_no_to",
    "ABC_older_younger": "yes_no_than",
    "ABC_taller_shorter": "yes_no_than",
    "ABC_faster_slower": "yes_no_than",
    "ABC_east_west": "yes_no_of",
    "ABC_north_south": "yes_no_of",
    "ABC_ahead_behind": "yes_no_of",
    "ABC_gene_regulatory_network": "does_binary",
    "ABC_neural_circuit": "does_binary",
    "ABC_proposition_equivalence": "is_binary",
    "ABC_variable_equivalence": "is_binary_to",
    "AB_CD_family_trees_who_is": "who_is_possessive",
    "AB_CD_family_trees_yes_no": "yes_no_possessive",
    "ABC_family_trees_descendant_ancestor": "is_binary_of",
    "ABC_older_younger_unique_locations_who_is": "who_is_binary",
    "ABC_older_younger_unique_locations_yes_no": "is_binary",
}


_SYMBOL_PATTERN = r"<?(?P<{name}>[A-Za-z]+)>?"
_RELATION_PATTERN = r"(?P<relation>.+)"
_QUERY_TEXT_PATTERNS = {
    "who_is_possessive": (
        rf"Who is {_SYMBOL_PATTERN.format(name='source')}'s {_RELATION_PATTERN}\?"
    ),
    "who_is_binary": (
        rf"Who is {_SYMBOL_PATTERN.format(name='source')} {_RELATION_PATTERN}\?"
    ),
    "yes_no_possessive": (
        rf"Is {_SYMBOL_PATTERN.format(name='source')}'s {_RELATION_PATTERN} "
        rf"{_SYMBOL_PATTERN.format(name='candidate')}\?"
    ),
    "yes_no_to": (
        rf"Is {_SYMBOL_PATTERN.format(name='source')} {_RELATION_PATTERN} to "
        rf"{_SYMBOL_PATTERN.format(name='candidate')}\?"
    ),
    "yes_no_than": (
        rf"Is {_SYMBOL_PATTERN.format(name='source')} {_RELATION_PATTERN} than "
        rf"{_SYMBOL_PATTERN.format(name='candidate')}\?"
    ),
    "yes_no_of": (
        rf"Is the {_SYMBOL_PATTERN.format(name='source')} {_RELATION_PATTERN} of the "
        rf"{_SYMBOL_PATTERN.format(name='candidate')}\?"
    ),
    "does_binary": (
        rf"Does {_SYMBOL_PATTERN.format(name='source')} {_RELATION_PATTERN} "
        rf"{_SYMBOL_PATTERN.format(name='candidate')}\?"
    ),
    "is_binary": (
        rf"Is {_SYMBOL_PATTERN.format(name='source')} {_RELATION_PATTERN} "
        rf"{_SYMBOL_PATTERN.format(name='candidate')}\?"
    ),
    "is_binary_of": (
        rf"Is {_SYMBOL_PATTERN.format(name='source')} {_RELATION_PATTERN} of "
        rf"{_SYMBOL_PATTERN.format(name='candidate')}\?"
    ),
    "is_binary_to": (
        rf"Is {_SYMBOL_PATTERN.format(name='source')} {_RELATION_PATTERN} to "
        rf"{_SYMBOL_PATTERN.format(name='candidate')}\?"
    ),
}


def _query_format_for_prompt_preset(prompt_preset: str) -> str:
    try:
        return _PROMPT_PRESET_TO_QUERY_FORMAT[prompt_preset]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt_preset: {prompt_preset!r}") from exc


def _format_query_symbol(symbol: str, *, bracket_symbols: bool) -> str:
    if bracket_symbols:
        return f"<{symbol}>"
    return symbol


def _require_candidate(parsed: QueryKey, *, query_format: str) -> str:
    if parsed.candidate_symbol is None:
        raise ValueError(
            f"Query format {query_format!r} requires a candidate symbol, "
            f"but query key has '?': {parsed!r}."
        )
    return parsed.candidate_symbol


def _require_no_candidate(parsed: QueryKey, *, query_format: str) -> None:
    if parsed.candidate_symbol is not None:
        raise ValueError(
            f"Query format {query_format!r} requires a who-is query key ending "
            f"in '?', but got candidate {parsed.candidate_symbol!r}."
        )
