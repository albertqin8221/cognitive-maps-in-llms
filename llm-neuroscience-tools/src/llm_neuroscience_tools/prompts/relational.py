import re
from itertools import permutations
from typing import Any, Literal, Mapping, cast
from dataclasses import dataclass, field

import numpy as np

from .base import PromptRuntime, PromptSchema


@dataclass(frozen=True)
class RelationPool:
    forward_relation:  str
    backward_relation: str
    query_relations:   tuple[str, ...]


@dataclass(frozen=True)
class RelationalPromptSchema:
    """A ``RelationalPromptSchema`` consists of a text template, along with a set
    of text symbols partitioned into four categories.

    Each symbol represents a base substitution key such as ``"A"`` or
    ``"R_AB"``. The template is expected to contain substitution slots whose
    base keys are drawn from these declared symbols.

    During ``__post_init__``, the schema concatenates the four declared symbol
    groups into ``substitution_keys`` so that the resulting object is
    compatible with the ``PromptSchema`` interface expected by ``PromptRuntime``.
    """

    template: str
    context_entity_symbols:   tuple[str, ...]
    context_relation_symbols: tuple[str, ...]
    query_entity_symbols:     tuple[str, ...]
    query_relation_symbols:   tuple[str, ...]
    query_mode:               Literal["yes_no", "who_is"]
    derived_landmarks:        dict[str, str]
    flags:                    Mapping[str, Any] = field(default_factory=dict)

    substitution_keys: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Derive PromptRuntime-compatible fields from schema data."""
        if not self.template:
            raise ValueError("template must be non-empty")
        if not self.context_entity_symbols:
            raise ValueError("context_entity_symbols must be non-empty")
        if self.query_mode not in ("yes_no", "who_is"):
            raise ValueError(
                "query_mode must be either 'yes_no' or 'who_is', got "
                f"{self.query_mode!r}"
            )
        if "query_mode" in self.flags:
            raise ValueError(
                "query_mode is a required schema field and must not also be "
                "provided in flags"
            )

        symbol_groups = {
            "context_entity_symbols": self.context_entity_symbols,
            "context_relation_symbols": self.context_relation_symbols,
            "query_entity_symbols": self.query_entity_symbols,
            "query_relation_symbols": self.query_relation_symbols,
        }
        for group_name, symbols in symbol_groups.items():
            for symbol in symbols:
                if not isinstance(symbol, str) or not symbol:
                    raise ValueError(
                        f"{group_name} must contain only non-empty strings"
                    )

        substitution_keys = (
            self.context_entity_symbols
            + self.context_relation_symbols
            + self.query_entity_symbols
            + self.query_relation_symbols
        )
        if len(set(substitution_keys)) != len(substitution_keys):
            duplicate_keys = sorted(
                {
                    key
                    for key in substitution_keys
                    if substitution_keys.count(key) > 1
                }
            )
            raise ValueError(
                "declared symbol groups contain duplicate substitution keys: "
                + ", ".join(sorted(duplicate_keys))
            )

        object.__setattr__(self, "substitution_keys", substitution_keys)



class RelationalPromptRuntime:

    def __init__(
            self,
            schema: RelationalPromptSchema,
            tokenizer: Any,
            name_pool: list[str],
            relation_pool: RelationPool
    ):
        """Initialize the runtime and construct an internal PromptRuntime.

        Args:
            schema: Relational prompt schema to use.
            tokenizer: Tokenizer used for rendering prompts. 
            name_pool: List of entity names for sampling contexts.
            relation_pool: Relation pool used to resolve abstract relations.
        """

        self.schema = schema
        self.tokenizer = tokenizer
        self.name_pool = name_pool
        self.relation_pool = relation_pool
        self.query_mode = schema.query_mode
        self.flags = dict(schema.flags)

        # Use an internal PromptRuntime to substitute into the prompt template
        self.prompt_runtime = PromptRuntime(
            schema = cast(PromptSchema, schema),
            tokenizer = tokenizer
        )
        self.landmarks = self.prompt_runtime.landmarks

    def sample_entities(self, seed: int | None = None) -> dict[str, str]:
        """Randomly draw names to instantiate the context entity symbols.

        Args:
            seed: Optional RNG seed for reproducibility.

        Returns:
            Mapping from entity symbol to sampled name.
        """
        if len(self.name_pool) < len(self.schema.context_entity_symbols):
            raise ValueError("name_pool is too small to draw a full context")

        rng = np.random.default_rng(seed)
        names = [
            str(n)
            for n in rng.choice(
                self.name_pool,
                size=len(self.schema.context_entity_symbols),
                replace=False,
            )
        ]
        return {
            symbol: names[i]
            for i, symbol in enumerate(self.schema.context_entity_symbols)
        }

    def sample_nonoverlapping_entities(
        self,
        seed: int | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Randomly draw two disjoint assignments for the context entity symbols.

        Args:
            seed: Optional RNG seed for reproducibility.

        Returns:
            Two context dictionaries mapping entity symbol to sampled name.
        """
        n_entities = len(self.schema.context_entity_symbols)
        if len(self.name_pool) < 2 * n_entities:
            raise ValueError("name_pool is too small to draw two disjoint contexts")

        rng = np.random.default_rng(seed)
        names = [
            str(n)
            for n in rng.choice(
                self.name_pool,
                size=2 * n_entities,
                replace=False,
            )
        ]
        context_a = {
            symbol: names[i]
            for i, symbol in enumerate(self.schema.context_entity_symbols)
        }
        context_b = {
            symbol: names[i + n_entities]
            for i, symbol in enumerate(self.schema.context_entity_symbols)
        }
        return context_a, context_b

    def build_query_specs(
        self,
        entity_symbols_to_names: Mapping[str, str],
    ) -> dict[str, dict[str, Any]]:
        """Build every query supported by the configured query mode.

        For ``"yes_no"`` prompts, this produces every ordered pair of
        distinct entities for every query relation. For ``"who_is"`` prompts,
        it produces every entity and query-relation combination.

        Args:
            entity_symbols_to_names: Entity symbols and the concrete names that
                may appear in queries and answers.

        Returns:
            A mapping from schematic query labels to specifications accepted by
            :meth:`render_prompt_as_tokens`.
        """
        query_specs: dict[str, dict[str, Any]] = {}

        if self.query_mode == "yes_no":
            for source_symbol, candidate_symbol in permutations(
                entity_symbols_to_names, 2
            ):
                for relation in self.relation_pool.query_relations:
                    label = f"{source_symbol} --{relation}--> {candidate_symbol}?"
                    query_specs[label] = {
                        "query_entities": {
                            "QE": entity_symbols_to_names[source_symbol],
                            "QCAND": entity_symbols_to_names[candidate_symbol],
                        },
                        "query_relations": {"QR": relation},
                        "answer_tokens": {
                            "Y": "Yes",
                            "N": "No",
                            "U": "Unknown",
                        },
                    }
            return query_specs

        if self.query_mode == "who_is":
            answer_tokens = {
                **dict(entity_symbols_to_names),
                "U": "Unknown",
            }
            for source_symbol, source_name in entity_symbols_to_names.items():
                for relation in self.relation_pool.query_relations:
                    label = f"{source_symbol} --{relation}--> ?"
                    query_specs[label] = {
                        "query_entities": {"QE": source_name},
                        "query_relations": {"QR": relation},
                        "answer_tokens": dict(answer_tokens),
                    }
            return query_specs

        raise ValueError(f"Unsupported query_mode: {self.query_mode!r}")
    
    def render_prompt(
        self,
        context_entities: dict[str, str] | None = None,
        context_relations: dict[str, str] | None = None,
        query_entities: dict[str, str] | None = None,
        query_relations: dict[str, str] | None = None,
    ) -> str:
        """Fill the prompt template. The special values ``"FWD"`` and ``"REV"``
        can be passed to either relations field to use the canonical forward
        and reverse relation in the relation pool

        Args:
            context_entities: Concrete names for the context entity symbols, or
                ``None`` to fill the context slots with the filler token.
            context_relations: Concrete relations for the context relation
                symbols, or ``None`` to fill the context slots. This must be
                omitted together with ``context_entities``.
            query_entities: Optional query-entity substitutions.
            query_relations: Optional query-relation substitutions. 

        Returns:
            Fully rendered prompt string.
        """
        self._validate_input(
            context_entities=context_entities,
            context_relations=context_relations,
            query_entities=query_entities,
            query_relations=query_relations,
        )
        return self.prompt_runtime.render_prompt(self._build_substitutions(
            context_entities=context_entities,
            context_relations=context_relations,
            query_entities=query_entities,
            query_relations=query_relations,
        ))
    
    def render_prompt_as_tokens(
        self,
        context_entities: dict[str, str] | None = None,
        context_relations: dict[str, str] | None = None,
        query_entities: dict[str, str] | None = None,
        query_relations: dict[str, str] | None = None,
    ) -> list[int]:
        """Fill the prompt template, and returns the result as a list of token 
        ids. The special values ``"FWD"`` and ``"REV"`` can be passed to either 
        relations field to use the canonical forward and reverse relation in the
        relation pool.

        Args:
            context_entities: Concrete names for the context entity symbols, or
                ``None`` to fill the context slots with the filler token.
            context_relations: Concrete relations for the context relation
                symbols, or ``None`` to fill the context slots. This must be
                omitted together with ``context_entities``.
            query_entities: Optional query-entity substitutions.
            query_relations: Optional query-relation substitutions. 

        Returns:
            Rendered prompt string as a list of token ids.
        """
        self._validate_input(
            context_entities=context_entities,
            context_relations=context_relations,
            query_entities=query_entities,
            query_relations=query_relations,
        )
        return self.prompt_runtime.render_prompt_as_tokens(
            self._build_substitutions(
                context_entities=context_entities,
                context_relations=context_relations,
                query_entities=query_entities,
                query_relations=query_relations,
        ))

    def _validate_input(
        self,
        context_entities: dict[str, str] | None,
        context_relations: dict[str, str] | None,
        query_entities: dict[str, str] | None,
        query_relations: dict[str, str] | None,
    ) -> None:
        """Verify that the provided dictionaries match the schema exactly."""
        if (context_entities is None) != (context_relations is None):
            raise ValueError(
                "context_entities and context_relations must either both be "
                "provided or both be omitted"
            )

        if (query_entities is None) != (query_relations is None):
            raise ValueError(
                "query_entities and query_relations must either both be "
                "provided or both be omitted"
            )

        context_is_empty = context_entities is None
        query_is_empty = query_entities is None
        if context_is_empty and query_is_empty:
            raise ValueError(
                "At least one of the context or query must be provided"
            )

        if not context_is_empty:
            assert context_entities is not None
            assert context_relations is not None

            expected_context_entities = set(self.schema.context_entity_symbols)
            if set(context_entities) != expected_context_entities:
                raise ValueError(
                    "context_entities must contain exactly the schema symbols: "
                    + ", ".join(sorted(expected_context_entities))
                )

            expected_context_relations = set(self.schema.context_relation_symbols)
            if set(context_relations) != expected_context_relations:
                raise ValueError(
                    "context_relations must contain exactly the schema symbols: "
                    + ", ".join(sorted(expected_context_relations))
                )

        if not query_is_empty:
            assert query_entities is not None
            assert query_relations is not None

            expected_query_entities = set(self.schema.query_entity_symbols)
            if set(query_entities) != expected_query_entities:
                raise ValueError(
                    "query_entities must contain exactly the schema symbols: "
                    + ", ".join(sorted(expected_query_entities))
                )

            expected_query_relations = set(self.schema.query_relation_symbols)
            if set(query_relations) != expected_query_relations:
                raise ValueError(
                    "query_relations must contain exactly the schema symbols: "
                    + ", ".join(sorted(expected_query_relations))
                )

    def _build_substitutions(
        self,
        context_entities: dict[str, str] | None,
        context_relations: dict[str, str] | None,
        query_entities: dict[str, str] | None,
        query_relations: dict[str, str] | None,
    ) -> dict[str, str | None]:
        """Build a substitutions dictionary."""

        substitutions: dict[str, str | None] = {}

        if context_entities is None and context_relations is None:
            substitutions |= {
                symbol: None
                for symbol in self.schema.context_entity_symbols
            }
            substitutions |= {
                symbol: None
                for symbol in self.schema.context_relation_symbols
            }
        else:
            assert context_entities is not None
            assert context_relations is not None
            substitutions |= dict(context_entities)
            substitutions |= {
                symbol: self._resolve_relation(relation)
                for symbol, relation in context_relations.items()
            }

        if query_entities is None and query_relations is None:
            substitutions |= {
                symbol: None
                for symbol in self.schema.query_entity_symbols
            }
            substitutions |= {
                symbol: None
                for symbol in self.schema.query_relation_symbols
            }
            return substitutions

        # Otherwise, add in the query substitutions
        assert query_entities is not None
        assert query_relations is not None
        substitutions |= dict(query_entities)
        substitutions |= {
            symbol: self._resolve_relation(relation)
            for symbol, relation in query_relations.items()
        }
        return substitutions

    def _resolve_relation(self, relation: str) -> str:
        """Resolve relation markers into concrete relation strings."""
        # ``FWD`` and ``REV`` are shorthand for the pool's canonical directions.
        if relation == "FWD":
            return self.relation_pool.forward_relation
        if relation == "REV":
            return self.relation_pool.backward_relation
        return relation
