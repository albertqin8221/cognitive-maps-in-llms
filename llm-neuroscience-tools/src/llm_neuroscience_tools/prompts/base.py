"""Base prompt schema and prompt runtime abstractions."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptSchema:
    """Schema for a family of prompts with substitution slots.

    Specifically, a ``PromptSchema`` defines:
    - a text template,
    - a set of base substitution keys such as ``"A"`` or ``"R_AB"``, and
    - optional derived landmarks that will later be compiled by a runtime.

    The template is expected to contain substitution slots of the form
    ``<KEYocc:width>``, where:

    - ``KEY`` is a base substitution key declared in ``substitution_keys``,
    - ``occ`` is an optional positive integer occurrence index, and
    - ``width`` is an optional positive integer giving the maximum token width
      of that substitution slot.

    Substitution slots are assumed to be preceded by a leading space.
    """

    template: str
    substitution_keys: tuple[str, ...]
    derived_landmarks: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate the declared substitution keys against the template."""
        
        # Validate the top-level schema fields.
        if not self.template:
            raise ValueError("template must be non-empty")
        if not self.substitution_keys:
            raise ValueError("substitution_keys must be non-empty")
        if any(not isinstance(key, str) or not key for key in self.substitution_keys):
            raise ValueError("substitution_keys must contain only non-empty strings")

        if len(set(self.substitution_keys)) != len(self.substitution_keys):
            raise ValueError("substitution_keys must be unique")

        slot_pattern = re.compile(
            r"<(?P<key>[A-Za-z_][A-Za-z0-9_]*?)(?P<occ>\d+)?(?:\:(?P<width>\d+))?>"
        )

        slots_by_key: dict[str, list[tuple[int | None, int]]] = {
            key: []
            for key in self.substitution_keys
        }

        # Parse template slots and collect data for declared keys.
        undeclared_keys: set[str] = set()
        for match in slot_pattern.finditer(self.template):
            key = match.group("key")
            if key not in slots_by_key:
                undeclared_keys.add(key)
                continue

            if match.start() == 0 or self.template[match.start() - 1] != " ":
                raise ValueError(
                    f"Slot {match.group(0)!r} for key {key!r} must be "
                    "preceded by a space."
                )

            occ_text = match.group("occ")
            width_text = match.group("width")
            occurrence = int(occ_text) if occ_text is not None else None
            width = int(width_text) if width_text is not None else 1

            if occurrence is not None and occurrence <= 0:
                raise ValueError(
                    f"Slot {match.group(0)!r} for key {key!r} must use a "
                    "positive occurrence index."
                )
            if width <= 0:
                raise ValueError(
                    f"Slot {match.group(0)!r} for key {key!r} must use a "
                    "positive slot width."
                )

            slots_by_key[key].append((occurrence, width))

        if undeclared_keys:
            raise ValueError(
                "template contains slots for undeclared substitution keys: "
                + ", ".join(sorted(undeclared_keys))
            )

        missing_slots = [
            key for key, slots in slots_by_key.items() if not slots
        ]
        if missing_slots:
            raise ValueError(
                "template is missing slots for substitution keys: "
                + ", ".join(missing_slots)
            )

        # Check per-key width consistency and occurrence-index rules.
        for key, slots in slots_by_key.items():
            widths = {width for _, width in slots}
            if len(widths) > 1:
                raise ValueError(
                    f"All slots for key {key!r} must declare the same width."
                )

            if len(slots) == 1:
                continue

            occurrences = [occ for occ, _ in slots]
            if any(occ is None for occ in occurrences):
                raise ValueError(
                    f"Repeated key {key!r} must use explicit occurrence indices."
                )

            explicit_occurrences = [occ for occ in occurrences if occ is not None]
            if len(set(explicit_occurrences)) != len(explicit_occurrences):
                raise ValueError(
                    f"Repeated key {key!r} must use unique occurrence indices."
                )
        

class PromptRuntime:
    """A ``PromptRuntime`` combines a prompt schema with a tokenizer. It is
    responsible for:
    - compiling landmarks into concrete token positions,
    - compiling template substitution slots, and
    - rendering supplied substitutions as strings or token sequences.
    """

    def __init__(
        self,
        schema: PromptSchema,
        tokenizer: Any,
    ) -> None:
        """Initialize the runtime and precompute template token positions.

        Args:
            schema: Prompt schema describing the template and landmarks.
            tokenizer: Tokenizer used to render prompts.
        """
        
        self.schema = schema
        self.tokenizer = tokenizer
        self.tokenized_template, self.slot_metadata, compiled_slots = (
            self._compile_template()
        )
        self.landmarks = self._build_landmarks(compiled_slots)

        whitespace_ids = self.tokenizer.encode(" ", add_special_tokens=False)
        if len(whitespace_ids) != 1:
            raise ValueError(
                "Tokenizer must encode a single space as exactly one token to "
                "use it as the slot filler token."
            )
        self.filler_token_id = whitespace_ids[0]

    def render_prompt(self, substitutions: Mapping[str, str | None]) -> str:
        """Fill the prompt template from a set of substitution rules. Returns
        the result as a string. Note that the string returned by this method 
        may be decorated by a chat template if the tokenizer has one. 

        Args:
            substitutions: Mapping from placeholder names to strings. A value
                of ``None`` fills that key's slots with the filler token.

        Returns:
            Tokenized prompt as a list of token ids.
        """
        return self.tokenizer.decode(self.render_prompt_as_tokens(substitutions))

    def render_prompt_as_tokens(
        self,
        substitutions: Mapping[str, str | None],
    ) -> list[int]:
        """Fill the prompt template from a set of substitution rules. Returns
        the result as a list of token IDs.

        Args:
            substitutions: Mapping from placeholder names to strings. A value
                of ``None`` fills that key's slots with the filler token.

        Returns:
            Tokenized prompt as a list of token ids.
        """
        expected_keys = set(self.schema.substitution_keys)
        actual_keys = set(substitutions)
        missing = sorted(expected_keys - actual_keys)
        if missing:
            raise KeyError("Missing substitutions for keys: " + ", ".join(missing))
        extra = sorted(actual_keys - expected_keys)
        if extra:
            raise KeyError("Extra substitutions for keys: " + ", ".join(extra))

        rendered_prompt_ids = self.tokenized_template.copy()

        # Fill each reserved token-space slot span with the tokenized
        # substitution value, padding with the configured filler token when
        # the value is shorter than the declared slot width.
        for key in self.schema.substitution_keys:
            slot_width = self.slot_metadata[key]["substitution_length"]
            substitution = substitutions[key]
            if substitution is None:
                substitution_ids = [self.filler_token_id] * slot_width
            else:
                substitution_ids = self.tokenizer.encode(
                    text=" " + substitution,
                    add_special_tokens=False,
                )
                if len(substitution_ids) > slot_width:
                    raise ValueError(
                        f"Substitution {(' '+substitution)!r} for key {key!r} "
                        f"tokenizes to {len(substitution_ids)} tokens, exceeding "
                        f"the declared slot width {slot_width}."
                    )
            if len(substitution_ids) < slot_width:
                substitution_ids = substitution_ids + (
                    [self.filler_token_id] * (slot_width - len(substitution_ids))
                )

            for begin, end in self.slot_metadata[key]["locations"]:
                rendered_prompt_ids[begin:end] = substitution_ids

        # Every reserved ``-1`` span should have been overwritten by the end of
        # the rendering pass.
        if any(token_id == -1 for token_id in rendered_prompt_ids):
            raise RuntimeError("Rendered prompt contains unfilled substitution slots.")

        return rendered_prompt_ids
            
    def _iter_template_slots(self) -> list[dict[str, Any]]:
        """Parse the template's substitution slots."""
        slot_pattern = re.compile(
            r"<(?P<key>[A-Za-z_][A-Za-z0-9_]*?)(?P<occ>\d+)?(?:\:(?P<width>\d+))?>"
        )

        # Parse each declared slot occurrence from the raw template text.
        # At this stage, the boundaries are still string indices rather than
        # token indices.
        slots: list[dict[str, Any]] = []
        for match in slot_pattern.finditer(self.schema.template):
            key = match.group("key")
            if key not in self.schema.substitution_keys:
                continue

            # Normalize omitted fields to the schema defaults so later
            # compilation logic can treat every slot occurrence uniformly.
            occ_text = match.group("occ")
            width_text = match.group("width")
            occurrence = int(occ_text) if occ_text is not None else None
            width = int(width_text) if width_text is not None else 1
            occurrence_name = f"{key}{occurrence}" if occurrence is not None else key

            slots.append(
                {
                    "key": key,
                    "occurrence": occurrence,
                    "occurrence_name": occurrence_name,
                    "width": width,
                    "text_start": match.start(),
                    "text_end": match.end(),
                }
            )
        return slots

    def _compile_template(
        self,
    ) -> tuple[list[int], dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """Compile the schema template into token space with reserved slot spans.

        Each substitution slot in the raw template is first replaced by a unique
        sentinel token. After the full prompt is tokenized under the tokenizer's
        chat-template behavior, each sentinel is expanded into a fixed-width ``-1``
        span in token space.

        Returns:
            A tuple ``(tokenized_template, slot_metadata, compiled_slots)`` where
            ``tokenized_template`` is the compiled prompt, ``slot_metadata`` maps each
            substitution key to the corresponding token-space slot spans, and
            ``compiled_slots`` stores per-slot data for landmark construction.
        """
        slots = self._iter_template_slots()

        # Replace each slot occurrence with a unique sentinel token before the
        # chat template is applied, then expand the sentinels into fixed-width
        # token spans.
        #
        # This preserves the tokenizer's usual behavior on the full prompt,
        # including any chat-template wrappers, while still giving us exact
        # token-space boundaries for each substitution slot.
        tokenizer_with_specials = deepcopy(self.tokenizer)
        sentinel_tokens = [f" <SLOT_{i}>" for i in range(len(slots))]
        tokenizer_with_specials.add_special_tokens(
            {"additional_special_tokens": sentinel_tokens}
        )

        synthetic_template_parts: list[str] = []
        cursor = 0
        for i, slot in enumerate(slots):
            # Drop the leading space that belongs to the original slot, since
            # the sentinel token itself is registered with its own leading
            # space and should occupy the same textual position.
            synthetic_template_parts.append(
                self.schema.template[cursor:slot["text_start"] - 1]
            )
            synthetic_template_parts.append(f" <SLOT_{i}>")
            cursor = slot["text_end"]
        synthetic_template_parts.append(self.schema.template[cursor:])

        tokenized_with_sentinels = tokenizer_with_specials.apply_chat_template(
            [{"role": "user", "content": "".join(synthetic_template_parts)}],
            return_dict           = False,
            tokenize              = True,
            add_generation_prompt = True,
            enable_thinking       = False
        )

        sentinel_ids = {
            i: len(self.tokenizer) + i
            for i in range(len(slots))
        }
        sentinel_id_to_index = {
            token_id: i
            for i, token_id in sentinel_ids.items()
        }

        compiled_tokens: list[int] = []
        compiled_slots: list[dict[str, Any]] = []
        slot_metadata: dict[str, dict[str, Any]] = {}

        # Walk the tokenized prompt and replace each sentinel with its reserved
        # token-space slot span.
        for token_id in tokenized_with_sentinels:
            slot_index = sentinel_id_to_index.get(token_id)
            if slot_index is None:
                # Ordinary prompt tokens are copied through unchanged.
                compiled_tokens.append(token_id)
                continue

            slot = slots[slot_index]
            begin = len(compiled_tokens)
            # Reserve one token position per declared slot width. These ``-1``
            # values are placeholders in token space and will be overwritten
            # later when a concrete prompt is rendered.
            compiled_tokens.extend([-1] * slot["width"])
            end = len(compiled_tokens)

            # Record both the per-key grouped metadata used for rendering and
            # the per-occurrence compiled span used for landmark construction.
            slot_metadata.setdefault(
                slot["key"],
                {
                    "substitution_length": slot["width"],
                    "locations": [],
                },
            )["locations"].append((begin, end))

            compiled_slots.append(slot | {"begin": begin, "end": end})

        return compiled_tokens, slot_metadata, compiled_slots

    def _build_landmarks(self, compiled_slots: list[dict[str, Any]]) -> dict[str, int]:
        """Build token landmarks for slot starts and derived landmarks."""
        landmarks: dict[str, int] = {"PROMPT_END": len(self.tokenized_template) - 1}

        # Materialize primitive token landmarks from the compiled slot spans.
        for slot in compiled_slots:
            begin = slot["begin"]
            width = slot["width"]
            occurrence_name = slot["occurrence_name"]

            if width == 1:
                # One-token slots keep the old scalar landmark behavior.
                landmarks[occurrence_name] = begin
                continue

            # Multi-token slots expose one landmark per token position in the
            # reserved span, indexed relative to the slot start.
            for offset in range(width):
                landmarks[f"{occurrence_name}.{offset}"] = begin + offset

        # Evaluate any user-defined landmark expressions against the primitive
        # landmarks from the compiled template.
        for name, expr in self.schema.derived_landmarks.items():
            landmarks[name] = self._eval_landmark_expr(
                expr,
                landmarks=landmarks,
                prompt_length=len(self.tokenized_template),
            )

        return landmarks
            
    def _eval_landmark_expr(
        self,
        expr: str,
        landmarks: Mapping[str, int],
        prompt_length: int,
    ) -> int:
        """Evaluate a landmark expression like ``C1+1`` or ``LEN-1``."""
        expr = expr.strip()
        match = re.fullmatch(r"([A-Za-z0-9_]+)([+-]\d+)?", expr)
        if not match:
            raise ValueError(f"Invalid landmark expression: {expr!r}")

        name, offset = match.group(1), match.group(2)
        base = prompt_length if name == "LEN" else landmarks[name]
        return base + (int(offset) if offset else 0)
