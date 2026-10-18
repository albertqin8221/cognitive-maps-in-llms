import pytest

from tokenizers import Tokenizer, models, decoders
from transformers import PreTrainedTokenizerFast

from llm_neuroscience_tools.prompts import (
    PromptRuntime,
    PromptSchema,
    RelationalPromptSchema,
)


# ---- Custom tokenizer definition ----


MICROVOCABLIST = {
    0:  '\n',
    1:  ':',
    2:  '.',
    3:  'Context',
    4:  'Then',
    5:  ' Alice',
    6:  ' Beth',
    7:  ' Carol',
    8:  ' has',
    9:  ' daughter',
    10: ' '
}

@pytest.fixture(scope='module')
def micro_tokenizer():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel())
    )
    tokenizer.backend_tokenizer.decoder = decoders.Fuse()
    tokenizer.add_tokens(list(MICROVOCABLIST.values()))
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    return tokenizer


# ---- Construction of prompt runtime ----


MICROPROMPTSCHEMA = PromptSchema(
    template = (
"""Context: <A1> <R_AB:2> <B1>. <B2> <R_BC:2> <C1>.
Then: <QE> has <QR>:"""
    ),
    substitution_keys = ('A', 'B', 'C', 'R_AB', 'R_BC', 'QE', 'QR'),
    derived_landmarks = {'CTXT_START': 'A1', 'CTXT_END': 'C1+1'}
)

@pytest.fixture(scope='module')
def micro_runtime(micro_tokenizer) -> PromptRuntime:
    return PromptRuntime(
        schema = MICROPROMPTSCHEMA,
        tokenizer = micro_tokenizer
    )


# ---- Prompt runtime tests ----


SUBSTITUTIONS = {
    'A':    'Alice',
    'B':    'Beth',
    'C':    'Carol',
    'R_AB': 'has daughter',
    'R_BC': 'has daughter',
    'QE':   'Beth',
    'QR':   'daughter'
}

EXPECTED_PROMPT = """Context: Alice has daughter Beth. Beth has daughter Carol.
Then: Beth has daughter:"""

EXPECTED_LANDMARKS = {
    'A1':     2,
    'R_AB.0': 3,
    'R_AB.1': 4,
    'B1':     5,
    'B2':     7,
    'R_BC.0': 8,
    'R_BC.1': 9,
    'C1':     10,
    'QE':     15,
    'QR':     17,
    'CTXT_START': 2,
    'CTXT_END':   11,
    'PROMPT_END': 18
}


def test_micro_runtime_renders_expected_prompt(micro_runtime):
    prompt = micro_runtime.render_prompt(SUBSTITUTIONS)
    assert prompt == EXPECTED_PROMPT


def test_micro_runtime_computes_expected_landmarks(micro_runtime):
    assert micro_runtime.landmarks == EXPECTED_LANDMARKS


@pytest.mark.codex
def test_micro_runtime_fills_none_substitution_with_filler_tokens(micro_runtime):
    substitutions = dict(SUBSTITUTIONS)
    substitutions["R_AB"] = None

    prompt = micro_runtime.render_prompt_as_tokens(substitutions)

    for begin, end in micro_runtime.slot_metadata["R_AB"]["locations"]:
        assert prompt[begin:end] == [
            micro_runtime.filler_token_id
        ] * (end - begin)


@pytest.mark.codex
def test_relational_prompt_schema_rejects_unknown_query_mode():
    with pytest.raises(ValueError, match="query_mode must be either"):
        RelationalPromptSchema(
            template="Context: <A>. Question: <QE> <QR>?",
            context_entity_symbols=("A",),
            context_relation_symbols=(),
            query_entity_symbols=("QE",),
            query_relation_symbols=("QR",),
            query_mode="multiple_choice",  # type: ignore[arg-type]
            derived_landmarks={},
        )
