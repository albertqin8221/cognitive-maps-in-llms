"""Equivalence prompt preset definitions."""

from dataclasses import replace

from llm_neuroscience_tools.prompts.relational import RelationPool, RelationalPromptSchema


ABC_proposition_equivalence_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a series of relationships between propositions.
2) For this task, propositions are numbered by a roman numeral.
3) At the end, determine whether the statement in the question is true. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
4) If the statement is true, output 'Yes'.
5) If the statement is false, output 'No'.
6) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
7) Output exactly one word (no punctuation or extra text).

Context: <A1> is <R_AB:2> <B1>. <B2> is <R_BC:2> <C1>.

Question: Is <QE> <QR:2> <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="yes_no",
)

proposition_equivalence_relations = RelationPool(
    forward_relation="equivalent to",
    backward_relation="incompatible with",
    query_relations=("equivalent to", "compatible with"),
)

ABC_variable_equivalence_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a series of relationships between variables.
2) For this task, variables are identified by a single letter.
3) At the end, determine whether the statement in the question is true. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
4) If the statement is true, output 'Yes'.
5) If the statement is false, output 'No'.
6) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
7) Output exactly one word (no punctuation or extra text).

Context: <A1> is <R_AB:2> to <B1>. <B2> is <R_BC:2> to <C1>.

Question: Is <QE> <QR:2> to <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="yes_no",
)

variable_equivalence_relations = RelationPool(
    forward_relation="equal",
    backward_relation="not equal",
    query_relations=("equal",),
)


ABC_alien_proposition_equivalence_schema = replace(
    ABC_proposition_equivalence_schema,
    template=ABC_proposition_equivalence_schema.template.replace(
        "For this task, propositions are numbered by a roman numeral.",
        "For this task, each proposition is identified by a name.",
    ),
)


ABC_alien_variable_equivalence_schema = replace(
    ABC_variable_equivalence_schema,
    template=ABC_variable_equivalence_schema.template.replace(
        "For this task, variables are identified by a single letter.",
        "For this task, each variable is identified by a name.",
    ),
)
