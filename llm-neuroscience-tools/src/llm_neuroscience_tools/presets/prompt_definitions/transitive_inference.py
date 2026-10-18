"""Transitive-inference prompt preset definitions."""

from llm_neuroscience_tools.prompts.relational import RelationPool, RelationalPromptSchema


ABC_transitive_inference_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a series of comparisons between people.
2) At the end, determine whether the statement in the question is true. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
3) If the statement is true, output 'Yes'.
4) If the statement is false, output 'No'.
5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
6) Output exactly one word (no punctuation or extra text).

Context: <A1> is <R_AB> than <B1>. <B2> is <R_BC> than <C1>.

Question: Is <QE> <QR> than <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="yes_no",
)

older_younger_relations = RelationPool(
    forward_relation="older",
    backward_relation="younger",
    query_relations=("older", "younger"),
)

taller_shorter_relations = RelationPool(
    forward_relation="taller",
    backward_relation="shorter",
    query_relations=("taller", "shorter"),
)

faster_slower_relations = RelationPool(
    forward_relation="faster",
    backward_relation="slower",
    query_relations=("faster", "slower"),
)
