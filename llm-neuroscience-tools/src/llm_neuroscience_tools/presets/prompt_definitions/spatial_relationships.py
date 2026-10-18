"""Spatial-relationship prompt preset definitions."""

from llm_neuroscience_tools.prompts.relational import RelationPool, RelationalPromptSchema


ABC_spatial_relationships_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a series of spatial relationships between locations.
2) At the end, determine whether the statement in the question is true. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
3) If the statement is true, output 'Yes'.
4) If the statement is false, output 'No'.
5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
6) Output exactly one word (no punctuation or extra text).

Context: The <A1> is <R_AB> of the <B1>. The <B2> is <R_BC> of the <C1>.

Question: Is the <QE> <QR> of the <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="yes_no",
)

east_west_relations = RelationPool(
    forward_relation="east",
    backward_relation="west",
    query_relations=("east", "west"),
)

north_south_relations = RelationPool(
    forward_relation="north",
    backward_relation="south",
    query_relations=("north", "south"),
)

ahead_behind_relations = RelationPool(
    forward_relation="ahead",
    backward_relation="behind",
    query_relations=("ahead", "behind"),
)
