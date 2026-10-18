"""Company prompt preset definitions."""

from llm_neuroscience_tools.prompts.relational import RelationPool, RelationalPromptSchema


ABC_companies_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing the organizational structure in a company.
2) At the end, determine whether the statement in the question is true. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
3) If the statement is true, output 'Yes'.
4) If the statement is false, output 'No'.
5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
6) Output exactly one word (no punctuation or extra text).

Context: <A1>'s <R_AB> is <B1>. <B2>'s <R_BC> is <C1>.

Question: Is <QE> <QR> to <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="yes_no",
)

companies_relations = RelationPool(
    forward_relation="boss",
    backward_relation="employee",
    query_relations=("junior", "senior"),
)
