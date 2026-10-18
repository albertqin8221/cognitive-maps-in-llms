"""Biology prompt preset definitions."""

from llm_neuroscience_tools.prompts.relational import RelationPool, RelationalPromptSchema


ABC_gene_regulatory_network_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a simple gene regulatory network.
2) For this task, genes are identified by three-letter strings.
3) At the end, determine whether the statement in the question is true. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
4) If the statement is true, output 'Yes'.
5) If the statement is false, output 'No'.
6) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
7) Output exactly one word (no punctuation or extra text).

Context: <A1> <R_AB:3> <B1>. <B2> <R_BC:3> <C1>.

Question: Does <QE> <QR:2> <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="yes_no",
)

activation_repression_relations = RelationPool(
    forward_relation="activates",
    backward_relation="represses",
    query_relations=("activate", "repress"),
)

ABC_neural_circuit_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a simple vertebrate neural circuit.
2) For this task, neurons are identified by three-letter strings.
3) At the end, determine whether the statement in the question is true. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
4) If the statement is true, output 'Yes'.
5) If the statement is false, output 'No'.
6) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
7) Output exactly one word (no punctuation or extra text).

Context: <A1> makes <R_AB:5> projections to <B1>. <B2> makes <R_BC:5> projections to <C1>.

Question: Does <QE> <QR:2> <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="yes_no",
)

excitation_inhibition_relations = RelationPool(
    forward_relation="glutamatergic",
    backward_relation="GABAergic",
    query_relations=("excite", "inhibit"),
)
