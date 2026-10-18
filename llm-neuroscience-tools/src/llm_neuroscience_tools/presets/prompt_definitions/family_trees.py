"""Family-tree prompt preset definitions."""

from llm_neuroscience_tools.prompts.relational import RelationPool, RelationalPromptSchema


ABC_family_tree_who_is_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a series of relationships between people in a family.
2) Keep track of who is related to whom as you read.
3) You may assume that everyone who appears in the context is female.
4) At the end, answer the question. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
6) Output exactly one word (no punctuation or extra text).

Context: <A1> has <R_AB> <B1>. <B2> has <R_BC> <C1>.

Question: Who is <QE>'s <QR>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE",),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="who_is",
)

ABC_family_tree_yes_no_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a series of relationships between people in a family.
2) Keep track of who is related to whom as you read.
3) You may assume that everyone who appears in the context is female.
4) At the end, answer the question. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
5) If the statement is true, output 'Yes'.
6) If the statement is false, output 'No'.
7) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
8) Output exactly one word (no punctuation or extra text).

Context: <A1> has <R_AB> <B1>. <B2> has <R_BC> <C1>.

Question: Is <QE>'s <QR> <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="yes_no",
)

family_relations = RelationPool(
    forward_relation="daughter",
    backward_relation="mother",
    query_relations=("daughter", "mother", "sister", "grandmother", "granddaughter"),
)


# ---------------- Alien instructions? ----------------

"""Instructions:
1) Read the context describing a series of relationships between aliens in an alien family.
2) Keep track of who is related to whom as you read.
3) You may assume that the family relationships obey the usual rules of kinship reasoning.
4) You may assume that everyone who appears in the context is female.
5) At the end, answer the question. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
6) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
7) Output exactly one word (no punctuation or extra text).

Context: <A1> has <R_AB> <B1>. <B2> has <R_BC> <C1>.

Question: Who is <QE>'s <QR>?

Answer: """
