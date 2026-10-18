"""Additional prompt preset definitions.
These prompts instantiate a few ideas I had to figure out what's going on in the
map-vs-graph experiments.
"""

from llm_neuroscience_tools.prompts.relational import RelationPool, RelationalPromptSchema


# ---------------- AB ; CD family trees ----------------


# Note that we'll still call the second relation "R_BC" for compatibility
# with the old experiments. However, this is actually the relationship between
# entities C and D.


AB_CD_family_tree_who_is_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a series of relationships between people in a family.
2) Keep track of who is related to whom as you read.
3) You may assume that everyone who appears in the context is female.
4) At the end, answer the question. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
6) Output exactly one word (no punctuation or extra text).

Context: <A1> has <R_AB> <B1>. <C1> has <R_BC> <D1>.

Question: Who is <QE>'s <QR>?

Answer: """,
    context_entity_symbols=("A", "B", "C", "D"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE",),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "D1+1"},
    query_mode="who_is",
)

AB_CD_family_tree_yes_no_schema = RelationalPromptSchema(
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

Context: <A1> has <R_AB> <B1>. <C1> has <R_BC> <D1>.

Question: Is <QE>'s <QR> <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C", "D"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "D1+1"},
    query_mode="yes_no",
)

family_relations_simple_queries = RelationPool(
    forward_relation="daughter",
    backward_relation="mother",
    query_relations=("daughter", "mother"),
)


# ---------------- Transitive inference, with unique locations ----------------


ABC_older_younger_unique_location_yes_no_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a series of comparisons between people.
2) At the end, determine whether the statement in the question is true. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
3) If the statement is true, output 'Yes'.
4) If the statement is false, output 'No'.
5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
6) Output exactly one word (no punctuation or extra text).

Context: <A1> is <R_AB:4> <B1>. <B2> is <R_BC:4> <C1>.

Question: Is <QE> <QR:4> <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="yes_no",
)

ABC_older_younger_unique_location_who_is_schema = RelationalPromptSchema(
    template="""Instructions:
1) Read the context describing a series of comparisons between people.
2) At the end, answer the question. You may use:
    (a) Information stated in the context.
    (b) Any facts that follow from the context by valid logical inference.
3) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.
4) Output exactly one word (no punctuation or extra text).

Context: <A1> is <R_AB:4> <B1>. <B2> is <R_BC:4> <C1>.

Question: Who is <QE> <QR:4>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE",),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="who_is",
)

unique_older_younger_relations = RelationPool(
    forward_relation="one year older than",
    backward_relation="one year younger than",
    query_relations=("equal in age to",
                     "one year older than", 
                     "one year younger than", 
                     "two years older than", 
                     "two years younger than",
                     "older than",
                     "younger than"),
)


# ---------------- Family trees, with nonunique locations ----------------


ABC_family_tree_descendant_ancestor_schema = RelationalPromptSchema(
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

Question: Is <QE> <QR:2> of <QCAND>?

Answer: """,
    context_entity_symbols=("A", "B", "C"),
    context_relation_symbols=("R_AB", "R_BC"),
    query_entity_symbols=("QE", "QCAND"),
    query_relation_symbols=("QR",),
    derived_landmarks={"CTXT_START": "A1", "CTXT_END": "C1+1"},
    query_mode="yes_no",
)

descendant_ancestor_relations = RelationPool(
    forward_relation="daughter",
    backward_relation="mother",
    query_relations=("an ancestor", "a descendant"),
)
