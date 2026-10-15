import json


def build_prompt_1_1(sample, error_category, subcat):
    return f"""
You are a data synthesis expert for Text-to-SQL evaluation.

Your goal is:
Given a CORRECT original Text-to-SQL question, rewrite it to deliberately introduce
a specific type of error while keeping the question otherwise natural and realistic.

----------------------------------------------------------------
Background knowledge

A necessary premise in Text-to-SQL refers to a query condition that MUST be specified
in order to uniquely determine the intended SQL query under a given database schema.

----------------------------------------------------------------
Error type definition (must strictly follow)

Ambiguity:
The rewritten question contains an ambiguous description of a NECESSARY SCHEMA-LEVEL
QUERY CONDITION, such that the condition may reasonably correspond to MULTIPLE
columns, tables, join paths, or aggregation targets in the given schema.

This ambiguity CANNOT be resolved using only the question text and the schema.

----------------------------------------------------------------
Task

Rewrite the original question so that:
- EXACTLY ONE necessary query condition becomes ambiguous at the schema level
- The ambiguity leads to AT LEAST TWO different and reasonable SQL interpretations
- Each interpretation must be valid under the given schema

----------------------------------------------------------------
Allowed ambiguity patterns (use ONE, include but not limited to):

- Entity reference ambiguity: A condition or object in the text may correspond to multiple columns, tables, or foreign key paths in the schema and cannot be uniquely determined.
- Aggregation ambiguity: The description of quantities, totals, averages, or other aggregation metrics is not precise.
- Time/Range ambiguity: Time or numerical ranges are not clearly specified.
- Status ambiguity: The state or category of an object is not clearly defined.
- Relationship ambiguity: The relationship between objects is not clearly defined, such as multiple possible join paths or foreign key relationships.

----------------------------------------------------------------
Strict requirements

- The rewritten question MUST contain only ONE error: schema-level ambiguity
- Do NOT invent tables or columns
- Do NOT clarify or hint at the ambiguity
- Do NOT simply paraphrase the original question

----------------------------------------------------------------
Give-up condition (IMPORTANT)

If it is NOT POSSIBLE to introduce the specified schema-level ambiguity WITHOUT:
- inventing schema elements,
- introducing non-schema-based vagueness,
- or violating the error definition,

then you MUST give up.

In this case:
- Set "give_up" to true
- Set all other fields to null EXCEPT "give_up_reason"
- Briefly explain why this question is not suitable for this error type

Do NOT force a rewrite.

----------------------------------------------------------------
Database schema:
{json.dumps(sample["schema"], indent=2)}

Foreign keys:
{json.dumps(sample["foreign_keys"], indent=2)}

Original question:
{sample["question"]}

Gold SQL (for reference only):
{sample["sql"]}

----------------------------------------------------------------
Output format

Return ONLY a JSON object with:
- rewritten_question (string or null)
- error_category (string or null)
- error_subcategory (string or null)
- give_up (boolean)
- give_up_reason (string or null)
""".strip()


##################################################################

def build_prompt_1_2(sample, error_category, subcat):
    return f"""
You are a data synthesis expert for Text-to-SQL evaluation.

Your goal is:
Given a CORRECT original Text-to-SQL question, rewrite it to deliberately introduce
a specific type of error while keeping the question otherwise natural and realistic.

----------------------------------------------------------------
Background knowledge

A necessary premise in Text-to-SQL refers to information that MUST be specified
to determine the OVERALL QUERY INTENT, including whether the user intends to:
- retrieve records,
- aggregate values,
- compare groups,
- rank or sort results.

If this intent is unclear, the SQL structure itself becomes indeterminate.

----------------------------------------------------------------
Error type definition (must strictly follow)

Unclear Intent:
The rewritten question fails to clearly specify the user's query intent
(e.g., selection vs aggregation vs comparison vs ranking),
such that it is NOT POSSIBLE to determine whether the SQL query should include
operations like GROUP BY, ORDER BY, aggregation functions, or simple selection.

This is a QUERY-LEVEL ambiguity, NOT a schema-level reference ambiguity.

----------------------------------------------------------------
Task

Rewrite the original question so that:
- The USER'S OVERALL QUERY INTENT becomes unclear
- It is impossible to determine whether the query requires:
  - simple SELECT,
  - aggregation (COUNT / SUM / AVG),
  - grouping (GROUP BY),
  - or ordering/comparison (ORDER BY / LIMIT)
- The ambiguity affects the GLOBAL SQL STRUCTURE, not a specific condition

----------------------------------------------------------------
Allowed unclear-intent patterns (use ONE, include but not limited to):

- Vague request for "status", "situation", or "overview" without specifying metrics
- Asking to "check", "see", or "look at" something without defining output form
- Referring to entities without clarifying whether details, counts, or summaries are desired
- Mixing potential intents implicitly (e.g., records vs statistics) without choosing one

----------------------------------------------------------------
Strict requirements

- The rewritten question MUST contain only ONE error: unclear query intent
- Do NOT introduce schema-level ambiguity
- Do NOT invent tables or columns
- Do NOT clarify, hint at, or resolve the intent ambiguity
- Do NOT simply paraphrase the original question

----------------------------------------------------------------
Give-up condition (IMPORTANT)

If it is NOT POSSIBLE to introduce unclear intent WITHOUT:
- changing the core entities of the question,
- introducing schema-level ambiguity,
- or violating the definition above,

then you MUST give up.

In this case:
- Set "give_up" to true
- Set all other fields to null EXCEPT "give_up_reason"
- Briefly explain why this question is not suitable for this error type

Do NOT force a rewrite.

----------------------------------------------------------------
Database schema:
{json.dumps(sample["schema"], indent=2)}

Foreign keys:
{json.dumps(sample["foreign_keys"], indent=2)}

Original question:
{sample["question"]}

Gold SQL (for reference only):
{sample["sql"]}

----------------------------------------------------------------
Output format

Return ONLY a JSON object with:
- rewritten_question (string or null)
- error_category (string or null)
- error_subcategory (string or null)
- give_up (boolean)
- give_up_reason (string or null)
""".strip()


######################################################################

def build_prompt_2_1(sample, error_category, subcat):
    return f"""
You are a data synthesis expert for Text-to-SQL evaluation.

Your goal is:
Given a CORRECT original Text-to-SQL question, rewrite it to deliberately introduce
a specific type of error while keeping the question otherwise natural and realistic.

----------------------------------------------------------------
Background knowledge

In Text-to-SQL, a necessary premise refers to a condition that MUST be true
for the intended SQL query to be logically valid.

If two or more necessary premises in the same question are mutually exclusive,
then no SQL query can satisfy all conditions simultaneously.

----------------------------------------------------------------
Error type definition (must strictly follow)

Internal Text Contradiction:
The rewritten question contains TWO OR MORE necessary premises that
LOGICALLY CANNOT be true at the same time.

This contradiction exists entirely within the question text itself,
and cannot be resolved by choosing different tables, columns, or SQL structures.

Even with a correct schema, NO valid SQL query can satisfy all conditions.

----------------------------------------------------------------
Task

Rewrite the original question so that:
- The question includes at least TWO necessary conditions
- These conditions are LOGICALLY INCOMPATIBLE or mutually exclusive
- The contradiction is explicit or logically unavoidable
- The conflict occurs at the TEXT / LOGIC level, not the schema level

----------------------------------------------------------------
Allowed contradiction patterns (use ONE, include but not limited to):

- Mutually exclusive numeric constraints
  (e.g., highest vs below a threshold)
- Conflicting comparative conditions
  (e.g., maximum but also minimum)
- Incompatible categorical states
  (e.g., active and inactive at the same time)
- Temporal contradictions
  (e.g., before and after the same event in incompatible ways)

----------------------------------------------------------------
Strict requirements

- The rewritten question MUST contain only ONE error: internal text contradiction
- Do NOT introduce schema-level ambiguity
- Do NOT invent tables or columns
- Do NOT rely on vague or underspecified language
- The contradiction must be inherent and unavoidable

----------------------------------------------------------------
Give-up condition (IMPORTANT)

If it is NOT POSSIBLE to introduce an internal text contradiction WITHOUT:
- changing the core entities of the question,
- inventing schema elements,
- or introducing schema-level ambiguity,

then you MUST give up.

In this case:
- Set "give_up" to true
- Set all other fields to null EXCEPT "give_up_reason"
- Briefly explain why this question is not suitable for this error type

Do NOT force a rewrite.

----------------------------------------------------------------
Database schema:
{json.dumps(sample["schema"], indent=2)}

Foreign keys:
{json.dumps(sample["foreign_keys"], indent=2)}

Original question:
{sample["question"]}

Gold SQL (for reference only):
{sample["sql"]}

----------------------------------------------------------------
Output format

Return ONLY a JSON object with:
- rewritten_question (string or null)
- error_category (string or null)
- error_subcategory (string or null)
- give_up (boolean)
- give_up_reason (string or null)
""".strip()


###############################################################################

def build_prompt_2_2(sample, error_category, subcat):
    return f"""
You are a data synthesis expert for Text-to-SQL evaluation.

Your goal is:
Given a CORRECT original Text-to-SQL question, rewrite it to deliberately introduce
a specific type of error while keeping the question otherwise natural and realistic.

----------------------------------------------------------------
Background knowledge

In Text-to-SQL, a question is only valid if every required table, column,
and relationship explicitly exists in the given database schema.

If a question necessarily depends on schema elements that do NOT exist,
then the question is logically impossible under the given schema.

----------------------------------------------------------------
Error type definition (must strictly follow)

Question–Schema Contradiction:
The rewritten question requires at least one table, column, or relationship
that is EXPLICITLY ABSENT from the given database schema.

This contradiction is caused by a DIRECT MISMATCH between the question
and the schema, not by ambiguity or underspecification.

No SQL query can be constructed to satisfy the question under the given schema.

----------------------------------------------------------------
Task

Rewrite the original question so that:
- The question necessarily depends on at least ONE schema element
  (table / column / relationship)
- That required element is CLEARLY NOT PRESENT in the given schema
- The contradiction is unavoidable and explicit
- The failure arises solely from schema incompatibility

----------------------------------------------------------------
Allowed contradiction patterns (use ONE, include but not limited to):

- Referring to a non-existent column
  (e.g., gender, status, category not defined in schema)
- Referring to a non-existent table or entity
- Requiring a relationship that is not supported by any foreign key
- Asking for attributes that the schema explicitly does not model

----------------------------------------------------------------
Strict requirements

- The rewritten question MUST contain only ONE error: question–schema contradiction
- Do NOT introduce schema-level ambiguity
- Do NOT invent schema elements beyond what is required for the contradiction
- Do NOT rely on vague language — the missing element must be necessary
- The impossibility must be caused ONLY by schema mismatch

----------------------------------------------------------------
Give-up condition (IMPORTANT)

If it is NOT POSSIBLE to introduce a question–schema contradiction WITHOUT:
- inventing completely unrelated entities,
- changing the core topic of the question,
- or introducing ambiguity instead of contradiction,

then you MUST give up.

In this case:
- Set "give_up" to true
- Set all other fields to null EXCEPT "give_up_reason"
- Briefly explain why this question is not suitable for this error type

Do NOT force a rewrite.

----------------------------------------------------------------
Database schema:
{json.dumps(sample["schema"], indent=2)}

Foreign keys:
{json.dumps(sample["foreign_keys"], indent=2)}

Original question:
{sample["question"]}

Gold SQL (for reference only):
{sample["sql"]}

----------------------------------------------------------------
Output format

Return ONLY a JSON object with:
- rewritten_question (string or null)
- error_category (string or null)
- error_subcategory (string or null)
- give_up (boolean)
- give_up_reason (string or null)
""".strip()


##################################################################

def build_prompt_3_1(sample, error_category, subcat):
    return f"""
You are a data synthesis expert for Text-to-SQL evaluation.

Your goal is:
Given a CORRECT original Text-to-SQL question, rewrite it to deliberately introduce
a specific type of error while keeping the question otherwise natural and realistic.

----------------------------------------------------------------
Background knowledge

In Text-to-SQL tasks, a query is only well-defined if all NECESSARY constraints
required to determine the intended result set are explicitly specified.

Constraints may include time ranges, object scopes, status filters,
or other limiting conditions that are required to make the query evaluable.

----------------------------------------------------------------
Error type definition (must strictly follow)

Missing Constraint:
The rewritten question omits at least ONE necessary filtering or limiting condition
required to uniquely determine or meaningfully evaluate the query result.

The query becomes underspecified, leading to multiple possible result sets
or an ill-defined aggregation scope.

This error is caused by ABSENCE of required information, not by contradiction
or schema mismatch.

----------------------------------------------------------------
Task

Rewrite the original question so that:
- At least ONE necessary constraint is intentionally omitted
- The missing constraint is REQUIRED to properly evaluate the query
- Without this constraint, the result set or aggregation scope is unclear
- The question remains fluent and natural but becomes underspecified

----------------------------------------------------------------
Allowed missing-constraint patterns (use ONE, include but not limited to):

- Missing time range or date constraint
- Missing status or category constraint
- Missing object scope or subset definition
- Missing numerical range or threshold
- Missing comparison baseline or reference group

----------------------------------------------------------------
Strict requirements

- The rewritten question MUST contain only ONE error: missing constraint
- Do NOT introduce ambiguity between multiple schema elements
- Do NOT introduce logical contradiction
- Do NOT invent schema elements
- The missing information must be NECESSARY, not optional

----------------------------------------------------------------
Give-up condition (IMPORTANT)

If it is NOT POSSIBLE to remove a necessary constraint WITHOUT:
- changing the core intent of the question,
- making the question trivially vague or meaningless,
- or introducing ambiguity or contradiction instead,

then you MUST give up.

In this case:
- Set "give_up" to true
- Set all other fields to null EXCEPT "give_up_reason"
- Briefly explain why this question is not suitable for this error type

Do NOT force a rewrite.

----------------------------------------------------------------
Database schema:
{json.dumps(sample["schema"], indent=2)}

Foreign keys:
{json.dumps(sample["foreign_keys"], indent=2)}

Original question:
{sample["question"]}

Gold SQL (for reference only):
{sample["sql"]}

----------------------------------------------------------------
Output format

Return ONLY a JSON object with:
- rewritten_question (string or null)
- error_category (string or null)
- error_subcategory (string or null)
- give_up (boolean)
- give_up_reason (string or null)
""".strip()


##############################################################

def build_prompt_3_2(sample, error_category, subcat):
    return f"""
You are a data synthesis expert for Text-to-SQL evaluation.

Your goal is:
Given a CORRECT original Text-to-SQL question, rewrite it to deliberately introduce
a specific type of error while keeping the question otherwise natural and realistic.

----------------------------------------------------------------
Background knowledge

In Text-to-SQL tasks, a user question must be fully grounded in the GIVEN database schema.
All required tables, columns, and relationships referenced by the question
must explicitly exist in the provided schema.

If a question depends on schema elements that are NOT provided,
the query becomes impossible to execute under the given schema.

----------------------------------------------------------------
Error type definition (must strictly follow)

Missing Schema Premise:
The rewritten question requires one or more tables, columns, or relationships
that are NECESSARY to execute the query but are NOT present in the given schema.

The schema explicitly lacks the required structural information,
making the query logically impossible under the provided schema.

This error is caused by missing schema support, not by ambiguity
or logical contradiction.

----------------------------------------------------------------
Task

Rewrite the original question so that:
- The question clearly requires at least ONE table, column, or relationship
  that does NOT exist in the given schema
- The missing schema element is NECESSARY to answer the question
- The question itself is clear and specific
- The failure arises solely because the schema does not support the query

----------------------------------------------------------------
Allowed missing-schema patterns (use ONE, include but not limited to):

- Referencing a non-existent table
- Referencing a non-existent column
- Requiring an unsupported relationship or join path
- Referring to attributes that are conceptually reasonable
  but explicitly absent from the schema

----------------------------------------------------------------
Strict requirements

- The rewritten question MUST contain only ONE error: missing schema premise
- Do NOT introduce ambiguity between existing schema elements
- Do NOT introduce logical contradiction within the question text
- Do NOT invent partial schema hints or explanations
- The schema mismatch must be explicit and unavoidable

----------------------------------------------------------------
Give-up condition (IMPORTANT)

If it is NOT POSSIBLE to introduce a missing schema premise WITHOUT:
- inventing unreasonable or unrealistic attributes,
- weakening the clarity of the question,
- or introducing ambiguity or contradiction instead,

then you MUST give up.

In this case:
- Set "give_up" to true
- Set all other fields to null EXCEPT "give_up_reason"
- Briefly explain why this question is not suitable for this error type

Do NOT force a rewrite.

----------------------------------------------------------------
Database schema:
{json.dumps(sample["schema"], indent=2)}

Foreign keys:
{json.dumps(sample["foreign_keys"], indent=2)}

Original question:
{sample["question"]}

Gold SQL (for reference only):
{sample["sql"]}

----------------------------------------------------------------
Output format

Return ONLY a JSON object with:
- rewritten_question (string or null)
- error_category (string or null)
- error_subcategory (string or null)
- give_up (boolean)
- give_up_reason (string or null)
""".strip()


##########################################################

def build_prompt_3_3(sample, error_category, subcat):
    return f"""
You are a data synthesis expert for Text-to-SQL evaluation.

Your goal is:
Given a CORRECT original Text-to-SQL question, rewrite it to deliberately introduce
a specific type of error while keeping the question otherwise natural and realistic.

----------------------------------------------------------------
Background knowledge

In Text-to-SQL tasks, user questions must map clearly and deterministically
to the given database schema.

If a question relies on abstract, business-level, or conceptual terms,
there must exist a clear and UNIQUE semantic mapping from those terms
to schema elements (columns, tables, relationships),
either explicitly or implicitly.

If such a mapping can only be guessed, the query is ill-defined.

----------------------------------------------------------------
Error type definition (must strictly follow)

Semantic Gap:
The rewritten question introduces a concept or term whose meaning
CANNOT be uniquely mapped to the given schema.

The schema may contain related raw data,
but there is NO direct field or unambiguous rule
that defines how the concept should be computed or identified.

The query therefore requires subjective interpretation or guessing.

This error is caused by missing semantic grounding,
not by missing schema elements or logical contradiction.

----------------------------------------------------------------
Task

Rewrite the original question so that:
- The question introduces ONE abstract or business-level concept
- The concept has NO direct column in the schema
- The schema does NOT provide a unique or implied rule
  to derive this concept
- The question does NOT explain or define the concept
- Answering the question would require guessing the intended mapping

----------------------------------------------------------------
Allowed semantic-gap patterns (use ONE, include but not limited to):

- Business concepts without definitions (e.g., high-value, active, loyal)
- Subjective or qualitative descriptors (e.g., important, risky, popular)
- Implicit classifications with no threshold or rule
- Derived metrics without computation logic
- Domain terms lacking schema-level grounding

----------------------------------------------------------------
Strict requirements

- The rewritten question MUST contain only ONE error: semantic gap
- Do NOT introduce schema-level ambiguity between multiple columns
- Do NOT require non-existent tables or columns
- Do NOT introduce logical contradiction
- The failure must be due solely to missing semantic mapping

----------------------------------------------------------------
Give-up condition (IMPORTANT)

If it is NOT POSSIBLE to introduce a semantic gap WITHOUT:
- inventing unrealistic or unnatural concepts,
- overlapping with missing schema premise or ambiguity,
- or changing the core intent of the question,

then you MUST give up.

In this case:
- Set "give_up" to true
- Set all other fields to null EXCEPT "give_up_reason"
- Briefly explain why this question is not suitable for this error type

Do NOT force a rewrite.

----------------------------------------------------------------
Database schema:
{json.dumps(sample["schema"], indent=2)}

Foreign keys:
{json.dumps(sample["foreign_keys"], indent=2)}

Original question:
{sample["question"]}

Gold SQL (for reference only):
{sample["sql"]}

----------------------------------------------------------------
Output format

Return ONLY a JSON object with:
- rewritten_question (string or null)
- error_category (string or null)
- error_subcategory (string or null)
- give_up (boolean)
- give_up_reason (string or null)
""".strip()


##################################################################

def build_prompt_4_1(sample, error_category, subcat):
    return f"""
You are a data synthesis expert for Text-to-SQL evaluation.

Your goal is:
Given a CORRECT original Text-to-SQL question, rewrite it to deliberately introduce
a specific type of error while keeping the question otherwise natural and realistic.

----------------------------------------------------------------
Background knowledge

Text-to-SQL tasks assume that the user intent can be expressed
as a deterministic SQL query over the given database schema.

SQL is limited to querying, filtering, aggregating, and joining
existing structured data.
It CANNOT perform prediction, reasoning about the future,
subjective judgment, or inference beyond the stored data.

----------------------------------------------------------------
Error type definition (must strictly follow)

Inexpressible in SQL:
The rewritten question requires an operation or intent
that CANNOT be precisely expressed using standard SQL
over the given schema.

Even if the schema is complete and correct,
there exists NO valid SQL query that can fully satisfy the request.

This limitation is due to SQL's expressive power,
not due to missing schema elements or ambiguity.

----------------------------------------------------------------
Task

Rewrite the original question so that:
- The question requires reasoning, prediction, or generation
  beyond querying existing data
- The intent CANNOT be reduced to filtering, aggregation,
  grouping, sorting, or joining
- The task would require external knowledge, modeling,
  or non-SQL computation
- The question remains fluent and realistic

----------------------------------------------------------------
Allowed beyond-ability patterns (use ONE, include but not limited to):

- Prediction or forecasting (e.g., future sales, trends)
- Classification or inference (e.g., likely to churn)
- Subjective judgment (e.g., best, most promising)
- Causal reasoning or explanation (e.g., why something happened)
- Non-structured analysis (e.g., sentiment, intent)

----------------------------------------------------------------
Strict requirements

- The rewritten question MUST contain only ONE error: inexpressible intent
- Do NOT rely on missing tables or columns
- Do NOT introduce schema-level ambiguity
- Do NOT introduce logical contradiction
- Failure must be due solely to SQL expressive limits

----------------------------------------------------------------
Give-up condition (IMPORTANT)

If it is NOT POSSIBLE to introduce an inexpressible intent WITHOUT:
- overlapping with missing schema premise,
- requiring non-existent schema elements,
- or changing the question into an unrealistic request,

then you MUST give up.

In this case:
- Set "give_up" to true
- Set all other fields to null EXCEPT "give_up_reason"
- Briefly explain why this question is not suitable for this error type

Do NOT force a rewrite.

----------------------------------------------------------------
Database schema:
{json.dumps(sample["schema"], indent=2)}

Foreign keys:
{json.dumps(sample["foreign_keys"], indent=2)}

Original question:
{sample["question"]}

Gold SQL (for reference only):
{sample["sql"]}

----------------------------------------------------------------
Output format

Return ONLY a JSON object with:
- rewritten_question (string or null)
- error_category (string or null)
- error_subcategory (string or null)
- give_up (boolean)
- give_up_reason (string or null)
""".strip()


######################################################################

def build_prompt_4_2(sample, error_category, subcat):
    return f"""
You are a data synthesis expert for Text-to-SQL evaluation.

Your goal is:
Given a CORRECT original Text-to-SQL question, rewrite it to deliberately introduce
a specific type of error while keeping the question otherwise natural and realistic.

----------------------------------------------------------------
Background knowledge

Text-to-SQL tasks assume that all required information to answer the question
exists in the database schema.

If the question requires facts, rules, or knowledge outside the database,
then the intent cannot be accurately expressed in SQL alone.

----------------------------------------------------------------
Error type definition (must strictly follow)

External Knowledge Required:
The rewritten question depends on information NOT present in the given schema,
such as common sense, external data, or hidden facts.

Even if the schema is complete,
there is no deterministic SQL query that can fully satisfy the question
without accessing external knowledge.

This limitation is due to missing data, not SQL expressive power.

----------------------------------------------------------------
Task

Rewrite the original question so that:
- The question depends on real-world knowledge or facts outside the database
- The database schema does NOT contain the required fields
- Completing the query would require external resources, lookup, or business rules
- The question remains fluent and realistic

----------------------------------------------------------------
Allowed external-knowledge patterns (use ONE, include but not limited to):

- Real-world calendars, holidays, or events
- Geographic or demographic facts
- Business rules or domain-specific knowledge
- Public statistics or standards
- Implicit temporal knowledge

----------------------------------------------------------------
Strict requirements

- The rewritten question MUST contain only ONE error: requires external knowledge
- Do NOT introduce schema-level ambiguity
- Do NOT invent non-existent tables or columns
- Do NOT introduce logical contradictions
- Failure must be solely due to missing external information

----------------------------------------------------------------
Give-up condition (IMPORTANT)

If it is NOT POSSIBLE to introduce external knowledge dependency WITHOUT:
- overlapping with missing schema premise or semantic gap,
- inventing unrealistic data or concepts,
- or changing the core intent of the question,

then you MUST give up.

In this case:
- Set "give_up" to true
- Set all other fields to null EXCEPT "give_up_reason"
- Briefly explain why this question is not suitable for this error type

Do NOT force a rewrite.

----------------------------------------------------------------
Database schema:
{json.dumps(sample["schema"], indent=2)}

Foreign keys:
{json.dumps(sample["foreign_keys"], indent=2)}

Original question:
{sample["question"]}

Gold SQL (for reference only):
{sample["sql"]}

----------------------------------------------------------------
Output format

Return ONLY a JSON object with:
- rewritten_question (string or null)
- error_category (string or null)
- error_subcategory (string or null)
- give_up (boolean)
- give_up_reason (string or null)
""".strip()
