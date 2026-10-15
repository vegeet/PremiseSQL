from typing import Dict, Any
import json


def build_review_prompt_1_1(sample: Dict[str, Any]) -> str:
    return f"""
You are a STRICT dataset auditor, not a helper.

===================== ABSOLUTE RULES =====================
- Default decision is FAIL.
- Your primary job is to FIND PROBLEMS, not to approve.
- You MUST actively look for reasons to reject the sample.
- If a case is borderline, unclear, or debatable, it MUST FAIL.
- Being generous, charitable, or cooperative is an error.
- In a healthy dataset, MANY samples should fail.

IMPORTANT:
You MUST first identify potential issues BEFORE deciding overall_pass.
If you cannot find any issue, you must explicitly explain why none exist.
=========================================================

Your task is to review whether the rewritten question correctly introduces
EXACTLY ONE schema-level ambiguity, as specified by the assigned error subcategory.

----------------------------------------------------------------
Database schema:
{json.dumps(sample.get('schema', {}), indent=2)}

Foreign keys:
{json.dumps(sample.get('foreign_keys', []), indent=2)}

Original question:
{sample.get('question')}

Rewritten question:
{sample.get('rewritten_question')}

Assigned error category / subcategory:
{sample.get('error_category')} / {sample.get('error_subcategory')}

Gold SQL (reference):
{sample.get('sql')}

Give-up info:
give_up: {sample.get('give_up')}
give_up_reason: {sample.get('give_up_reason')}

----------------------------------------------------------------
REVIEW PROCEDURE (MUST FOLLOW IN ORDER):

Step 1: Actively search for problems.
- List ANY potential issues, weaknesses, or ambiguities that could invalidate this sample.
- Even minor or borderline issues should be listed.

Step 2: Decide the strongest reason to FAIL.
- If the sample should be rejected, clearly state why.
- If you believe it should pass, explain why NO failure reason is valid.

Step 3: Decide the strongest reason to PASS.
- State the single best argument in favor of accepting this sample.

Step 4: Checklist evaluation (TRUE / FALSE / NULL):
1. correct_error_type
2. single_error_only
3. error_matches_subcategory
4. no_extra_tables_or_columns
5. ambiguity_clarity
6. sql_interpretation_ambiguity

Step 5: Final verdict.
- Set overall_pass = TRUE ONLY IF all checklist items are clearly satisfied
  AND the failure reasons are convincingly refuted.
- Otherwise, overall_pass = FALSE.

----------------------------------------------------------------
Return ONLY a JSON object with the following keys:
potential_issues,
strongest_reason_to_fail,
strongest_reason_to_pass,
correct_error_type,
single_error_only,
error_matches_subcategory,
no_extra_tables_or_columns,
ambiguity_clarity,
sql_interpretation_ambiguity,
overall_pass,
comments
""".strip()


######################################################################
def build_review_prompt_1_2(sample: Dict[str, Any]) -> str:
    return f"""
You are a STRICT dataset auditor, not a helper.

===================== ABSOLUTE RULES =====================
- Default decision is FAIL.
- Your primary job is to FIND PROBLEMS, not to approve.
- You MUST actively look for reasons to reject the sample.
- If a case is borderline, unclear, or debatable, it MUST FAIL.
- Being generous, charitable, or cooperative is an error.
- In a healthy dataset, MANY samples should fail.

IMPORTANT:
You MUST first identify potential issues BEFORE deciding overall_pass.
If you cannot find any issue, you must explicitly explain why none exist.
=========================================================

Your task is to review whether the rewritten question correctly introduces
EXACTLY ONE instance of UNCLEAR QUERY INTENT, as specified by the assigned
error subcategory.

----------------------------------------------------------------
Error type focus (CRITICAL)

Unclear Intent refers to a failure to specify the OVERALL QUERY OPERATION TYPE,
such that it is NOT POSSIBLE to determine whether the SQL should involve:
- simple selection (SELECT records),
- aggregation (COUNT / SUM / AVG),
- grouping (GROUP BY),
- sorting or ranking (ORDER BY / LIMIT),
- or comparison between groups.

This is a QUERY-LEVEL STRUCTURAL ambiguity,
NOT a schema-level reference ambiguity.

----------------------------------------------------------------
Database schema:
{json.dumps(sample.get('schema', {}), indent=2)}

Foreign keys:
{json.dumps(sample.get('foreign_keys', []), indent=2)}

Original question:
{sample.get('question')}

Rewritten question:
{sample.get('rewritten_question')}

Assigned error category / subcategory:
{sample.get('error_category')} / {sample.get('error_subcategory')}

Gold SQL (reference):
{sample.get('sql')}

Give-up info:
give_up: {sample.get('give_up')}
give_up_reason: {sample.get('give_up_reason')}

----------------------------------------------------------------
REVIEW PROCEDURE (MUST FOLLOW IN ORDER):

Step 1: Actively search for problems.
- List ANY potential issues that could invalidate this sample.
- Pay special attention to:
  * Whether the rewritten question clearly specifies the intended query operation.
  * Whether multiple SQL structures (SELECT vs GROUP BY vs ORDER BY) are plausible.
- Even minor or borderline issues MUST be listed.

Step 2: Decide the strongest reason to FAIL.
- Reject the sample if:
  * The intent is actually clear (e.g., explicit counts, rankings, filters).
  * More than one error type is introduced.
  * The ambiguity is schema-level rather than intent-level.
- If you believe it should pass, explain why NO failure reason applies.

Step 3: Decide the strongest reason to PASS.
- State the single strongest argument that the rewritten question truly leaves
  the overall SQL structure indeterminate.

Step 4: Checklist evaluation (TRUE / FALSE / NULL):
1. correct_error_type
   - Is the error about unclear QUERY INTENT rather than schema ambiguity?
2. single_error_only
   - Is unclear intent the ONLY error present?
3. error_matches_subcategory
   - Does the ambiguity concern selection vs aggregation vs ordering?
4. no_extra_tables_or_columns
   - Were no schema elements invented or misused?
5. intent_clarity_failure
   - Is it genuinely impossible to determine the required SQL structure?
6. multiple_sql_structures_possible
   - Are multiple reasonable SQL forms plausible under this question?

Step 5: Final verdict.
- Set overall_pass = TRUE ONLY IF:
  * Exactly one unclear-intent error exists, AND
  * The overall SQL structure cannot be uniquely determined.
- Otherwise, overall_pass = FALSE.

----------------------------------------------------------------
Return ONLY a JSON object with the following keys:
potential_issues,
strongest_reason_to_fail,
strongest_reason_to_pass,
correct_error_type,
single_error_only,
error_matches_subcategory,
no_extra_tables_or_columns,
intent_clarity_failure,
multiple_sql_structures_possible,
overall_pass,
comments
""".strip()


######################################################################
def build_review_prompt_2_1(sample: Dict[str, Any]) -> str:
    return f"""
You are a STRICT dataset auditor, not a helper.

===================== ABSOLUTE RULES =====================
- Default decision is FAIL.
- Your primary job is to FIND PROBLEMS, not to approve.
- You MUST actively look for reasons to reject the sample.
- If a case is borderline, unclear, or debatable, it MUST FAIL.
- Being generous, charitable, or cooperative is an error.
- In a healthy dataset, MANY samples should fail.

IMPORTANT:
You MUST first identify potential issues BEFORE deciding overall_pass.
If you cannot find any issue, you must explicitly explain why none exist.
=========================================================

Your task is to review whether the rewritten question correctly introduces
EXACTLY ONE instance of INTERNAL TEXT CONTRADICTION, as specified by the assigned
error subcategory.

----------------------------------------------------------------
Error type focus (CRITICAL)

Internal Text Contradiction occurs when the rewritten question contains TWO OR MORE necessary premises
that LOGICALLY CANNOT be true at the same time.

This contradiction exists entirely within the question text itself,
and cannot be resolved by choosing different tables, columns, or SQL structures.

Even with a correct schema, NO valid SQL query can satisfy all conditions simultaneously.

This is a TEXT/LOGIC-LEVEL contradiction, NOT a schema-level ambiguity.

----------------------------------------------------------------
Database schema:
{json.dumps(sample.get('schema', {}), indent=2)}

Foreign keys:
{json.dumps(sample.get('foreign_keys', []), indent=2)}

Original question:
{sample.get('question')}

Rewritten question:
{sample.get('rewritten_question')}

Assigned error category / subcategory:
{sample.get('error_category')} / {sample.get('error_subcategory')}

Gold SQL (reference):
{sample.get('sql')}

Give-up info:
give_up: {sample.get('give_up')}
give_up_reason: {sample.get('give_up_reason')}

----------------------------------------------------------------
REVIEW PROCEDURE (MUST FOLLOW IN ORDER):

Step 1: Actively search for problems.
- List ANY potential issues that could invalidate this sample.
- Pay special attention to:
  * Whether the rewritten question contains necessary premises that contradict each other.
  * Whether the contradiction is explicit or logically unavoidable.
- Even minor or borderline issues MUST be listed.

Step 2: Decide the strongest reason to FAIL.
- Reject the sample if:
  * No internal text contradiction is actually introduced.
  * More than one error type is present.
  * The issue arises from schema-level ambiguity rather than text-level contradiction.
- If you believe it should pass, explain why NO failure reason applies.

Step 3: Decide the strongest reason to PASS.
- State the single strongest argument that the rewritten question contains exactly one
  unavoidable internal text contradiction.

Step 4: Checklist evaluation (TRUE / FALSE / NULL):
1. correct_error_type
   - Is the error specifically an internal text contradiction?
2. single_error_only
   - Is internal text contradiction the ONLY error present?
3. error_matches_subcategory
   - Are the conflicting premises logically incompatible as required by this subcategory?
4. no_extra_tables_or_columns
   - Were no schema elements invented or misused?
5. contradiction_clarity
   - Is the contradiction explicit and logically unavoidable?
6. sql_interpretation_ambiguity
   - Is SQL interpretation logically impossible due to the contradiction?

Step 5: Final verdict.
- Set overall_pass = TRUE ONLY IF:
  * Exactly one internal text contradiction exists, AND
  * The contradiction is unavoidable and logically clear.
- Otherwise, overall_pass = FALSE.

----------------------------------------------------------------
Return ONLY a JSON object with the following keys:
potential_issues,
strongest_reason_to_fail,
strongest_reason_to_pass,
correct_error_type,
single_error_only,
error_matches_subcategory,
no_extra_tables_or_columns,
contradiction_clarity,
sql_interpretation_ambiguity,
overall_pass,
comments
""".strip()


#######################################################################
def build_review_prompt_2_2(sample: Dict[str, Any]) -> str:
    return f"""
You are a STRICT dataset auditor, not a helper.

===================== ABSOLUTE RULES =====================
- Default decision is FAIL.
- Your primary job is to FIND PROBLEMS, not to approve.
- You MUST actively look for reasons to reject the sample.
- If a case is borderline, unclear, or debatable, it MUST FAIL.
- Being generous, charitable, or cooperative is an error.
- In a healthy dataset, MANY samples should fail.

IMPORTANT:
You MUST first identify potential issues BEFORE deciding overall_pass.
If you cannot find any issue, you must explicitly explain why none exist.
=========================================================

Your task is to review whether the rewritten question correctly introduces
EXACTLY ONE instance of QUESTION–SCHEMA CONTRADICTION, as specified by the assigned
error subcategory.

----------------------------------------------------------------
Error type focus (CRITICAL)

Question–Schema Contradiction occurs when the rewritten question requires
at least one table, column, or relationship that is **explicitly absent**
from the given database schema.

This contradiction is caused by a direct mismatch between the question
and the schema, not by ambiguity or underspecification.

No SQL query can be constructed to satisfy the question under the given schema.

This is a SCHEMA-LEVEL contradiction, NOT a text/logic-level contradiction.

----------------------------------------------------------------
Database schema:
{json.dumps(sample.get('schema', {}), indent=2)}

Foreign keys:
{json.dumps(sample.get('foreign_keys', []), indent=2)}

Original question:
{sample.get('question')}

Rewritten question:
{sample.get('rewritten_question')}

Assigned error category / subcategory:
{sample.get('error_category')} / {sample.get('error_subcategory')}

Gold SQL (reference):
{sample.get('sql')}

Give-up info:
give_up: {sample.get('give_up')}
give_up_reason: {sample.get('give_up_reason')}

----------------------------------------------------------------
REVIEW PROCEDURE (MUST FOLLOW IN ORDER):

Step 1: Actively search for problems.
- List ANY potential issues that could invalidate this sample.
- Pay special attention to:
  * Whether the rewritten question depends on tables, columns, or relationships
    that do NOT exist in the schema.
  * Whether the contradiction is unavoidable and explicit.
- Even minor or borderline issues MUST be listed.

Step 2: Decide the strongest reason to FAIL.
- Reject the sample if:
  * No question–schema contradiction is actually introduced.
  * More than one error type is present.
  * The issue arises from text-level ambiguity rather than schema-level contradiction.
- If you believe it should pass, explain why NO failure reason applies.

Step 3: Decide the strongest reason to PASS.
- State the single strongest argument that the rewritten question contains exactly one
  unavoidable question–schema contradiction.

Step 4: Checklist evaluation (TRUE / FALSE / NULL):
1. correct_error_type
   - Is the error specifically a question–schema contradiction?
2. single_error_only
   - Is question–schema contradiction the ONLY error present?
3. error_matches_subcategory
   - Does the contradiction arise from explicitly missing schema elements?
4. no_extra_tables_or_columns
   - Were no schema elements invented beyond what is required?
5. contradiction_clarity
   - Is the contradiction explicit and logically unavoidable?
6. sql_interpretation_impossible
   - Is SQL interpretation impossible due to the missing schema elements?

Step 5: Final verdict.
- Set overall_pass = TRUE ONLY IF:
  * Exactly one question–schema contradiction exists, AND
  * The contradiction is unavoidable and explicit.
- Otherwise, overall_pass = FALSE.

----------------------------------------------------------------
Return ONLY a JSON object with the following keys:
potential_issues,
strongest_reason_to_fail,
strongest_reason_to_pass,
correct_error_type,
single_error_only,
error_matches_subcategory,
no_extra_tables_or_columns,
contradiction_clarity,
sql_interpretation_impossible,
overall_pass,
comments
""".strip()


########################################################################
def build_review_prompt_3_1(sample: Dict[str, Any]) -> str:
    return f"""
You are a STRICT dataset auditor, not a helper.

===================== ABSOLUTE RULES =====================
- Default decision is FAIL.
- Your primary job is to FIND REASONS TO REJECT the sample.
- Passing a sample is an exception, not the norm.
- If a case is borderline, debatable, or can be interpreted generously, it MUST FAIL.
- Assuming default interpretations such as "all records" is FORBIDDEN.

IMPORTANT:
You MUST identify concrete failure risks BEFORE considering acceptance.
=========================================================

Your task is to review whether the rewritten question introduces
EXACTLY ONE instance of **Missing Constraint (3.1)**.

----------------------------------------------------------------
CRITICAL DEFINITION (READ CAREFULLY)

A Missing Constraint (3.1) error exists ONLY IF:

- The original question required a SPECIFIC evaluation boundary
  (e.g., a partition, comparison context, scope, or reference group),
- The rewritten question REMOVES that boundary,
- AND as a result, it is IMPOSSIBLE to determine a unique and meaningful
  SQL query WITHOUT GUESSING.

If the rewritten question can still be reasonably interpreted as:
"compute this over ALL records by default",
then it is NOT a valid 3.1 error and MUST FAIL.

This error is about a missing TASK-DEFINING BOUNDARY,
NOT merely a missing optional filter.

----------------------------------------------------------------
Database schema:
{json.dumps(sample.get('schema', {}), indent=2)}

Foreign keys:
{json.dumps(sample.get('foreign_keys', []), indent=2)}

Original question:
{sample.get('question')}

Rewritten question:
{sample.get('rewritten_question')}

Assigned error category / subcategory:
{sample.get('error_category')} / {sample.get('error_subcategory')}

Gold SQL (reference only):
{sample.get('sql')}

Give-up info:
give_up: {sample.get('give_up')}
give_up_reason: {sample.get('give_up_reason')}

----------------------------------------------------------------
REVIEW PROCEDURE (MUST FOLLOW IN ORDER)

Step 1: Actively search for failure reasons.
- Identify what constraint was removed compared to the original question.
- Determine whether this constraint DEFINES the evaluation boundary
  (e.g., partition, comparison baseline, scope).
- Explicitly ask:
  "Can this rewritten question still be naturally answered
   by assuming 'all records' or a global aggregation?"

If YES → this is NOT Missing Constraint → MUST FAIL.

List ALL potential issues, including borderline ones.

----------------------------------------------------------------
Step 2: Strongest reason to FAIL.
Reject the sample if ANY of the following is true:
- No task-defining boundary was removed
- The question is merely broader, not underspecified
- A single reasonable SQL query still exists without guessing
- The error degenerates into unclear intent (1.2)
- Schema-level ambiguity or contradiction is introduced

State the strongest single rejection reason.

----------------------------------------------------------------
Step 3: Strongest reason to PASS.
Only if no failure condition applies:
- State ONE clear reason why the missing constraint makes the task
  fundamentally non-evaluable without guessing.

----------------------------------------------------------------
Step 4: Checklist (TRUE / FALSE / NULL)

1. correct_error_type
   - Is this truly a Missing Constraint (absence of task boundary)?

2. single_error_only
   - Is Missing Constraint the ONLY error present?

3. error_matches_subcategory
   - Is the removed constraint REQUIRED to define the evaluation scope?

4. no_extra_tables_or_columns
   - Were no schema elements invented or misused?

5. boundary_missing_clarity
   - Is the missing evaluation boundary clearly identifiable?

6. non_defaultable_scope
   - Is it INVALID to interpret the query as applying to all records?

----------------------------------------------------------------
Step 5: Final verdict.

Set overall_pass = TRUE ONLY IF:
- ALL checklist items are clearly satisfied
- AND the query cannot be answered by a single reasonable SQL
  without making an arbitrary assumption.

Otherwise, overall_pass = FALSE.

----------------------------------------------------------------
Return ONLY a JSON object with the following keys:
potential_issues,
strongest_reason_to_fail,
strongest_reason_to_pass,
correct_error_type,
single_error_only,
error_matches_subcategory,
no_extra_tables_or_columns,
boundary_missing_clarity,
non_defaultable_scope,
overall_pass,
comments
""".strip()



##################################################################
def build_review_prompt_3_2(sample: Dict[str, Any]) -> str:
    return f"""
You are a STRICT dataset auditor, not a helper.

===================== ABSOLUTE RULES =====================
- Default decision is FAIL.
- Your primary job is to FIND PROBLEMS, not to approve.
- You MUST actively look for reasons to reject the sample.
- If a case is borderline, unclear, or debatable, it MUST FAIL.
- Being generous, charitable, or cooperative is an error.
- In a healthy dataset, MANY samples should fail.

IMPORTANT:
You MUST first identify potential issues BEFORE deciding overall_pass.
If you cannot find any issue, you must explicitly explain why none exist.
=========================================================

Your task is to review whether the rewritten question correctly introduces
EXACTLY ONE instance of **Missing Schema Premise**, as specified by the assigned error subcategory.

----------------------------------------------------------------
Error type focus (CRITICAL)

Missing Schema Premise refers to a user question that requires one or more
tables, columns, or relationships that are NECESSARY to execute the query
but are NOT provided in the given database schema.

This makes the query logically impossible under the provided schema.
The error is caused solely by missing schema support, not by logical
contradiction or ambiguity in the text.

----------------------------------------------------------------
Database schema:
{json.dumps(sample.get('schema', {}), indent=2)}

Foreign keys:
{json.dumps(sample.get('foreign_keys', []), indent=2)}

Original question:
{sample.get('question')}

Rewritten question:
{sample.get('rewritten_question')}

Assigned error category / subcategory:
{sample.get('error_category')} / {sample.get('error_subcategory')}

Gold SQL (reference):
{sample.get('sql')}

Give-up info:
give_up: {sample.get('give_up')}
give_up_reason: {sample.get('give_up_reason')}

----------------------------------------------------------------
REVIEW PROCEDURE (MUST FOLLOW IN ORDER):

Step 1: Actively search for problems.
- Identify any missing schema elements required to execute the query.
- List any other issues that could invalidate this sample.
- Pay special attention to:
  * Tables, columns, or relationships referenced in the question but absent in the schema
  * Whether the missing element is NECESSARY to execute the SQL
- Even minor or borderline issues MUST be listed.

Step 2: Decide the strongest reason to FAIL.
- Reject the sample if:
  * No required schema element is missing
  * Multiple error types are introduced
  * Ambiguity or contradiction exists instead of missing schema
- If you believe it should pass, explain why no failure reason applies.

Step 3: Decide the strongest reason to PASS.
- State the single strongest argument that the rewritten question
  contains exactly one missing schema premise and is otherwise valid.

Step 4: Checklist evaluation (TRUE / FALSE / NULL):
1. correct_error_type
   - Is the error truly a missing schema premise rather than contradiction or underspecification?
2. single_error_only
   - Is missing schema premise the ONLY error present?
3. error_matches_subcategory
   - Is the missing element required to execute the query?
4. no_extra_tables_or_columns
   - Were no unrelated schema elements invented?
5. schema_missing_clarity
   - Is the missing schema element clearly necessary and absent?
6. query_executable_impossible
   - Does the missing schema make execution of the query impossible?

Step 5: Final verdict.
- Set overall_pass = TRUE ONLY IF all checklist items are clearly satisfied
  AND the failure reasons are convincingly refuted.
- Otherwise, overall_pass = FALSE.

----------------------------------------------------------------
Return ONLY a JSON object with the following keys:
potential_issues,
strongest_reason_to_fail,
strongest_reason_to_pass,
correct_error_type,
single_error_only,
error_matches_subcategory,
no_extra_tables_or_columns,
schema_missing_clarity,
query_executable_impossible,
overall_pass,
comments
""".strip()



########################################################################
def build_review_prompt_3_3(sample: Dict[str, Any]) -> str:
    return f"""
You are a STRICT dataset auditor, not a helper.

===================== ABSOLUTE RULES =====================
- Default decision is FAIL.
- Your primary job is to FIND PROBLEMS, not to approve.
- You MUST actively look for reasons to reject the sample.
- If a case is borderline, unclear, or debatable, it MUST FAIL.
- Being generous, charitable, or cooperative is an error.
- In a healthy dataset, MANY samples should fail.

IMPORTANT:
You MUST first identify potential issues BEFORE deciding overall_pass.
If you cannot find any issue, you must explicitly explain why none exist.
=========================================================

Your task is to review whether the rewritten question correctly introduces
EXACTLY ONE instance of **Semantic Gap**, as specified by the assigned error subcategory.

----------------------------------------------------------------
Error type focus (CRITICAL)

Semantic Gap refers to a question that introduces a concept, term, or
business-level notion that CANNOT be uniquely mapped to the given schema.

Even if the schema contains related raw data, there is:
- NO direct column, table, or relationship
- NO implicit or unambiguous derivation rule

Thus, answering the query would require guessing or subjective interpretation.

This error is caused solely by missing semantic grounding,
not by missing schema elements or logical contradiction.

----------------------------------------------------------------
Database schema:
{json.dumps(sample.get('schema', {}), indent=2)}

Foreign keys:
{json.dumps(sample.get('foreign_keys', []), indent=2)}

Original question:
{sample.get('question')}

Rewritten question:
{sample.get('rewritten_question')}

Assigned error category / subcategory:
{sample.get('error_category')} / {sample.get('error_subcategory')}

Gold SQL (reference):
{sample.get('sql')}

Give-up info:
give_up: {sample.get('give_up')}
give_up_reason: {sample.get('give_up_reason')}

----------------------------------------------------------------
REVIEW PROCEDURE (MUST FOLLOW IN ORDER):

Step 1: Actively search for problems.
- Identify any concept, term, or metric in the rewritten question
  that cannot be uniquely mapped to the schema.
- List any other issues that could invalidate this sample.
- Even minor or borderline issues MUST be listed.

Step 2: Decide the strongest reason to FAIL.
- Reject the sample if:
  * The concept can be clearly derived from schema elements
  * Multiple error types are present
  * Ambiguity, missing constraints, or schema mismatch is the cause instead
- If you believe it should pass, explain why no failure reason applies.

Step 3: Decide the strongest reason to PASS.
- State the single strongest argument that the rewritten question
  contains exactly one semantic gap and is otherwise clear.

Step 4: Checklist evaluation (TRUE / FALSE / NULL):
1. correct_error_type
   - Is the error truly a semantic gap rather than missing schema or contradiction?
2. single_error_only
   - Is semantic gap the ONLY error present?
3. error_matches_subcategory
   - Is the concept ungrounded in the schema?
4. no_extra_tables_or_columns
   - Were no schema elements invented or misused?
5. semantic_gap_clarity
   - Is the missing semantic mapping explicit and unavoidable?
6. query_requires_guessing
   - Does the question require subjective interpretation to answer?

Step 5: Final verdict.
- Set overall_pass = TRUE ONLY IF all checklist items are clearly satisfied
  AND the failure reasons are convincingly refuted.
- Otherwise, overall_pass = FALSE.

----------------------------------------------------------------
Return ONLY a JSON object with the following keys:
potential_issues,
strongest_reason_to_fail,
strongest_reason_to_pass,
correct_error_type,
single_error_only,
error_matches_subcategory,
no_extra_tables_or_columns,
semantic_gap_clarity,
query_requires_guessing,
overall_pass,
comments
""".strip()

####################################################################
def build_review_prompt_4_1(sample: Dict[str, Any]) -> str:
    return f"""
You are a STRICT dataset auditor, not a helper.

===================== ABSOLUTE RULES =====================
- Default decision is FAIL.
- Your primary job is to FIND PROBLEMS, not to approve.
- You MUST actively look for reasons to reject the sample.
- If a case is borderline, unclear, or debatable, it MUST FAIL.
- Being generous, charitable, or cooperative is an error.
- In a healthy dataset, MANY samples should fail.

IMPORTANT:
You MUST first identify potential issues BEFORE deciding overall_pass.
If you cannot find any issue, you must explicitly explain why none exist.
=========================================================

Your task is to review whether the rewritten question correctly introduces
EXACTLY ONE instance of **Inexpressible in SQL**, as specified by the assigned error subcategory.

----------------------------------------------------------------
Error type focus (CRITICAL)

Inexpressible in SQL refers to a question whose intent
CANNOT be precisely expressed using standard SQL over the given schema.

Even with a complete and correct schema, there is:
- NO valid SQL query that fully satisfies the user request

This error is due solely to the **expressive limits of SQL**, not missing tables,
columns, logical contradictions, or schema ambiguity.

----------------------------------------------------------------
Database schema:
{json.dumps(sample.get('schema', {}), indent=2)}

Foreign keys:
{json.dumps(sample.get('foreign_keys', []), indent=2)}

Original question:
{sample.get('question')}

Rewritten question:
{sample.get('rewritten_question')}

Assigned error category / subcategory:
{sample.get('error_category')} / {sample.get('error_subcategory')}

Gold SQL (reference):
{sample.get('sql')}

Give-up info:
give_up: {sample.get('give_up')}
give_up_reason: {sample.get('give_up_reason')}

----------------------------------------------------------------
REVIEW PROCEDURE (MUST FOLLOW IN ORDER):

Step 1: Actively search for problems.
- Identify whether the rewritten question requires operations, reasoning,
  or information beyond what SQL over the given schema can express.
- List any other issues that could invalidate this sample.
- Even minor or borderline issues MUST be listed.

Step 2: Decide the strongest reason to FAIL.
- Reject the sample if:
  * The intent can be fully expressed in SQL
  * Multiple error types are present
  * The failure arises from missing schema, contradiction, or ambiguity instead
- If you believe it should pass, explain why no failure reason applies.

Step 3: Decide the strongest reason to PASS.
- State the single strongest argument that the rewritten question
  contains exactly one inexpressible intent and is otherwise valid.

Step 4: Checklist evaluation (TRUE / FALSE / NULL):
1. correct_error_type
   - Is the error truly due to SQL inexpressibility rather than schema or logic issues?
2. single_error_only
   - Is inexpressible intent the ONLY error present?
3. error_matches_subcategory
   - Does the question require reasoning, prediction, or analysis beyond SQL?
4. no_extra_tables_or_columns
   - Were no schema elements invented or misused?
5. inexpressibility_clarity
   - Is the SQL limitation explicit and unavoidable?
6. requires_external_reasoning
   - Does the question require knowledge or computation beyond the schema?

Step 5: Final verdict.
- Set overall_pass = TRUE ONLY IF all checklist items are clearly satisfied
  AND the failure reasons are convincingly refuted.
- Otherwise, overall_pass = FALSE.

----------------------------------------------------------------
Return ONLY a JSON object with the following keys:
potential_issues,
strongest_reason_to_fail,
strongest_reason_to_pass,
correct_error_type,
single_error_only,
error_matches_subcategory,
no_extra_tables_or_columns,
inexpressibility_clarity,
requires_external_reasoning,
overall_pass,
comments
""".strip()


#################################################################
def build_review_prompt_4_2(sample: Dict[str, Any]) -> str:
    return f"""
You are a STRICT dataset auditor, not a helper.

===================== ABSOLUTE RULES =====================
- Default decision is FAIL.
- Your primary job is to FIND PROBLEMS, not to approve.
- You MUST actively look for reasons to reject the sample.
- If a case is borderline, unclear, or debatable, it MUST FAIL.
- Being generous, charitable, or cooperative is an error.
- In a healthy dataset, MANY samples should fail.

IMPORTANT:
You MUST first identify potential issues BEFORE deciding overall_pass.
If you cannot find any issue, you must explicitly explain why none exist.
=========================================================

Your task is to review whether the rewritten question correctly introduces
EXACTLY ONE instance of **External Knowledge Required**, as specified by the assigned error subcategory.

----------------------------------------------------------------
Error type focus (CRITICAL)

External Knowledge Required refers to a question whose correct execution
depends on facts or knowledge outside the given database schema.

Even if the schema is complete, there is:
- NO deterministic SQL query that can fully satisfy the request
without access to external information.

This error is caused solely by missing real-world knowledge,
not by missing schema elements, logical contradictions, or SQL expressive limits.

----------------------------------------------------------------
Database schema:
{json.dumps(sample.get('schema', {}), indent=2)}

Foreign keys:
{json.dumps(sample.get('foreign_keys', []), indent=2)}

Original question:
{sample.get('question')}

Rewritten question:
{sample.get('rewritten_question')}

Assigned error category / subcategory:
{sample.get('error_category')} / {sample.get('error_subcategory')}

Gold SQL (reference):
{sample.get('sql')}

Give-up info:
give_up: {sample.get('give_up')}
give_up_reason: {sample.get('give_up_reason')}

----------------------------------------------------------------
REVIEW PROCEDURE (MUST FOLLOW IN ORDER):

Step 1: Actively search for problems.
- Identify whether the rewritten question requires information, facts,
  or reasoning beyond what is available in the database schema.
- List any other issues that could invalidate this sample.
- Even minor or borderline issues MUST be listed.

Step 2: Decide the strongest reason to FAIL.
- Reject the sample if:
  * All required information exists in the schema and no external knowledge is needed
  * Multiple error types are present
  * The failure arises from missing schema, semantic gap, or contradiction instead
- If you believe it should pass, explain why no failure reason applies.

Step 3: Decide the strongest reason to PASS.
- State the single strongest argument that the rewritten question
  contains exactly one dependency on external knowledge and is otherwise valid.

Step 4: Checklist evaluation (TRUE / FALSE / NULL):
1. correct_error_type
   - Is the error truly due to external knowledge requirement rather than schema or logic issues?
2. single_error_only
   - Is external knowledge dependency the ONLY error present?
3. error_matches_subcategory
   - Does the question require real-world facts, business rules, or domain knowledge?
4. no_extra_tables_or_columns
   - Were no schema elements invented or misused?
5. external_dependency_clarity
   - Is the need for external knowledge explicit and unavoidable?
6. cannot_execute_with_schema_only
   - Is SQL over the given schema insufficient to answer the question?

Step 5: Final verdict.
- Set overall_pass = TRUE ONLY IF all checklist items are clearly satisfied
  AND the failure reasons are convincingly refuted.
- Otherwise, overall_pass = FALSE.

----------------------------------------------------------------
Return ONLY a JSON object with the following keys:
potential_issues,
strongest_reason_to_fail,
strongest_reason_to_pass,
correct_error_type,
single_error_only,
error_matches_subcategory,
no_extra_tables_or_columns,
external_dependency_clarity,
cannot_execute_with_schema_only,
overall_pass,
comments
""".strip()
