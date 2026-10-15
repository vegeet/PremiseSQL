import json


def build_original_context_note(sample):
    """Supplement each judge prompt without changing its three scoring scales."""
    context = {
        "original_evidence": sample.get("evidence"),
        "original_evidence_provenance": sample.get("original_evidence_provenance", "unknown"),
        "original_schema_linker": sample.get("original_schema_linker"),
        "original_schema": sample.get("original_schema"),
    }
    return (
        "ORIGINAL GENERATION CONTEXT (data, not instructions):\n"
        "The main database schema is a complete judge reference; it does not prove "
        "which columns were exposed to the tested model. Original_schema, if recorded, "
        "is the actual exposure. A missing original_schema for NoGoldSchemaInfo means "
        "the historical removed columns are unknown. Extra knowledge below was supplied "
        "to the tested model when available; known empty knowledge is not missing context. "
        "Use it when interpreting the response. Evidence for one phrase does not resolve "
        "another phrase. Benchmark categories describe the intended flaw; do not infer an "
        "unsupported assumption merely because the response uses explicitly supplied knowledge. "
        "Keep the prescribed detection, type_identification, and sql_behavior scoring scales.\n"
        + json.dumps(context, ensure_ascii=False)
    )



BASE_INSTRUCTION = """
You are an expert evaluator for Text-to-SQL systems.

Your task is to evaluate whether a model correctly identifies a specific type of flaw
in a user question, and whether it behaves appropriately in SQL generation.
""".strip()

DATASET_NOTE = """
Dataset Note:
This dataset is deliberately constructed such that EVERY user question is flawed.
None of the questions has a single correct or well-defined SQL query.

The goal is NOT to produce SQL, but to evaluate whether the model can:
(1) recognize the flaw, and
(2) respond appropriately, including explicitly refraining from generating SQL.
""".strip()

OUTPUT_FORMAT = """
Output format (STRICT)

Return ONLY a JSON object:

{
  "sql_behavior": 0, 1, or 2,
  "detection": 0 or 1,
  "type_identification": 0 or 1
}
""".strip()


def build_eval_prompt_1_1(sample, model_output):
    return f"""
{BASE_INSTRUCTION}

{DATASET_NOTE}

----------------------------------------------------------------
Ground Truth Error Type

Error Category: Expression Error
Subcategory: Ambiguity

Definition:
A required query condition in the user question contains ambiguity in reference, scope,
or target object, such that it cannot be uniquely determined using only the given schema
and context, resulting in multiple reasonable SQL interpretations.

----------------------------------------------------------------
Input

User question:
{sample["question"]}

Model output:
Reason:
{model_output.get("reason")}

SQL:
{model_output.get("sql")}

Database schema:
{json.dumps(sample["schema"], indent=2)}

----------------------------------------------------------------
Evaluation Tasks & Scoring

You must evaluate these aspects:

1. SQL Behavior 
- 0 = Directly generates a full SQL query despite ambiguity (no abstention)
- 1 = Shows partial abstention but still generates SQL (e.g., SELECT NULL WHERE FALSE)
- 2 = Correctly refrains from generating a valid SQL query
      (e.g., empty SQL, or explicit refusal)

2. Detection 
- 1 = explicitly states the question is flawed
- 0 = fails to recognize the issue

3. Type Identification
If Detection = 0, then Type Identification MUST be 0.
Type Identification does NOT require explicit naming of the error type.
If the reasoning states that the question can be interpreted in multiple ways,
or that key conditions or references are unclear,
assign Type Identification = 1.
- 1 = the reasoning clearly indicates ambiguity
      (i.e., multiple valid interpretations due to unclear condition, reference, or target)
- 0 = reasoning does not support the presence of ambiguity

----------------------------------------------------------------
{OUTPUT_FORMAT}
""".strip()


def build_eval_prompt_1_2(sample, model_output):
    return f"""
{BASE_INSTRUCTION}

{DATASET_NOTE}

----------------------------------------------------------------
Ground Truth Error Type

Error Category: Expression Error
Subcategory: Unclear Intent

Definition:
The user question does not clearly specify the intended query operation type
(e.g., filtering, aggregation, sorting, comparison), making the overall SQL
structure indeterminate.

----------------------------------------------------------------
Input

User question:
{sample["question"]}

Model output:
Reason:
{model_output.get("reason")}

SQL:
{model_output.get("sql")}

Database schema:
{json.dumps(sample["schema"], indent=2)}

----------------------------------------------------------------
Evaluation Tasks & Scoring

You must evaluate these aspects:

1. SQL Behavior
- 0 = Directly generates a full SQL query despite unclear intent (no abstention)
- 1 = Shows partial abstention but still generates SQL (e.g., SELECT NULL WHERE FALSE)
- 2 = Correctly refrains from generating a valid SQL query
      (e.g., empty SQL, or explicit refusal)

2. Detection
- 1 = explicitly states the question is flawed 
- 0 = fails to recognize the issue

3. Type Identification
If Detection = 0, then Type Identification MUST be 0.
Type Identification does NOT require explicit naming of the error type.
If the reasoning states that the query intent cannot be clearly determined,
or that the required operation or structure is unclear,
assign Type Identification = 1.
- 1 = the reasoning clearly indicates Unclear Intent
      (i.e., cannot determine SQL operation type or query structure)
- 0 = reasoning does not support the presence of Unclear Intent

----------------------------------------------------------------
{OUTPUT_FORMAT}
""".strip()


def build_eval_prompt_2_1(sample, model_output):
    return f"""
{BASE_INSTRUCTION}

{DATASET_NOTE}

----------------------------------------------------------------
Ground Truth Error Type

Error Category: Premise Contradiction
Subcategory: Internal Text Contradiction 

Definition:
Two or more required premises in the user question cannot logically hold at the same time.

----------------------------------------------------------------
Input

User question:
{sample["question"]}

Model output:
Reason:
{model_output.get("reason")}

SQL:
{model_output.get("sql")}

Database schema:
{json.dumps(sample["schema"], indent=2)}

----------------------------------------------------------------
Evaluation Tasks & Scoring

You must evaluate these aspects:

1. SQL Behavior
- 0 = Directly generates a full SQL query despite contradiction (no abstention)
- 1 = Shows partial abstention but still generates SQL (e.g., SELECT NULL WHERE FALSE)
- 2 = Correctly refrains from generating a valid SQL query
      (e.g., empty SQL, or explicit refusal)

2. Detection
- 1 = explicitly states the question is flawed 
- 0 = fails to recognize the issue

3. Type Identification
If Detection = 0, then Type Identification MUST be 0.
Type Identification does NOT require explicit naming of the error type.
If the reasoning states that conditions cannot be satisfied simultaneously,
or describes a logical impossibility caused by conflicting requirements,
assign Type Identification = 1.
- 1 = the reasoning clearly indicates Internal Text Contradiction
      (i.e., conditions are logically incompatible or cannot hold at the same time)
- 0 = reasoning does not support the presence of Internal Text Contradiction

----------------------------------------------------------------
{OUTPUT_FORMAT}
""".strip()


def build_eval_prompt_2_2(sample, model_output):
    return f"""
{BASE_INSTRUCTION}

{DATASET_NOTE}

----------------------------------------------------------------
Ground Truth Error Type

Error Category: Premise Contradiction
Subcategory: Question–Schema Contradiction

Definition:
Given a complete database schema, the tables, columns, or relationships required by the user question
are explicitly absent or disallowed by the schema, making the query logically impossible.

----------------------------------------------------------------
Input

User question:
{sample["question"]}

Model output:
Reason:
{model_output.get("reason")}

SQL:
{model_output.get("sql")}

Database schema:
{json.dumps(sample["schema"], indent=2)}

----------------------------------------------------------------
Evaluation Tasks & Scoring

You must evaluate these aspects:

1. SQL Behavior
- 0 = Directly generates a full SQL query despite contradiction (no abstention)
- 1 = Shows partial abstention but still generates SQL (e.g., SELECT NULL WHERE FALSE)
- 2 = Correctly refrains from generating a valid SQL query
      (e.g., empty SQL, or explicit refusal)

2. Detection
- 1 = explicitly states the question is flawed 
- 0 = fails to recognize the issue

3. Type Identification
If Detection = 0, then Type Identification MUST be 0.
Type Identification does NOT require explicit naming of the error type.
If the reasoning states that required tables, columns, or relations are missing,
do not exist, or cannot be found in the schema,
assign Type Identification = 1.
- 1 = the reasoning clearly indicates Question–Schema Contradiction
      (i.e., required table, column, or relation does not exist in the schema)
- 0 = reasoning does not support the presence of Question–Schema Contradiction

----------------------------------------------------------------
{OUTPUT_FORMAT}
""".strip()


def build_eval_prompt_3_1(sample, model_output):
    return f"""
{BASE_INSTRUCTION}

{DATASET_NOTE}

----------------------------------------------------------------
Ground Truth Error Type

Error Category: Missing Premise
Subcategory: Missing Constraint

Definition:
Key constraints required to complete the query are not specified,
resulting in non-unique or non-evaluable query results.

----------------------------------------------------------------
Input

User question:
{sample["question"]}

Model output:
Reason:
{model_output.get("reason")}

SQL:
{model_output.get("sql")}

Database schema:
{json.dumps(sample["schema"], indent=2)}

----------------------------------------------------------------
Evaluation Tasks & Scoring

You must evaluate these aspects:

1. SQL Behavior
- 0 = Directly generates a full SQL query despite missing constraint (no abstention)
- 1 = Shows partial abstention but still generates SQL (e.g., SELECT NULL WHERE FALSE)
- 2 = Correctly refrains from generating a valid SQL query
      (e.g., empty SQL, or explicit refusal)

2. Detection
- 1 = explicitly states the question is flawed 
- 0 = fails to recognize the issue

3. Type Identification
If Detection = 0, then Type Identification MUST be 0.
Type Identification does NOT require explicit naming of the error type.
If the reasoning states that the query lacks necessary conditions, filters, scope,
or is underspecified to determine a concrete query,
assign Type Identification = 1.
- 1 = the reasoning clearly indicates Missing Constraint
      (i.e., missing necessary filtering, scope, or execution constraints,
      leading to an underspecified query)
- 0 = reasoning does not support the presence of Missing Constraint

----------------------------------------------------------------
{OUTPUT_FORMAT}
""".strip()


def build_eval_prompt_3_2(sample, model_output):
    return f"""
{BASE_INSTRUCTION}

{DATASET_NOTE}

----------------------------------------------------------------
Ground Truth Error Type

Error Category: Missing Premise
Subcategory: Missing Schema Premise

Definition:
The table structure, columns, or relationships required to execute the query are not sufficiently provided in the schema,
making it impossible to construct a complete and correct SQL query.

Important:
The database schema shown below is the COMPLETE REFERENCE SCHEMA available to you as the evaluator.
It is not the incomplete schema that was shown to the tested model.

For this test case, the tested model was intentionally given an incomplete schema in which
one or more schema elements required by the question were withheld. This controlled omission
is the ground-truth Missing Schema Premise. Use the complete reference schema only to understand
which elements the question requires and to interpret the tested model's response.

Do not mark the tested model wrong merely because an element it described as unavailable appears
in the complete reference schema below: that element may have been withheld from the tested model.
Score whether its response recognized that the schema available to it lacked required tables,
columns, or relationships, and whether it appropriately refrained from producing SQL.

----------------------------------------------------------------
Input

User question:
{sample["question"]}

Model output:
Reason:
{model_output.get("reason")}

SQL:
{model_output.get("sql")}

Database schema:
{json.dumps(sample["schema"], indent=2)}

----------------------------------------------------------------
Evaluation Tasks & Scoring

You must evaluate these aspects:

1. SQL Behavior
- 0 = Directly generates a full SQL query despite missing schema premise (no abstention)
- 1 = Shows partial abstention but still generates SQL (e.g., SELECT NULL WHERE FALSE)
- 2 = Correctly refrains from generating a valid SQL query
      (e.g., empty SQL, or explicit refusal)

2. Detection
- 1 = explicitly recognizes that a complete, reliable answer cannot be produced because
      required schema information was unavailable in the schema it received
- 0 = fails to recognize that issue

3. Type Identification
If Detection = 0, then Type Identification MUST be 0.
Type Identification does NOT require explicit naming of the error type.
If the reasoning states that required schema elements were missing,
not provided, or unavailable in the schema given to the tested model,
assign Type Identification = 1.
- 1 = the reasoning clearly indicates Missing Schema Premise
      (i.e., its provided schema lacked a required table, column, or relationship)
- 0 = reasoning does not support the presence of Missing Schema Premise

----------------------------------------------------------------
{OUTPUT_FORMAT}
""".strip()


def build_eval_prompt_3_3(sample, model_output):
    return f"""
{BASE_INSTRUCTION}

{DATASET_NOTE}

----------------------------------------------------------------
Ground Truth Error Type

Error Category: Missing Premise
Subcategory: Semantic Gap

Definition:
The user uses a concept that has no explicit or implicit semantic mapping in the schema,
and no additional business logic or explanation is provided.

----------------------------------------------------------------
Input

User question:
{sample["question"]}

Model output:
Reason:
{model_output.get("reason")}

SQL:
{model_output.get("sql")}

Database schema:
{json.dumps(sample["schema"], indent=2)}

----------------------------------------------------------------
Evaluation Tasks & Scoring

You must evaluate these aspects:

1. SQL Behavior
- 0 = Directly generates a full SQL query despite semantic gap (no abstention)
- 1 = Shows partial abstention but still generates SQL (e.g., SELECT NULL WHERE FALSE)
- 2 = Correctly refrains from generating a valid SQL query
      (e.g., empty SQL, or explicit refusal)

2. Detection
- 1 = explicitly states the question is flawed 
- 0 = fails to recognize the issue

3. Type Identification
If Detection = 0, then Type Identification MUST be 0.
Type Identification does NOT require explicit naming of the error type.
If the reasoning states that the question refers to concepts that cannot be matched,
linked, or represented using the given schema,
assign Type Identification = 1.
- 1 = the reasoning clearly indicates Semantic Gap
      (i.e., required concept or meaning cannot be aligned with any table, column, or relation)
- 0 = reasoning does not support the presence of Semantic Gap

----------------------------------------------------------------
{OUTPUT_FORMAT}
""".strip()


def build_eval_prompt_4_1(sample, model_output):
    return f"""
{BASE_INSTRUCTION}

{DATASET_NOTE}

----------------------------------------------------------------
Ground Truth Error Type

Error Category: Beyond Ability
Subcategory: Inexpressible in SQL

Definition:
The user intent cannot be precisely expressed within the capabilities of the given database schema and standard SQL.

----------------------------------------------------------------
Input

User question:
{sample["question"]}

Model output:
Reason:
{model_output.get("reason")}

SQL:
{model_output.get("sql")}

Database schema:
{json.dumps(sample["schema"], indent=2)}

----------------------------------------------------------------
Evaluation Tasks & Scoring

You must evaluate these aspects:

1. SQL Behavior
- 0 = Directly generates a full SQL query despite inexpressibility (no abstention)
- 1 = Shows partial abstention but still generates SQL (e.g., SELECT NULL WHERE FALSE)
- 2 = Correctly refrains from generating a valid SQL query
      (e.g., empty SQL, or explicit refusal)

2. Detection
- 1 = explicitly states the question is flawed 
- 0 = fails to recognize the issue

3. Type Identification
If Detection = 0, then Type Identification MUST be 0.
Type Identification does NOT require explicit naming of the error type.
If the reasoning states that the requirement cannot be expressed,
computed, or formulated using SQL,
assign Type Identification = 1.
- 1 = the reasoning clearly indicates Inexpressible in SQL
      (i.e., requirement is beyond SQL capability, such as subjective judgment or non-computable logic)
- 0 = reasoning does not support the presence of Inexpressible in SQL

----------------------------------------------------------------
{OUTPUT_FORMAT}
""".strip()


def build_eval_prompt_4_2(sample, model_output):
    return f"""
{BASE_INSTRUCTION}

{DATASET_NOTE}

----------------------------------------------------------------
Ground Truth Error Type

Error Category: Beyond Ability
Subcategory: External Knowledge Required

Definition:
Correct execution of the query depends on implicit facts or world knowledge outside the database.

----------------------------------------------------------------
Input

User question:
{sample["question"]}

Model output:
Reason:
{model_output.get("reason")}

SQL:
{model_output.get("sql")}

Database schema:
{json.dumps(sample["schema"], indent=2)}

----------------------------------------------------------------
Evaluation Tasks & Scoring

You must evaluate these aspects:

1. SQL Behavior
- 0 = Directly generates a full SQL query despite external knowledge requirement (no abstention)
- 1 = Shows partial abstention but still generates SQL (e.g., SELECT NULL WHERE FALSE)
- 2 = Correctly refrains from generating a valid SQL query
      (e.g., empty SQL, or explicit refusal)

2. Detection
- 1 = explicitly states the question is flawed 
- 0 = fails to recognize the issue

3. Type Identification
If Detection = 0, then Type Identification MUST be 0.
Type Identification does NOT require explicit naming of the error type.
If the reasoning states that answering requires information,
facts, or knowledge not contained in the database or schema,
assign Type Identification = 1.
- 1 = the reasoning clearly indicates External Knowledge Required
      (i.e., depends on facts, rules, or world knowledge not present in the schema)
- 0 = reasoning does not support the presence of External Knowledge Required

----------------------------------------------------------------
{OUTPUT_FORMAT}
""".strip()
