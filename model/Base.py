"""NL2SQL prompting, structured response validation, and inference."""
import json
import logging

import language_model

logger = logging.getLogger(__name__)


class Base:
    def __init__(self, args, **kwargs):
        self.mode = args.mode
        self.model_name = args.language_model
        self.client = language_model.load_language_model(self.model_name, **kwargs)

    def __call__(self, schema: str, question: str, evidence: str = None):
        return self.inference(schema, question, evidence)

    def _response_schema(self):
        properties = {"sql": {"type": ["string", "null"]}}
        if self.mode != "base":
            properties["is_valid"] = {"type": "boolean"}
        properties["reason"] = {"type": "string", "minLength": 1}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def _empty_result(self):
        result = {"sql": None, "reason": None}
        if self.mode != "base":
            result["is_valid"] = False
        return result

    @staticmethod
    def _is_schema_placeholder(value):
        if isinstance(value, str):
            try:
                value = json.loads(value.replace("'", '"'))
            except (json.JSONDecodeError, TypeError):
                return False
        return isinstance(value, dict) and "type" in value

    def _parse_response(self, response):
        if isinstance(response, str):
            response = json.loads(response)
        if not isinstance(response, dict):
            raise ValueError("Structured output must be a JSON object")

        sql = response.get("sql")
        reason = response.get("reason")
        if self._is_schema_placeholder(sql) or self._is_schema_placeholder(reason):
            raise ValueError("Model returned a JSON-Schema placeholder instead of field content")
        if sql is not None and not isinstance(sql, str):
            raise ValueError("Structured output field 'sql' must be string or null")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Structured output field 'reason' must be a non-empty string")

        result = {"sql": sql.strip() or None if isinstance(sql, str) else None,
                  "reason": reason}
        if self.mode != "base":
            is_valid = response.get("is_valid")
            if not isinstance(is_valid, bool):
                raise ValueError("Structured output field 'is_valid' must be boolean")
            result["is_valid"] = is_valid
        return result

    def inference(self, schema: str, question: str, evidence: str = None):
        try:
            prompt = self.get_prompt(schema, question, evidence)
            response = self.client.chat(
                [{"role": "user", "content": prompt}],
                json_schema=self._response_schema(),
            )
            logger.debug("Model response: %s", response)
            return self._parse_response(response)
        except Exception as exc:
            # KeyboardInterrupt inherits BaseException and propagates naturally.
            logger.error("Inference failed: %s", exc)
            return self._empty_result()

    def get_prompt(self, schema: str, question: str, evidence: str = None) -> str:

        schema_str = "\n".join(schema) if isinstance(schema, list) else schema

        prompt = (
            f"The database schema is as follows:\n{schema_str}\n\n"
            f"Question: {question}\n\n"
        )

        if evidence:
            prompt += f"Extra knowledge: {evidence}\n\n"

        if self.mode == "base":
            prompt += (
                "Generate SQL to answer the question.\n"
                "Return structured JSON only.\n"
                "{\n"
                '  "sql": string or null,\n'
                '  "reason": string\n'
                "}\n"
                "Do NOT include extra explanations.\n"
                "The fields `reason` must always be provided.\n"
                "`reason` must be a non-empty explanation.\n"
            )

            if "glm" in self.model_name.lower():
                prompt += (
                    "Please output only the final answer.\n"
                    "Do not explain. Do not repeat. "
                    "Stop immediately after the output.\n"
                )

        if self.mode == "implicit":
            prompt += (
                "Generate SQL to answer the question.\n"
                "The question may contain errors. "
                "You must assess its validity before answering.\n"
                "Return structured JSON only.\n"
                "The JSON must include: sql, is_valid, reason.\n"
                "Set is_valid to true only if the question is logically valid and answerable; "
                "otherwise false.\n"
                "Do NOT include extra explanations.\n"
                "The fields `is_valid` and `reason` must always be provided.\n"
                "`reason` must be a non-empty explanation.\n"
            )

            if "glm" in self.model_name.lower():
                prompt += (
                    "Please output only the final answer. "
                    "Do not explain. Do not repeat. "
                    "Stop immediately after the output.\n"
                )

        if self.mode == "explicit":
            prompt += (
                "The question is logically flawed and contains incorrect premises.\n"
                "You must critically examine its assumptions and explain the problem.\n"
                "Return ONLY one valid JSON object.\n"
                "The JSON must include exactly these fields: sql, is_valid, reason.\n"
                "Set is_valid to true only if the question is logically valid and answerable; otherwise false.\n"
                "If is_valid is false, briefly explain the specific premise flaw in reason.\n"
                "If is_valid is true, briefly explain why it is valid in reason.\n"
                "The reason field must be no more than 2 sentences.\n"
                "Do NOT provide chain-of-thought, intermediate reasoning, self-debate, or repeated reconsideration.\n"
                "Do NOT include markdown, explanations, or any text outside the JSON object.\n"
                "Output only the final JSON object and stop immediately after it.\n"
            )

        return prompt
