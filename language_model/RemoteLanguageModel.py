"""Non-streaming requests with JSON extraction from plain or wrapped responses."""
import json
import re
from .LanguageModel import LanguageModel


class RemoteLanguageModel(LanguageModel):
    schema_repair_attempts = 1

    def _extract_json(self, text: str) -> dict:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
            else:
                return {
                    "sql": None,
                    "reason": None
                }

        try:
            return json.loads(json_str)
        except Exception:
            return {
                "sql": None,
                "reason": None
            }

    @staticmethod
    def _schema_issues(parsed, json_schema):
        required = set(json_schema.get("required", []))
        missing = required - set(parsed)
        issues = (["missing required fields: " + ", ".join(sorted(missing))]
                  if missing else [])
        json_types = {
            "string": str,
            "boolean": bool,
            "integer": int,
            "number": (int, float),
            "object": dict,
            "array": list,
            "null": type(None),
        }
        for field, value in parsed.items():
            expected = json_schema.get("properties", {}).get(field, {}).get("type")
            if not expected:
                continue
            expected = expected if isinstance(expected, list) else [expected]
            allowed = tuple(json_types[name] for name in expected if name in json_types)
            # bool is an int subclass, but JSON Schema keeps them distinct.
            valid = isinstance(value, allowed) if allowed else True
            if "integer" in expected and isinstance(value, bool):
                valid = False
            if not valid:
                issues.append(
                    f"field {field!r} must have type {' or '.join(expected)}"
                )
        return issues

    def chat(self, messages, json_schema=None):
        working_messages = list(messages)
        last_issues = []
        for repair_index in range(self.schema_repair_attempts + 1):
            completion = self._create_completion(working_messages, json_schema)
            content = completion.choices[0].message.content
            if not json_schema:
                return content

            parsed = self._extract_json(content)
            last_issues = self._schema_issues(parsed, json_schema)
            if not last_issues:
                return json.dumps(parsed)
            if repair_index >= self.schema_repair_attempts:
                break
            working_messages.extend([
                {"role": "assistant", "content": content or "{}"},
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON does not satisfy the required response "
                        "schema: " + "; ".join(last_issues) + ". Return one corrected "
                        "JSON object containing every required field with the required "
                        "types. Preserve the intended answer and output JSON only."
                    ),
                },
            ])
        raise ValueError("Structured output schema violation: " + "; ".join(last_issues))
