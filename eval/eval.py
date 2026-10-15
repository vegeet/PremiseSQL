import json
import language_model
from language_model.StreamLanguageModel import StreamLanguageModel
from .build_prompt import build_original_context_note


class Evaluator:
    def __init__(self, args, prompt_builder, **kwargs):
        self.model_name = args.language_model
        self.client = language_model.load_language_model(self.model_name, **kwargs)
        self.prompt_builder = prompt_builder

    def __call__(self, sample, model_output):
        try:
            return self.evaluate(sample, model_output)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("Error:", e)
            return self.empty_result()

    def evaluate(self, sample, model_output):
        prompt = self.prompt_builder(sample, model_output) + "\n\n" + build_original_context_note(sample)
        is_stream = isinstance(self.client, StreamLanguageModel)

        # -------------------------
        # 评估专用 JSON schema
        # -------------------------
        json_schema = {
            "type": "object",
            "properties": {
                "sql_behavior": {"type": "integer"},
                "detection": {"type": "integer"},
                "type_identification": {"type": "integer"}
            },
            "required": ["sql_behavior", "detection", "type_identification"],
            "additionalProperties": False
        }

        # -------------------------
        # 调模型
        # -------------------------
        if is_stream:
            response = self.client.chat(
                [{"role": "user", "content": prompt}]
            )
        else:
            response = self.client.chat(
                [{"role": "user", "content": prompt}],
                json_schema=json_schema
            )

        print(response)

        # -------------------------
        # 解析 JSON
        # -------------------------
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return self.empty_result()

        if not isinstance(response, dict):
            return self.empty_result()

        required_fields = {"sql_behavior", "detection", "type_identification"}
        if not required_fields.issubset(response):
            return self.empty_result()

        sql_behavior = response.get("sql_behavior", 0)
        detection = response.get("detection", 0)
        type_identification = response.get("type_identification", 0)

        if detection == 0:
            type_identification = 0

        return {
            "sql_behavior": sql_behavior,
            "detection": detection,
            "type_identification": type_identification,
        }

    def empty_result(self):
        return {
            "sql_behavior": -1,
            "detection": -1,
            "type_identification": -1,
        }
