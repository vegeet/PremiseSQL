"""Shared language-model client, request construction, and generation helpers."""
import os
from abc import abstractmethod
from typing import Dict, List, Union
from openai import OpenAI


class LanguageModel:
    default_timeout = 60
    request_defaults = {}
    strict_schema = True

    def __init__(self, model_name, **kwargs):
        self.max_new_tokens = kwargs.get("max_new_tokens", 512)
        self.system_prompt = kwargs.get("system_prompt", "")
        self.request_defaults = kwargs.get("request_defaults", self.request_defaults)
        self.strict_schema = kwargs.get("strict_schema", self.strict_schema)
        self.model_name = model_name
        self.temperature = kwargs.get("temperature", 0.0)
        self.timeout = kwargs.get("timeout", self.default_timeout)
        # Keep existing provider timeout/retry behavior during this refactor.
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE_URL"),
        )

    def _create_completion(self, messages, json_schema=None, request_overrides=None):
        request = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            **self.request_defaults,
        }
        if request_overrides:
            request.update(request_overrides)
        if json_schema:
            schema = {"name": "structured_output", "schema": json_schema}
            if self.strict_schema:
                schema["strict"] = True
            request["response_format"] = {"type": "json_schema", "json_schema": schema}
        return self.client.chat.completions.create(**request)

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError()

    @abstractmethod
    def chat_batch(self, messages: List[List[Dict[str, str]]]) -> List[str]:
        raise NotImplementedError()

    def generate(self, prompt: Union[str | List[str]], system_prompt: str = None) -> str:
        system_prompt = system_prompt or self.system_prompt

        if isinstance(prompt, list):
            messages = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": single_prompt}
                ]
                for single_prompt in prompt
            ]
            return self.chat_batch(messages)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            return self.chat(messages)

    def generate_sql(self, prompt):
        response: str = self.generate(prompt)

        # fetch code block
        if "```" in response:
            sql = response.split("```")[1]
        else:
            sql = response

        if 'sql\n' in sql[:4]:
            sql = sql[4:]

        sql = sql.strip()

        if sql == "":
            sql = ";"

        return sql
