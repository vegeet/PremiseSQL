"""Streaming adapter retaining reasoning collection and JSON recovery."""
import json
import re
from typing import Dict, List, Tuple
from .LanguageModel import LanguageModel


class StreamLanguageModel(LanguageModel):
    request_defaults = {
        "stream": True,
        "max_tokens": 8192,
        "extra_body": {
            "thinking": {"type": "disabled"},
            "reasoning_effort": "low",
        },
    }

    def chat(
            self,
            messages: List[Dict[str, str]],
            json_schema: dict = None,
            return_reasoning: bool = False
    ) -> str | Tuple[str, str]:
        reasoning_content = ""
        full_content = ""
        chunk_count = 0
        choice_count = 0
        finish_reasons = set()

        completion = self._create_completion(messages, json_schema)

        for chunk in completion:
            chunk_count += 1

            if not chunk.choices:
                continue

            choice_count += len(chunk.choices)
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason:
                finish_reasons.add(str(finish_reason))
            delta = getattr(choice, "delta", None)

            if delta:
                rc = (getattr(delta, "reasoning_content", None)
                      or getattr(delta, "reasoning", None))
                if rc:
                    reasoning_content += self._as_text(rc)

                c = getattr(delta, "content", None)
                if c:
                    full_content += self._as_text(c)

            message = getattr(choice, "message", None)

            if message:
                mc = getattr(message, "content", None)
                if mc:
                    full_content += self._as_text(mc)
                mr = (getattr(message, "reasoning_content", None)
                      or getattr(message, "reasoning", None))
                if mr:
                    reasoning_content += self._as_text(mr)

            choice_text = getattr(choice, "text", None)
            if choice_text:
                full_content += self._as_text(choice_text)

        stream_candidate = full_content or reasoning_content
        try:
            json_str = self._extract_json(stream_candidate)
        except ValueError as stream_error:
            # Some OpenAI-compatible providers occasionally close a stream
            # without content even though an equivalent non-stream request works.
            fallback = self._create_completion(
                messages, json_schema, request_overrides={"stream": False}
            )
            fallback_choices = getattr(fallback, "choices", None) or []
            fallback_message = (
                getattr(fallback_choices[0], "message", None)
                if fallback_choices else None
            )
            fallback_content = self._as_text(
                getattr(fallback_message, "content", None) if fallback_message else None
            )
            fallback_reasoning = self._as_text(
                (getattr(fallback_message, "reasoning_content", None)
                 or getattr(fallback_message, "reasoning", None))
                if fallback_message else None
            )
            fallback_candidate = fallback_content or fallback_reasoning
            try:
                json_str = self._extract_json(fallback_candidate)
                reasoning_content = reasoning_content or fallback_reasoning
            except ValueError as structured_fallback_error:
                # A few compatible gateways return an empty body specifically
                # when response_format=json_schema is present. The prompt itself
                # already requires JSON, so retry once without that transport hint
                # and keep the same semantic prompt and local validation.
                plain = self._create_completion(
                    messages, None, request_overrides={"stream": False}
                )
                plain_choices = getattr(plain, "choices", None) or []
                plain_message = (
                    getattr(plain_choices[0], "message", None)
                    if plain_choices else None
                )
                plain_content = self._as_text(
                    getattr(plain_message, "content", None)
                    if plain_message else None
                )
                plain_reasoning = self._as_text(
                    (getattr(plain_message, "reasoning_content", None)
                     or getattr(plain_message, "reasoning", None))
                    if plain_message else None
                )
                plain_candidate = plain_content or plain_reasoning
                if not plain_candidate:
                    diagnostics = {
                        "stream_chunks": chunk_count,
                        "stream_choices": choice_count,
                        "stream_content_chars": len(full_content),
                        "stream_reasoning_chars": len(reasoning_content),
                        "finish_reasons": sorted(finish_reasons),
                        "structured_non_stream_content_chars": len(fallback_content),
                        "structured_non_stream_reasoning_chars": len(fallback_reasoning),
                        "plain_non_stream_content_chars": 0,
                        "plain_non_stream_reasoning_chars": 0,
                    }
                    raise ValueError(
                        "Empty response after streaming, structured non-stream, "
                        "and plain non-stream attempts: "
                        + json.dumps(diagnostics, ensure_ascii=False)
                    ) from structured_fallback_error
                json_str = self._extract_json(plain_candidate)
                reasoning_content = reasoning_content or plain_reasoning

        if return_reasoning:
            return json_str, reasoning_content
        else:
            return json_str

    @staticmethod
    def _as_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(getattr(item, "text", "") or ""))
            return "".join(parts)
        return str(value)

    def _extract_json(self, text: str) -> str:
        if not text:
            raise ValueError("Empty response")

        # 去 markdown
        text = re.sub(r"```json|```", "", text).strip()

        json_str = self._find_first_json(text)

        if not json_str:
            raise ValueError(f"No JSON found:\n{text}")

        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            json_str = self._repair_json(json_str)
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False)

    def _repair_json(self, text: str) -> str:
        text = text.strip()
        text = text.replace("'", '"')
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)

        return text

    def _find_first_json(self, text: str) -> str | None:
        """
        用栈匹配第一个完整 JSON（避免正则误伤）
        """
        start = text.find("{")
        if start == -1:
            return None

        stack = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                stack += 1
            elif text[i] == "}":
                stack -= 1
                if stack == 0:
                    return text[start:i + 1]

        return None



