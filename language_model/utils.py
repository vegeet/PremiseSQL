from .RemoteLanguageModel import RemoteLanguageModel
from .StreamLanguageModel import StreamLanguageModel

model_name_dict = {
    "Qwen/QwQ3.5-397b-a17b": "qwen3.5-397b-a17b",
    "Qwen/QwQ3.5-122b-a10b": "qwen3.5-122b-a10b",
    "Qwen/QwQ3.5-35b-a3b": "qwen3.5-35b-a3b",

    "DeepSeek/DeepSeek-V3.2": "deepseek-v3.2",
    "DeepSeek/deepseek-v4.1-flash": "deepseek-v4.1-flash",

    "openai/gpt-5.4": "gpt-5.4",
    "openai/gpt-5.6-sol": "gpt-5.6-sol",
    "openai/gpt-5.6-terra": "gpt-5.6-terra",
    "openai/gpt-5.6-luna": "gpt-5.6-luna",
    "openai/gpt-6-luna": "gpt-6-luna",
    "openai/gpt-6.1-sol": "gpt-6.1-sol",

    "gemini/gemini-3.1-pro-preview": "gemini-3.1-pro-preview",

    "claude/claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude/claude-opus-4-6": "claude-opus-4-6",
    "claude/claude-opus-5-5": "claude-opus-5-5",

    "meta-llama/llama-4-maverick": "meta-llama/llama-4-maverick",

    "ChatGLM/glm-5.1": "glm-5.1",
    "Doubao/doubao-seed-2-0-pro-260215": "doubao-seed-2-0-pro-260215",

}


def load_language_model(model_name: str, **kwargs):
    if model_name in model_name_dict.keys():
        if "QwQ" in model_name or "gemini" in model_name.lower() or "seed" in model_name.lower() or "glm" in model_name.lower():
            return StreamLanguageModel(model_name_dict[model_name], **kwargs)
        else:
            # Provider options formerly used by the separate robust adapter.
            kwargs.setdefault("timeout", 90)
            kwargs.setdefault("strict_schema", False)
            kwargs.setdefault("request_defaults", {
                "max_tokens": 4096,
                "extra_body": {"thinking": {"type": "disabled"}},
            })
            return RemoteLanguageModel(model_name_dict[model_name], **kwargs)

