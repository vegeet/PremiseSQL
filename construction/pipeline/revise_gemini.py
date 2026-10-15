import json
from typing import Dict, Any
from tqdm import tqdm
import openai
import os
from multiprocessing import Process, Queue
from construction.pipeline.build_revise_prompt import *

# =========================
# API 配置
# =========================
MODEL_NAME = os.environ.get("REVIEW_MODEL", "gemini-2.5-pro-thinking")


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before reviewing questions")
    return openai.OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_API_BASE_URL"))

# =========================
# 审题输出 Schema
# =========================
REVIEW_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "potential_issues": {
            "type": "string",
            "description": "List any potential problems, weaknesses, or reasons this sample might FAIL."
        },
        "strongest_reason_to_fail": {
            "type": "string",
            "description": "The strongest single reason to reject this sample."
        },
        "strongest_reason_to_pass": {
            "type": "string",
            "description": "The strongest single reason to accept this sample."
        },

        "correct_error_type": {
            "type": ["boolean", "null"],
            "description": "Does the rewritten question exhibit the intended error type?"
        },

        "single_error_only": {
            "type": ["boolean", "null"],
            "description": "Does the question contain exactly one error and no additional issues?"
        },

        "error_matches_subcategory": {
            "type": ["boolean", "null"],
            "description": "Does the error precisely match the assigned subcategory definition?"
        },

        "no_extra_tables_or_columns": {
            "type": ["boolean", "null"],
            "description": "Does the question avoid inventing tables, columns, or schema elements?"
        },

        "issue_clarity": {
            "type": ["boolean", "null"],
            "description": "Is the introduced issue (ambiguity, unclear intent, contradiction, etc.) clearly perceivable?"
        },

        "sql_structure_uncertainty": {
            "type": ["boolean", "null"],
            "description": "Does the issue make the overall SQL structure indeterminate (e.g., SELECT vs GROUP BY vs ORDER BY)?"
        },

        "overall_pass": {
            "type": ["boolean"]
        },

        "comments": {
            "type": ["string", "null"]
        }
    },
    "required": [
        "potential_issues",
        "strongest_reason_to_fail",
        "strongest_reason_to_pass",
        "overall_pass"
    ],
    "additionalProperties": False
}

PROMPT_REGISTRY = {
    "1_1": build_review_prompt_1_1,
    "1_2": build_review_prompt_1_2,
    "2_1": build_review_prompt_2_1,
    "2_2": build_review_prompt_2_2,
    "3_1": build_review_prompt_3_1,
    "3_2": build_review_prompt_3_2,
    "3_3": build_review_prompt_3_3,
    "4_1": build_review_prompt_4_1,
    "4_2": build_review_prompt_4_2,
}


# =========================
# LLM 调用函数（带超时）
# =========================
def llm_worker(prompt: str, q: Queue):
    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
            stream=False
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response")

        # 尝试解析
        q.put(json.loads(content))
    except Exception as e:
        q.put({
            "overall_pass": False,
            "comments": f"API Error: {str(e)}"
        })


def llm_call_with_timeout(prompt: str, timeout_sec: int = 20) -> Dict[str, Any]:
    q = Queue()
    p = Process(target=llm_worker, args=(prompt, q))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "overall_pass": False,
            "comments": f"Timeout > {timeout_sec}s"
        }

    if q.empty():
        return {
            "overall_pass": False,
            "comments": "No output from LLM"
        }

    return q.get()


# =========================
# Prompt 构造函数（包含 schema）
# =========================
def sub_id_to_key(sub_id: str) -> str:
    return sub_id.replace(".", "_")  # 3.3 -> 3_3


def build_review_prompt(sample: Dict[str, Any], sub_id: str) -> str:
    key = sub_id_to_key(sub_id)

    if key not in PROMPT_REGISTRY:
        raise ValueError(f"No review prompt registered for sub_id={sub_id}")

    return PROMPT_REGISTRY[key](sample)


# =========================
# 主流程
# =========================
def review_dataset_gemini(
        input_path: str,
        output_path: str,
        sub_id: str,
        save_every: int = 50
):
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    reviewed_samples = []

    for idx, sample in enumerate(tqdm(samples, desc="Reviewing samples")):
        if sample.get("give_up", False):
            review_result = {
                "overall_pass": False,
                "comments": f"give_up=True, reason: {sample.get('give_up_reason')}"
            }
        else:
            prompt = build_review_prompt(sample, sub_id)
            review_result = llm_call_with_timeout(prompt, timeout_sec=30)

        new_sample = sample.copy()
        new_sample["review"] = review_result
        reviewed_samples.append(new_sample)

        # ====== 中间保存 ======
        if (idx + 1) % save_every == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(reviewed_samples, f, indent=2, ensure_ascii=False)
            print(f"[Checkpoint] 已处理 {idx + 1} 条样本，中间结果已保存。")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reviewed_samples, f, indent=2, ensure_ascii=False)

    print(f"[Finished] Reviewed {len(reviewed_samples)} samples. 输出保存到 {output_path}")


