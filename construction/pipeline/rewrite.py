import json
import os
from typing import Dict, Any, Optional
from tqdm import tqdm
import hashlib
import openai
from multiprocessing import Process, Queue
from construction.pipeline.build_prompt import *

# =========================
# API 配置
# =========================

MODEL_NAME = os.environ.get("REWRITE_MODEL", "gpt-5.1")


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before rewriting questions")
    return openai.OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_API_BASE_URL"))

# =========================
# 输出 JSON Schema
# =========================

REWRITE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_question": {"type": ["string", "null"]},
        "error_category": {"type": ["string", "null"]},
        "error_subcategory": {"type": ["string", "null"]},
        "give_up": {"type": "boolean"},
        "give_up_reason": {"type": ["string", "null"]}
    },
    "required": ["give_up"],
    "additionalProperties": False
}

# =========================
# Prompt 注册表
# =========================

PROMPT_REGISTRY = {
    "1_1": build_prompt_1_1,
    "1_2": build_prompt_1_2,
    "2_1": build_prompt_2_1,
    "2_2": build_prompt_2_2,
    "3_1": build_prompt_3_1,
    "3_2": build_prompt_3_2,
    "3_3": build_prompt_3_3,
    "4_1": build_prompt_4_1,
    "4_2": build_prompt_4_2,
}


# =========================
# 工具函数
# =========================
def llm_worker(prompt: str, q: Queue):
    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=False
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response")

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found")

        q.put(json.loads(content[start:end + 1]))

    except Exception as e:
        q.put({
            "give_up": True,
            "give_up_reason": f"LLM error: {e}",
            "rewritten_question": None,
            "error_category": None,
            "error_subcategory": None
        })


def llm_call_with_timeout(prompt: str, timeout_sec: int = 20):
    q = Queue()
    p = Process(target=llm_worker, args=(prompt, q))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        result = {
            "give_up": True,
            "give_up_reason": f"Timeout > {timeout_sec}s",
            "rewritten_question": None,
            "error_category": None,
            "error_subcategory": None
        }
        print("\n[LLM RESULT - TIMEOUT]")
        print(result)
        return result

    if q.empty():
        result = {
            "give_up": True,
            "give_up_reason": "No output from LLM",
            "rewritten_question": None,
            "error_category": None,
            "error_subcategory": None
        }
        print("\n[LLM RESULT - EMPTY]")
        print(result)
        return result

    result = q.get()
    print("\n[LLM RESULT - OK]")
    print(result)
    return result


def generate_schema_name(schema: Dict[str, Any]) -> str:
    return "schema_" + hashlib.md5(
        json.dumps(schema, sort_keys=True).encode()
    ).hexdigest()[:8]


def llm_call(
        prompt: str,
        json_schema: Dict[str, Any],
        temperature: float = 0.2,
        stream_print: bool = True
) -> Optional[Dict[str, Any]]:
    stream = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": generate_schema_name(json_schema),
                "schema": json_schema
            }
        },
        temperature=temperature,
        stream=True
    )

    full_text = ""

    for chunk in stream:
        if not hasattr(chunk, "choices") or not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            if stream_print:
                print(delta.content, end="", flush=True)
            full_text += delta.content

    if stream_print:
        print()

    try:
        return json.loads(full_text)
    except json.JSONDecodeError:
        print("\n[ERROR] JSON parse failed. Raw model output:")
        print(full_text)
        return None


def sub_id_to_key(sub_id: str) -> str:
    return sub_id.replace(".", "_")  # 3.3 -> 3_3


def build_prompt(sample, error_category, subcat, sub_id: str):
    key = sub_id_to_key(sub_id)

    if key not in PROMPT_REGISTRY:
        raise ValueError(f"No build_prompt function registered for sub_id={sub_id}")

    return PROMPT_REGISTRY[key](sample, error_category, subcat)


# =========================
# 主流程
# =========================

def rewrite_dataset(
        samples_path: str,
        taxonomy_path: str,
        output_path: str,
        error_category: str,
        error_sub_id: str,
        target_subcategory: str = None,  # 可指定只改写该子类别
        save_every: int = 50  # 每 N 条样本保存一次
):
    # 读取样本
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # 读取 taxonomy
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)

    # 指定子类别对象
    subcat = taxonomy[error_category]["subcategories"][error_sub_id]

    # 筛选指定子类别
    if target_subcategory is not None:
        filtered_samples = [
            s for s in samples
            if s.get("assigned_subcategory") == target_subcategory
        ]
        print(f"已筛选 {len(filtered_samples)} 条样本用于子类别 {target_subcategory}")
    else:
        filtered_samples = samples

    rewritten_samples = []
    success_cnt = 0
    give_up_cnt = 0

    for idx, sample in enumerate(tqdm(filtered_samples, desc="Rewriting questions")):
        try:
            prompt = build_prompt(sample, error_category, subcat, error_sub_id)
            rewritten = llm_call_with_timeout(prompt, timeout_sec=30)

            if rewritten is None:
                rewritten = {"give_up": True, "give_up_reason": "LLM 输出解析失败"}

            # 在原样本基础上增加/覆盖改写相关字段
            new_sample = sample.copy()
            if rewritten.get("give_up", False):
                give_up_cnt += 1
                new_sample.update({
                    "give_up": True,
                    "give_up_reason": rewritten.get("give_up_reason"),
                    "rewritten_question": None,
                    "error_category": None,
                    "error_subcategory": None
                })
            else:
                success_cnt += 1
                new_sample.update({
                    "give_up": False,
                    "give_up_reason": None,
                    "rewritten_question": rewritten.get("rewritten_question"),
                    "error_category": rewritten.get("error_category"),
                    "error_subcategory": rewritten.get("error_subcategory")
                })

            rewritten_samples.append(new_sample)

        except Exception as e:
            print(f"[ERROR] Sample 出错: {e}\nQuestion: {sample.get('question', '')}")
            new_sample = sample.copy()
            new_sample.update({
                "give_up": True,
                "give_up_reason": str(e),
                "rewritten_question": None,
                "error_category": None,
                "error_subcategory": None
            })
            rewritten_samples.append(new_sample)
            give_up_cnt += 1

        # 定期保存中间结果
        if (idx + 1) % save_every == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rewritten_samples, f, indent=2, ensure_ascii=False)
            print(f"[Checkpoint] 已处理 {idx + 1} 条样本，已保存中间结果。")

    # 保存最终结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rewritten_samples, f, indent=2, ensure_ascii=False)

    print(f"[Summary] success: {success_cnt}, give_up: {give_up_cnt}")


