import json


def is_passed(overall_pass):
    return overall_pass is True or overall_pass in ("TRUE", "True")


# 递归展开 JSON
def flatten_samples(data):
    flat = []

    if isinstance(data, dict):
        flat.append(data)

    elif isinstance(data, list):
        for item in data:
            flat.extend(flatten_samples(item))

    return flat


def extract_overall_pass(review):

    # review 是 dict
    if isinstance(review, dict):
        return review.get("overall_pass")

    # review 是 list
    if isinstance(review, list):

        for r in review:
            if isinstance(r, dict) and "overall_pass" in r:
                return r["overall_pass"]

    return None


def keep_true(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_data = flatten_samples(data)

    print("原始样本数:", len(flat_data))

    passed_samples = []
    skipped = 0

    for sample in flat_data:

        if not isinstance(sample, dict):
            skipped += 1
            continue

        review = sample.get("review")

        overall_pass = extract_overall_pass(review)

        if is_passed(overall_pass):
            passed_samples.append(sample)
        else:
            skipped += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(passed_samples, f, indent=2, ensure_ascii=False)

    print("===== 过滤完成 =====")
    print("保留样本数:", len(passed_samples))
    print("剔除样本数:", skipped)
    print("输出文件:", output_path)


if __name__ == "__main__":

    input_path = "reviewed_1_1_gemini.json"
    output_path = "re_1_1_gemini.json"

    keep_true(input_path, output_path)