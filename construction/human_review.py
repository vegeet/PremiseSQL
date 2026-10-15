import json
import os
import argparse
import tkinter as tk
from datetime import datetime

# =========================
# 配置
# =========================
SOURCE_JSON = None
FINAL_DIR = None
FINAL_JSON = None
REVIEWER_NAME = "human"


# =========================
# GUI
# =========================
class ReviewGUI:
    def __init__(self, root, source_samples, reviewed_samples):
        self.root = root
        self.source_samples = source_samples
        self.reviewed = {s["sample_id"]: s for s in reviewed_samples}
        self.queue = self.build_queue()
        self.index = 0

        root.title("人工审查工具（Final 输出）")
        root.geometry("1000x820")

        # ===== 主文本区 =====
        self.text = tk.Text(
            root,
            wrap=tk.WORD,
            font=("Consolas", 11),
            padx=10,
            pady=10
        )
        self.text.pack(expand=True, fill=tk.BOTH)

        # ===== 人工改写输入区（可选）=====
        rewrite_frame = tk.Frame(root)
        rewrite_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            rewrite_frame,
            text="【人工改写 Rewritten Question（可选，不填则不生效）】",
            anchor="w"
        ).pack(fill=tk.X)

        self.rewrite_entry = tk.Entry(
            rewrite_frame,
            font=("Consolas", 11)
        )
        self.rewrite_entry.pack(fill=tk.X)

        # ===== 状态栏 =====
        self.status = tk.Label(root, anchor="w")
        self.status.pack(fill=tk.X)

        self.bind_keys()
        self.render()

    # =========================
    # 数据逻辑
    # =========================
    def build_queue(self):
        return [
            s for s in self.source_samples
            if s["sample_id"] not in self.reviewed
        ]

    def save_final(self):
        os.makedirs(FINAL_DIR, exist_ok=True)
        with open(FINAL_JSON, "w", encoding="utf-8") as f:
            json.dump(
                list(self.reviewed.values()),
                f,
                indent=2,
                ensure_ascii=False
            )

    # =========================
    # 键盘绑定
    # =========================
    def bind_keys(self):
        self.text.focus_set()
        self.text.bind("<space>", self._pass)
        self.text.bind("n", self._fail)
        self.text.bind("g", self._giveup)
        self.text.bind("q", self._quit)

    def _pass(self, event):
        self.mark(True)
        return "break"

    def _fail(self, event):
        self.mark(False)
        return "break"

    def _giveup(self, event):
        self.give_up()
        return "break"

    def _quit(self, event):
        self.quit()
        return "break"

    # =========================
    # 渲染
    # =========================
    def render(self):
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.rewrite_entry.delete(0, tk.END)

        if self.index >= len(self.queue):
            self.text.insert(tk.END, "🎉 所有样本已完成审查")
            self.text.config(state=tk.DISABLED)
            self.status.config(text="")
            return

        s = self.queue[self.index]

        content = f"""
【进度】 {self.index + 1} / {len(self.queue)}

Sample ID: {s.get("sample_id")}
DB: {s.get("db_id")}
错误类型: {s.get("error_category")} / {s.get("error_subcategory")}
子类: {s.get("assigned_subcategory")}

────────────────────────────
【Question】
{s.get("question")}

【Rewritten Question】
{s.get("rewritten_question")}

【SQL】
{s.get("sql")}

────────────────────────────
【Schema】
{self.format_schema(s.get("schema", {}))}

【Foreign Keys】
{self.format_fks(s.get("foreign_keys", []))}

────────────────────────────
【最强失败理由】
{s.get("review", {}).get("strongest_reason_to_fail")}

【最强通过理由】
{s.get("review", {}).get("strongest_reason_to_pass")}
        """.strip()

        self.text.insert(tk.END, content)
        self.text.config(state=tk.DISABLED)

        self.status.config(
            text="[Space] 通过   [N] 失败   [G] 放弃   [Q] 退出并保存"
        )

    # =========================
    # 格式化工具
    # =========================
    def format_schema(self, schema):
        lines = []
        for table, cols in schema.items():
            lines.append(f"- {table}")
            for c in cols:
                lines.append(f"    • {c}")
        return "\n".join(lines)

    def format_fks(self, fks):
        if not fks:
            return "(None)"
        return "\n".join(f"- {fk['from']} → {fk['to']}" for fk in fks)

    # =========================
    # 人工改写（只覆盖，不留痕）
    # =========================
    def apply_human_rewrite_if_any(self, s):
        text = self.rewrite_entry.get().strip()
        if text:
            s["rewritten_question"] = text

    # =========================
    # 审查动作
    # =========================
    def mark(self, passed: bool):
        s = self.queue[self.index]

        self.apply_human_rewrite_if_any(s)

        s["human_review"] = {
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
            "reviewer": REVIEWER_NAME
        }

        self.reviewed[s["sample_id"]] = s
        self.index += 1
        self.save_final()
        self.render()

    def give_up(self):
        s = self.queue[self.index]

        self.apply_human_rewrite_if_any(s)

        s["human_review"] = {
            "passed": False,
            "give_up": True,
            "timestamp": datetime.now().isoformat(),
            "reviewer": REVIEWER_NAME
        }

        self.reviewed[s["sample_id"]] = s
        self.index += 1
        self.save_final()
        self.render()

    def quit(self):
        self.save_final()
        self.root.destroy()


# =========================
# 主入口
# =========================
def main():
    global SOURCE_JSON, FINAL_DIR, FINAL_JSON, REVIEWER_NAME
    parser = argparse.ArgumentParser(description="Human review of accepted premise-error questions")
    parser.add_argument("input", help="Accepted questions JSON")
    parser.add_argument("output", help="Review file; resumed when present")
    parser.add_argument("--reviewer", default="human")
    args = parser.parse_args()
    SOURCE_JSON = args.input
    FINAL_JSON = args.output
    FINAL_DIR = os.path.dirname(os.path.abspath(FINAL_JSON))
    REVIEWER_NAME = args.reviewer
    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        source_samples = json.load(f)

    if os.path.exists(FINAL_JSON):
        with open(FINAL_JSON, "r", encoding="utf-8") as f:
            reviewed_samples = json.load(f)
    else:
        reviewed_samples = []

    root = tk.Tk()
    ReviewGUI(root, source_samples, reviewed_samples)
    root.mainloop()


if __name__ == "__main__":
    main()
