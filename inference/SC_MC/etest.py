import json
import os
import argparse
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finmtm_eval.metrics import objective_set_score

# ------------------------------------------
# 抽取文本 + 图片
# ------------------------------------------




def extract_text_and_images(messages):
    contents = messages[0]["content"]
    txt = ""
    images = []

    for part in contents:
        if part["type"] == "text":
            txt += part["text"]
        elif part["type"] == "image_url":
            url = part["image_url"]["url"]
            images.append(url)
    # txt+="\n 注意：不要输出<|begin_of_box|> <|end_of_box|>"
    return txt, images

import re

def strip_box_markers(text: str) -> str:
    if not isinstance(text, str):
        return text
    # 去掉 begin / end 标记
    text = re.sub(r"<\|begin_of_box\|>", "", text)
    text = re.sub(r"<\|end_of_box\|>", "", text)
    return text.strip()


# ------------------------------------------
# ✨ 强化后的 JSON 解析器
# ------------------------------------------
def parse_model_answer(answer_str):
    """返回 ['A'] 或 ['A','C']，解析失败返回 []"""
    # print(answer_str)

    if not answer_str:
        return []

    s = answer_str.strip()

    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]

    s = s.replace("\\\"", "\"")

    try:
        obj = json.loads(s)
        ans = obj.get("answer", None)
    except Exception:
        return []

    # --- 标准化输出 ---
    if isinstance(ans, str):
        return [ans]
    if isinstance(ans, list):
        return ans

    return []


# ------------------------------------------
# Ground Truth
# ------------------------------------------
def get_ground_truth(record):
    try:
        txt = record["choices"][0]["message"]["content"][0]["text"]
        obj = json.loads(txt)
        ans = obj.get("answer", [])

        if isinstance(ans, str):
            return [ans]
        if isinstance(ans, list):
            return ans
        return []
    except Exception:
        return []


# ------------------------------------------
# 论文公式 (1)：禁止过选，否则按正确选项覆盖率给部分分
# ------------------------------------------
def score_answer(pred_list, gt_list):
    return objective_set_score(pred_list, gt_list)


def is_correct(pred_list, gt_list):
    """Backward-compatible exact-correct flag derived from Equation (1)."""

    return score_answer(pred_list, gt_list) == 1.0



# ------------------------------------------
# 主函数
# ------------------------------------------
def evaluate_jsonl_with_accuracy(
    input_jsonl,
    output_jsonl="eval_results.jsonl",
    summary_json="eval_summary.json",
    api_base="http://localhost:8000/v1",
    model="Qwen3-VL-30B-A3B-Instruct",
    image_root=None,
):
    try:
        from .qwen3vl import Qwen3VLClient
    except ImportError:
        from qwen3vl import Qwen3VLClient

    client = Qwen3VLClient(api_base=api_base, model=model)

    total = 0
    correct = 0
    score_sum = 0.0

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(tqdm(fin, desc="Evaluating")):
            total += 1
            record = json.loads(line)

            # -------- 1) 抽取题干 + 图片 --------
            text, images = extract_text_and_images(record["messages"])


            # print(text)

            real_images = []
            for img in images:
                if os.path.exists(img):
                    real_images.append(img)
                elif image_root:
                    path2 = os.path.join(image_root, img)
                    if os.path.exists(path2):
                        real_images.append(path2)
                    else:
                        real_images.append(img)
                else:
                    real_images.append(img)

            # -------- 2) 调模型 --------
            try:
                pred_str = client.chat(
                    image=real_images,
                    text=text,
                    temperature=0.0
                )
            except Exception as e:
                pred_str = f"[ERROR]{e}"

            # -------- 3) 解析预测 --------

            pred_str = pred_str.replace("```json", "").replace("```", "").strip()

            pred_str = strip_box_markers(pred_str)
            print(pred_str)
            pred_ans = parse_model_answer(pred_str)

            # -------- 4) 解析 GT --------
            gt_ans = get_ground_truth(record)
            print(pred_ans,gt_ans)

            # -------- 5) 论文公式 (1) 评分 --------
            item_score = score_answer(pred_ans, gt_ans)
            hit = item_score == 1.0
            score_sum += item_score
            if hit:
                correct += 1

            # -------- 6) 写入输出 --------
            out = {
                "sample_id": line_idx,
                "query": record.get("query", ""),
                "gt_answer": gt_ans,
                "model_answer_raw": pred_str,
                "model_answer": pred_ans,
                "score": item_score,
                "correct": hit
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    # -----------------------------------
    # 统计
    # -----------------------------------
    acc = correct / total if total else 0
    mean_score = score_sum / total if total else 0

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "mean_score": mean_score,
        "score_percent": mean_score * 100.0,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n==== 评测完成 ====")
    print("Total:", total)
    print("Correct:", correct)
    print("Accuracy:", f"{acc:.4f}")
    print("Mean Eq.(1) score:", f"{mean_score:.4f}")
    print("Reported score:", f"{mean_score * 100.0:.2f}")
    print("结果保存到:", output_jsonl)
    print("统计保存到:", summary_json)



# ------------------------------------------
# main
# ------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run objective-question inference and paper-aligned evaluation."
    )
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", default="outputs/objective_eval_results.jsonl")
    parser.add_argument("--summary", default="outputs/objective_eval_summary.json")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional base directory for relative image paths",
    )
    args = parser.parse_args()
    evaluate_jsonl_with_accuracy(
        input_jsonl=args.input,
        output_jsonl=args.output,
        summary_json=args.summary,
        api_base=args.api_base,
        model=args.model,
        image_root=args.image_root,
    )


if __name__ == "__main__":
    main()
