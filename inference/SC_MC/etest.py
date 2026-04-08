import json
import os
from tqdm import tqdm
from qwen3vl import Qwen3VLClient


# ------------------------------------------
# 抽取文本 + 图片
# ------------------------------------------




def extract_text_and_images(messages):
    if not messages or not messages[0].get("content"):
        return "", []
    contents = messages[0]["content"]
    txt = ""
    images = []

    for part in contents:
        if part["type"] == "text":
            txt += part["text"]
        elif part["type"] == "image_url":
            url = part["image_url"]["url"]
            images.append(url)
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
    if not answer_str:
        return []

    s = answer_str.strip()

    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]

    s = s.replace("\\\"", "\"")

    # --- 尝试解析 JSON 格式 {"answer": "A"} 或 {"answer": ["A","C"]} ---
    try:
        obj = json.loads(s)
        ans = obj.get("answer", None)
        if isinstance(ans, str):
            return [ans]
        if isinstance(ans, list):
            return ans
    except Exception:
        pass

    # --- 回退：裸字母或逗号分隔列表，如 "A" / "A,B,C" / "ABC" ---
    # 去掉首尾空白，优先按逗号分隔，否则按字符逐个分割（过滤非字母）
    fallback = s.strip().strip("'\"").strip()
    if not fallback:
        return []
    if "," in fallback:
        parts = [p.strip().strip("'\"") for p in fallback.split(",")]
        return [p for p in parts if p]
    # 单个字母或连续字母视为选项列表
    letters = re.findall(r"[A-Za-z]", fallback)
    return letters if letters else []


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
# 判定是否正确
# ------------------------------------------
def is_correct(pred_list, gt_list):
    return set(pred_list) == set(gt_list)



# ------------------------------------------
# 主函数
# ------------------------------------------
def evaluate_jsonl_with_accuracy(
    input_jsonl,
    output_jsonl="eval_results.jsonl",
    summary_json="eval_summary.json",
    api_base="http://localhost:8000/v1",
    model="Qwen3-VL-30B-A3B-Instruct"
):

    client = Qwen3VLClient(api_base=api_base, model=model)

    total = 0
    correct = 0

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

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
                else:
                    # 拼接你的默认图片目录
                    path2 = f"/mnt/HithinkOmni/user_workspace/zhangchenxi4/MMfin/gemi/chinese/output/images/{img}"
                    if os.path.exists(path2):
                        real_images.append(path2)
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

            # -------- 5) 判对 --------
            hit = is_correct(pred_ans, gt_ans)
            if hit:
                correct += 1

            # -------- 6) 写入输出 --------
            out = {
                "sample_id": line_idx,
                "query": record.get("query", ""),
                "gt_answer": gt_ans,
                "model_answer_raw": pred_str,
                "model_answer": pred_ans,
                "correct": hit
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    # -----------------------------------
    # 统计
    # -----------------------------------
    acc = correct / total if total else 0

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": acc
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n==== 评测完成 ====")
    print("Total:", total)
    print("Correct:", correct)
    print("Accuracy:", f"{acc:.4f}")
    print("结果保存到:", output_jsonl)
    print("统计保存到:", summary_json)



# ------------------------------------------
# main
# ------------------------------------------
if __name__ == "__main__":
    input_jsonl = "/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/OQ/output.jsonl"

    evaluate_jsonl_with_accuracy(
        input_jsonl=input_jsonl,
        output_jsonl="/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/OQ/Qwen3/choice/eval_results_s.jsonl",
        summary_json="/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/OQ/Qwen3/choice/eval_summary_s.json",
        model="Qwen3-VL-30B-A3B-Instruct"
    )
