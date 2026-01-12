#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import json
import glob
import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from util import safe_json_parse
from api2 import GEMINIClient

ALLOWED_PLOT_TYPES = {
    "line","bar","histogram","pie","scatter plot","candlestick","mixed","others","timeline"
}
DEFAULT_DIR2PLOT = {
    "line":"line","bar":"bar","hist":"histogram","histogram":"histogram",
    "pie":"pie","scatter":"scatter plot","scatter_plot":"scatter plot",
    "scatter-plot":"scatter plot","candlestick":"candlestick","mixed":"mixed"
}

def _safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-一-龥]+", "_", (s or "unknown")).strip("_") or "unknown"

def clean_text(v: str) -> str:
    return re.sub(r"\s+", " ", str(v or "").strip())

def normalize_plot_type_from_dirname(dirname: str, extra_map: Dict[str,str]) -> str:
    name = dirname.lower().replace("-", "_").replace(" ", "_")
    if name in extra_map:
        pt = extra_map[name].strip().lower()
        return pt if pt in ALLOWED_PLOT_TYPES else "others"
    if name in DEFAULT_DIR2PLOT:
        return DEFAULT_DIR2PLOT[name]
    for k,v in {**DEFAULT_DIR2PLOT, **extra_map}.items():
        if k in name:
            return v if v in ALLOWED_PLOT_TYPES else "others"
    return "others"

# ---------- few-shot（可选，不提供也能跑） ----------
def fewshot_from_json(json_path: str) -> List[Dict[str,Any]]:
    p = Path(json_path)
    if not p.exists(): return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def select_fewshot_for_plot(fewshot_all: List[Dict[str,Any]], plot_type: str, k_local: int, k_global: int) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    local = [x for x in fewshot_all if str(x.get("plot_type","")).lower()==plot_type.lower()]
    global_ = [x for x in fewshot_all if x not in local]
    return local[:k_local], global_[:k_global]

def build_fewshot_block(local_fs: List[Dict[str,Any]], global_fs: List[Dict[str,Any]]) -> str:
    fs = local_fs + global_fs
    # 仅作风格参考，禁止直接抄写内容
    return json.dumps(fs, ensure_ascii=False, indent=2) if fs else "[]"

# ---------- Prompt：要求输出极简多轮会话 JSON ----------
# def build_prompt_multiturn(plot_type: str, fewshot_block: str) -> str:
#     return f"""
# 你是一名金融多轮对话数据集生成专家。请**仅根据给定图片**构造一个“多轮问答 + 上下文记忆/一致性推理”的会话（3~5轮），并**严格输出 JSON 对象**，字段最小化如下：

# {{
#   "session_id": "任意占位",
#   "image_path": "任意占位（将被上游覆盖为真实路径）",
#   "turns": [
#     {{
#       "turn_id": "T1",
#       "question": "中文问题（必须仅凭当前图片可答）",
#       "gold_answer": "中文答案（自然语言）",
#       "vars_out": {{
#         "变量名": {{"value": 数值或字符串, "unit": "可选"}}
#       }}
#     }},
#     {{
#       "turn_id": "T2",
#       "question": "必须在语义上承接T1，并正确引用T1中的变量或结论进行计算/推理",
#       "depends_on": ["T1.变量名", "..."],   // 可选，若有引用请写明
#       "gold_answer": "中文答案",
#       "vars_out": {{
#         "变量名": {{"value": 数值或字符串, "unit": "可选"}}
#       }}
#     }}
#     // 可以继续到 T3, T4, T5（总轮次 3~5）
#   ]
# }}

# **要求与约束**
# - 图表类型（plot_type）="{plot_type}"；问题必须与图像内容/刻度/要素强关联，避免图外知识。
# - 轮次 4；每一轮的提问要在语义上承接前文，考察“记忆与一致性”：引用前轮变量/结论并保持逻辑连贯。
# - 至少一轮包含“轻微干预/口径修正”（例如：上调/下调某起点1%、剔除一次性收益、换一个均线参数），并要求**只影响相关变量**。
# - 数值相关请给到结构化 `vars_out`（最小包含 "value"，"unit" 可选）；变量名尽量与问题语义相符（如 rev_2024Q1, ret_Mar）。
# - 若图片为 K线/指标/双轴/表格等，优先选择：区间涨跌幅、均线相对、同比/环比、比率/比例类的链式问题。
# - 禁止幻觉：答案必须可由图像读数或简单计算得到；引用上一轮必须正确，不可自相矛盾。
# - **仅输出 JSON；不要有多余文字、注释或 Markdown 反引号**。

# 【few-shot 风格示例（仅供结构/语气参考，禁止抄写具体内容）】
# {fewshot_block}
# """.strip()


def build_prompt_multiturn(plot_type: str, fewshot_block: str) -> str:
    return f"""
你是一名**金融多轮对话数据集生成专家**。  
请仅依据给定图像构造一个**固定 4 轮会话**（线索定位 → 引申 → 想象扩展 → 实际观察），并**只输出 1 个合法 JSON 对象**。  
禁止输出任何额外文字、注释、说明或 Markdown 反引号。

---

【对象定义规则】
根据图表类型选择最合适的唯一对象并命名（需可复现）：
- 折线/柱状/蜡烛图：点 A（极值、交叉）、区间 R1（时间段）、区域 Z1（值域/象限）
- 饼图：扇区 S1（最大占比、指定标签）
- 表格：单元格 C1（某行列交叉）、行/列 R1（整体行列）
- 文本：段落 P1 / 句子 S1（包含关键词的段落/句子）

对象命名需唯一且语义明确。  
后续各轮必须**复用相同命名**（隐式记忆），但每轮题面必须**重述其最小定义**，确保独立可判分。  
禁止引用上一轮的数值或结果。

---

【固定四轮结构】

1️⃣ **T1 线索定位**  
   - 在图中唯一确定并命名一个对象（A/R1/Z1/S1/C1/P1）。  
   - 题面须给出“最小定义”（颜色/标签/位置/时间/关键词），保证唯一性。  
   - gold_answer：对象的直接读数或内容（如日期、值、文本）。  
   - vars_out：对象属性（如 A_ts, A_close, S1_ratio, C1_value, P1_text）。

2️⃣ **T2 线索引申**  
   - 基于同一对象（重述最小定义）提出独立可计算的度量或比较问题。  
   - 示例：  
     - A：A→右端区间涨跌幅、A 相对均线位置  
     - R1：区间均值/波动率/占比  
     - Z1：值域比较/样本比例  
     - S1：扇区间对比、累计占比  
     - C1/R1：同比/环比/平均值  
     - P1：提取句意、判断是否含特定信息  
   - gold_answer：可从图像直接推算的结果。  
   - vars_out：指标名称与数值。

3️⃣ **T3 想象扩展（轻微干预）**  
   - 不改变对象定义，执行一次**轻微且可计算的调整**（如上调1%、换均线周期、剔除一次性收益）。  
   - 题面必须说明如何取原值及调整方式。  
   - gold_answer：调整后的新结论。  
   - vars_out：调整后变量及说明（如 ret_adj, adj_note）。

4️⃣ **T4 实际观察（验证/对照）**  
   - 围绕同一对象提出**直接可观测的验证/对照问题**。  
   - 示例：  
     - A：A 后 5 根是否出现更高高点  
     - R1：R1 与相邻区间对比  
     - Z1：Z1 与补集差异  
     - S1：S1 是否超过 50%  
     - C1/R1：该行最大值在哪列、该列是否递增  
     - P1：文中是否包含展望/结论倾向  
   - gold_answer：是/否或具体可观察结论。  
   - vars_out：检验结果字段。

---

【JSON 输出模板】

{{
  
  "image_path": "占位（由上游覆盖）",
  "turns": [
    {{
      "turn_id": "T1",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "obj_type": {{"value": "A|R1|Z1|S1|C1|P1"}},
        "obj_attr": {{"value": "..."}}
      }}
    }},
    {{
      "turn_id": "T2",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "metric_name": {{"value": "..."}},
        "metric_value": {{"value": 0, "unit": ""}}
      }}
    }},
    {{
      "turn_id": "T3",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "metric_name":      {{"value": "..."}},
        "metric_value_adj": {{"value": 0, "unit": ""}},
        "adj_note":         {{"value": "..."}}
      }}
    }},
    {{
      "turn_id": "T4",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "check_name":   {{"value": "..."}},
        "check_result": {{"value": "是/否|数值"}}
      }}
    }}
  ]
}}

---

【要求与约束】

1. **固定轮次**
   - 必须严格包含 4 轮（T1~T4），不可多或少。

2. **上下文一致性**
   - 每轮问题需语义连贯，复用同一对象命名。  
   - 必须重述对象的最小定义，确保独立判分。  
   - 禁止出现“上一轮结果”“你刚才算的”等显式引用。

3. **轻微干预必选**
   - 至少一轮（通常为 T3）包含轻微可计算的修正，仅影响相关变量。  
   - 干预示例：上调/下调1%、更换均线周期、剔除异常点、调整分位区间。  
   - 题面需说明取值与修正方式，不得随意虚构。

4. **图像内容强关联**
   - 图表类型（plot_type）="{plot_type}"。  
   - 问题必须与图中元素、刻度、标签、数值强相关，禁止引入图外知识。  
   - 优先选择以下问题类型：  
     - K线/折线：区间涨跌幅、均线比较、交叉点、极值  
     - 柱状/直方：同比/环比、均值、波动率、占比  
     - 饼图：最大扇区、累计占比、比例关系  
     - 表格：单元格值、行列比较、同比环比  
     - 文本：关键词句含义、同比描述、情感倾向

5. **结构化输出**
   - 每轮必须有 `vars_out`。  
   - `vars_out` 为字典，变量至少包含 `"value"`；`"unit"` 可选。  
   - 变量命名需语义化，反映题面（如 `ret_Mar`, `rev_2024Q1`, `S1_ratio`, `higher_high_within_5d`）。  
   - 数值必须来自图像读数或可验证计算。

6. **一致性与正确性**
   - 对象命名、定义、变量名在 4 轮内保持一致。  
   - 若存在 `depends_on` 字段，必须引用前轮已存在变量。  
   - 不允许自相矛盾的叙述或重复定义。

7. **禁止幻觉**
   - 所有答案必须由图像可见信息或直接计算得到。  
   - 禁止虚构时间、指标、概念、外部知识或推测性内容。

8. **输出格式**
   - 必须输出合法 JSON 对象，字段包括：  
     - `session_id`, `image_path`, `turns`  
     - 每个 turn 含 `turn_id`, `question`, `gold_answer`, `vars_out`（可选 `depends_on`）  
   - JSON 内禁止注释、Markdown、尾逗号、额外文本。

---

few-shot 风格参考（仅供语气/结构参考，禁止抄写具体内容）：
{fewshot_block}
""".strip()

# ---------- 结果校验（保持最小字段，做必要检查） ----------
def validate_multiturn_session(obj: Any) -> Optional[Dict[str,Any]]:
    if not isinstance(obj, dict): return None
    if "turns" not in obj or not isinstance(obj["turns"], list): return None
    turns = obj["turns"]
    if not (3 <= len(turns) <= 5): return None

    seen_ids = set()
    for t in turns:
        if not isinstance(t, dict): return None
        tid = str(t.get("turn_id","")).strip()
        q   = clean_text(t.get("question",""))
        ga  = clean_text(t.get("gold_answer",""))
        vo  = t.get("vars_out", {})
        if not tid or tid in seen_ids: return None
        seen_ids.add(tid)
        if not q or not ga: return None
        if not isinstance(vo, dict): return None
        # 简单校验 vars_out 的最小字段
        for vname, vobj in vo.items():
            if not isinstance(vobj, dict): return None
            if "value" not in vobj: return None
            # unit 可选，不强制

        # depends_on 可选；若存在，需要是 list[str]
        if "depends_on" in t and not isinstance(t["depends_on"], list):
            return None

    return obj

# ---------- 模型调用（await + 重试） ----------
async def call_gemini_multiturn(gpt, image_uri: str, prompt_text: str, trials: int = 1):
    last_raw = None
    for _ in range(max(1, trials)):
        try:
            resp = await gpt.image2text(prompt_text, image=image_uri)
            raw  = (resp if isinstance(resp, str) else str(resp)).strip()
            last_raw = raw
            data = safe_json_parse(raw)
            ok = validate_multiturn_session(data)
            if ok:
                return ok, {"raw": raw}
        except Exception as e:
            print(f"[ERROR] call_gemini_multiturn: {e}")
    return None, {"raw": last_raw}

# ---------- 主流程 ----------
async def main():
    ap = argparse.ArgumentParser(description="L1")
    ap.add_argument("--img_root", required=False,default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans", help="根目录：其子目录名即 plot_type")
    ap.add_argument("--out",      required=False,default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/muti_out/", help="输出 JSONL（每行一条会话）")
    ap.add_argument("--subdirs",  type=str, default="line", help="仅处理这些子目录（逗号分隔）；留空=全部")
    ap.add_argument("--glob",     type=str, default="**/*.*", help="图片通配符（默认递归）")
    ap.add_argument("--limit",    type=int, default=1000, help="每个子目录最多处理多少张（0=不限）")
    ap.add_argument("--trials",   type=int, default=1, help="每张图最多重试几次")
    ap.add_argument("--map",      type=str, default="", help='目录名->plot_type 映射，如 "scatterplot=scatter plot;bars=bar"')
    ap.add_argument("--fewshot_json", type=str, default="", help="few-shot JSON（可选）")
    ap.add_argument("--fewshot_local", type=int, default=2, help="同类型 few-shot 数量")
    ap.add_argument("--fewshot_global", type=int, default=2, help="通用 few-shot 数量")
    ap.add_argument("--dump_raw_fail", action="store_true", help="失败时将原始返回落到 _raw/*.txt")
    args = ap.parse_args()

    root = Path(args.img_root)
    if not root.exists():
        raise FileNotFoundError(f"img_root 不存在：{root}")

    # 解析 --map
    extra_map: Dict[str,str] = {}
    if args.map.strip():
        for p in [p for p in args.map.split(";") if p.strip()]:
            if "=" in p:
                k, v = p.split("=", 1)
                k = k.strip().lower().replace("-", "_").replace(" ", "_")
                v = (v or "").strip().lower()
                extra_map[k] = v

    # few-shot
    fewshot_all: List[Dict[str,Any]] = []
    if args.fewshot_json:
        try:
            fewshot_all.extend(fewshot_from_json(args.fewshot_json))
        except Exception as e:
            print(f"[WARN] few-shot 加载失败：{e}")

    # 子目录筛选
    plot_dirs = [d for d in root.iterdir() if d.is_dir()]
    only_dirs = {d.strip() for d in args.subdirs.split(",") if d.strip()}
    if only_dirs:
        plot_dirs = [d for d in plot_dirs if d.name in only_dirs]

    gpt = GEMINIClient()
    total = 0
    out_path = Path(args.out+'line.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as fout:
        for d in plot_dirs:
            plot_type = normalize_plot_type_from_dirname(d.name, extra_map)
            if plot_type not in ALLOWED_PLOT_TYPES:
                plot_type = "others"

            files = glob.glob(str(d / args.glob), recursive=True)
            exts  = {".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"}
            img_files = [f for f in files if Path(f).suffix.lower() in exts and "checkpoint" not in Path(f).name.lower()]
            if args.limit and args.limit > 0:
                img_files = img_files[:args.limit]
            if not img_files:
                print(f"[WARN] 子目录无图片：{d}")
                continue

            print(f"\n=== 目录：{d} | plot_type={plot_type} | {len(img_files)} 张 ===")

            # few-shot 组装
            local_fs, global_fs = select_fewshot_for_plot(fewshot_all, plot_type, args.fewshot_local, args.fewshot_global)
            fewshot_block = build_fewshot_block(local_fs, global_fs)
            prompt_text = build_prompt_multiturn(plot_type, fewshot_block)

            for fp in img_files:
                total += 1
                image_uri  = fp  # 直接用文件路径（你的 GEMINIClient 已支持）
                if os.path.exists(image_uri):
                  print("")
                else :
                  continue
                sess, detail = await call_gemini_multiturn(gpt, image_uri=image_uri, prompt_text=prompt_text, trials=args.trials)

                if sess:
                    # 强制覆盖最小必要字段
                    # sess["session_id"] = sess.get("session_id") or _safe_name(f"{Path(fp).stem}")
                    sess["image_path"] = fp
                    # 统一清洗 turn 文本
                    for t in sess.get("turns", []):
                        t["turn_id"]    = clean_text(t.get("turn_id",""))
                        t["question"]   = clean_text(t.get("question",""))
                        t["gold_answer"]= clean_text(t.get("gold_answer",""))
                    fout.write(json.dumps(sess, ensure_ascii=False) + "\n")
                    print(f"[OK] {fp} -> appended")
                else:
                    failed = {
                        "image_path": fp,
                        "plot_type": plot_type,
                        "error": "generation_failed_or_malformed",
                        "raw": detail.get("raw")
                    }
                    (Path(args.out).parent / "failed.jsonl").open("a", encoding="utf-8").write(json.dumps(failed, ensure_ascii=False) + "\n")
                    print(f"[WARN] 生成异常：{fp} 已写入 failed.jsonl")

                    if args.dump_raw_fail and detail.get("raw"):
                        raw_dir = Path(args.out).parent / "_raw"
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        (raw_dir / f"{_safe_name(Path(fp).stem)}.txt").write_text(detail["raw"] or "", encoding="utf-8")

    print(f"\n完成：共处理 {total} 张图片；多轮会话已追加到 {args.out}")

if __name__ == "__main__":
    asyncio.run(main())
