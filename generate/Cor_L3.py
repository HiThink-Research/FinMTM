#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多轮对话金融 Benchmark 生成器（极简字段）
- 与你现有的单轮脚本风格一致：async/await、GEMINIClient、safe_json_parse
- 每张图生成 1 条多轮会话（3~5轮），字段最小化：
  {
    "session_id": "...",
    "image_path": "...",
    "turns": [
      {
        "turn_id": "T1",
        "question": "...",
        "gold_answer": "...",
        "vars_out": { "var_name": {"value": <number or string>, "unit": "可选"} }
      },
      {
        "turn_id": "T2",
        "question": "...",
        "depends_on": ["T1.rev_2024Q1", "..."],   # 可选
        "gold_answer": "...",
        "vars_out": {...}
      }
    ]
  }

用法：
  python generate_multiturn_finbench.py \
      --img_root /path/to/images_root \
      --out fin_multiturn.jsonl \
      --subdirs candlestick,line,bar \
      --limit 50 \
      --trials 2 \
      --fewshot_json /path/to/fewshot_multiturn.json
"""

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
strs={
  "session_id": "dialogue_behavior_001",
  "image_path": "bar/company_profit_2024.png",
  "turns": [
    {
      "turn_id": "T1",
      "question": "在图中，蓝色柱体代表公司季度净利润。请读取2024年第一季度的净利润数值。",
      "gold_answer": "2024年第一季度净利润约为1.2亿元人民币。",
      "vars_out": {
        "np_2024Q1": {"value": 1.2e8, "unit": "CNY"}
      }
    },
    {
      "turn_id": "T2",
      "question": "根据同一系列柱体，2024年第一季度的净利润相较于2023年第四季度是上升还是下降？",
      "gold_answer": "净利润环比上升约10%。",
      "vars_out": {
        "growth_qoq": {"value": 0.10, "unit": ""}
      }
    },
    {
      "turn_id": "T3",
      "question": "如果假设2024年第一季度净利润较上一季度下降了5%，这一假设是否符合图中趋势？",
      "gold_answer": "不符合，图中显示净利润是上升的而非下降。",
      "vars_out": {
        "correction_flag": {"value": "错误假设"},
        "true_trend": {"value": "上升"}
      }
    },
    {
      "turn_id": "T4",
      "question": "请基于正确趋势重新陈述公司季度净利润的变化方向与幅度。",
      "gold_answer": "修正后：2024年第一季度净利润环比上升约10%。",
      "vars_out": {
        "growth_qoq_corrected": {"value": 0.10, "unit": ""}
      }
    },
    {
      "turn_id": "T5",
      "question": "外部新闻提到“分析师预计公司利润将继续保持增长趋势”。结合图中数据，你认为这一判断合理吗？",
      "gold_answer": "合理。根据图中季度利润持续上升的趋势，该预测具备合理性。",
      "vars_out": {
        "integrated_view": {"value": "合理"},
        "confidence_level": {"value": 0.85, "unit": ""}
      }
    }
  ]
}

def build_prompt_dialogue_behavior(plot_type: str, fewshot_block: str) -> str:
    return f"""
你是一名**金融多轮行为推理数据集生成专家**。  
请依据给定图像（图表类型="{plot_type}"）与相关文本描述，构造一个包含**五轮连续对话（T1–T5）**的场景，  
用于评测模型的长期记忆、一致性保持、自我纠错与跨来源整合能力。  
请仅输出一个合法 JSON 对象，不含任何解释或 Markdown。

---

【测试目标】
本场景属于「对话行为层（Dialogue Behavior Layer）」：
- ⑥ **上下文一致性**：能否在多轮中持续保持结论与口径一致，不自相矛盾；
- ⑦ **连续任务理解（Context Carryover）**：能否正确延续前轮逻辑和变量；
- ⑧ **自我纠错（Self-Correction）**：在发现或被提示错误后，是否能在后续轮次中修正；
- ⑨ **外部信息整合（Multi-source Retrieval）**：在最后阶段，结合外部金融文本或新闻修正推理；
- **长期记忆（Episodic Memory）**：贯穿全对话，考察对早期事实与数值的稳定引用与修正。
---

【固定五轮结构】

1️⃣ **T1 初始理解（Fact Identification）**  
   - 读取图像中关键对象的事实信息（如季度净利润、股价峰值、表格中的指标）。  
   - gold_answer：直接读数。  
   - vars_out：主要实体及属性（如 `rev_2024Q1`, `np_2024Q1`）。

2️⃣ **T2 延伸推理（Context Carryover）**  
   - 基于T1提到的对象进行计算或趋势判断（如同比、环比、增长率）。  
   - 问题需自然延续，但禁止显式引用“上一轮”。  
   - gold_answer：推理结论。  
   - vars_out：新指标（如 `growth_yoy`, `trend_label`）。

3️⃣ **T3 误判干预（Injected Misunderstanding）**  
   - 在问题中引入**轻微错误假设**（如“上一季度利润下降”，实际应为上升），  
     观察模型能否保持自洽或指出错误。  
   - gold_answer：明确指出错误并给出正确说明。  
   - vars_out：修正标志（如 `correction_flag`, `true_trend`）。

4️⃣ **T4 自我修正（Self-Correction）**  
   - 模型在上一轮被提示后，应能在此轮重新计算或陈述更新结论。  
   - gold_answer：修正后的新结果或一致性说明。  
   - vars_out：修正后变量（如 `growth_yoy_corrected`, `revised_np`）。

5️⃣ **T5 外部整合（Multi-source Retrieval）**  
   - 引入额外的金融新闻、分析师报告、或宏观数据描述片段（简短1–2句）。  
   - 问题要求结合图像信息与外部描述判断合理性或趋势修正。  
   - gold_answer：整合后的结论（如“结合报告判断增长具备持续性”）。  
   - vars_out：最终判断结果（如 `integrated_view`, `confidence_level`）。

---

【输出格式】

{{
  "session_id": "占位（由上游覆盖）",
  "image_path": "占位（由上游覆盖）",
  "turns": [
    {{
      "turn_id": "T1",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "rev_2024Q1": {{"value": 1.2e9, "unit": "CNY"}},
        "np_2024Q1": {{"value": 1.1e8, "unit": "CNY"}}
      }}
    }},
    {{
      "turn_id": "T2",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "growth_yoy": {{"value": 0.15, "unit": ""}}
      }}
    }},
    {{
      "turn_id": "T3",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "correction_flag": {{"value": "误判已纠正"}},
        "true_trend": {{"value": "上升"}}
      }}
    }},
    {{
      "turn_id": "T4",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "growth_yoy_corrected": {{"value": 0.15, "unit": ""}}
      }}
    }},
    {{
      "turn_id": "T5",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "integrated_view": {{"value": "增长趋势可持续"}},
        "confidence_level": {{"value": 0.9, "unit": ""}}
      }}
    }}
  ]
}}

---

【生成要求】
1. 必须严格包含 T1–T5 五轮；
2. 各轮逻辑自然连贯，不显式引用“上一轮”，但语义上保持一致；
3. 第3轮必须包含轻微错误假设；
4. 第4轮需明确修正；
5. 第5轮整合外部描述；
6. 所有答案必须从图像和描述可验证；
7. 输出为纯 JSON，禁止注释或Markdown。

---

【few-shot 风格参考】（仅供语气参考，禁止抄写内容）：
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
import random
# ---------- 主流程 ----------
async def main():
    ap = argparse.ArgumentParser(description="L3")
    ap.add_argument("--img_root", required=False,default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans", help="根目录：其子目录名即 plot_type")
    ap.add_argument("--out",      required=False,default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/muti_out/dialogue_behavior", help="输出 JSONL（每行一条会话）")
    ap.add_argument("--subdirs",  type=str, default="line", help="仅处理这些子目录（逗号分隔）；留空=全部")
    ap.add_argument("--glob",     type=str, default="**/*.*", help="图片通配符（默认递归）")
    ap.add_argument("--limit",    type=int, default=700, help="每个子目录最多处理多少张（0=不限）")
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
            random.shuffle(img_files)

            if args.limit and args.limit > 0:
                img_files = img_files[:args.limit]
            if not img_files:
                print(f"[WARN] 子目录无图片：{d}")
                continue

            print(f"\n=== 目录：{d} | plot_type={plot_type} | {len(img_files)} 张 ===")

            # few-shot 组装
            local_fs, global_fs = select_fewshot_for_plot(fewshot_all, plot_type, args.fewshot_local, args.fewshot_global)
          
            prompt_text = build_prompt_dialogue_behavior(plot_type, str(build_prompt_dialogue_behavior))

            for fp in img_files:
                total += 1
                image_uri  = fp  # 直接用文件路径（你的 GEMINIClient 已支持）
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
