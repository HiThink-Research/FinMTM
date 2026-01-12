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

def build_prompt_multiview_memory_eval(doc_type: str = "行业研报") -> str:
    """
    用于评测 VLM 在多页研报中的长上下文记忆与整合能力（递进式依赖）。
    约束：
    - 多图输入（多页/多图一起送入）
    - 问题(question)中禁止出现任何页码/图表编号等引用
    - 引用只允许出现在答案(gold_answer)中
    - 四轮任务严格递进，后轮必须隐式依赖前轮，不使用“上一轮/如前所述”等显式词
    - 输出 JSON 中包含 depends_on 字段表达依赖关系h
    - 去掉 vars_out
    """
    prompt = f"""
你是一位具备专业金融分析能力的多模态大模型评测助手，
任务是评估模型在阅读 {doc_type} 时的 **记忆与上下文整合能力（Memory Capability）**。
模型将接收多页图片（或PDF页），并根据内容回答多类问题。

请严格遵守以下硬性规范：
- 准确提取数值、年份、同比/环比、趋势方向；
- 若涉及计算，请展示中间步骤（写出公式与代入过程）；
- 若涉及引用，请明确指出来源页或图表编号；
- 若问题跨页，请整合多页信息；
- 若信息不足，请回答“**不足以回答**”并说明缺失信息种类；
- **禁止在 question 中加入任何引用（页码、图表/表格编号、正文页号等）；所有引用只能出现在 gold_answer 中；**
- 四轮问答**必须递进式关联**：T2 必须依赖 T1；T3 必须同时依赖 T1 与 T2；T4 必须依赖 T1–T3；
- **禁止使用“上一轮/刚才/如前所述”等显式指代词**，但语义上必须依赖之前已建立的事实；
- 每条 **gold_answer** 都必须附带 citation（如：[图2，第1页]、[表1，第2页]、[正文，第3页]）。

---

### 【评测任务类型（四轮，递进依赖）】

#### T1  单页事实问答（Single-page factual QA）
目标：在一页内提取**可核验的单一事实**（数值/同比/单位/方向）。
- **question**：仅描述信息需求，不得出现引用位置。
- **gold_answer**：给出准确数值 + 时间 + 单位 + 同比/环比 + 趋势（若有），并附 citation。

示例（仅示意）：
- question：本期行业的累计销量与同比增速分别是多少？
- gold_answer：2025年1–8月累计销量为 **154,181 台**，同比增长 **17.2%**。[图2，第1页]

---

#### T2  多模态跨段融合（Cross-page & multi-modal reasoning） —— 依赖 T1
目标：将 T1 的关键量参与到**跨页计算**（占比、差值、同比/环比等）。
- **question**：提出需要组合多页数据完成计算的需求，不出现引用。
- **gold_answer**：写出公式、代入过程、结果，并给出涉及的每个来源的 citation。

示例（仅示意）：
- question：基于相关数据，计算内销与出口在累计销量中的占比。
- gold_answer：累计销量 **154,181** [表2/正文，第1页]；内销 **80,628**、出口 **73,553** [正文，第1页]。
  内销占比 = 80,628 / 154,181 ≈ **52.3%**；出口占比 = 73,553 / 154,181 ≈ **47.7%**。[表2/正文，第1页]

---

#### T3  行业/个股逻辑与趋势分析（Analytical reasoning, Multi-choice） —— 依赖 T1 与 T2
目标：基于 T1–T2 已建立/计算出的事实，构造**多选题**进行逻辑检验。
- **question**：给出 4 个选项（A–D），不得出现引用；题干要与前述事实存在**隐式依赖**（需用到 T1/T2 的量或结论）。
- **gold_answer**：给出正确选项数组 `["A","C",...]` + 逐项推理说明；每条论据都要带 citation。

示例（仅示意）：
- question：以下判断哪些成立？A. 行业累计销量同比增长  B. 内销占比超过 50%  C. 出口占比低于 40%  D. 新能源发电盈利恶化
- gold_answer：正确答案 **A、B**。A：累计销量同比 +17.2% [图2，第1页]；B：内销占比 ≈52.3%（由 T2 计算，数据源见正文/表2）；
  C：出口约 47.7%，非低于 40% [正文/表2，第1页]；D：报告为“趋于稳健”而非“恶化” [正文，第8页]。

---

#### T4  概括总结（Summary synthesis） —— 依赖 T1–T3
目标：在长上下文中做**结构化摘要**，覆盖：  
【主要观点】【相对指数表现】【风险提示】【投资建议】  
- **question**：提出总结需求，不出现引用。
- **gold_answer**：以要点列表形式输出，**每个要点都包含具体数字**，并在句尾附 citation。

示例（仅示意）：
- gold_answer：
  【主要观点】行业延续修复，挖掘机销量 **16,523 台**，同比 **+12.8%**；电动化率 **26.2%** [第1页]。  
  【相对指数表现】沪深300 **+1.38%**；公用事业 **+0.80%**、环保 **+1.00%**，相对收益分别 **-0.58%**、**-0.38%** [第6–9页]。  
  【风险提示】经济复苏放缓、原材料波动、海外不确定性 [正文，第1页]。  
  【投资建议】关注 X、Y、Z，维持“强推/增持”评级 [正文，第8–9页]。

---

### 【输出 JSON 格式（必须四轮且递进依赖）】

```json
{{
  "image_paths": ["<img_1>", "<img_2>", "<img_3>", "..."],  // 外层会注入公司下所有图片
  "turns": [
    {{
      "turn_id": "T1",
      "task_type": "单页事实问答",
      "question": "（不得包含任何页码/图表编号的引用）",
      "gold_answer": "…… [图X，第Y页]"
    }},
    {{
      "turn_id": "T2",
      "task_type": "多模态跨段融合",
      "depends_on": ["T1"],
      "question": "（不得包含任何页码/图表编号的引用）",
      "gold_answer": "（公式 + 代入 + 结果）…… [图/表/正文+页码]"
    }},
    {{
      "turn_id": "T3",
      "task_type": "行业逻辑与趋势分析（多选）",
      "depends_on": ["T1","T2"],
      "question": "（不得包含任何引用）\\nA. …\\nB. …\\nC. …\\nD. …",
      "options": {{
        "A": "……",
        "B": "……",
        "C": "……",
        "D": "……"
      }},
      "gold_answer": {{
        "correct": ["A","C"], 
        "reasoning": "逐项给出依据，并在每条依据末尾添加 citation，如 [图X，第Y页]/[表Z，第W页]/[正文，第K页]"
      }}
    }},
    {{
      "turn_id": "T4",
      "task_type": "概括总结",
      "depends_on": ["T1","T2","T3"],
      "question": "（不得包含任何页码/图表编号的引用）",
      "gold_answer": "【主要观点】…[来源]\\n【相对指数表现】…[来源]\\n【风险提示】…[来源]\\n【投资建议】…[来源]"
    }}
  ]
}}
""".strip()

    return prompt




# ---------- 结果校验（保持最小字段，做必要检查） ----------
def validate_multiturn_session(obj: Any) -> Optional[Dict[str,Any]]:
    if not isinstance(obj, dict): 
      print("1")
      return None
    if "turns" not in obj or not isinstance(obj["turns"], list): 
      print("2")
      return None
    turns = obj["turns"]
    # if not (3 <= len(turns) <= 5): return None

    seen_ids = set()
    for t in turns:
        if not isinstance(t, dict): return None
        tid = str(t.get("turn_id","")).strip()
        q   = clean_text(t.get("question",""))
        ga  = clean_text(t.get("gold_answer",""))
        vo  = t.get("vars_out", {})
        if not tid or tid in seen_ids:
          print("not tid")
          return None
        seen_ids.add(tid)
        if not q or not ga: 
          print("not q ga")
          return None
        if not isinstance(vo, dict): return None
        # 简单校验 vars_out 的最小字段
        # for vname, vobj in vo.items():
        #     if not isinstance(vobj, dict): return None
        #     if "value" not in vobj: return None
        #     # unit 可选，不强制

        # depends_on 可选；若存在，需要是 list[str]
        # if "depends_on" in t and not isinstance(t["depends_on"], list):
        #     return None

    return obj
import traceback
async def call_gemini_multiview(gpt, image_list, prompt_text):
    """
    多图输入，调用 Gemini 模型，返回解析后的结果
    """
    try:
        # Gemini 支持多图输入，因此这里直接传入 list
        # print(prompt_text,image_list)
        resp = await gpt.image2text2(prompt_text, image=image_list)
        raw = resp if isinstance(resp, str) else str(resp)
        data = safe_json_parse(raw)
        if data and isinstance(data, dict) and "turns" in data:
            data["image_paths"] = image_list
            return data, raw
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] Gemini 调用异常: {e}")
    return None, None

# ========== 主逻辑 ==========
from PIL import Image
async def main():
    base_dir = Path("/mnt/HithinkOmni/user_workspace/zhangchenxi4/reports_hs300_3/images")
    out_file ="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/muti_out/memory_pdfv3.jsonl"
    gpt = GEMINIClient()

    # 输出文件打开为追加模式
    with open(out_file, "a", encoding="utf-8") as fout:
        for company_dir in sorted(base_dir.iterdir()):
            if not company_dir.is_dir():
                continue
            company_name = company_dir.name
            image_files = sorted([str(p) for p in company_dir.glob("*.jpg")])[:5]
            if len(image_files) == 0:
                print(f"[WARN] {company_name} 无图片，跳过。")
                continue
            with Image.open(image_files[0]) as img:
              width, height = img.size
              print(f"📏 第一张图片分辨率: {width} x {height}")

            print(f"\n=== 正在生成公司：{company_name} ({len(image_files)} 张图) ===")

            # 构造 prompt
            prompt_text = build_prompt_multiview_memory_eval(doc_type=f"{company_name}研报")

            # 调用模型
            session, raw = await call_gemini_multiview(gpt, image_files, prompt_text)

            if session:
                fout.write(json.dumps(session, ensure_ascii=False) + "\n")
                print(f"[OK] 已生成：{company_name}")
            else:
                fail_path = base_dir / "_failed"
                fail_path.mkdir(exist_ok=True)
                (fail_path / f"{company_name}.txt").write_text(raw or "返回为空", encoding="utf-8")
                print(f"[FAIL] {company_name} 生成失败，已记录。")

    print(f"\n✅ 全部完成，结果保存在：{out_file}")

if __name__ == "__main__":
    asyncio.run(main())