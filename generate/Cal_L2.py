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

def build_prompt_logical_reasoning_implicit(plot_type: str, fewshot_block: str) -> str:
    return f"""
你是一名**金融多轮逻辑推理数据生成专家**。  
请仅依据给定图像（图表类型="{plot_type}"）构造一个**固定 4 轮对话**，要求问题具有逻辑承接与隐式依赖关系。  
请只输出**一个合法 JSON 对象**，不含任何多余文字、注释或 Markdown 符号。

---

【核心目标】
- 模拟真实金融分析师的多轮对话：从事实读取 → 计算 → 条件变化 → 对照验证。
- 每轮问题隐式依赖前轮定义的对象和变量，但**不出现显式引用词**（如“上一轮”、“你刚才”）。
- 模型必须通过语义记忆理解上下文并保持逻辑一致。

---

【四轮固定结构】

1️⃣ **T1 基础读取（Fact Extraction）**  
   - 定义并读取一个唯一的金融对象（如 A 点、R1 区间、C1 单元格）。  
   - 问题应明确说明对象的最小定义（如“蓝色折线中2024年3月末的A点”）。  
   - gold_answer：直接读数（如价格、比率、金额）。  
   - vars_out：记录该对象的主要属性（如 A_close, A_date, rev_2024Q1）。

2️⃣ **T2 一步计算（Direct Reasoning）**  
   - 使用 T1 中定义的对象变量进行一次明确计算（如同比增长、涨跌幅、利润率）。  
   - 问题需自然衔接 T1 对象的定义，但**不出现任何显式承接词**。  
   - gold_answer：计算结果。  
   - vars_out：记录新指标（如 A_growth, ret_R1, margin_2024Q1）。

3️⃣ **T3 条件变化（Counterfactual Adjustment）**  
   - 在 T1/T2 的对象上施加轻微可计算的假设调整（如上调1%、延长周期、剔除一次性收益）。  
   - 问题要自然说明调整条件，但禁止出现“重新计算”、“上轮”等字样。  
   - gold_answer：调整后的结果。  
   - vars_out：记录调整后变量（如 A_growth_adj, ret_R1_adj, margin_adj）。

4️⃣ **T4 对照验证（Comparative Validation）**  
   - 对同一对象执行对照推理（如与相邻时间段或其他指标比较）。  
   - 问题必须仍指向同一对象的定义，测试模型的长期语义记忆与逻辑保持。  
   - gold_answer：可验证结论（是/否、较大者、趋势方向等）。  
   - vars_out：记录最终度量（如 higher_than_peer, change_dir）。

---

【输出格式】
{{
  "image_path": "占位（由上游覆盖）",
  "turns": [
    {{
      "turn_id": "T1",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "A_close": {{"value": 0, "unit": ""}},
        "A_date":  {{"value": "2024-03-31"}}
      }}
    }},
    {{
      "turn_id": "T2",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "A_growth": {{"value": 0.05, "unit": ""}}
      }}
    }},
    {{
      "turn_id": "T3",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "A_growth_adj": {{"value": 0.06, "unit": ""}}
      }}
    }},
    {{
      "turn_id": "T4",
      "question": "...",
      "gold_answer": "...",
      "vars_out": {{
        "compare_result": {{"value": "A更高" }}
      }}
    }}
  ]
}}

---

【生成要求】
1. 四轮固定，顺序严格为 T1–T4。  
2. 每轮问题逻辑承接自然，不显式引用上一轮。  
3. 对象命名保持一致（如 A/R1/C1），变量名统一前缀，便于评测器隐式解析依赖。  
4. 所有 gold_answer 均应由图像中可读元素或其直接计算结果得出，禁止幻觉或外部知识。  
5. 数值型变量需附单位（如 %, CNY, USD），时间应为 YYYY-MM 或 YYYY-Qn 格式。  
6. 输出必须是合法 JSON，无多余字段、注释或 Markdown。

---

【few-shot 风格参考】（仅供语言风格参考，禁止抄写内容）：
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
    ap = argparse.ArgumentParser(description="L2")
    ap.add_argument("--img_root", required=False,default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans", help="根目录：其子目录名即 plot_type")
    ap.add_argument("--out",      required=False,default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/muti_out/multi_reasoning", help="输出 JSONL（每行一条会话）")
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
          
            prompt_text = build_prompt_logical_reasoning_implicit(plot_type, "")

            for fp in img_files:
                total += 1
                image_uri  = fp  # 直接用文件路径（你的 GEMINIClient 已支持）
                sess, detail = await call_gemini_multiturn(gpt, image_uri=image_uri, prompt_text=prompt_text, trials=args.trials)

                if sess:
                    # 强制覆盖最小必要字段
                    # sess["session_id"] = str()
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
