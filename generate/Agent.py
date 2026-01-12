#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版：自动生成多轮金融Agent评测样本（全流程异步版本）
流程：
  Step 1: Gemini_caption 提取图像要素（公司、指标、时间、趋势）
  Step 2: 调用 MCP 工具链（FinQuery / NoticeSearch / StockNews...）补充上下文知识
  Step 3: Gemini 综合图像与知识上下文，生成三轮对话样本 JSON
  Step 4: 输出 JSONL，每行对应一张图的完整 session
"""

import os
import re
import json
import glob
import random
import asyncio
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack

# -------- 外部依赖模块（用户已有） --------
from util import safe_json_parse
from api2 import GEMINIClient
from tooluse import ToolCaller
from mcp import ClientSession
from mcp.client.sse import sse_client


# ============================================================
# Step 1: Gemini Caption 提取图像要素
# ============================================================
async def extract_chart_info(gpt: GEMINIClient, image_uri: str) -> Dict[str, str]:
    """从图像中提取关键信息（标的、指标、时间、趋势）"""
    prompt = """你是一名金融图表解析专家。
只提取公司名字
输出 JSON：
{
  "chart_type": "",
  "target_name": "公司名称",
  "metric": "",
  "period": "",
  "trend": ""
}"""
    resp = await gpt.image2text(prompt, image=image_uri)
    return safe_json_parse(resp) or {}


# ============================================================
# Step 2: 工具知识补全
# ============================================================
import datetime
# ============================================================
# Step 2: 从本地知识库检索工具结果
# ============================================================
import json
from typing import Dict, Any, List

async def collect_agent_knowledge_from_kb(kb_path: str, company_name: str) -> Dict[str, Any]:
    """
    从本地知识库中匹配公司名，提取结构化知识，用于Agent上下文。
    输出格式包含工具、查询参数、返回内容、发布时间等字段。
    """

    # 加载知识库
    kb_data = []
    if kb_path.endswith(".jsonl"):
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = [json.loads(line) for line in f if line.strip()]
    elif kb_path.endswith(".json"):
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
            if isinstance(kb_data, dict):
                kb_data = [kb_data]

    # 匹配公司
    matched_record = next((r for r in kb_data if company_name in r.get("name", "")), None)
    if not matched_record:
        return {"company": company_name, "knowledge": [], "note": "知识库中未找到匹配公司"}

    dims = matched_record.get("dimensions", {})
    evidence = matched_record.get("evidence", [])

    knowledge_structured = []

    # 遍历每个维度
    for dim_name, dim_val in dims.items():
        raw_text = dim_val.get("raw_text", "").strip()
        if not raw_text:
            continue

        # 查找该维度对应的 evidence
        related_evidence = [ev for ev in evidence if ev["dimension"] == dim_name]
        if related_evidence:
            for ev in related_evidence:
                knowledge_structured.append({
                    "dimension": dim_name,
                    "tool": ev.get("tool", "未知"),
                    "query": ev.get("query", "未知"),
                    "timestamp": ev.get("timestamp", ""),
                    "raw_text": raw_text
                })
        else:
            # 没有 evidence 时，仍保留原始内容
            knowledge_structured.append({
                "dimension": dim_name,
                "tool": "未知",
                "query": "（无对应查询记录）",
                "timestamp": "",
                "raw_text": raw_text
            })

    return {
        "company": company_name,
        "knowledge": knowledge_structured,
        "note": f"提取到 {len(knowledge_structured)} 条知识项"
    }
import json
from typing import Dict, Any, List

async def collect_agent_knowledge_from_kb2(kb_path: str, info: str) -> Dict[str, Any]:
    """
    从本地知识库中匹配图像或公司名，提取完整的工具查询知识。
    输出格式包含每个工具的调用参数、查询语句、返回结果，方便直接用于Prompt。
    """
    company = info

    # 加载知识库文件
    kb_data = []
    if kb_path.endswith(".jsonl"):
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = [json.loads(line) for line in f if line.strip()]
    elif kb_path.endswith(".json"):
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
            if isinstance(kb_data, dict):  # 单记录
                kb_data = [kb_data]

    # 匹配记录：根据 image 或 company 名称
    matched_record = None
    for record in kb_data:
        if company in record.get("company", "") or company in record.get("image", ""):
            matched_record = record
            break

    if not matched_record:
        return {
            "company": company,
            "knowledge_text": "（知识库中未找到匹配公司）",
            "knowledge_list": []
        }

    # 提取工具调用及结果
    action_list = matched_record.get("ActionList", [])
    tool_results = matched_record.get("ToolResults", [])

    # 整理结构化信息
    knowledge_list = []
    for action in action_list:
        tool_name = action.get("tool_name", "")
        query = action.get("query", "")

        # 尝试匹配对应的返回结果
        result_obj = next((r for r in tool_results if r.get("tool_name") == tool_name and r.get("query") == query), None)
        result_text = ""
        if result_obj:
            raw_result = result_obj.get("result", "")
            try:
                # 解析嵌套的字符串形式JSON
                if isinstance(raw_result, str) and raw_result.strip().startswith("{"):
                    result_text = eval(raw_result).get("result", raw_result)
                else:
                    result_text = raw_result
            except Exception:
                result_text = raw_result

        knowledge_list.append({
            "tool": tool_name,
            "query": query,
            "result": result_text.strip() # 限制过长文本
        })

    # 格式化成自然语言知识文本
    formatted_texts = []
    for k in knowledge_list:
        formatted_texts.append(
            f"【{k['tool']}】使用指令：{k['query']}；\n返回结果摘要：{k['result'][:600]}..."
        )

    knowledge_text = "\n\n".join(formatted_texts)

    return {
        "company": matched_record.get("company", company),
        "image": matched_record.get("image", ""),
        "knowledge_text": knowledge_text,
        "knowledge_list": knowledge_list,
        "timestamp": matched_record.get("timestamp", "")
    }

# ============================================================
# Step 3: 构造评测 Prompt
# ============================================================
def build_prompt(chart_info: str, knowledge: str, knowledge2: str, fewshot: str = "") -> str:
    """
    构造单轮问题，严格限定为客观数值或逻辑计算类问题，
    要求模型结合视觉观察与结构化知识（FinQuery、ReportQuery、StockNews）进行计算或判断，
    并生成完整的 depend_on 字段，列出每个数据的来源、工具与获取方式。
    输出为单个 JSON 对象。
    """
    return f"""
你是一名**金融Agentic评测数据构造专家**。  
你的任务是生成一个**完全客观、基于数据计算或逻辑判断**的问题与标准答案。  
模型必须基于图像（chart_info）与知识库（knowledge / knowledge2）中的数值信息进行事实性计算或逻辑推导，问题必须要通过工具调用和视觉观察共同解决，如果无法保证，就优先工具调用解决,

注意：问题中不要出现**根据知识库**，也不要带公司名，可以说**图中的公司**
1、在生成问题时，不直接点名公司名称，而是利用知识库1中工具调用得到的事实信息（如股价、行业、业绩、政策背景等），把这些事实编织进问题中，形成模糊化的公司描述
2、question1 是不模糊的，question2 是模糊后的,
3、并且 tool_call里也应该有如何找到这个公司的工具调用过程
4、Thought里也要写出来如何锁定这家公司的过程，必须是通过工具调用，根据你怎么模糊的来锁定它呗，不能是通过观察的，也要有如何规划工具调用的过程


---

### 【任务类型】
你只能生成以下类型的问题：
1. **数值计算类**：根据图表和知识库数值计算涨跌幅、比值、差额、比例等；
2. **逻辑判断类**：判断两个指标是否同向变化；
3. **比较关系类**：比较两个时间点或两个指标的大小；
4. **趋势一致性类**：判断图表趋势与指标变化是否一致；
5. **比例推导类**：根据市值、利润、营收等字段计算相关比率。

---

### 【禁止项（若出现立即重写）】
- 不得包含主观或模糊词：如“可能”、“明显”、“合理”、“显著”、“乐观”、“悲观”、“受影响”等；
- 不得生成预测性、原因分析或建议类问题；
- 不得生成非数值可验证的问题；
- 所有问题答案必须**可通过数据直接推导或验证**；
- 若信息不足，生成限定性问题（例如：“根据已知数据，是否可以判断……”）。
- ActionTrace不能为空，必须要结合知识库

---

### 【输入信息】
#### 图像解析结果：
{json.dumps(chart_info, ensure_ascii=False, indent=2)}

#### 知识库1（基础面信息）：
{knowledge or "（暂无）"}

#### 知识库2（结构化工具数据）：
{knowledge2 or "（暂无）"}

---

### 【输出 JSON 模板】
输出必须为**单个合法 JSON 对象**（不得包含 Markdown、解释或注释）。
"reason" 字段中直接嵌入原始工具调用 JSON 信息，让模糊化的溯源更清晰。
也就是说，reason 不仅要描述模糊化逻辑，还要引用对应的工具调用对象,reason是描述采用了什么知识来模糊问题的
{{
    "reason": {{
    "tool_call": {{
      "tool_name": "FinQuery",
      "query": "查询邮储银行在2025年第三季度的股价表现，包括收盘价、涨跌幅和成交量"
    }},
    "explanation": "根据FinQuery工具调用结果，原问题涉及邮储银行在2025年第三季度的股价表现。为模糊公司身份，保留了'2025年第三季度'与'股价走势'等事实限定条件，并将公司实体改为'大型国有商业银行'以避免直接暴露。"
  }},
  "image_path": "占位或可为空",
  "turn": {{
    "question": "计算中芯国际近一个月的股价涨幅百分比。",
    "question2": "模糊后的问题",
    "Thought": "图像中价格点随时间上升；知识库提供起始与结束价格，可计算涨幅。注意要有工具调用过程",
    "VisualObservation": [
      "图表中起始价格约为22.5元，结束价格约为24.0元"
    ],
    "ActionTrace": [
      {{
        "tool": "FinQuery",
        "query": "中芯国际 股价数据 2025.09-2025.10",
        "observation": "起始价格22.5元，结束价格24.0元"
      }}
    ],
    "gold_answer": {{
      "final_conclusion": "根据价格差异 (24.0 - 22.5) / 22.5 = 6.7%，可得涨幅约为6.7%。",
      "depend_on": {{
        "items": [
          {{
            "name": "起始价格",
            "content": "22.5元",
            "source_type": "视觉观察",
            "source_method": "从图表左端读取价格刻度"
          }},
          {{
            "name": "结束价格",
            "content": "24.0元",
            "source_type": "视觉观察",
            "source_method": "从图表右端读取价格刻度"
          }},
          {{
            "name": "涨幅计算",
            "content": "(24.0 - 22.5) / 22.5 = 6.7%",
            "source_type": "逻辑计算",
            "source_tool": "无",
            "source_method": "根据价格差计算涨幅百分比"
          }}
        ],
      }}
    }}
  }}
}}

---

### 【最终约束】
1. 问题必须为可计算、可验证的客观数值或逻辑判断题；
2. 不得出现预测性、情绪性、主观性描述；
3. 不得虚构任何未在图像或知识库出现的数值；
4. 答案需展示中间计算步骤；
5. 依赖字段 depend_on 中必须列出数据来源与计算方式；
6. 若数据不足以计算，必须输出限定性问题（如“根据现有数据，是否可以判断…”）；
7. 模型在输出前必须自检，确保：
   - 问题可由现有数据回答；
   - 问题包含明确计算目标；
   - 无任何主观、预测或情绪性词汇。
{fewshot}
""".strip()


# ============================================================
# Step 4: 生成多轮问答 JSON
# ============================================================
async def generate_session(gpt: GEMINIClient, img: str, chart_info: str, knowledge: str,knowledge2: str, fewshot: str) -> Optional[Dict[str, Any]]:
    """调用 Gemini 生成完整三轮对话 session"""
    prompt = build_prompt(chart_info, knowledge,knowledge2, fewshot)
    resp = await gpt.image2text(prompt, image=img)
    data = safe_json_parse(resp)
    # data["image_path"] = img
    print(data)
    if isinstance(data, dict) and "turn" in data:
        data["image_path"] = img
        return data
    return None

import traceback
# ============================================================
# Step 5: 主流程控制
# ============================================================
def extract_company_name(file_path: str) -> str:
    """从文件路径中提取公司名"""
    filename = os.path.basename(file_path)  # 取文件名，比如 '中芯国际_1.jpg'
    name_no_ext = os.path.splitext(filename)[0]  # 去掉扩展名 '.jpg'
    company_name = re.sub(r'_\d+$', '', name_no_ext)  # 去掉末尾编号 '_1' 或 '_23'
    return company_name
async def process_image(gpt, img, fewshot,name):
    """单张图片完整处理流程"""
    print(f"\n🖼️ 处理图像: {img}")
    try:
        # Step 1
        print(1)
        # chart_info = await extract_chart_info(gpt, img)
        # name=extract_company_name(img)
        
        print("图像要素:", name)

        # Step 2
        knowledge = await collect_agent_knowledge_from_kb(
    kb_path="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/knowledge_base2.jsonl",
    company_name=name
)
        knowledge2 = await collect_agent_knowledge_from_kb2(
    kb_path="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/action_kb.jsonl",
    info=name
)     
        knowledge=json.dumps(knowledge, ensure_ascii=False, indent=2)
        knowledge2=json.dumps(knowledge2, ensure_ascii=False, indent=2)
        print()
        # if "未找到匹配公司" in knowledge2 or not knowledge2.strip():
        #    print(f"⚠️ {name} 在 action_kb.jsonl 中未找到匹配记录，跳过。")
        #    return None

        print("知识上下文长度:", len(knowledge))
        print("知识上下文长度:", len(knowledge2))

        # Step 3
        sess = await generate_session(gpt, img, name, knowledge, knowledge2,fewshot)
        if sess:
            print("✅ 已成功生成样本")

            # ✳️ 立即写入文件
            out_path = Path("/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/muti_out/multi_round_outv3.jsonl")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(sess, ensure_ascii=False) + "\n")

            return sess
        else:
            print("⚠️ 生成失败，返回空")
            return None
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback; traceback.print_exc()
        return None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/reports_hs300/images")
    parser.add_argument("--out", default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/muti_out/multi_round_outv3.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    base_dir = Path("/mnt/HithinkOmni/user_workspace/zhangchenxi4/reports_hs300_3/images")
    gpt = GEMINIClient()

    # 初始化 MCP 工具会话
    async with AsyncExitStack() as stack:
        # transport = await stack.enter_async_context(sse_client("http://localhost:8081/sse"))
        # session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
        # await session.initialize()
        # tool_caller = ToolCaller(session)

        # few-shot 样本可选
        fewshot = ""
        for company_dir in sorted(base_dir.iterdir()):
            if not company_dir.is_dir():
                continue
            company_name = company_dir.name
            # if company_name=="万泰生物": 
            #     continue
            img_files = sorted([str(p) for p in company_dir.glob("*.jpg")])[:3]
            # img_files = glob.glob(os.path.join(args.img_root, "**/*.jpg"), recursive=True)[:args.limit]
            print(f"共找到 {len(img_files)} 张图片")

            # 并发执行
            results = await asyncio.gather(*[
                process_image(gpt, img, fewshot,company_name) for img in img_files
            ])

        # 写出结果
            # out_path = Path(args.out)
            # out_path.parent.mkdir(parents=True, exist_ok=True)
            # with open(out_path, "a", encoding="utf-8") as fout:
            #     for sess in results:
            #         if sess:
            #             fout.write(json.dumps(sess, ensure_ascii=False) + "\n")

            # print(f"\n✅ 全部完成，输出文件：{out_path}")


if __name__ == "__main__":
    asyncio.run(main())
