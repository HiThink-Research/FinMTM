#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📘 图像→公司+工具调用检索器
--------------------------------------
1️⃣ Qwen3-VL 提取行业、指标、时间
2️⃣ 生成公司名与工具调用ActionList
3️⃣ 执行真实工具搜索 (Clarify / Search / FinQuery / StockNews / ReportQuery)
4️⃣ 保存为结构化 JSONL
"""
from api2 import GEMINIClient
import os, re, json, glob, asyncio, argparse, traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from contextlib import AsyncExitStack

from qwen3vl import Qwen3VLClient
from tooluse import ToolCaller
from mcp import ClientSession
from mcp.client.sse import sse_client
from util import safe_json_parse

# ============ JSON 解析 ============
def extract_json_from_text(text: str):
    """
    从文本中提取第一个合法的 JSON 块，忽略多余内容。
    """
    if not text:
        return None

    # 匹配所有 {...} 块（跨行匹配）
    matches = re.findall(r'\{[\s\S]*?\}', text)
    if not matches:
        return None

    for block in matches:
        json_str = block.strip()
        try:
            return json.loads(json_str)  # 只要解析成功就返回
        except json.JSONDecodeError:
            # 清理可能的中文引号、单引号再尝试
            cleaned = (
                json_str
                .replace("”", "\"")
                .replace("“", "\"")
                .replace("'", "\"")
            )
            try:
                return json.loads(cleaned)
            except Exception:
                continue  # 尝试下一个匹配块

    # 所有都失败，返回 None
    return None


# ============ Step 1️⃣ 图表信息提取 ============
async def extract_chart_info(qwen: GEMINIClient, image_uri: str) -> Dict[str, str]:
    prompt = """你是一名金融图表解析专家。
请从下图中提取关键信息，仅输出JSON：
{
  "company": "公司名称",
  "metric": "指标，如产能利用率、价格、营收等",
  "period": "时间范围，如2023Q4-2024Q3"
}"""
    resp = qwen.chat(image=image_uri, text=prompt)
    parsed = extract_json_from_text(resp)
    print(f"📊 图表识别结果: {parsed}")
    return parsed or {}

def extract_company_name(file_path: str) -> str:
    """从文件路径中提取公司名"""
    filename = os.path.basename(file_path)  # 取文件名，比如 '中芯国际_1.jpg'
    name_no_ext = os.path.splitext(filename)[0]  # 去掉扩展名 '.jpg'
    company_name = re.sub(r'_\d+$', '', name_no_ext)  # 去掉末尾编号 '_1' 或 '_23'
    return company_name
# ============ Step 2️⃣ 生成 ActionList ============
async def generate_action_plan(qwen: GEMINIClient, chart_info: Dict[str, str], image_uri: str,name:str) -> Dict[str, Any]:
    # period = chart_info.get("period", "近期")
    # name = chart_info.get("company", "近期")
    
    # name=extract_company_name(image_uri)
    
    prompt = f"""
你是一名金融智能体。首先分析这份财报是从几个维度来描述{name}公司的,请你根据图表中的时间、行业、指标与公司，自动生成一个多工具查询计划 (ActionList)，要求如下。
-图表中已有的时间段、指标、公司数据视为“已知事实”，禁止重复查询。
-请扩展到“紧邻或后续”时间段（如图表为2023–2024，则查询2024–2025或2025–2026),查询的时间不能越过2025.10。
-或者扩展到“相关但不同”的指标（如图表为营收，则可查询净利润、毛利率、研发支出等）
-查询必须与图表主题在逻辑上相关（如图显示“产量增长”，则可查“销售额”“出口量”“成本变化”等）
-查询中必须包含公司名{name}与新时间段。
-输出应在语义上互补，覆盖多维角度（数据、新闻、研报、宏观）
-全部用中文描述
每个动作应包含 "tool_name" 与 "query" 字段。查询语句必须带上时间段  和 公司名。

工具可选：
- Clarify：仅在问题含糊或违背常识时使用；**出现 Clarify 的该轮 ActionList 只能包含 Clarify 一个动作**。
- Search：非结构化网页检索，关键词≤5，可分多次。
- FinQuery：结构化金融数据（股票、指数、基金、期货、宏观、公司财务、行情、事件、技术形态、自选股等）。  
  若未指明标的类型，默认“股票”。筛选类需一次写全条件；查询类用“标的+指标+时间”。
- StockNews：新闻/资讯/消息；可加日期或情绪过滤。
- ReportQuery：研报观点查询。


图表信息：
{json.dumps(chart_info, ensure_ascii=False, indent=2)}

输出严格JSON：
{{
  "company": "公司名",
  "ActionList": [
    {{
      "tool_name": "FinQuery",
      "query": "查询通威股份在2024Q1-2024Q4期间的硅料价格、毛利率、营收变化"
    }},
    {{"tool_name": "StockNews",
      "query": "搜索中芯国际在2023Q4-2024Q3期间的主要新闻及市场舆情"}},
      {{"tool_name": "StockNews",
      "query": "搜索中芯国际在2023Q4-2024Q3期间的主要新闻及市场舆情"}},
  ]
}}
"""
    # resp = qwen.image2text(prompt, image=image_uri)
    try:
        # Gemini 支持多图输入，因此这里直接传入 list
        # print(prompt_text,image_list)
        resp = await qwen.image2text(prompt, image=image_uri)
        raw = resp if isinstance(resp, str) else str(resp)
        data = safe_json_parse(raw)
        print(f"🧭 生成 ActionList: {data}")
        if data and isinstance(data, dict):
            return data
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] Gemini 调用异常: {e}")
    print(resp)
    
    # resp=resp.split("</think>", 1)[-1]
    # parsed = extract_json_from_text(resp)
    
    print(f"🧭 生成 ActionList: {data}")
    return data or {}


# ============ Step 3️⃣ 执行工具调用 ============
async def execute_actions(tool_caller: ToolCaller, action_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results = []
    for action in action_list:
        tool = action.get("tool_name")
        query = action.get("query")
        try:
            print(f"🔍 调用 {tool}: {query}")
            out = await tool_caller.call_one_tool(tool_name=tool, tool_args={"query": query})
            results.append({
                "tool_name": tool,
                "query": query,
                "result": str(out)[:1000]  # 截断保存摘要
            })
        except Exception as e:
            results.append({
                "tool_name": tool,
                "query": query,
                "error": str(e)
            })
    return results


# ============ Step 4️⃣ JSONL 写入 ============
class KnowledgeBaseBuilder:
    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]):
        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============ Step 5️⃣ 主流程 ============
async def process_image(qwen, tool_caller, kb: KnowledgeBaseBuilder, img: str,name:str):
    print(f"\n🖼️ 处理图片: {img}")
    try:
        # chart_info = await extract_chart_info(qwen, img)
        # if not chart_info.get("company"):
        #     print("⚠️ 未识别到行业，跳过。")
        #     return

        plan = await generate_action_plan(qwen, [],img,name)
        actions = plan.get("ActionList", [])
        if not actions:
            print("⚠️ 无可执行动作，跳过。")
            return

        results = await execute_actions(tool_caller, actions)

        record = {
            "image": img,
            # "chart_info": chart_info,
            "company": name,
            "ActionList": actions,
            "ToolResults": results,
            "timestamp": datetime.now().isoformat()
        }
        kb.append(record)
        print(f"✅ 已写入 {record['image']}")

    except Exception as e:
        print(f"❌ 处理异常: {e}")
        traceback.print_exc()


# ============ Step 6️⃣ 启动 ============
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/reports_hs300/images")
    parser.add_argument("--kb_path", default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/action_kb.jsonl")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    qwen = GEMINIClient()

    async with AsyncExitStack() as stack:
        transport = await stack.enter_async_context(sse_client("http://localhost:8081/sse"))
        session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
        await session.initialize()

        tool_caller = ToolCaller(session)
        kb = KnowledgeBaseBuilder(args.kb_path)
        base_dir = Path("/mnt/HithinkOmni/user_workspace/zhangchenxi4/reports_hs300_3/images")
        # imgs = glob.glob(os.path.join(args.img_root, "**/*.jpg"), recursive=True)
        for company_dir in sorted(base_dir.iterdir()):
            if not company_dir.is_dir():
                continue
            company_name = company_dir.name
            image_files = sorted([str(p) for p in company_dir.glob("*.jpg")])[:3]
            for img in image_files:
                await process_image(qwen, tool_caller, kb, img,company_name)

        # print(f"\n✅ 全部完成，共处理 {len(imgs)} 张图像。")
        print(f"📄 输出文件：{args.kb_path}")


if __name__ == "__main__":
    asyncio.run(main())
