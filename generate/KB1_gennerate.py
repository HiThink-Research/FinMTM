#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📘 行业-公司六维金融知识库构建器
------------------------------------------------
1️⃣ 从图像提取行业、指标、时间等信息（Qwen3VL）
2️⃣ 搜索龙头公司（FinQuery/ReportQuery）
3️⃣ 对每家龙头公司，按六维度调用工具获取详细知识
4️⃣ 输出结构化JSONL知识库（每张图一条）
"""

import os
import re
import json
import glob
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from contextlib import AsyncExitStack

# ===== 用户已有模块 =====
from util import safe_json_parse
from tooluse import ToolCaller
from mcp import ClientSession
from mcp.client.sse import sse_client
from qwen3vl import Qwen3VLClient
qwen = Qwen3VLClient(api_base="http://localhost:8000/v1", model="Qwen3-VL-30B-A3B-Instruct")
# ---------------------------
import re, json

import re
import json

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


# ============================================================
# Step 1️⃣: 图表解析
# ============================================================
async def extract_chart_info(qwen: Qwen3VLClient, image_uri: str) -> Dict[str, str]:
    """提取行业、指标、时间信息"""
    prompt = """你是一名金融图表解析专家。
请从下图中提取关键信息，只输出JSON：
{
  "industry": "行业名称，如钢铁、汽车、光伏",
  "metric": "指标，如产能利用率、价格、营收等",
  "period": "时间范围，如2023Q4-2024Q3"
}"""
    resp = qwen.chat(image=image_uri, text=prompt)
    print("📊 图像识别结果：", extract_json_from_text(resp))
    return extract_json_from_text(resp)


# ============================================================
# Step 2️⃣: 获取行业龙头公司列表
# ============================================================
async def refine_with_qwen(qwen: Qwen3VLClient,  query: str, raw_text: str) -> Dict[str, Any]:

    prompt = f"""
你是一名金融分析专家。请根据以下工具结果，提炼出维度相关的核心信息,不要输出思考过程。
输出必须是简明的 JSON 格式，仅包含 "result" 字段。

示例：
{{
  "result": "公司名字，如中信特钢，宝钢集团，阿里巴巴"
}}

请严格遵守：只输出 JSON，不要输出任何思考或解释。
---
【查询指令】：
{query}

【工具返回原文】：
{str(raw_text)}
"""

    refined = qwen.chat(text=prompt)
    refined=refined.split("</think>", 1)[-1]
    print(refined)
    parsed = extract_json_from_text(refined)
    print("qwen:",parsed)
    return parsed or {"summary": parsed}


async def get_leading_companies(qwen: Qwen3VLClient,tool_caller: ToolCaller, industry: str,per: str) -> List[str]:
    """用FinQuery/ReportQuery获取行业龙头公司"""
    tools = [
        {"tool_name": "FinQuery", "tool_args": {"query": f"{industry}行业龙头公司有哪些?"}},
        # {"tool_name": "ReportQuery", "tool_args": {"query": f"{industry}行业龙头公司有哪些"}}
    ]
    results = await tool_caller.batch_call_tools(tools)
    print(results)
    results=await refine_with_qwen(qwen=qwen,raw_text=results,query="这里面行业龙头公司有哪些"
                     )
    print("result:",results)
    if isinstance(results, str):
     results = [{"result": results}]
    elif isinstance(results, dict):
     results = [results]
    elif results is None:
     results = []


    text_results = " ".join([str(r.get("result", "")) for r in results ])
    print(2)
    companies = re.split(r"[，,、；;]", text_results)
    print(companies)
    companies = [c.strip() for c in companies if c.strip()]
    # 简单提取公司名（用中文公司名匹配）
    # candidates = re.findall(r"[一-龥]{2,6}(股份|集团|科技|实业|能源|公司)", text_results)
    unique_companies = list(companies)[:5]  # 去重取前5个
    print(f"🏭 行业【{industry}】龙头公司前5:", unique_companies)
    return unique_companies


async def refine_with_qwen2(qwen: Qwen3VLClient, query: str, raw_text: str) -> str:
    """
    调用 Qwen 对工具返回结果进行总结。
    仅输出简明 JSON 格式或一句话摘要。
    """
    if not raw_text:
        return ""

    prompt = f"""
你是一名金融分析专家。请根据以下工具返回内容，精简百分之5的信息。
示例：
{{
  "result": ""
}}

【问题】：
{query}

【工具返回内容】：
{raw_text}

"""

    try:
        resp = qwen.chat(text=prompt)
        resp=resp.split("</think>", 1)[-1]
        refined=resp.strip()
        print(refined)
        refined=extract_json_from_text(refined)
        return refined
    except Exception as e:
        print(f"⚠️ Qwen 总结异常: {e}")
        return ""

# ============================================================
# Step 3️⃣: 六维信息采集
# ============================================================
import re
import traceback
from datetime import datetime
from typing import Dict, Any, List

async def collect_company_knowledge(
    qwen, tool_caller, company: str, industry: str, p: str,
    dim_configs: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    通用化版本：为单个公司采集多维知识（可扩展维度）
    Args:
        qwen: Qwen3VLClient 实例
        tool_caller: 工具调用器
        company: 公司名称
        industry: 所属行业
        p: 时间参数（如“2024年Q4”或“2025-10-26”）
        dim_configs: 维度配置列表（默认加载基础维度）
    """
    import traceback

    # ---------- 默认维度配置 ----------
    default_dims = [
        {"dim": "市值", "tool": "FinQuery", "query": f"截止到{p}，{company}当前总市值及流通市值"},
        {"dim": "PE", "tool": "FinQuery", "query": f"截止到{p}，{company}当前市盈率PE(TTM)"},
        {"dim": "PB", "tool": "FinQuery", "query": f"截止到{p}，{company}当前市净率PB"},
        {"dim": "PS", "tool": "FinQuery", "query": f"截止到{p}，{company}当前市销率PS"},
        {"dim": "行业竞争", "tool": "ReportQuery", "query": f"截止到{p}，{industry}的主要竞争格局与景气度"},
        {"dim": "财务健康", "tool": "FinQuery", "query": f"截止到{p}，{company}近三年关键财务指标（ROE、ROA、净利润率、负债率、现金流比率）趋势 为每条给发布时间、来源、摘要、潜在影响（利好/利空/中性）；同步给出股价在新闻后1个、3个交易日的超额收益。"},
        {"dim": "市场表现", "tool": "FinQuery", "query": f"截止到{p}，{company}近1个月的涨跌幅、最大回撤及成交额变化,并结合新闻/公告列出前3个可能催化/压制因素"},
        {"dim": "新闻", "tool": "StockNews", "query": f"截止到{p}，{company}的新闻"},
    ]

    queries = dim_configs or default_dims
    tool_tasks = [
        {"tool_name": q["tool"], "tool_args": {"query": q["query"]}, "dim": q["dim"]}
        for q in queries
    ]

    # ---------- 初始化结果结构 ----------
    company_record = {
        "name": company,
        "dimensions": {},
        "evidence": []
    }

    # ---------- 执行批量工具调用 ----------
    try:
        results = await tool_caller.batch_call_tools(tool_tasks)
    except Exception as e:
        print(f"❌ 批量工具调用异常: {e}")
        traceback.print_exc()
        return company_record

    # ---------- 结果解析 ----------
    for i, result in enumerate(results):
        try:
            dim = tool_tasks[i]["dim"]
            query = tool_tasks[i]["tool_args"]["query"]

            if not isinstance(result, dict):
                result = {"status": "success", "result": str(result)}

            if result.get("status") == "success" and result.get("result"):
                text = re.sub(r"\s+", " ", str(result["result"]))
                company_record["dimensions"].setdefault(dim, {})["raw_text"] = text
                company_record["evidence"].append({
                    "dimension": dim,
                    "tool": tool_tasks[i]["tool_name"],
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                company_record["dimensions"].setdefault(dim, {})["raw_text"] = "无结果"

        except Exception as e:
            print(f"❌ 解析维度 {dim} 时出错: {e}")
            traceback.print_exc()

    print(f"✅ {company} 信息采集完成，共 {len(company_record['dimensions'])} 个维度")
    return company_record

import traceback
# ============================================================
# Step 4️⃣: 构建行业知识库记录
# ============================================================
class KnowledgeBaseBuilder:
    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]):
        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# Step 5️⃣: 主控制流程
# ============================================================
async def process_image(qwen, tool_caller, kb: KnowledgeBaseBuilder, img: str):
    print(f"\n🖼️ 正在处理图像: {img}")
    try:
        # chart_info = await extract_chart_info(qwen, img)
        # industry = chart_info.get("industry", "")
        # if not industry:
        #     print("⚠️ 未识别到行业，跳过。")
        #     return

        # Step 1: 获取行业龙头公司
        # leading_companies = await get_leading_companies(qwen,tool_caller, industry,chart_info.get("period", ""))

        # Step 2: 为每家公司收集六维信息
        company_records = []
        leading_companies=["同花顺"]
        leading_companies = [
    "中芯国际", "天合光能", "百利天恒", "阿特斯", "华润微", "大全能源", "联影医疗", "寒武纪", "晶科能源", "时代电气",
    "石头科技", "沪硅产业", "金山办公", "盛美上海", "龙芯中科", "海光信息", "传音控股", "中微公司", "中国通号", "澜起科技",
    "东鹏饮料", "德业股份", "洛阳钼业", "兆易创新", "欧派家居", "福斯特", "华友钴业", "豪威集团", "万泰生物", "今世缘",
    "华勤技术", "海天味业", "合盛硅业", "药明康德", "公牛集团", "中科曙光", "中信银行", "中金公司", "中国银行", "中国核电",
    "建设银行", "中远海控", "浙商银行", "方正证券", "紫金矿业", "中煤能源", "中国中免", "中国银河", "浙商证券", "正泰电器",
    "招商轮船", "中国能建", "福莱特", "中国石油", "成都银行", "沪农商行", "光大银行", "京沪高铁", "中海油服", "中国交建",
    "星宇股份", "光大证券", "中国中车", "中国电信", "潞安环能", "中国卫通", "拓普集团", "华泰证券", "中国电建", "中国建筑",
    "邮储银行", "长城汽车", "中国人寿", "中国中冶", "上海医药", "中国太保", "中国铝业", "工商银行", "中国中铁", "兴业证券",
    "三六零", "新华保险", "交通银行", "中国人保", "中国平安", "青岛港", "农业银行", "广汽集团", "红塔证券", "上海银行",
    "陕西煤业", "国泰海通", "中国铁建", "北京银行", "兴业银行", "工业富联", "首创证券", "赛力斯", "中国化学", "中国国航"
]

        for c in leading_companies:
            # rec = await collect_company_knowledge(qwen,tool_caller, c, industry,chart_info.get("period", ""))
            rec = await collect_company_knowledge(qwen,tool_caller, c, "","2025.10.27")
            company_records.append(rec)

            # # # Step 3: 整合成行业知识记录
            # record = {
            #     # "industry": industry,
            #     # "metric": chart_info.get("metric", ""),
            #     # "period": chart_info.get("period", ""),
            #     "leading_companies": leading_companies,
            #     "companies": company_records,
            #     # "source_image": os.path.basename(img),
            #     "timestamp": datetime.now().isoformat()
            # }

            kb.append(rec)
        # print(f"✅ 行业【{industry}】知识库记录已写入。")

    except Exception as e:
        print(f"❌ 处理异常: {e}")
        traceback.print_exc() 


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/line")
    parser.add_argument("--kb_path", default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/knowledge_base2.jsonl")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    qwen = Qwen3VLClient(api_base="http://localhost:8000/v1", model="Qwen3-VL-30B-A3B-Instruct")

    async with AsyncExitStack() as stack:
        transport = await stack.enter_async_context(sse_client("http://localhost:8081/sse"))
        session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
        await session.initialize()

        tool_caller = ToolCaller(session)
        kb = KnowledgeBaseBuilder(args.kb_path)

        img_files = glob.glob(os.path.join(args.img_root, "**/*.jpg"), recursive=True)[:args.limit]
        print(f"共找到 {len(img_files)} 张图片")

        for img in img_files:
            await process_image(qwen, tool_caller, kb, img)

        print(f"\n✅ 知识库构建完成，共处理 {len(img_files)} 张图像。")
        print(f"📄 输出文件：{args.kb_path}")


if __name__ == "__main__":
    asyncio.run(main())
