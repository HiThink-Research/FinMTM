#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import os
import re
import time
import traceback
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import aiofiles
from mcp import ClientSession
from mcp.client.sse import sse_client
from qwen3vl import Qwen3VLClient  # 你的封装类


# ----------------------------
# 全局配置
# ----------------------------
VLLM_MODEL_NAME = os.environ.get("CURRENT_VLM_MODEL", "MIMO7BVL")

SERVER_URL = "http://localhost:8081/sse"

INPUT_FILE = "/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/generate_qes_ans/eval/agent.jsonl"

OUTPUT_DIR = os.path.join(
    "/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/eval/agent_val/out",
    VLLM_MODEL_NAME,
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "agent_results.jsonl")

MAX_ITER = 4


# ----------------------------
# 日志配置
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------
# 工具函数
# ----------------------------
def ensure_dir_for_file(path: str) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def extract_json_block(text: str) -> Optional[str]:
    """
    从文本中提取最外层 { ... } 的 JSON 块（尽量贪婪匹配）。
    """
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else None


def parse_model_json(text: str) -> Dict[str, Any]:
    """
    尝试从模型输出中解析 JSON。
    允许处理中文引号、单引号等常见错误。
    """
    block = extract_json_block(text)
    if not block:
        return {}

    # 先直接解析
    try:
        return json.loads(block)
    except Exception:
        pass

    # 再做一次轻量修复
    fixed = (
        block.replace("“", '"')
        .replace("”", '"')
        .replace("‘", '"')
        .replace("’", '"')
        .replace("'", '"')
    )
    try:
        return json.loads(fixed)
    except Exception:
        logger.debug("Failed to parse JSON block. Raw snippet: %s", fixed[:300])
        return {}


def parse_double_brace_json(text: str) -> Optional[Dict[str, Any]]:
    """
    解析模型输出中可能出现的 {{ }} 包裹 JSON 内容。
    若不存在 {{}}，则退化为解析普通 {} JSON。
    """
    if not text:
        return None

    # 优先匹配 {{ ... }}
    m = re.search(r"\{\{.*\}\}", text, re.DOTALL)
    if m:
        content = m.group(0).replace("{{", "{").replace("}}", "}")
    else:
        # 退化为普通 {}
        content = extract_json_block(text)
        if not content:
            return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.debug("Double-brace JSON parse failed. Snippet: %s", content[:300])
        return None


# ----------------------------
# 核心评测类
# ----------------------------
class QwenMultiRoundEvaluator:
    def __init__(self, session: ClientSession, qwen_client: Qwen3VLClient):
        self.session = session
        self.qwen = qwen_client

    async def run_multiround(self, query: str, image_url: str = "") -> Dict[str, Any]:
        """
        Qwen3 多轮规划与 MCP 工具调用。
        """
        rounds: List[Dict[str, Any]] = []
        all_tool_calls: List[Dict[str, str]] = []  # 只保存调用计划（不含结果）
        all_thoughts: List[str] = []

        final_answer: Optional[str] = None
        visual_observation: str = ""
        tool_feedback_all: str = ""

        # 首轮提示
        prompt = self._get_prompt(query)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        for round_id in range(1, MAX_ITER + 1):
            logger.info("Round %d: start planning", round_id)
            start_time = time.time()

            # 调用模型规划
            try:
                response_text = self.qwen.chat(
                    image=image_url,
                    text=json.dumps(messages, ensure_ascii=False),
                )
            except Exception as e:
                logger.error("Round %d: model call failed: %s", round_id, e)
                break

            logger.info("Round %d: raw plan output:\n%s", round_id, response_text)

            plan_rec: Dict[str, Any] = {
                "round": round_id,
                "plan": response_text,
                "tool_calls": [],
            }

            # 是否结束
            if "<FINISHED>" in response_text:
                final_answer = response_text
                rounds.append(plan_rec)
                logger.info("Round %d: finished token detected", round_id)
                break

            # 解析规划 JSON
            parsed_json = parse_model_json(response_text)
            action_trace = parsed_json.get("ActionTrace", []) or []
            visual_obs = parsed_json.get("VisualObservation", []) or []
            thought = parsed_json.get("Thought", "") or ""

            if thought.strip():
                all_thoughts.append(thought.strip())

            # 记录视觉观察
            if isinstance(visual_obs, list) and visual_obs:
                visual_observation = "\n".join([str(x) for x in visual_obs])
                plan_rec["visual_observation"] = visual_observation

            # 保存工具调用信息
            for call in action_trace:
                simple_call = {
                    "tool": str(call.get("tool", "")),
                    "query": str(call.get("query", "")),
                }
                plan_rec["tool_calls"].append(simple_call)
                all_tool_calls.append(simple_call)

            # 实际调用工具（本轮反馈单独累计，避免越滚越长）
            round_tool_feedback = ""
            for call in action_trace:
                tool_name = str(call.get("tool", "")).strip()
                tool_query = str(call.get("query", "")).strip()
                if not tool_name:
                    continue

                try:
                    logger.info("Round %d: call tool %s | query=%s", round_id, tool_name, tool_query)
                    result = await asyncio.wait_for(
                        self.session.call_tool(tool_name, {"query": tool_query}),
                        timeout=30,
                    )
                    parsed = self._parse_tool_result(result)
                    round_tool_feedback += f"{tool_name}: {parsed}\n"
                    logger.info("Round %d: tool %s returned (snippet): %s", round_id, tool_name, str(parsed)[:160])
                except asyncio.TimeoutError:
                    logger.error("Round %d: tool %s timeout", round_id, tool_name)
                except Exception as e:
                    logger.error("Round %d: tool %s failed: %s", round_id, tool_name, e)

            if round_tool_feedback:
                tool_feedback_all += round_tool_feedback

            # 构造下一轮输入
            next_prompt = self._get_prompt(query)
            composed = (
                next_prompt
                + f"\n上一轮思考:\n{thought}\n"
                + f"\n图像观察 (VisualObservation):\n{visual_observation}\n"
                + f"\n工具反馈 (ToolFeedback):\n{round_tool_feedback}\n"
                + "请综合以上内容继续规划，不要重复调用工具。"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": composed},
                        {"type": "image", "image_url": image_url},
                    ],
                }
            ]

            rounds.append(plan_rec)
            logger.info(
                "Round %d: done, elapsed=%.2fs",
                round_id,
                time.time() - start_time,
            )

        # 若未生成 <FINISHED>，强制总结
        if not final_answer:
            logger.warning("Model did not output <FINISHED>. Trigger final summary.")
            summary_prompt = (
                "请根据以下视觉观察和工具反馈，总结出最终答案。\n"
                f"问题：{query}\n\n"
                f"视觉观察：\n{visual_observation}\n\n"
                f"工具反馈：\n{tool_feedback_all}\n\n"
                "请直接给出结论，并在最后一行输出 <FINISHED>。"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": summary_prompt},
                        {"type": "image", "image_url": image_url},
                    ],
                }
            ]

            try:
                final_answer = self.qwen.chat(
                    image=image_url,
                    text=json.dumps(messages, ensure_ascii=False),
                )
                logger.info("Final answer generated:\n%s", final_answer)
            except Exception as e:
                logger.error("Final answer generation failed: %s", e)
                final_answer = "模型未能生成最终答案。"

        summary_text = "\n\n---\n\n".join(
            [f"Round {i+1} Thought:\n{t}" for i, t in enumerate(all_thoughts)]
        )

        return {
            "final_answer": final_answer,
            "rounds": rounds,
            "thought": summary_text,
            "tool_calls": all_tool_calls,
            "visual_observation": visual_observation,
        }

    def _get_prompt(self, query: str) -> str:
        return f"""
你是一名金融多轮分析 Agent，需要逐步规划 MCP 工具调用，并结合图像中的信息完成任务。问题是：{query}
请先根据工具识别该公司是哪一家。

---

可使用的工具：
1) FinQuery
   用途：金融数据查询工具，用于获取目标标的的多类型金融数据（宏观、财务、行情、交易、基金、指数、期货等）。
   输入要求：需指定具体金融指标或带时间的指标；可输入多个指标筛选；指标过多请拆分查询。
   示例：FinQuery: 查询苹果公司近5天股价以及涨跌幅

2) Search
   用途：通用搜索工具，用于检索新闻、概念或知识信息。
   输入要求：自然语言短语或关键词（不超过5个）。
   示例：Search: 苹果公司近期新闻

3) StockNews
   用途：财经新闻检索工具，用于获取股票、指数、概念或大宗商品的最新资讯。
   使用规范：查询时在输入中明确添加属性后缀（如“新闻/资讯/消息”），可指定日期或情绪（利好/利空）。
   示例：StockNews: 查询宁德时代近期新闻（利好）

4) ReportQuery
   用途：研报查询工具，用于检索个股或行业的研报观点与总结。
   使用规范：在输入中添加属性后缀（如“研报”）。
   示例：ReportQuery: 查询宁德时代2024年研报观点

---

内容与生成要求：
1) 表层分析为主：只需简要推理与任务分解，不需要深度金融分析。
2) Thought 中不要出现具体工具名（如 FinQuery、Search 等），可使用自然描述（如“我将查询相关数据”“我计划检索近期新闻”）。
3) Thought 要简洁，不要过长。
4) 若多次查询仍无法获得结果，可选择结束计划并直接进入回答阶段。
5) ActionTrace 中的工具输入必须包含明确对象，不可出现代词或引用（如“该公司”“上文”）。
6) 一次输出中工具调用不超过 5 个。
7) 通常 2–3 轮规划即可结束任务。
8) 输出内容必须为中文。

---

输出格式要求：
请仅输出一个 JSON 对象，包含以下 4 个字段：

{{
  "Thought": "你对任务的总体思考、推理步骤。",
  "VisualObservation": ["你从图像中观察或识别到的金融关键信息（如数值、时间、指标）"],
  "ActionTrace": [
    {{"tool": "FinQuery", "query": "查询宁德时代2024年净利润"}},
    {{"tool": "Search", "query": "查询宁德时代近期政策消息"}}
  ]
}}

若任务完成，请输出 <FINISHED> 并总结最终结论。
""".strip()

    def _parse_tool_result(self, result: Any) -> str:
        """
        MCP 工具返回解析。
        """
        if hasattr(result, "content") and result.content:
            first = result.content[0]
            raw = getattr(first, "text", None)
            if raw:
                try:
                    # 你的工具返回可能是双层 JSON 字符串
                    return json.loads(json.loads(raw))["response"]["result"][0]["text"]
                except Exception:
                    return raw
        return str(result)


# ----------------------------
# 主执行入口（批量）
# ----------------------------
async def main():
    ensure_dir_for_file(OUTPUT_FILE)

    async with AsyncExitStack() as stack:
        transport = await stack.enter_async_context(sse_client(SERVER_URL))
        session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
        await session.initialize()

        qwen = Qwen3VLClient(api_base="http://localhost:8000/v1", model="Qwen3-VL-30B-A3B-Instruct")
        evaluator = QwenMultiRoundEvaluator(session, qwen)

        async with aiofiles.open(INPUT_FILE, "r", encoding="utf-8") as fin:
            lines = await fin.readlines()

        logger.info("Loaded %d samples, start evaluation.", len(lines))

        async with aiofiles.open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
            for idx, line in enumerate(lines, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.error("Line %d: JSON parse failed, skipped.", idx)
                    continue

                turn = item.get("turn", {}) or {}
                question = turn.get("question2", "")
                image_path = item.get("image_path", "")

                ref_thought = turn.get("Thought", {})
                ref_visual_obs = turn.get("VisualObservation", {})
                ref_action_trace = turn.get("ActionTrace", [])

                gold_answer = (turn.get("gold_answer", {}) or {}).get("final_conclusion", {})

                if not question:
                    logger.warning("Sample %d: empty question, skipped.", idx)
                    continue

                logger.info("Sample %d: %s", idx, question[:120])

                try:
                    result = await evaluator.run_multiround(question, image_url=image_path)
                except Exception as e:
                    logger.error("Sample %d: evaluation failed: %s", idx, e)
                    logger.debug("Traceback:\n%s", traceback.format_exc())
                    continue

                output_record = {
                    "sample_id": idx,
                    "question": question,
                    "image_path": image_path,

                    "model_final_answer": result.get("final_answer", ""),
                    "reference_gold_answer": gold_answer,

                    "model_visual_observation": result.get("visual_observation", ""),
                    "reference_visual_observation": ref_visual_obs,

                    "model_tool_calls": result.get("tool_calls", []),
                    "reference_tool_calls": ref_action_trace,

                    "model_thought": result.get("thought", ""),
                    "reference_thought": ref_thought,

                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                await fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                logger.info("Sample %d: done, written to output.", idx)

        logger.info("All samples finished. Output: %s", OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())
