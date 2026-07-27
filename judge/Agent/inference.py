#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import traceback
from contextlib import AsyncExitStack
from typing import Any

import aiofiles

from .mcp_client import connect_mcp
from .agent import MultiRoundAgent

async def run_inference_pipeline(input_file: str, output_file: str, llm: Any, logger):
    if os.path.exists(output_file):
        logger.warning(f"输出文件 {output_file} 已存在，将被覆盖。")

    async with AsyncExitStack() as stack:
        _session = None

        async def get_session(reconnect=False):
            nonlocal _session
            if _session is None or reconnect:
                _session = await connect_mcp(stack)
            return _session

        try:
            await get_session()
            logger.info("✅ MCP Server 连接成功")
        except Exception as e:
            logger.error(f"❌ MCP 连接失败: {e}")
            return

        agent = MultiRoundAgent(get_session=get_session, llm_client=llm, logger=logger)

        logger.info(f"📂 读取输入文件: {input_file}")
        async with aiofiles.open(input_file, "r", encoding="utf-8") as fin:
            lines = await fin.readlines()

        logger.info(f"🚀 开始批量处理 {len(lines)} 条样本...")

        async with aiofiles.open(output_file, "w", encoding="utf-8") as fout:
            for idx, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    logger.error(f"❌ Line {idx} JSON解析失败")
                    continue

                turn = item.get("turn", {})
                question = turn.get("question") or item.get("question", "")
                image_path = item.get("image_path", "")

                ref_thought = turn.get("Thought", {})
                ref_obs = turn.get("VisualObservation", {})
                ref_action = turn.get("ActionTrace", [])
                ref_tool_results = (
                    turn.get("ToolResults")
                    or turn.get("tool_results")
                    or item.get("reference_tool_results")
                    or []
                )
                ref_tool_feedback = (
                    turn.get("ToolFeedback")
                    or turn.get("tool_feedback")
                    or item.get("reference_tool_feedback")
                    or ""
                )
                raw_ref_gold = turn.get("gold_answer", "")
                if isinstance(raw_ref_gold, dict):
                    ref_gold = raw_ref_gold.get("final_conclusion", "")
                else:
                    ref_gold = str(raw_ref_gold or "")
                fallback_gold = item.get("gold_answer", "")
                if isinstance(fallback_gold, dict):
                    fallback_gold = fallback_gold.get("final_conclusion", "")
                ref_gold = ref_gold or str(fallback_gold or "")

                logger.info(f"🧩 Processing [{idx}/{len(lines)}]: {question[:30]}...")

                try:
                    res = await agent.run_multiround(question, image_url=image_path)
                except Exception as e:
                    logger.error(f"❌ 推理异常: {e}")
                    traceback.print_exc()
                    res = {
                        "final_answer": "Error",
                        "rounds": [],
                        "tool_calls": [],
                        "tool_results": [],
                        "tool_feedback": "",
                        "thought": "",
                        "visual_observation": "",
                    }

                record = {
                    "sample_id": item.get("sample_id", idx),
                    "question": question,
                    "image_path": image_path,

                    "model_final_answer": res.get("final_answer", ""),
                    "reference_gold_answer": ref_gold,

                    "model_visual_observation": res.get("visual_observation", ""),
                    "reference_visual_observation": ref_obs,

                    "model_tool_calls": res.get("tool_calls", []),
                    "reference_tool_calls": ref_action,
                    "model_tool_results": res.get("tool_results", []),
                    "reference_tool_results": ref_tool_results,
                    "model_tool_feedback": res.get("tool_feedback", ""),
                    "reference_tool_feedback": ref_tool_feedback,

                    "model_thought": res.get("thought", ""),
                    "reference_thought": ref_thought,
                }

                await fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(f"💾 Sample {idx} Saved.")

    logger.info(f"🎉 Stage 1 推理完成，结果保存在: {output_file}")
