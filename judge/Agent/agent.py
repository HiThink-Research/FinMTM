#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional

from .config import MAX_ITER
from .mcp_client import call_tool_with_retry
from .utils import safe_parse_json
from .prompts import agent_sys_prompt

class MultiRoundAgent:
    """
    纯粹负责：多轮规划 -> 调工具 -> 累积反馈 -> 终止/强制总结
    """

    def __init__(self, get_session: Callable[..., Any], llm_client: Any, logger):
        # get_session(reconnect=False/True) -> ClientSession
        self.get_session = get_session
        self.llm = llm_client
        self.logger = logger

    async def run_multiround(self, query: str, image_url: str = "") -> Dict[str, Any]:
        rounds: List[Dict[str, Any]] = []
        all_tool_calls: List[Dict[str, str]] = []
        all_tool_results: List[Dict[str, Any]] = []
        all_thoughts: List[str] = []
        all_visual_observations: List[str] = []

        final_answer: Optional[str] = None
        visual_observation: str = ""
        tool_feedback: str = ""

        # 首轮消息
        prompt = agent_sys_prompt(query)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image_url": image_url} if image_url else {"type": "text", "text": ""}
            ]
        }]
        messages[0]["content"] = [c for c in messages[0]["content"] if c.get("text") or c.get("image_url")]

        for round_id in range(1, MAX_ITER + 1):
            self.logger.info(f"🌀 [Round {round_id}] 模型规划中...")
            _ = time.time()

            # 1) 模型生成规划
            try:
                prompt_text = json.dumps(messages, ensure_ascii=False)
                resp = await self.llm.image2text(
                    instruction=prompt_text,
                    image=image_url if image_url else None,
                    temperature=0.0,
                    pbar=None,
                    output_file=None,
                )
                if isinstance(resp, dict):
                    resp = json.dumps(resp, ensure_ascii=False)
                response_text = str(resp)
            except Exception as e:
                self.logger.error(f"❌ 模型请求失败: {e}")
                break

            plan_rec: Dict[str, Any] = {"round": round_id, "plan": response_text, "tool_calls": []}

            # 2) 结束条件
            if "<FINISHED>" in response_text:
                final_answer = response_text
                rounds.append(plan_rec)
                self.logger.info("🏁 检测到 <FINISHED>，结束规划。")
                break

            # 3) 解析 JSON
            parsed_json = safe_parse_json(response_text)
            action_trace = parsed_json.get("ActionTrace", []) or []
            visual_obs = parsed_json.get("VisualObservation", [])
            thought = parsed_json.get("Thought", "")

            if thought:
                all_thoughts.append(str(thought).strip())

            if visual_obs:
                if isinstance(visual_obs, list):
                    current_visual_observation = "\n".join(
                        [str(x) for x in visual_obs]
                    )
                else:
                    current_visual_observation = str(visual_obs)
                all_visual_observations.append(
                    f"Round {round_id}: {current_visual_observation}"
                )
                visual_observation = "\n".join(all_visual_observations)
                plan_rec["visual_observation"] = current_visual_observation

            # 4) 执行工具
            current_round_feedback = ""
            for call in action_trace:
                tool_name = (call or {}).get("tool")
                tool_query = (call or {}).get("query", "")

                if not tool_name:
                    continue

                simple_call = {"tool": tool_name, "query": tool_query}
                plan_rec["tool_calls"].append(simple_call)
                all_tool_calls.append(simple_call)

                try:
                    self.logger.info(f"⚙️ 调用工具: {tool_name}({tool_query})")
                    result = await call_tool_with_retry(
                        self.get_session,
                        tool_name,
                        tool_query,
                        timeout=90,
                        retries=3,
                    )
                    parsed_res = self._parse_mcp_tool_result(result)
                    current_round_feedback += f"{tool_name}: {parsed_res}\n"
                    tool_result = {
                        "round": round_id,
                        "tool": tool_name,
                        "query": tool_query,
                        "status": "ok",
                        "result": parsed_res,
                    }
                    plan_rec.setdefault("tool_results", []).append(tool_result)
                    all_tool_results.append(tool_result)
                    self.logger.info(f"✅ 工具返回: {parsed_res[:60]}...")
                except asyncio.TimeoutError:
                    self.logger.error(f"❌ 工具 {tool_name} 超时")
                    current_round_feedback += f"{tool_name}: Error (Timeout)\n"
                    tool_result = {
                        "round": round_id,
                        "tool": tool_name,
                        "query": tool_query,
                        "status": "timeout",
                        "result": "",
                    }
                    plan_rec.setdefault("tool_results", []).append(tool_result)
                    all_tool_results.append(tool_result)
                except Exception as e:
                    self.logger.error(f"❌ 工具 {tool_name} 失败: {e}")
                    current_round_feedback += f"{tool_name}: Error ({str(e)})\n"
                    tool_result = {
                        "round": round_id,
                        "tool": tool_name,
                        "query": tool_query,
                        "status": "error",
                        "result": str(e),
                    }
                    plan_rec.setdefault("tool_results", []).append(tool_result)
                    all_tool_results.append(tool_result)

            tool_feedback += current_round_feedback
            plan_rec["tool_feedback"] = current_round_feedback
            rounds.append(plan_rec)

            # 5) 下一轮输入
            next_sys_prompt = agent_sys_prompt(query)
            composed_text = (
                next_sys_prompt
                + f"\n上一轮思考:\n{thought}\n"
                + f"\n图像观察 (VisualObservation):\n{visual_observation}\n"
                + f"\n工具反馈 (ToolFeedback):\n{tool_feedback}\n"
                + "请综合以上内容继续规划，不要重复调用工具。如果信息足够，请输出 <FINISHED> 并给出结论。"
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": composed_text},
                    {"type": "image", "image_url": image_url} if image_url else {"type": "text", "text": ""}
                ]
            }]
            messages[0]["content"] = [c for c in messages[0]["content"] if c.get("text") or c.get("image_url")]

        # 强制总结
        if not final_answer:
            self.logger.warning("⚠️ 达到最大轮次，触发强制总结。")
            summary_prompt = (
                "请根据以下视觉观察和工具反馈，总结出最终答案。\n"
                f"问题：{query}\n\n"
                f"视觉观察：\n{visual_observation}\n\n"
                f"工具反馈：\n{tool_feedback}\n\n"
                "请直接给出结论，并在最后一行输出 <FINISHED>。"
            )
            try:
                summary_msgs = [{"role": "user", "content": [{"type": "text", "text": summary_prompt}]}]
                if image_url:
                    summary_msgs[0]["content"].append({"type": "image", "image_url": image_url})

                prompt_text = json.dumps(summary_msgs, ensure_ascii=False)
                resp = await self.llm.image2text(
                    instruction=prompt_text,
                    image=image_url if image_url else None,
                    temperature=0.0,
                    pbar=None,
                    output_file=None,
                )
                if isinstance(resp, dict):
                    resp = json.dumps(resp, ensure_ascii=False)
                final_answer = str(resp)
            except Exception as e:
                self.logger.error(f"❌ 强制总结失败: {e}")
                final_answer = "模型未能生成最终答案。"

        summary_thought = "\n\n---\n\n".join([f"Round {i+1} Thought:\n{t}" for i, t in enumerate(all_thoughts)])

        return {
            "final_answer": final_answer,
            "rounds": rounds,
            "thought": summary_thought,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "tool_feedback": tool_feedback,
            "visual_observation": visual_observation,
        }

    def _parse_mcp_tool_result(self, result: Any) -> str:
        """
        解析 MCP 返回的复杂结构（保持你原来的逻辑）
        """
        if hasattr(result, "content") and result.content and getattr(result.content[0], "text", None):
            raw = result.content[0].text
            try:
                inner = json.loads(json.loads(raw))
                if "response" in inner and "result" in inner["response"]:
                    return inner["response"]["result"][0]["text"]
                return raw
            except Exception:
                return raw
        return str(result)
