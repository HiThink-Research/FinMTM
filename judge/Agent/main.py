#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import datetime
from pathlib import Path

from .config import DEFAULT_INPUT_FILE
from .logging import setup_logger
from .utils import safe_name
from .model import build_models
from .inference import run_inference_pipeline
from .judge import run_evaluation_pipeline

logger = setup_logger()

async def run_one_model(mode: str, input_path: str, out_dir: Path, llm_obj, judge_obj):
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    score_path = out_dir / "score.jsonl"

    if mode in ("all", "inference"):
        await run_inference_pipeline(input_path, str(trace_path), llm=llm_obj, logger=logger)

    if mode in ("all", "eval"):
        if not trace_path.exists():
            logger.error(f"❌ 找不到推理结果文件 {trace_path}，无法进行评分。")
            return
        await run_evaluation_pipeline(str(trace_path), str(score_path), judge_client=judge_obj, logger=logger)

def cli():
    parser = argparse.ArgumentParser(description="Agent Inference & Evaluation Pipeline (Multi-Model)")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "inference", "eval"], help="运行模式")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE, help="原始输入数据")
    parser.add_argument("--out_root", type=str, default="/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/eval/agent_val/out", help="输出根目录 (每个模型一个子目录)")
    parser.add_argument("--with_ts", action="store_true", help="输出目录添加时间戳")
    parser.add_argument("--models", type=str, default="gpt5,gemini", help="只跑指定模型key，逗号分隔；空=全跑")
    return parser.parse_args()

def main():
    args = cli()
    models = build_models()

    if args.models.strip():
        allow = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in models if m["key"] in allow]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if args.with_ts else ""
    out_root = Path(args.out_root)

    # judge 默认用 gemini（你也可以改成跟随某个 key）
    # 这里简单做成：固定 GEMINIClient() 做 judge
    from api2 import GEMINIClient
    judge_obj = GEMINIClient()

    for m in models:
        tag = safe_name(m["key"])
        out_dir = out_root / (f"{tag}_{ts}" if ts else tag)
        logger.info(f"🚀 Running model={m.get('display', m['key'])}, out_dir={out_dir}")

        # 每个模型一个 event loop：最稳
        asyncio.run(run_one_model(args.mode, args.input, out_dir, m["obj"], judge_obj))

    logger.info("✅ All done.")

if __name__ == "__main__":
    main()
