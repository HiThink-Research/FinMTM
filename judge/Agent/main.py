#!/usr/bin/env python3
"""CLI for financial-agent inference and evaluation."""

from __future__ import annotations

import argparse
import asyncio
import datetime
from pathlib import Path

from . import config
from .judge import run_evaluation_pipeline
from .logging import setup_logger
from .utils import safe_name


logger = setup_logger()


async def run_pipeline(
    mode: str,
    input_path: str,
    output_dir: Path,
    model_client,
    judge_client,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = (
        Path(input_path) if mode == "eval" else output_dir / "trace.jsonl"
    )
    score_path = output_dir / "score.jsonl"

    if mode in ("all", "inference"):
        from .inference import run_inference_pipeline

        await run_inference_pipeline(
            input_path,
            str(trace_path),
            llm=model_client,
            logger=logger,
        )
    if mode in ("all", "eval"):
        if not trace_path.exists():
            raise FileNotFoundError(
                f"trace file is required for evaluation: {trace_path}"
            )
        await run_evaluation_pipeline(
            str(trace_path),
            str(score_path),
            judge_client=judge_client,
            logger=logger,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinMTM financial-agent inference and evaluation"
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "inference", "eval"],
    )
    parser.add_argument("--input", required=True, help="Input agent JSONL")
    parser.add_argument("--out-root", default="outputs/agent")
    parser.add_argument("--with-timestamp", action="store_true")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--mcp-url", default=config.MCP_SERVER_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from .model import build_model

    config.MCP_SERVER_URL = args.mcp_url
    model_client = build_model(
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
    )
    judge_client = build_model(
        api_base=args.judge_api_base or args.api_base,
        model=args.judge_model or args.model,
        api_key=args.judge_api_key or args.api_key,
    )

    timestamp = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.with_timestamp
        else ""
    )
    tag = safe_name(args.model)
    output_name = f"{tag}_{timestamp}" if timestamp else tag
    output_dir = Path(args.out_root) / output_name
    asyncio.run(
        run_pipeline(
            args.mode,
            args.input,
            output_dir,
            model_client,
            judge_client,
        )
    )


if __name__ == "__main__":
    main()
