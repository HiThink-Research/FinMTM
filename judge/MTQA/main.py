# eval_runner/main.py
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import asyncio

import config
from io_utils import ensure_dir
from client.factory import build_client
from evaluator import run_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=config.DEFAULT_DIRS,
        help="包含 L*_with_id_vlm.jsonl 的目录列表（空格分隔）",
    )
    parser.add_argument("--pattern", default=config.DEFAULT_PATTERN, help="输入文件 glob pattern")
    parser.add_argument("--out_subdir", default=config.DEFAULT_OUT_SUBDIR, help="输出子目录名")
    parser.add_argument("--client", default="qwen", choices=["qwen", "gemini"], help="评测客户端")
    parser.add_argument("--api_base", default=config.DEFAULT_API_BASE, help="Qwen API base")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Qwen model name")

    args = parser.parse_args()

    eval_client = build_client(args.client, api_base=args.api_base, model=args.model)

    for base_dir in args.dirs:
        if not os.path.isdir(base_dir):
            print(f"⚠️ 目录不存在，跳过: {base_dir}")
            continue

        pattern = os.path.join(base_dir, args.pattern)
        input_files = sorted(glob.glob(pattern))
        if not input_files:
            print(f"⚠️ 未找到输入文件: {pattern}")
            continue

        print(f"\n📂 目录: {base_dir}")
        out_dir = os.path.join(base_dir, args.out_subdir)
        ensure_dir(os.path.join(out_dir, "dummy.txt"))

        for inp in input_files:
            fname = os.path.basename(inp)
            outp = os.path.join(out_dir, fname.replace("_with_id_vlm.jsonl", "_score.jsonl"))
            print(f"=== Running: {inp} -> {outp} ===")
            asyncio.run(run_file(inp, outp, eval_client))

    print("\n全部任务完成！")


if __name__ == "__main__":
    main()
