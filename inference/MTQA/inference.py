#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic multi-turn VLM batch runner (open-source friendly)

Features
- Read *.jsonl samples where each line is an object:
  {
    "turns": [{"turn_id": 1, "question": "..."} , ...],
    "image_paths": ["path1", "path2"]  # or "image_path": "path"
  }
- For each sample: run multi-turn chat WITH memory (assistant responses appended to history).
- Backend pluggable: "qwen3vl" or "openai" (OpenAI-compatible chat API).
- Robust error handling, retries, and clean logs.
- No internal paths or company identifiers.

Usage
  python inference.py \
    --backend qwen3vl \
    --api-base http://localhost:8000/v1 \
    --model qwen3vl-4b-instruct \
    --input-dir ./inputs \
    --output-dir ./outputs \
    --include "*.jsonl"

  # OpenAI-compatible:
  export OPENAI_API_KEY=xxx
  python run_vlm_batch.py \
    --backend openai \
    --api-base https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --input-dir ./inputs \
    --output-dir ./outputs
"""

import os
import re
import json
import time
import glob
import argparse
import traceback
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

# optional imports (lazy)
_QWEN_AVAILABLE = False
_OPENAI_AVAILABLE = False

try:
    from qwen3vl import Qwen3VLClient  # pip install qwen3vl (your client)
    _QWEN_AVAILABLE = True
except Exception:
    pass

try:
    from openai import OpenAI  # pip install openai
    _OPENAI_AVAILABLE = True
except Exception:
    pass


# -------------------- Utilities --------------------
def log(msg: str):
    print(msg, flush=True)


def normalize_image_list(image_paths) -> List[str]:
    if image_paths is None:
        return []
    if isinstance(image_paths, str):
        return [image_paths]
    if isinstance(image_paths, list):
        return image_paths
    return []


def safe_write_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_jsonl(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_user_message(text: str) -> Dict[str, Any]:
    # OpenAI-compatible "content" supports list of parts; here we keep simple text node
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def build_assistant_message(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


# -------------------- Backends --------------------
class BaseClient:
    def chat_with_memory(
        self,
        image_paths: List[str],
        text: str,
        messages: List[Dict[str, Any]],
        timeout: float = 120.0,
    ) -> str:
        raise NotImplementedError


class Qwen3VLBackend(BaseClient):
    def __init__(self, api_base: str, model: str):
        if not _QWEN_AVAILABLE:
            raise RuntimeError("qwen3vl not available. Please `pip install qwen3vl`.")
        self.client = Qwen3VLClient(api_base=api_base, model=model)

    def chat_with_memory(
        self,
        image_paths: List[str],
        text: str,
        messages: List[Dict[str, Any]],
        timeout: float = 120.0,
    ) -> str:
        # Qwen3VLClient signature based on user's snippet
        # image can be a list of local paths or URLs.
        return self.client.chat_with_memory(image=image_paths, text=text, messages=messages)


class OpenAIBackend(BaseClient):
    """
    OpenAI-compatible backend via /chat/completions.
    For vision: expects images as URLs/base64; here we pass as 'image_url' if path looks like URL.
    If you need local file to base64, extend `paths_to_image_contents`.
    """

    def __init__(self, api_base: str, model: str):
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai not available. Please `pip install openai`.")
        self.client = OpenAI(base_url=api_base) if api_base else OpenAI()
        self.model = model

    @staticmethod
    def paths_to_image_contents(paths: List[str]) -> List[Dict[str, Any]]:
        contents = []
        for p in paths:
            # naive: treat http(s) as url; skip local files unless user extends to base64
            if isinstance(p, str) and re.match(r"^https?://", p):
                contents.append({"type": "image_url", "image_url": {"url": p}})
            # else: ignore local files by default
        return contents

    def chat_with_memory(
        self,
        image_paths: List[str],
        text: str,
        messages: List[Dict[str, Any]],
        timeout: float = 120.0,
    ) -> str:
        # Convert our message schema to OpenAI's
        # Flatten each message["content"] list into a single list for multimodal
        converted_messages = []
        for m in messages:
            role = m["role"]
            parts = m.get("content", [])
            # Convert to OpenAI content format (list of dicts)
            oai_parts = []
            for part in parts:
                if part.get("type") == "text":
                    oai_parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    oai_parts.append({"type": "image_url", "image_url": {"url": part["image_url"]["url"]}})
            if not oai_parts:
                # If empty, keep a blank text part to avoid API errors
                oai_parts.append({"type": "text", "text": ""})
            converted_messages.append({"role": role, "content": oai_parts})

        # Append current user turn: text + optional images
        img_contents = self.paths_to_image_contents(image_paths)
        user_turn = [{"type": "text", "text": text}] + img_contents
        converted_messages.append({"role": "user", "content": user_turn})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=converted_messages,
            timeout=timeout,
        )
        # robust extract
        try:
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""


def build_backend(backend: str, api_base: str, model: str) -> BaseClient:
    backend = backend.lower().strip()
    if backend == "qwen3vl":
        return Qwen3VLBackend(api_base=api_base, model=model)
    elif backend == "openai":
        return OpenAIBackend(api_base=api_base, model=model)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'qwen3vl' or 'openai'.")


# -------------------- Core Logic --------------------
def chat_with_memory(
    client: BaseClient,
    turns: List[Dict[str, Any]],
    image_paths: List[str],
    max_retries: int = 2,
    retry_sleep: float = 1.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run multi-turn conversation with memory.
    """
    messages: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    for turn in turns:
        q = turn.get("question", "")
        if not q:
            turn["model_answer"] = ""
            all_results.append({"turn_id": turn.get("turn_id", None), "answer": ""})
            continue

        messages.append(build_user_message(q))

        # retry loop
        answer = ""
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                answer = client.chat_with_memory(image_paths=image_paths, text=q, messages=messages)
                break
            except Exception as e:
                last_err = e
                traceback.print_exc()
                time.sleep(retry_sleep)

        if answer == "" and last_err is not None:
            log(f"[WARN] turn_id={turn.get('turn_id')} generation failed: {last_err}")
            answer = "（生成失败）"

        # record + extend memory
        turn["model_answer"] = answer
        all_results.append({"turn_id": turn.get("turn_id"), "answer": answer})
        messages.append(build_assistant_message(answer))

    return turns, all_results


def process_file(
    client: BaseClient,
    in_path: str,
    out_path: str,
):
    ensure_dir(os.path.dirname(out_path) or ".")
    # write to a temp then move for safety
    tmp_out = out_path + ".tmp"
    if os.path.exists(tmp_out):
        os.remove(tmp_out)

    with open(tmp_out, "w", encoding="utf-8") as fout:
        for sample in tqdm(read_jsonl(in_path), desc=f"Processing {os.path.basename(in_path)}"):
            # image fields
            image_paths = sample.get("image_paths")
            if image_paths is None and "image_path" in sample:
                image_paths = sample["image_path"]

            image_paths = normalize_image_list(image_paths)

            # optional: filter invalid files (for local pipelines)
            # valid = (len(image_paths) > 0) and any(os.path.exists(p) for p in image_paths)
            # if not valid:
            #     continue

            turns = sample.get("turns", [])
            turns, results = chat_with_memory(client, turns, image_paths)
            sample["turns"] = turns
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # atomic move
    os.replace(tmp_out, out_path)
    log(f"✅ Done → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generic multi-turn VLM batch runner")
    parser.add_argument("--backend", type=str, default=os.environ.get("VLM_BACKEND", "qwen3vl"),
                        choices=["qwen3vl", "openai"], help="Backend type.")
    parser.add_argument("--api-base", type=str, default=os.environ.get("VLM_API_BASE", "http://localhost:8000/v1"),
                        help="API base URL (qwen3vl server or OpenAI-compatible endpoint).")
    parser.add_argument("--model", type=str, default=os.environ.get("VLM_MODEL", "qwen3vl"),
                        help="Model name (e.g., qwen3vl-4b-instruct or gpt-4o-mini).")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input *.jsonl files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--include", type=str, default="*.jsonl", help="Glob pattern for input files.")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per turn.")
    parser.add_argument("--retry-sleep", type=float, default=1.5, help="Sleep seconds between retries.")
    args = parser.parse_args()

    log(f"Backend: {args.backend}")
    log(f"API Base: {args.api_base}")
    log(f"Model:   {args.model}")

    ensure_dir(args.output_dir)
    client = build_backend(args.backend, args.api_base, args.model)

    files = sorted(glob.glob(os.path.join(args.input_dir, args.include)))
    if not files:
        log("No input files found.")
        return

    log("🔍 Found input files:")
    for f in files:
        log(f" - {f}")

    for in_path in files:
        base = os.path.basename(in_path)
        out_path = os.path.join(args.output_dir, base.replace(".jsonl", "_vlm.jsonl"))
        log(f"\n🚀 Processing: {in_path}")
        log(f"📤 Output:     {out_path}")
        process_file(client, in_path, out_path)

    log("🎉 All done!")


if __name__ == "__main__":
    main()

