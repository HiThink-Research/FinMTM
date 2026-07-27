"""Shared OpenAI-compatible multimodal client."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge.Agent.qwen3vl import Qwen3VLClient

__all__ = ["Qwen3VLClient"]
