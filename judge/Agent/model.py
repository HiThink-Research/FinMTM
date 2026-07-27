#!/usr/bin/env python3
"""Model-client construction without private repository dependencies."""

from __future__ import annotations

from typing import Any

from .qwen3vl import Qwen3VLClient


def build_model(
    *,
    api_base: str,
    model: str,
    api_key: str | None = None,
) -> Any:
    """Build an OpenAI-compatible multimodal client."""

    return Qwen3VLClient(
        api_base=api_base,
        model=model,
        api_key=api_key,
    )
