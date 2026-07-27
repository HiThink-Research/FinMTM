"""Evaluator-client factory."""

from .qwen3vl import Qwen3VLClient


def build_client(kind: str, api_base: str = None, model: str = None):
    """Build a client for an OpenAI-compatible multimodal endpoint."""

    kind = (kind or "").lower().strip()
    if kind not in {"qwen", "openai"}:
        raise ValueError(
            "Supported public adapters are 'qwen' and 'openai'; both use an "
            "OpenAI-compatible multimodal endpoint."
        )
    return Qwen3VLClient(api_base=api_base, model=model)
