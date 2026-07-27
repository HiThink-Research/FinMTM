"""Public client adapter retained for compatibility with early imports."""

from .qwen3vl import Qwen3VLClient

OpenAICompatibleClient = Qwen3VLClient

__all__ = ["OpenAICompatibleClient", "Qwen3VLClient"]
