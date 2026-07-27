"""OpenAI-compatible multimodal client used by the public pipeline."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


class Qwen3VLClient:
    """Small adapter for local or hosted OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen3-VL-30B-A3B-Instruct",
        timeout: int = 120,
        api_key: str | None = None,
    ):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(api_key=resolved_key, base_url=api_base)
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _to_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _image_part(image_path_or_url: str) -> dict[str, Any] | None:
        if not image_path_or_url:
            return None
        if image_path_or_url.startswith(("http://", "https://", "data:")):
            url = image_path_or_url
        else:
            path = Path(image_path_or_url)
            if not path.exists():
                raise FileNotFoundError(f"image not found: {path}")
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            url = f"data:{mime_type};base64,{encoded}"
        return {"type": "image_url", "image_url": {"url": url}}

    def chat(
        self,
        image=None,
        text=None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        content = []
        images = [image] if isinstance(image, str) else (image or [])
        for item in images:
            part = self._image_part(item)
            if part:
                content.append(part)
        content.append({"type": "text", "text": self._to_str(text)})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        answer = (response.choices[0].message.content or "").strip()
        return answer.split("</think>", 1)[-1].strip()

    def chat_with_memory(
        self,
        text=None,
        image=None,
        messages=None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Generate one turn from prior messages without mutating the caller."""

        content = []
        images = [image] if isinstance(image, str) else (image or [])
        for item in images:
            part = self._image_part(item)
            if part:
                content.append(part)
        content.append({"type": "text", "text": self._to_str(text)})
        request_messages = list(messages or []) + [
            {"role": "user", "content": content}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=request_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        answer = (response.choices[0].message.content or "").strip()
        return answer.split("</think>", 1)[-1].strip()

    async def image2text(
        self,
        instruction,
        image=None,
        temperature: float = 0.0,
        pbar=None,
        output_file=None,
    ) -> str:
        del pbar, output_file
        return await asyncio.to_thread(
            self.chat,
            image=image,
            text=instruction,
            temperature=temperature,
        )
