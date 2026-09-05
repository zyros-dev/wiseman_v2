# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from pydantic import TypeAdapter

from app.types import JsonObject

if TYPE_CHECKING:
    from app.clients.client_interfaces import ClientSettings, PromptClient


class OpenRouter:
    def __init__(self, settings: ClientSettings, prompts: PromptClient, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings, self.prompts = settings, prompts
        self.http = httpx.AsyncClient(
            base_url=settings.provider_url,
            headers={"authorization": f"Bearer {settings.provider_key}"},
            timeout=httpx.Timeout(300, connect=15),
            transport=transport,
        )

    async def responses(self, payload: JsonObject) -> httpx.Response:
        if not self.settings.provider_key:
            raise RuntimeError("OpenRouter is not configured")
        request = self.http.build_request("POST", "/api/v1/responses", json=payload)
        return await self.http.send(request, stream=True)

    async def describe(self, url: str, question: str) -> dict[str, object]:
        template = SandboxedEnvironment(autoescape=False, undefined=StrictUndefined).from_string(await self.prompts.source("vision-question"))
        response = await self.http.post(
            "/api/v1/chat/completions",
            json={
                "model": self.settings.vision_model,
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": template.render(question=question)},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
            },
        )
        response.raise_for_status()
        data = TypeAdapter(JsonObject).validate_python(response.json())
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("Vision provider returned no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TypeError("Vision provider returned no text")
        usage = data.get("usage")
        return {
            "text": message["content"],
            "model": data.get("model", self.settings.vision_model),
            "usage": usage,
            "cost": usage.get("cost") if isinstance(usage, dict) else data.get("cost"),
            "question": question or None,
        }

    async def close(self) -> None:
        await self.http.aclose()
