# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from app.admission import image_tool_instruction
from app.phoenix import json_text

if TYPE_CHECKING:
    from app.clients.client_interfaces import PhoenixClient, PromptClient
    from app.models import Message


@dataclass(frozen=True, slots=True)
class PromptRequest:
    phoenix: PhoenixClient
    prompts: PromptClient
    trace: str
    trigger: Message
    current: dict[str, object]
    grammar: dict[str, object]


async def build_prompt(request: PromptRequest) -> str:
    parts = {
        "soul": await request.prompts.source("wiseman-soul"),
        "runtime": await request.prompts.source("wiseman-runtime"),
        "memories": os.getenv("WISEMAN_MEMORIES", ""),
        "context": cast("str", request.grammar["rendered"]),
        "user": "\n\n".join(
            part
            for part in (
                request.trigger.content,
                image_tool_instruction(
                    request.trigger.model_dump(mode="json"),
                    cast("list[dict[str, object]]", request.current["reply_ancestors"]),
                ),
            )
            if part
        ),
    }
    prompt = json_text(parts)
    await request.phoenix.record(request.trace, "prompt", parts=parts, final_input=prompt)
    return prompt
