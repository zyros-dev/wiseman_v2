# Copyright (c) 2026 Nick van der Merwe
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import discord
from pydantic import BaseModel, ConfigDict, Field

ACTION_ROW_TYPE = 1
BUTTON_TYPE = 2
LINK_BUTTON_STYLE = 5


class MuteDuration(StrEnum):
    ONE_HOUR = "1h"
    SIX_HOURS = "6h"
    ONE_DAY = "24h"
    SEVEN_DAYS = "7d"


@dataclass(frozen=True, slots=True)
class AlertButton:
    fingerprint: str
    duration: MuteDuration


class AlertEmbed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=4_096)
    color: int = Field(ge=0, le=0xFFFFFF)


class AlertMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(default="", max_length=2_000)
    embeds: list[AlertEmbed] = Field(min_length=1, max_length=1)
    components: list[dict[str, Any]] = Field(default_factory=list, max_length=1)


def parse_custom_id(value: str) -> AlertButton | None:
    match value.split(":"):
        case ["heimdall", "mute", fingerprint, duration] if re.fullmatch(r"[0-9a-fA-F]{6,64}", fingerprint):
            try:
                return AlertButton(fingerprint, MuteDuration(duration))
            except ValueError:
                return None
        case _:
            return None


def view_from_components(components: list[dict[str, Any]]) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    for row in components:
        if row.get("type") != ACTION_ROW_TYPE or not isinstance(row.get("components"), list):
            raise ValueError("alert components require action rows")
        for raw in row["components"]:
            if not isinstance(raw, dict) or raw.get("type") != BUTTON_TYPE or raw.get("style") == LINK_BUTTON_STYLE:
                raise ValueError("alert components require an interactive button")
            custom_id = raw.get("custom_id")
            label = raw.get("label")
            if not isinstance(custom_id, str) or parse_custom_id(custom_id) is None or not isinstance(label, str):
                raise ValueError("alert components contain an invalid mute button")
            view.add_item(discord.ui.Button(label=label, style=discord.ButtonStyle.secondary, custom_id=custom_id))
    return view
