# Copyright (c) 2026 Nick van der Merwe

from typing import TypedDict

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]


type StateData = JsonObject


class EngineResult(TypedDict, total=False):
    trace: str
    status: str
    kind: str
    error: str
    output: str
    selected_ids: list[str]
    reactions: list[str]
    progress: list[str]
    state: StateData
