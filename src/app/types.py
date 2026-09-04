# Copyright (c) 2026 Nick van der Merwe

from typing import TypedDict

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


type StateData = dict[str, object]


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
