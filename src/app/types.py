# Copyright (c) 2026 Nick van der Merwe

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]


type StateData = JsonObject
