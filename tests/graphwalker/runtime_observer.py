# Copyright (c) 2026 Nick van der Merwe
"""Build graph observations from Temporal state and Phoenix evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from tests.graphwalker.model import ObservedChatState, ObservedWisemanState, RuntimeObservation, Vertex

if TYPE_CHECKING:
    from app.clients.client_interfaces import ClientContainer
    from app.types import JsonObject


class RuntimeObserver:
    def __init__(self, clients: ClientContainer) -> None:
        self.clients = clients

    async def observe(self, thread_id: str) -> RuntimeObservation:
        snapshot = await self.clients.temporal.snapshot(thread_id)
        session = _object(snapshot)
        active_id = _string(session.get("active_message"))
        turn_snapshot = _object(session.get("active_turn_snapshot"))
        work = _object(turn_snapshot.get("work"))
        state = _object(work.get("state"))
        delivery = _object(state.get("delivery"))
        records = self._records(active_id, session)
        phase = _phase(session, turn_snapshot, work, delivery)
        terminal = phase in {Vertex.DELIVERING, Vertex.ERROR}
        error = bool(work.get("error")) or any(record.get("node") == "failure" for record in records)
        pending = _ids(session.get("pending_message_ids"))
        pending = ((active_id,) if active_id else ()) + tuple(item for item in pending if item != active_id)
        answer_id = _string(delivery.get("answer_message_id"))
        progress_id = _string(delivery.get("progress_message_id"))
        return RuntimeObservation(
            chat=ObservedChatState(
                message_ids=_ids(session.get("message_ids")),
                background_context_ids=_ids(state.get("background_context_ids", session.get("background_context_ids"))),
                consumed_context_ids=_ids(state.get("consumed_context_ids", session.get("consumed_context_ids"))),
                steering_ids=_ids(state.get("steering_ids", session.get("steering_ids"))),
                stop_command_ids=_ids(state.get("stop_command_ids", session.get("stop_command_ids"))),
                answer_message_ids=(answer_id,) if answer_id and phase is Vertex.DELIVERING else (),
                progress_message_ids=(progress_id,) if progress_id and phase in {Vertex.PREPARING, Vertex.RUNNING} else (),
                progress_edit_count=_integer(delivery.get("progress_edit_count")),
                answer_edit_count=_integer(delivery.get("answer_edit_count")),
                typing=bool(delivery.get("typing")),
                archived=bool(session.get("closed")),
            ),
            wiseman=ObservedWisemanState(
                phase=phase,
                active_question=active_id or None,
                pending_question_ids=pending,
                session_id=_string(state.get("codex_thread")),
                turn=_integer(state.get("turn", session.get("turn"))),
                active_turn=_integer(state.get("turn")) + 1 if active_id else None,
                result_known=terminal or any(record.get("node") == "completed" for record in records),
                error=error,
                stop_target_question_id=active_id if phase is Vertex.CANCELLING else None,
            ),
        )

    def _records(self, active_id: str, session: JsonObject) -> list[dict[str, object]]:
        message_ids = {active_id, *_ids(session.get("message_ids"))}
        traces = {f"discord-{message_id}" for message_id in message_ids if message_id}
        return [record for record in self.clients.phoenix.records if str(record.get("trace", "")) in traces]


def _phase(session: JsonObject, turn_snapshot: JsonObject, work: JsonObject, delivery: JsonObject) -> Vertex:
    phase = Vertex.PREPARING
    if session.get("closed"):
        phase = Vertex.RETIRED
    elif not session.get("active_message"):
        phase = Vertex.IDLE
    elif work.get("error"):
        phase = Vertex.ERROR
    elif turn_snapshot.get("stop_requested"):
        phase = Vertex.CANCELLING
    elif delivery.get("phase") == "answer":
        phase = Vertex.DELIVERING
    elif turn_snapshot.get("inferencing"):
        phase = Vertex.RUNNING
    return phase


def _object(value: object) -> JsonObject:
    return cast("JsonObject", value) if isinstance(value, dict) else {}


def _ids(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value if item) if isinstance(value, list | tuple) else ()


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
