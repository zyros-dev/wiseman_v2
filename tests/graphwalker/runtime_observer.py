# Copyright (c) 2026 Nick van der Merwe
"""Build graph observations from Temporal state and Phoenix evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from tests.graphwalker.model import ObservedChatState, ObservedWisemanState, PhoenixEvidence, RuntimeObservation, Vertex

if TYPE_CHECKING:
    from app.clients.client_interfaces import ClientContainer
    from app.types import JsonObject, JsonValue


class RuntimeObserver:
    def __init__(self, clients: ClientContainer) -> None:
        self.clients = clients

    async def observe(self, thread_id: str) -> RuntimeObservation:
        snapshot = await self.clients.temporal.snapshot(thread_id)
        session = _object(snapshot)
        active_id = _string(session.get("active_message"))
        turn_snapshot = _object(session.get("active_turn_snapshot"))
        work = _object(turn_snapshot.get("work"))
        state = _object(work.get("state")) or session
        delivery = _object(state.get("delivery"))
        phase = _phase(session, turn_snapshot, work, delivery)
        records = self._records(active_id)
        trace = f"discord-{active_id}" if active_id else ""
        phoenix = _phoenix_evidence(records, self.clients.phoenix.audit(trace) if trace else None)
        discord = _discord_observation(self.clients.discord)
        terminal = phase in {Vertex.DELIVERING, Vertex.ERROR}
        error = bool(work.get("error")) or any(record.get("node") == "failure" for record in records)
        pending = _ids(session.get("pending_message_ids"))
        pending = ((active_id,) if active_id else ()) + tuple(item for item in pending if item != active_id)
        answer_id = _string(delivery.get("answer_message_id"))
        progress_id = _string(delivery.get("progress_message_id"))
        return RuntimeObservation(
            chat=ObservedChatState(
                message_ids=_ordered_ids(
                    (session.get("message_ids"), session.get("message_timestamps")),
                    (state.get("message_ids"), state.get("message_timestamps")),
                ),
                background_context_ids=_merge_ids(session.get("background_context_ids"), state.get("background_context_ids")),
                consumed_context_ids=_merge_ids(session.get("consumed_context_ids"), state.get("consumed_context_ids")),
                steering_ids=_merge_ids(session.get("steering_ids"), state.get("steering_ids")),
                stop_command_ids=_merge_ids(session.get("stop_command_ids"), state.get("stop_command_ids")),
                answer_message_ids=(answer_id,) if answer_id and phase is Vertex.DELIVERING else (),
                progress_message_ids=(progress_id,) if progress_id and phase in {Vertex.PREPARING, Vertex.RUNNING} else (),
                progress_edit_count=_integer(delivery.get("progress_edit_count")),
                answer_edit_count=_integer(delivery.get("answer_edit_count")),
                typing=bool(delivery.get("typing")),
                archived=bool(session.get("closed")),
                edited_message_ids=discord[0],
                typing_operations=discord[1],
                reaction_operations=discord[2],
            ),
            wiseman=ObservedWisemanState(
                phase=phase,
                active_question=active_id or None,
                pending_question_ids=pending,
                session_id=_optional_string(state.get("codex_thread")),
                turn=_integer(state.get("turn", session.get("turn"))),
                active_turn=_integer(state.get("turn")) if phase is Vertex.DELIVERING and active_id else _integer(state.get("turn")) + 1 if active_id else None,
                result_known=terminal,
                error=error,
                stop_target_question_id=active_id if phase is Vertex.CANCELLING else None,
                processed_question_ids=_merge_ids(session.get("processed"), state.get("processed")),
                inferencing=bool(turn_snapshot.get("inferencing")),
                stop_requested=bool(turn_snapshot.get("stop_requested")),
                resume_requested=bool(turn_snapshot.get("resume_requested")),
                resumed=bool(turn_snapshot.get("resumed")),
                recovering=bool(turn_snapshot.get("recovering")),
                outcome_unknown=bool(turn_snapshot.get("outcome_unknown")),
                cancellation_unknown=bool(turn_snapshot.get("cancellation_unknown")),
                delivery_phase=_string(delivery.get("phase")) or "idle",
                reaction_phase=_string(delivery.get("reaction_phase")) or "none",
            ),
            phoenix_nodes=phoenix.nodes,
            phoenix_sequence=phoenix.sequence,
            phoenix=phoenix,
        )

    def _records(self, active_id: str) -> list[dict[str, object]]:
        trace = f"discord-{active_id}" if active_id else ""
        return [record for record in self.clients.phoenix.records if str(record.get("trace", "")) == trace]


def _phoenix_evidence(records: list[dict[str, object]], audit: dict[str, object] | None) -> PhoenixEvidence:
    nodes = tuple(str(record["node"]) for record in records if record.get("node"))
    admission = _last_record(records, "admission")
    context = _last_record(records, "context")
    grammar = _last_record(records, "grammar")
    prompt = _last_record(records, "prompt")
    turn = _last_record(records, "turn")
    codex = _last_record(records, "codex")
    completed = _last_record(records, "completed")
    provider = _last_record(records, "provider")
    reaction_records = tuple(record for record in records if record.get("node") == "reaction")
    raw_request = _json_value(admission.get("raw_request"))
    normalized_request = _json_value(admission.get("normalized_request"))
    audit_raw = _json_value(audit.get("raw_request")) if audit else None
    audit_normalized = _json_value(audit.get("normalized_request")) if audit else None
    context_data = _object(context.get("normalized"))
    context_messages = tuple(_object(item) for item in _sequence(context_data.get("messages")))
    context_attachment_urls = tuple(
        str(_object(attachment).get("url"))
        for message in (*context_messages, _object(context_data.get("trigger")))
        for attachment in _sequence(message.get("attachments"))
        if _object(attachment).get("url")
    )
    route = _object(turn.get("route"))
    result = codex or completed
    billing = result or provider
    model = _string(billing.get("model"))
    return PhoenixEvidence(
        trace=_string(records[0].get("trace")) if records else "",
        audit_id=_string(admission.get("audit_id")) or None,
        audit_present=audit is not None,
        audit_matches_admission=audit_raw == raw_request and audit_normalized == normalized_request,
        raw_request=raw_request,
        normalized_request=normalized_request,
        context_message_ids=tuple(str(item) for item in _sequence(context_data.get("selected_ids")) if item),
        context_author_ids=tuple(_string(message.get("author_id")) for message in context_messages if _string(message.get("author_id"))),
        context_reply_ids=tuple(_string(message.get("reply_to")) for message in context_messages if _string(message.get("reply_to"))),
        context_attachment_urls=context_attachment_urls,
        grammar_rendered=_string(grammar.get("rendered")),
        prompt_final_input=_string(prompt.get("final_input")),
        codex_thread_id=_string(result.get("codex_thread_id")) or None,
        requested_model=_string(route.get("requested_model")) or _string(provider.get("requested_model")) or None,
        served_model=model,
        usage=_json_value(billing.get("usage")),
        cost=_json_value(billing.get("cost")),
        transport_complete=_bool_or_none(provider.get("transport_complete")),
        reaction_operations=tuple(operation for record in reaction_records for operation in _string_sequence(record.get("operations"))),
        nodes=tuple(sorted(set(nodes))),
        sequence=nodes,
    )


def _last_record(records: list[dict[str, object]], node: str) -> JsonObject:
    return next((cast("JsonObject", record) for record in reversed(records) if record.get("node") == node), {})


def _discord_observation(discord: object) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    state = getattr(discord, "state", None)
    calls = getattr(state, "calls", ())
    edited: list[str] = []
    typing: list[str] = []
    reactions: list[str] = []
    if not isinstance(calls, list):
        return (), (), ()
    for call in calls:
        if not isinstance(call, tuple) or len(call) != 3:
            continue
        client, operation, values = call
        if client != "discord" or not isinstance(values, tuple):
            continue
        if operation == "edit" and values:
            edited.append(str(values[0]))
        elif operation in {"start_typing", "stop_typing"} and values:
            typing.append(f"{operation.removesuffix('_typing')}:{values[0]}")
        elif operation in {"add_reaction", "remove_reaction"} and len(values) >= 2:
            reactions.append(f"{operation.removesuffix('_reaction')}:{values[0]}:{values[1]}")
    return tuple(edited), tuple(typing), tuple(reactions)


def _phase(session: JsonObject, turn_snapshot: JsonObject, work: JsonObject, delivery: JsonObject) -> Vertex:
    phase = Vertex.PREPARING
    if session.get("closed"):
        phase = Vertex.RETIRED
    elif not session.get("active_message"):
        phase = Vertex.IDLE
    elif turn_snapshot.get("stop_requested"):
        phase = Vertex.CANCELLING
    elif turn_snapshot.get("resumed"):
        phase = Vertex.RUNNING
    elif turn_snapshot.get("outcome_unknown"):
        phase = Vertex.OUTCOME_UNKNOWN
    elif turn_snapshot.get("recovering"):
        phase = Vertex.RECOVERING
    elif delivery.get("phase") == "answer":
        phase = Vertex.ERROR if work.get("error") else Vertex.DELIVERING
    elif turn_snapshot.get("inferencing"):
        phase = Vertex.RUNNING
    return phase


def _object(value: object) -> JsonObject:
    return cast("JsonObject", value) if isinstance(value, dict) else {}


def _ids(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value if item) if isinstance(value, list | tuple) else ()


def _merge_ids(*values: object) -> tuple[str, ...]:
    merged: list[str] = []
    for value in values:
        for item in _ids(value):
            if item not in merged:
                merged.append(item)
    return tuple(merged)


def _ordered_ids(*sources: tuple[object, object]) -> tuple[str, ...]:
    timestamps: dict[str, str] = {}
    order: list[str] = []
    for values, raw_timestamps in sources:
        for message_id in _ids(values):
            if message_id not in order:
                order.append(message_id)
        if isinstance(raw_timestamps, dict):
            timestamps.update({str(key): str(value) for key, value in raw_timestamps.items()})
    return tuple(sorted(order, key=lambda message_id: (timestamps.get(message_id, ""), order.index(message_id))))


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _string_sequence(value: object) -> tuple[str, ...]:
    return tuple(item for item in _sequence(value) if isinstance(item, str))


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)
