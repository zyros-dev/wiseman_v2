# Copyright (c) 2026 Nick van der Merwe
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from app.clients.discord_client import RealDiscord
from app.clients.mock_clients import MockDiscord, mock_container
from app.engine import Engine
from app.models import Event, Message, MessageRef, State, TurnWork, Upload
from app.presentation import render_progress

if TYPE_CHECKING:
    from app.clients.client_interfaces import DiscordClient


async def test_engine_uses_the_shared_discord_client_for_both_turns():
    clients = mock_container()
    engine = Engine(clients=clients)
    first = await engine.handle(
        Event(trigger=Message(id="first", author_id="alice", channel_id="parent", thread_id="thread"))
    )
    second = await Engine(clients=clients).handle(
        Event(trigger=Message(id="second", author_id="bob", channel_id="thread", thread_id="thread")),
        state_data=first["state"],
    )
    assert isinstance(clients.discord, MockDiscord)
    state = clients.discord.state
    assert len(state.messages) == 3
    assert state.reactions == {"first": ["✅"], "second": ["✅"]}
    assert second["state"]["owner_id"] == "alice"
    assert second["state"]["codex_thread"] == first["state"]["codex_thread"]
    assert sum(call.operation == "send" for call in state.calls) == 3
    assert first["output"] in state.messages.values()
    assert second["output"] in state.messages.values()


async def test_real_discord_adapter_preserves_address_nonce_and_upload():
    channel = Mock(spec=discord.TextChannel)
    channel.send = AsyncMock(return_value=SimpleNamespace(id=12))
    message = SimpleNamespace(edit=AsyncMock(), add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    channel.get_partial_message.return_value = message
    gateway = Mock()
    gateway.get_channel.return_value = channel
    gateway.user = SimpleNamespace(id=42)
    client: DiscordClient = RealDiscord(gateway)
    assert await client.send("10", "answer", nonce="trigger-id") == "12"
    channel.send.assert_awaited_once_with("answer", embeds=[], nonce="trigger-id")
    ref = MessageRef("10", "12")
    await client.edit(ref, "preview", upload=Upload("response.md", b"full response"))
    upload = message.edit.await_args.kwargs["attachments"][0]
    assert upload.filename == "response.md"
    assert upload.fp.read() == b"full response"
    await client.add_reaction(ref, "done")
    await client.remove_reaction(ref, "working")
    message.add_reaction.assert_awaited_once_with("done")
    message.remove_reaction.assert_awaited_once_with("working", gateway.user)
    assert all(call.args == (12,) for call in channel.get_partial_message.call_args_list)


async def test_long_answer_is_one_edit_with_the_complete_attachment():
    gateway = Mock()
    channel = Mock(spec=discord.TextChannel)
    gateway.get_channel.return_value = channel
    message = AsyncMock()
    channel.get_partial_message.return_value = message
    clients = mock_container()
    clients.discord = RealDiscord(gateway)
    output = "Paragraph\n" + "word " * 1500
    work = TurnWork(
        event=Event(trigger=Message(id="1", author_id="alice", channel_id="10", thread_id="20")),
        state=State(delivery_id="30"),
        output=output,
    )
    await Engine(clients=clients).deliver(work)
    message.edit.assert_awaited_once()
    sent = message.edit.await_args.kwargs
    assert len(sent["content"]) <= 2000
    assert sent["attachments"][0].fp.read().decode() == output
    channel.send.assert_not_called()


def test_progress_bounds_each_preview_and_the_complete_discord_message():
    steps = [f"step {number}: " + "output" * 1000 for number in range(32)]
    content = render_progress(steps, 8)
    assert len(content) <= 2000
    assert content.startswith("⏳ Working · Gurt 8")
    assert "step 31:" in content
    assert "step 0:" not in content


async def test_real_history_keeps_context_evidence():
    channel = Mock(spec=discord.Thread)
    item = SimpleNamespace(
        id=11,
        author=SimpleNamespace(id=42, name="Alice", bot=False),
        content="image <@7>",
        created_at=SimpleNamespace(isoformat=lambda: "2026-09-05T00:00:00+00:00"),
        reference=SimpleNamespace(message_id=9),
        raw_mentions=[7],
        attachments=[
            SimpleNamespace(id=6, filename="a.png", url="https://cdn.example/a.png", content_type="image/png", size=8)
        ],
    )

    async def history(**kwargs):
        assert kwargs == {"limit": 12}
        yield item

    channel.history = history
    gateway = Mock()
    gateway.get_channel.return_value = channel
    messages = await RealDiscord(gateway).history("10", 12)
    record = Message.model_validate(messages[0])
    assert record.author_name == "Alice"
    assert record.thread_id == record.channel_id == "10"
    assert record.reply_to == "9"
    assert "7" in record.mentions
    assert record.attachments[0]["filename"] == "a.png"
    assert record.timestamp == "2026-09-05T00:00:00+00:00"


@pytest.mark.parametrize("operation", ["archive_thread", "lock_thread"])
async def test_thread_mutations_reject_parent_channels(operation):
    gateway = Mock()
    gateway.get_channel.return_value = Mock(spec=discord.TextChannel)
    with pytest.raises(TypeError):
        await getattr(RealDiscord(gateway), operation)("10")
