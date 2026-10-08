# Copyright (c) 2026 Nick van der Merwe
"""Scheduled Discord posts for Wiseman V2, run by Kubernetes CronJobs.

Ported from the v1 Hermes cron jobs (leetcode-daily, weather-sydney,
hn-daily-top-posts, artista_drop_monitor, birthday_reminder). Stdlib only so it
runs on a stock Python image. Usage: ``scheduled_post.py JOB [--dry-run]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

DISCORD_API = "https://discord.com/api/v10"
DISCORD_LIMIT = 2000
SUPPRESS_EMBEDS = 1 << 2
PUBLIC_THREAD = 11
TIMEOUT = 15
ZWSP = "\u200b"
NO_EMBED = "[no-embed]"
NOON = 12

WISEMAN_HOME = "1501008202630696981"
LEETCODE_CHANNEL = "1360191262334128148"
ARTISTA_THREAD = "1504389612842258474"
BIRTHDAY_THREAD = "1535070488542453820"

SYDNEY = (-33.8688, 151.2093)
WMO = {
    0: ("☀️", "clear"),
    1: ("🌤", "mainly clear"),
    2: ("⛅", "partly cloudy"),
    3: ("☁️", "overcast"),
    45: ("🌫", "fog"),
    48: ("🌫", "freezing fog"),
    51: ("🌦", "light drizzle"),
    53: ("🌦", "drizzle"),
    55: ("🌧", "heavy drizzle"),
    61: ("🌦", "light rain"),
    63: ("🌧", "rain"),
    65: ("🌧", "heavy rain"),
    66: ("🌧", "freezing rain"),
    67: ("🌧", "freezing rain"),
    71: ("🌨", "light snow"),
    73: ("🌨", "snow"),
    75: ("❄️", "heavy snow"),
    77: ("🌨", "snow grains"),
    80: ("🌦", "rain showers"),
    81: ("🌧", "rain showers"),
    82: ("⛈", "violent showers"),
    85: ("🌨", "snow showers"),
    86: ("🌨", "snow showers"),
    95: ("⛈", "thunderstorm"),
    96: ("⛈", "thunderstorm w/ hail"),
    99: ("⛈", "thunderstorm w/ hail"),
}
DIFFICULTY = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}

ARTISTA_TICKER = "ANET"
ARTISTA_WINDOW = timedelta(hours=48)
ARTISTA_STEP = timedelta(minutes=10)
ARTISTA_THRESHOLD = 0.01
ARTISTA_POSITION = 200_000
ARTISTA_HOURLY_AUD = 150_000 / (52 * 40)
AEST = timezone(timedelta(hours=10))


@dataclass(frozen=True)
class Post:
    channel_id: str
    content: str
    thread_name: str | None = None


def _request(url: str, *, method: str = "GET", body: object = None, headers: dict[str, str] | None = None) -> object:
    data = json.dumps(body).encode() if body is not None else None
    merged = {"User-Agent": "wiseman-v2-scheduled-posts/1.0", **({"Content-Type": "application/json"} if data else {}), **(headers or {})}
    request = urllib.request.Request(url, method=method, data=data, headers=merged)  # noqa: S310 - fixed https endpoints
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - fixed https endpoints
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        message = f"{method} {url.split('?', maxsplit=1)[0]} -> HTTP {exc.code}: {detail}"
        raise RuntimeError(message) from None
    return json.loads(raw) if raw else {}


def _clock(iso: str) -> str:
    hour, minute = iso.split("T")[1].split(":")[:2]
    half, hour12 = divmod(int(hour), NOON)
    return f"{hour12 or NOON}:{minute} {'pm' if half else 'am'}"


def leetcode() -> Post:
    query = "query questionOfToday { activeDailyCodingChallengeQuestion { date link question { title difficulty topicTags { name } } } }"
    payload = _request("https://leetcode.com/graphql/", method="POST", body={"query": query}, headers={"User-Agent": "Mozilla/5.0"})
    daily = payload["data"]["activeDailyCodingChallengeQuestion"]  # type: ignore[index]
    question = daily["question"]
    tags = ", ".join(tag["name"] for tag in question["topicTags"][:4]) or "—"
    content = (
        f"{NO_EMBED}\n>>> 📅  **LeetCode Daily** — {daily['date']}\n"
        f"{DIFFICULTY.get(question['difficulty'], '⚪')}  **{question['difficulty']}** · "
        f"[{question['title']}](https://leetcode.com{daily['link']}){ZWSP}\n"
        f"🏷️  {tags}\n\n💡  Drop your solution / approach below."
    )
    # The bot can create public threads in this channel but not post at its top level.
    return Post(LEETCODE_CHANNEL, content, thread_name=f"{daily['date']} · {question['title']}"[:100])


def weather() -> Post:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={SYDNEY[0]}&longitude={SYDNEY[1]}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
        "&hourly=temperature_2m,precipitation_probability,weather_code"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "precipitation_probability_max,sunrise,sunset,uv_index_max,wind_speed_10m_max"
        "&timezone=Australia%2FSydney&forecast_days=1"
    )
    forecast = _request(url)
    now, day, hourly = forecast["current"], forecast["daily"], forecast["hourly"]  # type: ignore[index]
    emoji, label = WMO.get(int(now["weather_code"] or 0), ("🌡", "—"))
    day_emoji, day_label = WMO.get(int(day["weather_code"][0] or 0), ("🌡", "—"))
    rows = []
    for target in ("T09:", "T12:", "T15:", "T18:"):
        index = next((i for i, stamp in enumerate(hourly["time"]) if target in stamp), None)
        if index is not None:
            symbol = WMO.get(int(hourly["weather_code"][index] or 0), ("🌡", "—"))[0]
            temp, rain = hourly["temperature_2m"][index], hourly["precipitation_probability"][index]
            rows.append(f"{_clock(hourly['time'][index]):>7} {symbol} {temp:>4.0f}°C  rain {rain:>3.0f}%")
    content = (
        f"{NO_EMBED}\n>>> 🌏  **Sydney Weather** — {day['time'][0]}\n"
        f"{day_emoji}  **{day_label.title()}** · high **{day['temperature_2m_max'][0]:.0f}°C** / low **{day['temperature_2m_min'][0]:.0f}°C**\n"
        f"☔  {day['precipitation_sum'][0]:.1f}mm  ·  {day['precipitation_probability_max'][0]:.0f}% chance  ·  "
        f"💨 {day['wind_speed_10m_max'][0]:.0f} km/h  ·  🔆 UV {day['uv_index_max'][0]:.0f}\n"
        f"🌅  {_clock(day['sunrise'][0])}  →  🌆  {_clock(day['sunset'][0])}\n\n"
        f"**now**  {emoji} {label}  ·  {now['temperature_2m']:.0f}°C (feels {now['apparent_temperature']:.0f}°C)  ·  "
        f"💧 {now['relative_humidity_2m']:.0f}%  ·  💨 {now['wind_speed_10m']:.0f} km/h\n\n"
        "```\n" + "\n".join(rows) + "\n```"
    )
    return Post(WISEMAN_HOME, content)


def hacker_news() -> Post:
    lines = [NO_EMBED, ">>> 📰  **Hacker News Top 15**"]
    for rank, story in enumerate(_request("https://hacker-news.firebaseio.com/v0/topstories.json")[:15], 1):  # type: ignore[index]
        try:
            item = _request(f"https://hacker-news.firebaseio.com/v0/item/{story}.json")
        except (RuntimeError, OSError):
            continue
        thread = f"https://news.ycombinator.com/item?id={story}"
        title = (item.get("title") or "(untitled)").strip()  # type: ignore[union-attr]
        link = item.get("url") or thread  # type: ignore[union-attr]
        lines.append(f"`{rank:>2}` [{title}]({link}){ZWSP} · ⬆️ {item.get('score', 0)} · 💬 [{item.get('descendants', 0)}]({thread}){ZWSP}")  # type: ignore[union-attr]
    return Post(WISEMAN_HOME, "\n".join(lines))


def artista(now: datetime | None = None) -> Post | None:
    """Alert once when ANET first falls >=1% below its 48h high (stateless: compares with 10 minutes ago)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ARTISTA_TICKER}?interval=5m&range=5d"
    result = _request(url, headers={"User-Agent": "Mozilla/5.0"})["chart"]["result"][0]  # type: ignore[index]
    closes = result["indicators"]["quote"][0]["close"]
    series = [(datetime.fromtimestamp(ts, UTC), price) for ts, price in zip(result["timestamp"], closes, strict=False) if price]
    now = now or datetime.now(UTC)

    def drop_at(moment: datetime) -> tuple[float, datetime, float, float] | None:
        window = [(ts, price) for ts, price in series if moment - ARTISTA_WINDOW <= ts <= moment]
        if not window:
            return None
        peak_at, peak = max(window, key=lambda point: point[1])
        current = window[-1][1]
        return (peak - current) / peak, peak_at, peak, current

    latest, earlier = drop_at(now), drop_at(now - ARTISTA_STEP)
    if latest is None or latest[0] < ARTISTA_THRESHOLD or (earlier is not None and earlier[0] >= ARTISTA_THRESHOLD):
        return None
    drop, peak_at, peak, current = latest
    loss = ARTISTA_POSITION * drop
    content = (
        f"poggers_fish ANET fell {drop * 100:.2f}% in the last 48h "
        f"(from ${peak:.2f} at {peak_at.astimezone(AEST):%Y-%m-%d %H:%M} AEST to ${current:.2f} at {now.astimezone(AEST):%Y-%m-%d %H:%M} AEST). "
        f"If you had ${ARTISTA_POSITION:,} invested, you just lost ${loss:,.2f}. "
        f"That's ~{loss / ARTISTA_HOURLY_AUD:.1f} hours of your time (@ $150,000/yr AUD)."
    )
    return Post(ARTISTA_THREAD, content)


def birthday() -> Post:
    return Post(BIRTHDAY_THREAD, "Reminder: Happy birthday to <@263187621256101888>!")


JOBS = {"leetcode": leetcode, "weather": weather, "hn": hacker_news, "artista": artista, "birthday": birthday}


def chunks(text: str) -> list[str]:
    parts: list[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > DISCORD_LIMIT and current:
            parts.append(current)
            current = line[:DISCORD_LIMIT]
        else:
            current = candidate[:DISCORD_LIMIT]
    return [*parts, current] if current else parts


def deliver(post: Post, token: str) -> list[str]:
    auth = {"Authorization": f"Bot {token}"}
    flags = SUPPRESS_EMBEDS if post.content.startswith(NO_EMBED) else 0
    content = post.content.removeprefix(NO_EMBED).lstrip("\n")
    channel = post.channel_id
    if post.thread_name:
        body = {"name": post.thread_name, "auto_archive_duration": 1440, "type": PUBLIC_THREAD}
        thread = _request(f"{DISCORD_API}/channels/{channel}/threads", method="POST", body=body, headers=auth)
        channel = str(thread["id"])  # type: ignore[index]
    sent = []
    quoted = content.startswith(">>> ")
    for index, part in enumerate(chunks(content)):
        if quoted and index:
            part = f">>> {part}"  # noqa: PLW2901 - keep the block quote on continuation messages
        # Wiseman V2 never pings: mentions render but notify nobody.
        body = {"content": part, "flags": flags, "allowed_mentions": {"parse": []}}
        sent.append(str(_request(f"{DISCORD_API}/channels/{channel}/messages", method="POST", body=body, headers=auth)["id"]))  # type: ignore[index]
    return sent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=sorted(JOBS))
    parser.add_argument("--dry-run", action="store_true", help="print the post and verify the bot token instead of posting")
    args = parser.parse_args(argv)
    post = JOBS[args.job]()
    if post is None:
        sys.stdout.write(f"{args.job}: nothing to post\n")
        return 0
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if args.dry_run:
        if token:
            me = _request(f"{DISCORD_API}/users/@me", headers={"Authorization": f"Bot {token}"})
            sys.stdout.write(f"token ok: {me['username']} ({me['id']})\n")  # type: ignore[index]
        sys.stdout.write(f"channel={post.channel_id} thread={post.thread_name!r} chunks={len(chunks(post.content))}\n{post.content}\n")
        return 0
    if not token:
        sys.stderr.write("DISCORD_BOT_TOKEN is not set\n")
        return 2
    ids = deliver(post, token)
    sys.stdout.write(f"{args.job}: posted {len(ids)} message(s) to {post.channel_id}: {','.join(ids)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
