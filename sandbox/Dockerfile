FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl sudo util-linux passwd wget \
    && printf '%s\n' 'wsm_* ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/wiseman \
    && chmod 0440 /etc/sudoers.d/wiseman \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
COPY sandbox/src ./sandbox/src
COPY sandbox/codex-as-user /usr/local/bin/codex-as-user
COPY sandbox/wiseman-image /usr/local/bin/wiseman-image
COPY sandbox/wiseman-discord /usr/local/bin/wiseman-discord
RUN chmod 0755 /usr/local/bin/codex-as-user
RUN chmod 0755 /usr/local/bin/wiseman-image
RUN chmod 0755 /usr/local/bin/wiseman-discord
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project
ENV PYTHONPATH=/app/sandbox/src WISEMAN_CODEX_BIN=/usr/local/bin/codex-as-user
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "runner.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8090"]
