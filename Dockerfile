FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY contracts ./contracts
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "app.http_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
