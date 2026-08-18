FROM python:3.12-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    VOXADAPT_DB="/data/voxadapt.db"
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "voxadapt.api:app", "--host", "0.0.0.0", "--port", "8000"]
