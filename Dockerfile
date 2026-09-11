FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8501

CMD ["/app/.venv/bin/streamlit", "run", "app_annotator.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]