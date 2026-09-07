FROM ghcr.io/astral-sh/uv:0.12.10 AS uv
FROM python:3.12-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
ARG MODEL_EXTRA=fake
RUN case "$MODEL_EXTRA" in \
    fake) uv sync --locked --no-dev --no-editable ;; \
    custom|hf) uv sync --locked --no-dev --no-editable --extra "$MODEL_EXTRA" ;; \
    *) exit 1 ;; esac
RUN useradd --uid 10001 --create-home app
USER app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"
CMD ["uvicorn", "inference_service.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
