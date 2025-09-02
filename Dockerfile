FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv

COPY pyproject.toml ./
RUN uv venv && uv sync --no-dev
# (якщо треба) PyTorch:
RUN /app/.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.2+cpu

COPY . .

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn","src.backend.main:app","--host","0.0.0.0","--port","8000"]
