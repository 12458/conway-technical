FROM python:3.12 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN python -m venv .venv
COPY pyproject.toml ./
RUN .venv/bin/pip install .
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .

# Ensure config directory exists and is readable
RUN ls -la config/ || echo "WARNING: config directory not found"

CMD ["/app/.venv/bin/python", "main.py", "--host", "0.0.0.0"]
