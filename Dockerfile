FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

RUN pip install --no-cache-dir uv
RUN pip install --no-cache-dir fastapi[standard] uvicorn[standard]

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen 

COPY . .

EXPOSE 7860

CMD ["uv", "run", "fastapi", "run", "gomoku/app.py", "--host", "0.0.0.0", "--port", "7860"]
