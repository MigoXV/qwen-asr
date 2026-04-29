FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg sox libsox-fmt-all git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock README.md ./
RUN pip install --no-cache-dir poetry && poetry config virtualenvs.create false && poetry install --no-root

COPY . .
RUN poetry install

EXPOSE 50051

CMD ["poetry", "run", "python", "-m", "qwen_asr.commands.app", "serve", "--backend", "transformers", "--device", "cpu", "--model", "Qwen/Qwen3-ASR-0.6B"]
