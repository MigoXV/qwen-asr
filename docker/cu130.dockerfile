FROM vllm/vllm-openai:v0.14.0-x86_64-cu130

COPY . .
RUN mkdir -p src/qwen_asr && \
    touch src/qwen_asr/__init__.py && \
    pip install .

EXPOSE 50051

ENTRYPOINT ["python3", "-m", "qwen_asr.commands.app"]
