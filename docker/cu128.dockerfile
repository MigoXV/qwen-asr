FROM registry.cn-hangzhou.aliyuncs.com/migo-dl/vllm:0.14.1a1-torch291-cu128

# 设置工作目录
WORKDIR /app

# 拷贝必要的文件以安装依赖
COPY pyproject.toml poetry.lock README.md ./
RUN mkdir -p src/qwen_asr && \
    touch src/qwen_asr/__init__.py && \
    poetry install --no-root

# 拷贝源代码文件
COPY . .

# 安装当前包
RUN poetry install

# 暴露 gRPC 服务端口
EXPOSE 50051

# 默认入口
CMD ["poetry", "run", "python", "-m", "qwen_asr.commands.app"]
