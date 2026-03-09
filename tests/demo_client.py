"""
gRPC 流式 ASR 客户端 Demo

仿照 demo01.py，下载示例音频，通过 gRPC StreamingRecognize 接口发送，
实时打印服务端流式返回的识别结果。

使用方法：
  1. 启动服务端（先确保模型已加载）：
       QWEN_ASR_MODEL=/workspace/model-bin/Qwen/Qwen3-ASR-0.6B \
       python -m qwen_asr.commands.app

  2. 运行本客户端：
       python tests/demo_client.py
"""
import asyncio
import io
import time
import urllib.request
from typing import AsyncIterator, Tuple

import grpc
import numpy as np
import soundfile as sf

from qwen_asr.protos.asr.ux_speech_pb2 import (
    RecognitionConfig,
    StreamingRecognitionConfig,
    StreamingRecognizeRequest,
)
from qwen_asr.protos.asr.ux_speech_pb2_grpc import UxSpeechStub

# ── 服务端地址 ────────────────────────────────────────────────────────────────
SERVER_ADDR = "localhost:50018"

SAMPLE_RATE = 16000


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def load_audio_from_url(url: str) -> Tuple[np.ndarray, int]:
    """从 URL 下载音频，返回 (float32 waveform, sample_rate)。"""
    with urllib.request.urlopen(url) as resp:
        audio_bytes = resp.read()
    with io.BytesIO(audio_bytes) as f:
        audio, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sr)


def float32_to_pcm16_bytes(wav: np.ndarray) -> bytes:
    """将 float32 waveform 转换为 16-bit signed PCM little-endian bytes。"""
    pcm16 = np.clip(wav * 32768.0, -32768, 32767).astype(np.int16)
    return pcm16.tobytes()


async def request_generator(
    wav: np.ndarray,
    language: str = "",
    interim_results: bool = True,
) -> AsyncIterator[StreamingRecognizeRequest]:
    """
    生成 StreamingRecognizeRequest 序列：
      - 第一条：streaming_config（含采样率、语言等配置）
      - 第二条：完整 audio_content（LINEAR16 PCM）
    """
    yield StreamingRecognizeRequest(
        streaming_config=StreamingRecognitionConfig(
            config=RecognitionConfig(
                encoding=RecognitionConfig.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=language,
            ),
            interim_results=interim_results,
        )
    )
    yield StreamingRecognizeRequest(audio_content=float32_to_pcm16_bytes(wav))


async def stream_recognize(
    stub,
    wav: np.ndarray,
    language: str = "",
    interim_results: bool = True,
) -> None:
    """
    发起一次 StreamingRecognize 调用，实时打印流式返回结果。
    interim_results=True 时会看到逐步更新的识别文本。
    """
    requests = request_generator(wav, language=language, interim_results=interim_results)
    responses = stub.StreamingRecognize(requests)

    last_final = ""
    t_first = None
    t_start = time.perf_counter()

    async for response in responses:
        for result in response.results:
            transcript = result.alternative.transcript
            is_final = result.is_final

            if t_first is None:
                t_first = time.perf_counter()

            if is_final:
                # 覆盖当前行并打印最终结果
                print(f"\r[final]   {transcript}")
                last_final = transcript
            else:
                # 实时更新同一行
                print(f"\r[interim] {transcript}", end="", flush=True)

    elapsed = time.perf_counter() - t_start
    ttfb = (t_first - t_start) if t_first else elapsed
    print(f"[timing]  TTFB={ttfb:.3f}s  total={elapsed:.3f}s")
    return last_final


async def main():
    urls = [
        ("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav", "Chinese"),
        ("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav", "English"),
    ]

    print("正在下载示例音频…")
    samples = [(load_audio_from_url(url), lang) for url, lang in urls]
    print("下载完成。\n")

    channel = grpc.aio.insecure_channel(SERVER_ADDR)
    stub = UxSpeechStub(channel)

    for (audio, sr), lang in samples:
        # 若采样率不是 16kHz，在本地先 resample
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)

        print(f"{'=' * 60}")
        print(f"[stream] language={lang}  duration={len(audio)/SAMPLE_RATE:.1f}s")
        await stream_recognize(stub, audio, language=lang, interim_results=True)
        print()

    await channel.close()


if __name__ == "__main__":
    asyncio.run(main())
