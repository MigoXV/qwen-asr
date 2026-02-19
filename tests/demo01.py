import io
import urllib.request

import numpy as np
import soundfile as sf
import time
from qwen_asr import Qwen3ASRModel


def load_audio_from_url(url: str):
    """从 URL 下载音频并返回 (np.ndarray, sample_rate)"""
    with urllib.request.urlopen(url) as resp:
        audio_bytes = resp.read()
    with io.BytesIO(audio_bytes) as f:
        audio, sr = sf.read(f, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    sr = int(sr)
    return audio, sr


if __name__ == '__main__':
    # 预先下载音频，得到 numpy 数组
    urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    ]
    audio_list = [load_audio_from_url(u) for u in urls]
    languages = ["Chinese", "English"]

    model = Qwen3ASRModel.LLM(
        model="/workspace/model-bin/Qwen/Qwen3-ASR-0.6B",
        gpu_memory_utilization=0.5,
        max_model_len=4096,
        max_inference_batch_size=32,  # Batch size limit for inference. -1 means unlimited.
        max_new_tokens=512,  # Maximum number of tokens to generate.
    )

    # 逐条调用单条推理
    for audio, lang in zip(audio_list, languages):
        print(f"\n[stream] language={lang}")
        parts = []
        t_start = time.perf_counter()
        for chunk in model.transcribe_stream(
            audio=audio,
            language=lang,  # or None for automatic language detection
        ):
            if chunk:
                parts.append(chunk)
                print(chunk, end="", flush=True)
        elapsed = time.perf_counter() - t_start
        print()
        print(lang, "".join(parts))
        print(f"[timing] {lang} 转写耗时: {elapsed:.3f}s")
