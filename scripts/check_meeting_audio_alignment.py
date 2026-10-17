"""隔离 AGC 与语言提示对 30 秒转写的影响；复用已运行的单模型服务。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import wave

import grpc

from audio_harness.media.agc import RealtimeAutomaticGainControl
from audio_harness.protos.asr import ux_speech_pb2 as pb
from audio_harness.protos.asr.ux_speech_pb2_grpc import UxSpeechStub
from benchmark_ai_meeting import inputs


async def run(output):
    output.mkdir(parents=True, exist_ok=True)
    raw = inputs(Path("/root/repositories/qwen-asr/data-bin/rita.wav"), [30])[-1]["pcm"]
    agc = RealtimeAutomaticGainControl(sample_rate=16000, attack_seconds=0,
        decay_seconds=2.5, soft_knee_db=6, initial_level_db=-12, output_gain_db=4.5)
    adjusted = b"".join(agc.process(raw[start:start + 3200]) for start in range(0, len(raw), 3200))
    agc.close()
    if len(adjusted) != len(raw) or len(raw) != 480000 * 2:
        raise RuntimeError("AGC changed 30-second audio coverage")
    with wave.open(str(output / "rita-30s-agc.wav"), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(adjusted)
    cases = []
    async with grpc.aio.insecure_channel("localhost:50011") as channel:
        stub = UxSpeechStub(channel)
        for name, pcm in (("raw", raw), ("agc", adjusted)):
            for language in ("", "zh-CN"):
                for repeat in range(3):
                    async def requests():
                        yield pb.StreamingRecognizeRequest(streaming_config=pb.StreamingRecognitionConfig(
                            config=pb.RecognitionConfig(encoding=pb.RecognitionConfig.LINEAR16,
                                sample_rate_hertz=16000, language_code=language), interim_results=True))
                        yield pb.StreamingRecognizeRequest(audio_content=pcm)
                    transcript = ""
                    final = False
                    async for response in stub.StreamingRecognize(requests(), timeout=180):
                        for result in response.results:
                            transcript = result.alternative.transcript
                            final = final or result.is_final
                    if not final or not transcript.strip():
                        raise RuntimeError("Missing final transcript")
                    cases.append(dict(audio=name, language=language, round=repeat,
                        samples=480000, duration_s=30, pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                        transcript=transcript, final=final))
    (output / "alignment.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    for name in ("raw", "agc"):
        for language in ("", "zh-CN"):
            texts={c["transcript"] for c in cases if c["audio"] == name and c["language"] == language}
            print(json.dumps(dict(audio=name, language=language, unique_transcripts=len(texts),
                                  characters=sorted({len(t) for t in texts}), endings=[t[-20:] for t in texts]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
