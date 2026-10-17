"""不加载模型权重，验证本机 vLLM 的真实音频输入处理接口。"""

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: Path = typer.Option(..., envvar="QWEN_ASR_MODEL"),
    audio: Path = typer.Option(Path("data-bin/rita.wav")),
):
    from benchmark_asr import variants
    import numpy as np
    from qwen_asr.engines.vllm.plugin import register_qwen_asr

    register_qwen_asr()
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    from qwen_asr.models.transformers.processing_qwen3_asr import Qwen3ASRProcessor

    path = str(model.resolve())
    config = ModelConfig(model=path, dtype="bfloat16", max_model_len=1024)
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    hf = Qwen3ASRProcessor.from_pretrained(path, fix_mistral_regex=True)
    prompt = hf.apply_chat_template(
        [{"role": "system", "content": ""},
         {"role": "user", "content": [{"type": "audio", "audio": ""}]}],
        add_generation_prompt=True, tokenize=False,
    )
    sr, items = variants(audio, [10, 20, 30])
    rows = []
    for item in items:
        waveform = np.frombuffer(item["pcm"], dtype="<i2").astype(np.float32) / 32768
        parsed = processor.info.parse_mm_data({"audio": [(waveform, sr)]})
        inputs = ProcessorInputs(prompt, parsed)
        result = processor.apply(inputs, TimingContext(enabled=False))
        rows.append({"duration_s": item["duration_s"],
                     "prompt_tokens": len(result["prompt_token_ids"]),
                     "fields": list(result)})
    typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
