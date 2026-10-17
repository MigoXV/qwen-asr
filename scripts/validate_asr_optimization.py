"""中英文、混合长度和非整数批次的真实 RPC 一致性与长稳态验证。"""

import asyncio
import hashlib
import json
from pathlib import Path

import grpc
import numpy as np
import typer

from benchmark_asr import request, variants
from qwen_asr.protos.asr.ux_speech_pb2_grpc import UxSpeechStub

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output: Path = typer.Option(...),
    audio_root: Path = typer.Option(...),
    baseline: Path | None = typer.Option(None),
    server: str = typer.Option('127.0.0.1:50011'),
    rounds: int = typer.Option(3,min=1),
    soak_rounds: int = typer.Option(0,min=0),
):
    import soundfile as sf

    output.mkdir(parents=True,exist_ok=False)
    sr, source = variants(Path('data-bin/rita.wav'),[10,20,30])
    inputs = [{**r,'sample_rate':sr} for r in source]
    for name in ('asr_zh.wav','asr_en.wav'):
        waveform,rate = sf.read(audio_root/name,dtype='float32')
        pcm = (np.clip(waveform,-1,32767/32768)*32768).astype('<i2').tobytes()
        inputs.append({'name':name,'duration_s':len(waveform)/rate,'pcm':pcm,
                       'sample_rate':rate,'sha256':hashlib.sha256(pcm).hexdigest()})
    oracle = json.loads(baseline.read_text())['transcripts'] if baseline else None
    transcripts = {r['name']:set() for r in inputs}
    errors = []

    async def run():
        async with grpc.aio.insecure_channel(server) as channel:
            await asyncio.wait_for(channel.channel_ready(),30)
            stub = UxSpeechStub(channel)
            async def batch(size,iteration,label):
                async def one(index):
                    item=inputs[(index+iteration) % len(inputs)]
                    result=await request(stub,item['pcm'],item['sample_rate'],180)
                    transcripts[item['name']].add(result['transcript'])
                    mismatch = oracle is not None and result['transcript'] not in oracle[item['name']]
                    row={'input':item['name'],'clients':size,'iteration':iteration,'label':label,
                         'mismatch':mismatch,**result}
                    if result['error'] or mismatch:errors.append(row)
                    with (output/'raw.jsonl').open('a') as stream:
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                await asyncio.gather(*(one(i) for i in range(size)))
            for iteration in range(rounds):
                for size in (1,2,3,4,5,6,7,8,16,32):
                    await batch(size,iteration,'mixed')
            for iteration in range(soak_rounds):
                await batch(32,iteration,'soak')
                if (iteration+1)%10 == 0:typer.echo(f'completed soak rounds={iteration+1}')
    asyncio.run(run())
    result={'transcripts':{k:sorted(v) for k,v in transcripts.items()},'errors':errors,
            'inputs':[{k:v for k,v in item.items() if k!='pcm'} for item in inputs],
            'rounds':rounds,'soak_rounds':soak_rounds,
            'requests':sum(1 for _ in (output/'raw.jsonl').open())}
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    typer.echo(f"requests={result['requests']} errors/mismatches={len(errors)}")
    if errors:raise typer.Exit(1)


if __name__=='__main__':app()
