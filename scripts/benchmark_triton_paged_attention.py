"""实际 910B shape 上对照 FIA，验证 Triton 解码注意力及动态图回放。"""

import json
from pathlib import Path
import time

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(output: Path = typer.Option(...), block: int = typer.Option(128), grouped: bool = typer.Option(False), online: bool = typer.Option(False), production: bool = typer.Option(False)):
    import torch
    import torch_npu
    from experimental_paged_attention import paged_attention, make_workspace
    if production:
        from qwen_asr.engines.vllm.ascend.paged_decode import make_workspace as production_workspace
        from qwen_asr.engines.vllm.ascend.paged_decode import paged_attention as production_attention
        if block != 64:raise typer.BadParameter('Production kernel requires --block 64')
        def make_workspace(batch,heads,dim,max_length,device,*_args):
            return production_workspace(batch,heads,dim,max_length,device)
        def paged_attention(query,key,value,table,lengths,result,workspace,*_args):
            return production_attention(query,key,value,table,lengths,result,workspace)

    torch.manual_seed(731)
    torch.set_num_threads(4)
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    mask = torch.ones((2048, 2048), device='npu', dtype=torch.bool).triu(1)
    for batch in range(1, 9):
        q = torch.randn(batch, 16, 128, device='npu', dtype=torch.bfloat16)
        k = torch.randn(batch * 8, 128, 8, 128, device='npu', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        table = torch.arange(batch * 8, device='npu', dtype=torch.int32).view(batch, 8)
        lengths = torch.empty(batch, device='npu', dtype=torch.int32)
        actual = torch.empty_like(q)
        workspace = make_workspace(batch, 16, 128, 1024, q.device, block, online)
        compiled_start = time.perf_counter()
        lengths.fill_(1024)
        paged_attention(q, k, v, table, lengths, actual, workspace, block, grouped, online)
        torch.npu.synchronize()
        compile_s = time.perf_counter() - compiled_start
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            paged_attention(q, k, v, table, lengths, actual, workspace, block, grouped, online)
        for iteration in range(3):
            for length in (1, 7, 63, 127, 128, 129, 255, 384, 511, 512, 1023, 1024):
                current = [max(1, length - 7 * i) for i in range(batch)]
                lengths.copy_(torch.tensor(current, device='npu', dtype=torch.int32))
                blocks = torch.arange(batch * 8, device='npu', dtype=torch.int32).view(batch, 8)
                if iteration % 2:blocks = blocks.flip(1)
                table.copy_(blocks)
                q.normal_()
                reference = torch_npu.npu_fused_infer_attention_score(
                    q, k.flatten(2, 3), v.flatten(2, 3), block_table=table, block_size=128, atten_mask=mask,
                    actual_seq_lengths=list(range(1, batch + 1)), actual_seq_lengths_kv=current,
                    input_layout='TND', num_heads=16, num_key_value_heads=8,
                    scale=128 ** -.5, sparse_mode=3,
                )[0]
                graph.replay()
                torch.npu.synchronize()
                error = (actual.float()-reference.float()).abs()
                row = {'batch':batch, 'length':length, 'iteration':iteration,
                       'max_abs':error.max().item(), 'mean_abs':error.mean().item(),
                       'compile_s':compile_s, 'block':block}
                if iteration == 0 and length in (7,129,1024):
                    indices=table[0].long()
                    keys=k[indices].flatten(0,1)[:current[0]].transpose(0,1).cpu().float().repeat_interleave(2,dim=0)
                    values=v[indices].flatten(0,1)[:current[0]].transpose(0,1).cpu().float().repeat_interleave(2,dim=0)
                    oracle=torch.nn.functional.scaled_dot_product_attention(q[0].cpu().float().unsqueeze(1),keys,values).squeeze(1)
                    fp32_error=(actual[0].cpu().float()-oracle).abs()
                    row['fp32_max_abs']=fp32_error.max().item()
                    row['fp32_mean_abs']=fp32_error.mean().item()
                    torch.testing.assert_close(actual[0].cpu().float(),oracle,atol=.015625,rtol=.02)
                rows.append(row)
                (output/'correctness.json').write_text(json.dumps(rows,indent=2))
                torch.testing.assert_close(actual, reference, atol=.03125, rtol=.02)
        # Graphs amortize host enqueue overhead equally for both implementations.
        for length in (64, 384, 1024):
            lengths.fill_(length)
            for label in ('triton', 'fia'):
                measure = torch.npu.NPUGraph()
                with torch.npu.graph(measure):
                    for _ in range(32):
                        if label == 'triton':
                            paged_attention(q,k,v,table,lengths,actual,workspace,block,grouped,online)
                        else:
                            vendor = torch_npu.npu_fused_infer_attention_score(
                                q,k.flatten(2,3),v.flatten(2,3),block_table=table,block_size=128,atten_mask=mask,
                                actual_seq_lengths=list(range(1,batch+1)),actual_seq_lengths_kv=[length]*batch,
                                input_layout='TND',num_heads=16,num_key_value_heads=8,
                                scale=128**-.5,sparse_mode=3)[0]
                for _ in range(3):measure.replay()
                torch.npu.synchronize()
                timings=[]
                for _ in range(3):
                    start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):measure.replay()
                    end.record();end.synchronize()
                    timings.append(start.elapsed_time(end)/320)
                with (output/'timings.jsonl').open('a') as stream:
                    stream.write(json.dumps({'batch':batch,'length':length,'implementation':label,
                                             'ms':timings,'block':block})+'\n')
        typer.echo(f'batch={batch}: numerical checks and graph replay passed')
    stride_rows=[]
    for batch in (1,8):
        q=torch.randn(batch,16,256,device='npu',dtype=torch.bfloat16)[...,::2]
        k=torch.randn(batch*8,128,8,128,device='npu',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        table=torch.empty(batch,16,device='npu',dtype=torch.int32)[:,::2]
        table.copy_(torch.arange(batch*8,device='npu',dtype=torch.int32).view(batch,8).flip(1))
        lengths=torch.full((batch,),129,device='npu',dtype=torch.int32)
        actual=torch.empty(batch,16,256,device='npu',dtype=torch.bfloat16)[...,::2]
        workspace=make_workspace(batch,16,128,1024,q.device,block,online)
        paged_attention(q,k,v,table,lengths,actual,workspace,block,grouped,online)
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):paged_attention(q,k,v,table,lengths,actual,workspace,block,grouped,online)
        for length in (1,63,128,129,511,1024):
            lengths.fill_(length)
            q.normal_()
            graph.replay();torch.npu.synchronize()
            reference=torch_npu.npu_fused_infer_attention_score(
                q.contiguous(),k.contiguous().flatten(2,3),v.contiguous().flatten(2,3),
                block_table=table.contiguous(),block_size=128,atten_mask=mask,
                actual_seq_lengths=list(range(1,batch+1)),actual_seq_lengths_kv=[length]*batch,
                input_layout='TND',num_heads=16,num_key_value_heads=8,scale=128**-.5,sparse_mode=3)[0]
            error=(actual.float()-reference.float()).abs()
            stride_rows.append({'batch':batch,'length':length,'max_abs':error.max().item(),
                                'mean_abs':error.mean().item(),'q_stride':list(q.stride()),
                                'kv_stride':list(k.stride()),'table_stride':list(table.stride())})
            torch.testing.assert_close(actual,reference,atol=.03125,rtol=.02)
        lengths[-1]=0
        graph.replay();torch.npu.synchronize()
        assert torch.count_nonzero(actual[-1]).item()==0,'padding query must produce zero'
        if production:
            bad_k=torch.empty(batch*8,128,8,256,device='npu',dtype=torch.bfloat16)[...,::2]
            try:paged_attention(q,bad_k,v,table,lengths,actual,workspace,block,grouped,online)
            except ValueError as exc:
                assert 'contiguous ND KV cache' in str(exc)
            else:raise AssertionError('Unsupported strided KV cache must be rejected before compilation')
    (output/'strides.json').write_text(json.dumps(stride_rows,indent=2))


if __name__=='__main__':app()
