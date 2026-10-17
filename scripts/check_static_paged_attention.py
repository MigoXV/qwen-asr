"""验证 PA 图是否能直接读取变化的设备端长度与 block table。"""

import json
from pathlib import Path
import time

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(output: Path = typer.Option(...)):
    import torch
    import torch_npu

    torch.manual_seed(731)
    torch.set_num_threads(4)
    rows = []
    started = time.perf_counter()
    # Qwen3-ASR-1.7B: Q=16 heads, KV=8 heads, head_dim=128, block_size=128.
    for batch in range(1, 9):
        q = torch.randn(batch, 16, 128, device="npu", dtype=torch.bfloat16)
        k = torch.randn(batch * 8, 128, 8, 128, device="npu", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        table = torch.arange(batch * 8, device="npu", dtype=torch.int32).view(batch, 8)
        lengths = torch.full((batch,), 1024, device="npu", dtype=torch.int32)
        captured = torch.empty_like(q)
        reference = torch.empty_like(q)
        kwargs = dict(query=q, key_cache=k, value_cache=v, num_kv_heads=8,
                      num_heads=16, scale_value=128 ** -.5, block_table=table,
                      context_lens=lengths)
        workspace = torch_npu._npu_paged_attention_get_workspace(**kwargs, out=captured)
        for _ in range(3):
            torch_npu._npu_paged_attention(**kwargs, out=captured, workspace=workspace)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            torch_npu._npu_paged_attention(**kwargs, out=captured, workspace=workspace)
        for iteration in range(3):
            for length in (1, 7, 63, 64, 127, 128, 129, 255, 511, 512, 1023, 1024):
                current = [max(1, length - i * 7) for i in range(batch)]
                lengths.copy_(torch.tensor(current, dtype=torch.int32, device="npu"))
                # Alternate cache block order in place; captured addresses remain stable.
                blocks = torch.arange(batch * 8, device="npu", dtype=torch.int32).view(batch, 8)
                if iteration % 2:
                    blocks = blocks.flip(1)
                table.copy_(blocks)
                q.normal_()
                torch_npu._npu_paged_attention(**kwargs, out=reference, workspace=workspace)
                graph.replay()
                torch.npu.synchronize()
                error = (captured.float() - reference.float()).abs()
                rows.append({"batch": batch, "lengths": current, "iteration": iteration,
                             "max_abs": error.max().item(), "mean_abs": error.mean().item(),
                             "equal": bool(torch.equal(captured, reference))})
                torch.testing.assert_close(captured, reference, rtol=.02, atol=.03125)
                if iteration == 0 and length in (7, 129, 1024):
                    # FP32 oracle for the first sequence, preserving the current block order.
                    indices = table[0].long()
                    keys = k[indices].flatten(0, 1)[:current[0]].transpose(0, 1).float()
                    values = v[indices].flatten(0, 1)[:current[0]].transpose(0, 1).float()
                    keys = keys.repeat_interleave(2, dim=0)
                    values = values.repeat_interleave(2, dim=0)
                    oracle = torch.nn.functional.scaled_dot_product_attention(
                        q[0].float().unsqueeze(1), keys, values).squeeze(1)
                    ref_error = (captured[0].float() - oracle).abs()
                    rows[-1]["fp32_max_abs"] = ref_error.max().item()
                    rows[-1]["fp32_mean_abs"] = ref_error.mean().item()
                    torch.testing.assert_close(captured[0].float(), oracle, rtol=.02, atol=.03125)
    output.write_text(json.dumps({"cases": rows, "elapsed_s": time.perf_counter()-started,
                                  "atol": .03125, "rtol": .02}, indent=2))
    typer.echo(f"Validated {len(rows)} graph replays with changing lengths, blocks and queries")


if __name__ == "__main__":
    app()
