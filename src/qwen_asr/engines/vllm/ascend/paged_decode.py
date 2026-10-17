"""910B BF16 Qwen3-ASR-1.7B 解码：GQA 共享 KV 的 Triton split-KV。

长度和页表由设备张量读取，图回放无需逐层更新 Host tiling。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_gqa(Q, K, V, Table, Lengths, Partial, Maxima, Sums,
                 QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
                 KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr, KS3: tl.constexpr,
                 VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr, VS3: tl.constexpr,
                 TS0: tl.constexpr, TS1: tl.constexpr,
                 HEADS: tl.constexpr, KV_HEADS: tl.constexpr, DIM: tl.constexpr,
                 PARTS: tl.constexpr, BLOCK: tl.constexpr, PAGE: tl.constexpr,
                 SCALE: tl.constexpr):
    job = tl.program_id(0)
    part = job % PARTS
    bkv = job // PARTS
    batch = bkv // KV_HEADS
    kv_head = bkv % KV_HEADS
    head = kv_head * 2
    length = tl.load(Lengths + batch)
    dims = tl.arange(0, DIM)
    span = tl.cdiv(tl.cdiv(length, BLOCK), PARTS) * BLOCK
    end = tl.minimum(length, (part+1)*span)
    q0 = tl.load(Q + batch * QS0 + head * QS1 + dims * QS2).to(tl.float32)
    q1 = tl.load(Q + batch * QS0 + (head+1) * QS1 + dims * QS2).to(tl.float32)
    max0, max1 = tl.full((),-1.e30,tl.float32), tl.full((),-1.e30,tl.float32)
    sum0, sum1 = tl.full((),0.,tl.float32), tl.full((),0.,tl.float32)
    acc0, acc1 = tl.full((DIM,),0.,tl.float32), tl.full((DIM,),0.,tl.float32)
    for step in range(4):
        indices = part * span + step * BLOCK + tl.arange(0, BLOCK)
        if part * span + step * BLOCK < end:
            pages = tl.load(Table + batch * TS0 + (indices // PAGE) * TS1,
                            mask=indices < end, other=0)
            key = tl.load(K + pages[:, None] * KS0 + (indices % PAGE)[:, None] * KS1
                          + kv_head * KS2 + dims[None, :] * KS3,
                          mask=indices[:, None] < end, other=0).to(tl.float32)
            scores0 = tl.sum(key * q0[None, :], axis=1) * SCALE
            scores1 = tl.sum(key * q1[None, :], axis=1) * SCALE
            scores0 = tl.where(indices < end, scores0, float("-inf"))
            scores1 = tl.where(indices < end, scores1, float("-inf"))
            new0 = tl.maximum(max0,tl.max(scores0,axis=0))
            new1 = tl.maximum(max1,tl.max(scores1,axis=0))
            alpha0, alpha1 = tl.exp(max0-new0), tl.exp(max1-new1)
            p0, p1 = tl.exp(scores0-new0), tl.exp(scores1-new1)
            sum0 = sum0 * alpha0 + tl.sum(p0,axis=0)
            sum1 = sum1 * alpha1 + tl.sum(p1,axis=0)
            value = tl.load(V + pages[:, None] * VS0 + (indices % PAGE)[:, None] * VS1
                            + kv_head * VS2 + dims[None, :] * VS3,
                            mask=indices[:, None] < end, other=0).to(tl.float32)
            acc0 = acc0 * alpha0 + tl.sum(p0[:,None]*value,axis=0)
            acc1 = acc1 * alpha1 + tl.sum(p1[:,None]*value,axis=0)
            max0, max1 = new0, new1
    bh = batch * HEADS + head
    tl.store(Partial + (bh * PARTS + part) * DIM + dims, acc0)
    tl.store(Partial + ((bh+1) * PARTS + part) * DIM + dims, acc1)
    tl.store(Maxima + bh * PARTS + part, max0)
    tl.store(Maxima + (bh+1) * PARTS + part, max1)
    tl.store(Sums + bh * PARTS + part, sum0)
    tl.store(Sums + (bh+1) * PARTS + part, sum1)


@triton.jit
def _merge(Partial, Maxima, Sums, Out,
           OS0: tl.constexpr, OS1: tl.constexpr, OS2: tl.constexpr,
           HEADS: tl.constexpr, DIM: tl.constexpr, PARTS: tl.constexpr):
    bh = tl.program_id(0)
    parts = tl.arange(0, PARTS)
    dims = tl.arange(0, DIM)
    maxima = tl.load(Maxima + bh * PARTS + parts)
    sums = tl.load(Sums + bh * PARTS + parts)
    maximum = tl.max(maxima, axis=0)
    factors = tl.exp(maxima - maximum)
    total = tl.sum(sums * factors, axis=0)
    partial = tl.load(Partial + (bh * PARTS + parts[:, None]) * DIM + dims[None, :])
    value = tl.sum(partial * factors[:, None], axis=0) / tl.maximum(total, 1.0e-20)
    tl.store(Out + (bh // HEADS) * OS0 + (bh % HEADS) * OS1 + dims * OS2, value)


def make_workspace(batch, heads, dim, max_length, device, block=64):
    if max_length != 1024 or block != 64:
        raise ValueError("Validated profile requires max_length=1024 and block=64")
    parts = 4
    return (torch.empty((batch * heads, parts, dim), device=device, dtype=torch.float32),
            torch.empty((batch * heads, parts), device=device, dtype=torch.float32),
            torch.empty((batch * heads, parts), device=device, dtype=torch.float32))


def paged_attention(query, key, value, table, lengths, output, workspace, block=64):
    batch, heads, dim = query.shape
    if (not 1 <= batch <= 8 or dim != 128 or heads != 16 or key.ndim != 4
            or key.shape[2:] != (8,128) or key.shape[1] != 128
            or value.shape != key.shape or output.shape != query.shape
            or any(t.dtype != torch.bfloat16 for t in (query,key,value,output))
            or any(t.device != query.device for t in (key,value,output,table,lengths))
            or query.device.type != 'npu' or lengths.dtype != torch.int32
            or lengths.ndim != 1 or lengths.stride(0) != 1 or len(lengths) < batch
            or table.ndim != 2 or table.shape[0] < batch or table.shape[1] < 8
            or table.dtype != torch.int32):
        raise ValueError('Prototype requires ASR BF16 Q16/KV8/D128/page128 and device lengths')
    if not key.is_contiguous() or not value.is_contiguous():
        raise ValueError('Triton decode requires contiguous ND KV cache; use the upstream attention path for other layouts')
    # Only small query/table/output views may need normalization. Never copy
    # the entire KV cache or let unsupported cache strides reach the compiler.
    destination = output
    query = query.contiguous()
    table = table.contiguous()
    if not output.is_contiguous():
        output = torch.empty_like(query)
    partial, maxima, sums = workspace
    parts = maxima.shape[1]
    _partial_gqa[(batch * key.shape[2] * parts,)](
        query, key, value, table, lengths, partial, maxima, sums,
        *query.stride(), *key.stride(), *value.stride(), *table.stride(),
        heads, key.shape[2], dim, parts, block, key.shape[1], dim ** -.5,
        num_warps=4,
    )
    _merge[(batch * heads,)](partial, maxima, sums, output, *output.stride(),
                           heads, dim, parts, num_warps=4)
    if destination is not output:
        destination.copy_(output)
    return destination
