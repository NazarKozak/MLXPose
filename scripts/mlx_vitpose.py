#!/usr/bin/env python3
"""
mlx_vitpose.py — Python MLX reference implementation of ViTPose (base, simple decoder).

This is the executable spec the MLX Swift `MLXPose` port mirrors. Run as __main__ to
verify numerical parity against the Hugging Face reference (hf_reference.npz).

Architecture (from the converted weights):
  backbone.embeddings.patch_embeddings.projection : Conv2d 3->768, k16 s16   (NHWC weights)
  backbone.embeddings.position_embeddings          : (1, 193, 768)  -> add [:, 1:]
  backbone.encoder.layer.{0..11}                   : pre-norm ViT block (12 heads, dim 768)
  backbone.layernorm                               : final LN
  head.conv                                        : Conv2d 768->17, k3 s1 p1 (NHWC weights)
  decoder = upsample x4 (bilinear, align_corners=False) -> ReLU -> head.conv
"""
from __future__ import annotations
import math
import mlx.core as mx
import mlx.nn as nn

LN_EPS = 1e-12
NUM_HEADS = 12
HIDDEN = 768
HEAD_DIM = HIDDEN // NUM_HEADS
SCALE = 1.0 / math.sqrt(HEAD_DIM)


def layernorm(x, w, b, eps=LN_EPS):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / mx.sqrt(var + eps) * w + b


def linear(x, w, b):
    # HF/PyTorch Linear weight is (out, in): y = x @ w.T + b
    return x @ w.T + b


def bilinear_upsample_x(x, scale, axis):
    """Bilinear upsample along one spatial axis, align_corners=False, integer scale."""
    n = x.shape[axis]
    out = n * scale
    pos = (mx.arange(out) + 0.5) / scale - 0.5          # source coords
    pos = mx.clip(pos, 0.0, n - 1)
    i0 = mx.floor(pos)
    w1 = pos - i0
    i0 = i0.astype(mx.int32)
    i1 = mx.clip(i0 + 1, 0, n - 1)
    w1 = w1.reshape([-1] + [1] * (x.ndim - axis - 1))   # broadcast over trailing dims
    w0 = 1.0 - w1
    a = mx.take(x, i0, axis=axis)
    b = mx.take(x, i1, axis=axis)
    return a * w0 + b * w1


def backbone_forward(weights, pixel_values_nchw):
    """pixel_values_nchw (1,3,256,192) -> NHWC feature (1,16,12,768)."""
    w = {k: v.astype(mx.float32) for k, v in weights.items()}
    x = mx.transpose(pixel_values_nchw.astype(mx.float32), (0, 2, 3, 1))   # NHWC (1,256,192,3)

    # Patch embedding (Conv2d stride16)
    pe_w = w["backbone.embeddings.patch_embeddings.projection.weight"]     # (768,16,16,3)
    pe_b = w["backbone.embeddings.patch_embeddings.projection.bias"]
    x = mx.conv2d(x, pe_w, stride=16, padding=2) + pe_b                    # (1,16,12,768) — HF uses padding=2
    Hp, Wp = x.shape[1], x.shape[2]
    x = x.reshape(1, Hp * Wp, HIDDEN)                                      # (1,192,768)
    pos = w["backbone.embeddings.position_embeddings"]
    x = x + pos[:, 1:, :] + pos[:, :1, :]                                  # HF: patch positions + cls position (broadcast)

    for i in range(12):
        p = f"backbone.encoder.layer.{i}."
        h = x
        y = layernorm(x, w[p + "layernorm_before.weight"], w[p + "layernorm_before.bias"])
        q = linear(y, w[p + "attention.attention.query.weight"], w[p + "attention.attention.query.bias"])
        k = linear(y, w[p + "attention.attention.key.weight"], w[p + "attention.attention.key.bias"])
        v = linear(y, w[p + "attention.attention.value.weight"], w[p + "attention.attention.value.bias"])
        T = q.shape[1]
        def heads(t):
            return mx.transpose(t.reshape(1, T, NUM_HEADS, HEAD_DIM), (0, 2, 1, 3))
        q, k, v = heads(q), heads(k), heads(v)
        attn = mx.softmax((q @ mx.transpose(k, (0, 1, 3, 2))) * SCALE, axis=-1)
        ctx = mx.transpose(attn @ v, (0, 2, 1, 3)).reshape(1, T, HIDDEN)
        ctx = linear(ctx, w[p + "attention.output.dense.weight"], w[p + "attention.output.dense.bias"])
        x = h + ctx
        h2 = x
        y = layernorm(x, w[p + "layernorm_after.weight"], w[p + "layernorm_after.bias"])
        y = nn.gelu(linear(y, w[p + "mlp.fc1.weight"], w[p + "mlp.fc1.bias"]))
        y = linear(y, w[p + "mlp.fc2.weight"], w[p + "mlp.fc2.bias"])
        x = h2 + y

    x = layernorm(x, w["backbone.layernorm.weight"], w["backbone.layernorm.bias"])
    return x.reshape(1, Hp, Wp, HIDDEN)                                    # (1,16,12,768) NHWC


def decoder_forward(weights, feat_nhwc):
    """feat_nhwc (1,16,12,768) -> heatmaps (1,17,64,48). HF VitPoseSimpleDecoder order."""
    w = {k: v.astype(mx.float32) for k, v in weights.items()}
    x = nn.relu(feat_nhwc.astype(mx.float32))
    x = bilinear_upsample_x(x, 4, axis=1)
    x = bilinear_upsample_x(x, 4, axis=2)                                  # (1,64,48,768)
    x = mx.conv2d(x, w["head.conv.weight"], stride=1, padding=1) + w["head.conv.bias"]
    return mx.transpose(x, (0, 3, 1, 2))                                   # (1,17,64,48)


def forward(weights, pixel_values_nchw):
    """pixel_values_nchw (1,3,256,192) -> heatmaps (1,17,64,48)."""
    feat = backbone_forward(weights, pixel_values_nchw)
    return decoder_forward(weights, feat)


if __name__ == "__main__":
    import numpy as np, sys
    base = "weights/vitpose-base-simple-mlx"
    weights = mx.load(f"{base}/weights.safetensors")
    ref = np.load(f"{base}/hf_reference.npz")
    pv = mx.array(ref["pixel_values"])
    hm_mlx = np.array(forward(weights, pv))
    hm_hf = ref["heatmaps"]
    diff = np.abs(hm_mlx - hm_hf)
    print("MLX heatmaps:", hm_mlx.shape, "HF:", hm_hf.shape)
    print(f"max|Δ| = {diff.max():.6f}   mean|Δ| = {diff.mean():.6e}")
    # keypoint argmax agreement (per-channel peak location)
    def peaks(h):
        f = h.reshape(17, -1).argmax(1)
        return np.stack([f % 48, f // 48], 1)  # (x,y)
    pk_mlx, pk_hf = peaks(hm_mlx[0]), peaks(hm_hf[0])
    px = np.abs(pk_mlx - pk_hf)
    print(f"keypoint peak max pixel error = {px.max()} (x,y), exact matches {(px.sum(1)==0).sum()}/17")
    ok = diff.max() < 1e-2
    print("PARITY:", "PASS" if ok else "CHECK (see decoder upsample/order if large)")
    sys.exit(0 if ok else 1)
