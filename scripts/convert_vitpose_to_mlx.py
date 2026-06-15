#!/usr/bin/env python3
"""
convert_vitpose_to_mlx.py — Convert a Hugging Face ViTPose checkpoint to MLX safetensors.

Phase 0 of MLXPose. Produces MLX-format weights (+ config.json + a parameter report)
that the MLX Swift `MLXPose` package will load at runtime.

What it handles
---------------
- Downloads a ViTPose checkpoint from the HF Hub (default: usyd-community/vitpose-base-simple).
- Loads the PyTorch state dict (safetensors preferred; .bin fallback).
- Casts bf16/fp16 -> fp32 (or a chosen --dtype) safely.
- Re-lays-out conv weights from PyTorch to MLX:
    * Conv2d:          [O, I, kH, kW]  -> [O, kH, kW, I]
    * ConvTranspose2d: [I, O, kH, kW]  -> [O, kH, kW, I]   (used by the "classic" decoder / ViTPose++)
- Writes weights.safetensors (MLX), config.json, and conversion_report.txt.

Why a script (and not just transformers at runtime)
---------------------------------------------------
No MLX or mlx-community ViTPose conversion exists yet. Publishing the converted MLX
weights to HF is itself an adoption artifact. The Swift side then loads pure MLX —
no PyTorch, no cloud.

Usage
-----
    pip install huggingface_hub safetensors mlx torch numpy
    python convert_vitpose_to_mlx.py \
        --model usyd-community/vitpose-base-simple \
        --out ./weights/vitpose-base-simple-mlx \
        --dtype float16

Notes
-----
- The Swift module names are still being designed; HF parameter keys are preserved
  verbatim and a `normalize_key()` hook is provided so the Swift loader's expected
  names can be matched without re-running the heavy download. Adjust there, not here.
- Decoder type: `vitpose-base-simple` uses the *simple* decoder (bilinear upsample +
  Conv2d) -> only Conv2d re-layout is needed. The "plus"/classic-decoder checkpoints
  add ConvTranspose2d; pass --transpose-conv-substrings to flag those keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(f"[convert] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a HF ViTPose checkpoint to MLX safetensors.")
    p.add_argument("--model", default="usyd-community/vitpose-base-simple",
                   help="HF repo id of the ViTPose checkpoint.")
    p.add_argument("--out", default="./weights/vitpose-mlx",
                   help="Output directory for MLX weights + config.")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                   help="Target dtype for the MLX weights.")
    p.add_argument("--transpose-conv-substrings", nargs="*", default=["deconv"],
                   help="Key substrings whose 4-D weights are ConvTranspose2d (layout [I,O,kH,kW]).")
    p.add_argument("--revision", default=None, help="Optional HF revision/commit.")
    return p.parse_args()


def load_state_dict(model: str, revision: str | None):
    """Return (state_dict: dict[str, torch.Tensor], config: dict)."""
    from huggingface_hub import hf_hub_download

    # config.json — copied through to the MLX bundle.
    config_path = hf_hub_download(model, "config.json", revision=revision)
    with open(config_path) as f:
        config = json.load(f)

    # Try safetensors first, fall back to pytorch_model.bin.
    try:
        weights_path = hf_hub_download(model, "model.safetensors", revision=revision)
        from safetensors.torch import load_file
        log(f"loading safetensors: {weights_path}")
        sd = load_file(weights_path)
    except Exception as e:  # noqa: BLE001 — fall back path is intentional
        log(f"safetensors not available ({e}); trying pytorch_model.bin")
        import torch
        weights_path = hf_hub_download(model, "pytorch_model.bin", revision=revision)
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    return sd, config


def to_numpy_fp32(tensor):
    """torch.Tensor -> contiguous float32 numpy (handles bf16/fp16)."""
    return tensor.detach().to("cpu").float().contiguous().numpy()


def relayout(key: str, arr, transpose_conv_substrings: list[str]):
    """Apply PyTorch -> MLX conv weight layout changes. arr is a numpy array."""
    if arr.ndim != 4:
        return arr  # linear / norm / bias / embeddings: identical layout
    is_transpose = any(s in key for s in transpose_conv_substrings)
    if is_transpose:
        # ConvTranspose2d: [in, out, kH, kW] -> [out, kH, kW, in]
        return arr.transpose(1, 2, 3, 0)
    # Conv2d: [out, in, kH, kW] -> [out, kH, kW, in]
    return arr.transpose(0, 2, 3, 1)


def normalize_key(key: str) -> str:
    """
    Hook to remap HF parameter names to whatever the MLX Swift loader expects.
    Kept identity for now so the raw HF graph is preserved; edit here when the
    Swift module structure is finalized (cheap — no re-download needed).
    """
    return key


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mlx.core as mx
        import numpy as np  # noqa: F401  (used indirectly via numpy arrays)
    except ImportError:
        log("ERROR: `mlx` is required. Install with `pip install mlx` (Apple Silicon).")
        return 1

    log(f"model={args.model}  out={out_dir}  dtype={args.dtype}")
    sd, config = load_state_dict(args.model, args.revision)
    log(f"{len(sd)} tensors in state dict")

    mlx_dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[args.dtype]

    weights: dict = {}
    report_lines: list[str] = []
    total_params = 0
    relaid = 0
    for key, tensor in sd.items():
        arr = to_numpy_fp32(tensor)
        new_arr = relayout(key, arr, args.transpose_conv_substrings)
        if new_arr.shape != arr.shape:
            relaid += 1
        a = mx.array(new_arr).astype(mlx_dtype)
        out_key = normalize_key(key)
        weights[out_key] = a
        n = 1
        for d in a.shape:
            n *= d
        total_params += n
        report_lines.append(f"{out_key}\t{tuple(a.shape)}\t{args.dtype}")

    weights_path = out_dir / "weights.safetensors"
    mx.save_safetensors(str(weights_path), weights)
    log(f"wrote {weights_path}  ({total_params/1e6:.1f}M params, {relaid} conv tensors re-laid-out)")

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log("wrote config.json")

    report = out_dir / "conversion_report.txt"
    with open(report, "w") as f:
        f.write(f"source_model: {args.model}\n")
        f.write(f"revision: {args.revision}\n")
        f.write(f"dtype: {args.dtype}\n")
        f.write(f"total_params: {total_params}\n")
        f.write(f"conv_tensors_relaid_out: {relaid}\n")
        f.write("transpose_conv_substrings: " + ", ".join(args.transpose_conv_substrings) + "\n")
        f.write("\n# key\tshape\tdtype\n")
        f.write("\n".join(sorted(report_lines)) + "\n")
    log(f"wrote {report}")

    log("DONE. Next: load weights.safetensors in MLX (Python) and verify a forward pass "
        "against the HF model before wiring the Swift side.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
