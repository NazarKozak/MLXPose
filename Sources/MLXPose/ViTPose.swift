//
//  ViTPose.swift
//  MLXPose
//
//  Created by Nazar Kozak on 15.06.2026.
//
//  Native MLX Swift port of ViTPose (base, simple decoder).
//  Mirrors the numerically-verified Python reference (scripts/mlx_vitpose.py):
//  full-pipeline parity vs Hugging Face was max|Δ|=0.00045, 17/17 keypoint peaks exact.
//
//  Two HF quirks replicated here (do not "simplify" them):
//    1. Patch-embed Conv2d uses padding = 2 (kernel 16, stride 16) -> 16x12 grid for 256x192.
//    2. Embeddings add position_embeddings[:, 1:] + position_embeddings[:, :1]
//       (patch positions PLUS the cls position broadcast to every token).
//

import Foundation
import MLX
import MLXNN

/// ViTPose backbone + simple decoder, operating on MLX arrays.
public struct ViTPose {
    public struct Config: Sendable {
        public var hidden = 768
        public var heads = 12
        public var layers = 12
        public var numKeypoints = 17
        public var lnEps: Float = 1e-12
        public init() {}
    }

    let cfg: Config
    let w: [String: MLXArray]   // weights cast to float32

    public init(weights: [String: MLXArray], config: Config = Config()) {
        self.cfg = config
        self.w = weights.mapValues { $0.asType(.float32) }
    }

    /// pixelValues: NCHW (1, 3, 256, 192) -> heatmaps (1, numKeypoints, 64, 48)
    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        decoder(backbone(pixelValues))
    }

    // MARK: - Backbone (1,3,256,192) -> NHWC feature (1,16,12,768)

    public func backbone(_ pixelValues: MLXArray) -> MLXArray {
        let headDim = cfg.hidden / cfg.heads
        let scale = 1.0 / Float(headDim).squareRoot()

        var x = pixelValues.asType(.float32).transposed(0, 2, 3, 1)   // NHWC (1,256,192,3)

        // Patch embedding (Conv2d, padding = 2)
        let peW = w["backbone.embeddings.patch_embeddings.projection.weight"]!
        let peB = w["backbone.embeddings.patch_embeddings.projection.bias"]!
        x = conv2d(x, peW, stride: 16, padding: 2) + peB              // (1,16,12,768)
        let hp = x.shape[1], wp = x.shape[2]
        x = x.reshaped(1, hp * wp, cfg.hidden)                        // (1,192,768)

        let pos = w["backbone.embeddings.position_embeddings"]!       // (1,193,768)
        x = x + pos[0..., 1...] + pos[0..., 0 ..< 1]                  // patch + cls positions

        for i in 0 ..< cfg.layers {
            let p = "backbone.encoder.layer.\(i)."
            let h = x
            var y = layerNorm(x, p + "layernorm_before")
            let t = y.shape[1]
            func splitHeads(_ a: MLXArray) -> MLXArray {
                a.reshaped(1, t, cfg.heads, headDim).transposed(0, 2, 1, 3)
            }
            let q = splitHeads(linear(y, p + "attention.attention.query"))
            let k = splitHeads(linear(y, p + "attention.attention.key"))
            let v = splitHeads(linear(y, p + "attention.attention.value"))
            let attn = softmax(matmul(q, k.transposed(0, 1, 3, 2)) * scale, axis: -1)
            var ctx = matmul(attn, v).transposed(0, 2, 1, 3).reshaped(1, t, cfg.hidden)
            ctx = linear(ctx, p + "attention.output.dense")
            x = h + ctx

            let h2 = x
            y = layerNorm(x, p + "layernorm_after")
            y = gelu(linear(y, p + "mlp.fc1"))
            y = linear(y, p + "mlp.fc2")
            x = h2 + y
        }

        x = layerNorm(x, "backbone.layernorm")
        return x.reshaped(1, hp, wp, cfg.hidden)                      // (1,16,12,768) NHWC
    }

    // MARK: - Simple decoder: ReLU -> bilinear upsample x4 -> Conv2d 3x3

    public func decoder(_ featNHWC: MLXArray) -> MLXArray {
        var x = relu(featNHWC.asType(.float32))
        x = bilinearUpsample(x, scale: 4, axis: 1)
        x = bilinearUpsample(x, scale: 4, axis: 2)                    // (1,64,48,768)
        x = conv2d(x, w["head.conv.weight"]!, stride: 1, padding: 1) + w["head.conv.bias"]!
        return x.transposed(0, 3, 1, 2)                              // (1,17,64,48)
    }

    // MARK: - Helpers

    private func linear(_ x: MLXArray, _ prefix: String) -> MLXArray {
        matmul(x, w[prefix + ".weight"]!.transposed(1, 0)) + w[prefix + ".bias"]!
    }

    private func layerNorm(_ x: MLXArray, _ prefix: String) -> MLXArray {
        let mu = x.mean(axis: -1, keepDims: true)
        let d = x - mu
        let variance = (d * d).mean(axis: -1, keepDims: true)
        return d / sqrt(variance + cfg.lnEps) * w[prefix + ".weight"]! + w[prefix + ".bias"]!
    }

    /// Bilinear upsample along one spatial axis, align_corners = false, integer scale.
    private func bilinearUpsample(_ x: MLXArray, scale: Int, axis: Int) -> MLXArray {
        let n = x.shape[axis]
        let out = n * scale
        let coords = (MLXArray((0 ..< out).map { Float($0) }) + 0.5) / Float(scale) - 0.5
        let pos = clip(coords, min: 0.0, max: Float(n - 1))
        let i0f = floor(pos)
        let w1flat = pos - i0f
        let i0 = i0f.asType(.int32)
        let i1 = clip(i0 + 1, min: MLXArray(Int32(0)), max: MLXArray(Int32(n - 1)))
        let shape = axis == 1 ? [1, out, 1, 1] : [1, 1, out, 1]
        let w1 = w1flat.reshaped(shape)
        let w0 = 1.0 - w1
        let a = take(x, i0, axis: axis)
        let b = take(x, i1, axis: axis)
        return a * w0 + b * w1
    }
}
