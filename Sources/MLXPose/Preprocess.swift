//
//  Preprocess.swift
//  MLXPose
//
//  Created by Nazar Kozak on 15.06.2026.
//
//  UDP affine warp (theta = 0) from a person box into normalized model input
//  (1, 3, 256, 192). Verified against HF VitPoseImageProcessor (max|Δ| ≈ 0.009).
//  Reads the CVPixelBuffer's BGRA bytes directly (top-left origin), bilinear,
//  zero outside the source — matching scipy/cv2 warpAffine with cval 0.
//

import Foundation
import CoreGraphics
import CoreVideo
import MLX

enum Preprocess {
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    enum Error: Swift.Error { case unsupportedPixelFormat(OSType) }

    /// Returns NCHW (1, 3, 256, 192) float32 normalized input for one person box.
    static func makeInput(_ pixelBuffer: CVPixelBuffer, cs: CenterScale) throws -> MLXArray {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_32BGRA else { throw Error.unsupportedPixelFormat(format) }

        let outW = Geometry.inputWidth, outH = Geometry.inputHeight
        let (sx, sy, tx, ty) = Geometry.warpCoefficients(cs)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let base = CVPixelBufferGetBaseAddress(pixelBuffer)!.assumingMemoryBound(to: UInt8.self)

        // BGRA sample (returns RGB), zero outside.
        func sample(_ fx: Float, _ fy: Float) -> (Float, Float, Float) {
            let x0 = Int(floor(fx)), y0 = Int(floor(fy))
            let wx = fx - Float(x0), wy = fy - Float(y0)
            func px(_ x: Int, _ y: Int) -> (Float, Float, Float) {
                guard x >= 0, x < srcW, y >= 0, y < srcH else { return (0, 0, 0) }
                let p = y * rowBytes + x * 4
                return (Float(base[p + 2]), Float(base[p + 1]), Float(base[p]))  // R,G,B from BGRA
            }
            let (r00, g00, b00) = px(x0, y0), (r10, g10, b10) = px(x0 + 1, y0)
            let (r01, g01, b01) = px(x0, y0 + 1), (r11, g11, b11) = px(x0 + 1, y0 + 1)
            func lerp(_ a: Float, _ b: Float, _ c: Float, _ d: Float) -> Float {
                a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy
            }
            return (lerp(r00, r10, r01, r11), lerp(g00, g10, g01, g11), lerp(b00, b10, b01, b11))
        }

        var chw = [Float](repeating: 0, count: 3 * outH * outW)
        for v in 0 ..< outH {
            for u in 0 ..< outW {
                let fx = (Float(u) - tx) / sx
                let fy = (Float(v) - ty) / sy
                let (r, g, b) = sample(fx, fy)
                let rgb = [r, g, b]
                for c in 0 ..< 3 {
                    chw[c * outH * outW + v * outW + u] = (rgb[c] / 255.0 - mean[c]) / std[c]
                }
            }
        }
        return MLXArray(chw, [1, 3, outH, outW])
    }
}
