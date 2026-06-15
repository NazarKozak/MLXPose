//
//  Preprocess.swift
//  MLXPose
//
//  Created by Nazar Kozak on 15.06.2026.
//
//  Crop a person box and produce normalized model input (1, 3, 256, 192).
//  v0: aspect-agnostic resize. TODO(parity): affine warp + aspect padding to match
//  HF VitPoseImageProcessor (center/scale, normalize_factor 200) for sub-pixel accuracy.
//

import Foundation
import CoreVideo
import CoreImage
import MLX

enum Preprocess {
    static let inputWidth = 192
    static let inputHeight = 256
    // ImageNet normalization (HF default).
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    /// Returns NCHW (1, 3, 256, 192) float32 normalized input for one person box.
    static func makeInput(_ pixelBuffer: CVPixelBuffer, box: CGRect) -> MLXArray {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let cropped = image.cropped(to: box.integral)

        let sx = CGFloat(inputWidth) / max(cropped.extent.width, 1)
        let sy = CGFloat(inputHeight) / max(cropped.extent.height, 1)
        let scaled = cropped
            .transformed(by: CGAffineTransform(translationX: -cropped.extent.minX, y: -cropped.extent.minY))
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        let w = inputWidth, h = inputHeight
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        rgba.withUnsafeMutableBytes { ptr in
            ciContext.render(
                scaled,
                toBitmap: ptr.baseAddress!,
                rowBytes: w * 4,
                bounds: CGRect(x: 0, y: 0, width: w, height: h),
                format: .RGBA8,
                colorSpace: CGColorSpaceCreateDeviceRGB()
            )
        }

        // -> normalized CHW floats
        var chw = [Float](repeating: 0, count: 3 * h * w)
        for y in 0 ..< h {
            for x in 0 ..< w {
                let p = (y * w + x) * 4
                for c in 0 ..< 3 {
                    let v = Float(rgba[p + c]) / 255.0
                    chw[c * h * w + y * w + x] = (v - mean[c]) / std[c]
                }
            }
        }
        return MLXArray(chw, [1, 3, h, w])
    }
}
