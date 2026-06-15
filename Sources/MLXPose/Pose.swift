//
//  Pose.swift
//  MLXPose
//
//  Created by Nazar Kozak on 15.06.2026.
//

import Foundation
import MLX

/// COCO-17 keypoint names (ViTPose `vitpose-base-simple` output order).
public enum COCOKeypoint: Int, CaseIterable, Sendable {
    case nose, leftEye, rightEye, leftEar, rightEar,
         leftShoulder, rightShoulder, leftElbow, rightElbow,
         leftWrist, rightWrist, leftHip, rightHip,
         leftKnee, rightKnee, leftAnkle, rightAnkle
}

/// A single keypoint in image coordinates.
public struct Keypoint: Sendable, Equatable {
    public let name: COCOKeypoint
    public let x: Float
    public let y: Float
    public let confidence: Float
}

/// A detected person's pose.
public struct Pose: Sendable {
    public let keypoints: [Keypoint]
    /// Source bounding box (image pixels): x, y, width, height.
    public let box: CGRect

    public func keypoint(_ name: COCOKeypoint) -> Keypoint {
        keypoints[name.rawValue]
    }
}

/// Decodes ViTPose heatmaps into keypoints using DARK sub-pixel refinement,
/// mirroring HF `post_process_pose_estimation` (argmax -> Gaussian modulation
/// kernel 11 + Taylor expansion on the log-heatmap -> transform_preds).
enum HeatmapDecoder {
    /// heatmaps: (1, K, H, W). Maps refined heatmap coords into image pixels via `cs`.
    static func decode(_ heatmaps: MLXArray, cs: CenterScale, kernel: Int = 11) -> [Keypoint] {
        let k = heatmaps.shape[1], h = heatmaps.shape[2], w = heatmaps.shape[3]
        let raw = heatmaps.asType(.float32).asArray(Float.self)   // (1,K,H,W) row-major

        return (0 ..< k).map { c in
            var map = Array(raw[(c * h * w) ..< ((c + 1) * h * w)])  // H*W

            // 1) argmax peak + score on raw heatmap
            var bestIdx = 0, best = -Float.greatestFiniteMagnitude
            for i in 0 ..< map.count where map[i] > best { best = map[i]; bestIdx = i }
            var px = Float(bestIdx % w), py = Float(bestIdx / w)
            let score = best

            // 2) DARK: gaussian blur (sigma 0.8, radius (kernel-1)/2) -> clip -> log
            let radius = (kernel - 1) / 2
            map = gaussianBlur(map, width: w, height: h, sigma: 0.8, radius: radius)
            for i in 0 ..< map.count { map[i] = Foundation.log(min(max(map[i], 0.001), 50)) }

            // 3) Taylor refinement at the integer peak (edge-clamped neighbours)
            let xi = Int(px), yi = Int(py)
            func at(_ yy: Int, _ xx: Int) -> Float {
                map[min(max(yy, 0), h - 1) * w + min(max(xx, 0), w - 1)]
            }
            let i0 = at(yi, xi)
            let ix1 = at(yi, xi + 1), ixm = at(yi, xi - 1)
            let iy1 = at(yi + 1, xi), iym = at(yi - 1, xi)
            let ix1y1 = at(yi + 1, xi + 1), ixmym = at(yi - 1, xi - 1)
            let dx = 0.5 * (ix1 - ixm)
            let dy = 0.5 * (iy1 - iym)
            let dxx = ix1 - 2 * i0 + ixm
            let dyy = iy1 - 2 * i0 + iym
            let dxy = 0.5 * (ix1y1 - ix1 - iy1 + i0 + i0 - ixm - iym + ixmym)
            let eps: Float = 1.1920929e-7
            let a = dxx + eps, d = dyy + eps, b = dxy
            let det = a * d - b * b
            if abs(det) > 1e-12 {
                // offset = inv(H) * [dx, dy];  coord -= offset
                let ox = (d * dx - b * dy) / det
                let oy = (-b * dx + a * dy) / det
                px -= ox; py -= oy
            }

            let (imgX, imgY) = Geometry.transformPred(x: px, y: py, cs: cs, heatmapW: w, heatmapH: h)
            return Keypoint(name: COCOKeypoint(rawValue: c)!, x: imgX, y: imgY, confidence: score)
        }
    }

    /// Separable Gaussian blur, scipy `gaussian_filter` semantics (reflect/half-sample).
    private static func gaussianBlur(_ src: [Float], width w: Int, height h: Int,
                                     sigma: Float, radius: Int) -> [Float] {
        var kern = [Float](repeating: 0, count: 2 * radius + 1)
        var sum: Float = 0
        for i in -radius ... radius {
            let v = Foundation.exp(-Float(i * i) / (2 * sigma * sigma))
            kern[i + radius] = v; sum += v
        }
        for i in 0 ..< kern.count { kern[i] /= sum }

        func reflect(_ i: Int, _ n: Int) -> Int {
            var j = i
            while j < 0 || j >= n { if j < 0 { j = -j - 1 }; if j >= n { j = 2 * n - j - 1 } }
            return j
        }

        // horizontal
        var tmp = [Float](repeating: 0, count: src.count)
        for y in 0 ..< h {
            for x in 0 ..< w {
                var acc: Float = 0
                for kk in -radius ... radius { acc += kern[kk + radius] * src[y * w + reflect(x + kk, w)] }
                tmp[y * w + x] = acc
            }
        }
        // vertical
        var out = [Float](repeating: 0, count: src.count)
        for y in 0 ..< h {
            for x in 0 ..< w {
                var acc: Float = 0
                for kk in -radius ... radius { acc += kern[kk + radius] * tmp[reflect(y + kk, h) * w + x] }
                out[y * w + x] = acc
            }
        }
        return out
    }
}
