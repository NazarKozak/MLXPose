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

/// Decodes ViTPose heatmaps into keypoints.
///
/// v0: per-channel argmax peak. TODO(parity): add Gaussian modulation (kernel 11)
/// to match HF `post_process_pose_estimation` sub-pixel accuracy.
enum HeatmapDecoder {
    /// heatmaps: (1, K, H, W). Returns keypoints mapped from heatmap space into `box`.
    static func decode(_ heatmaps: MLXArray, box: CGRect) -> [Keypoint] {
        let k = heatmaps.shape[1]
        let h = heatmaps.shape[2]
        let wd = heatmaps.shape[3]

        let flat = heatmaps.reshaped(k, h * wd)
        let idx = argMax(flat, axis: 1).asArray(Int32.self)
        let maxv = flat.max(axis: 1).asArray(Float.self)

        return (0 ..< k).map { i in
            let pos = Int(idx[i])
            let hx = Float(pos % wd)
            let hy = Float(pos / wd)
            // heatmap -> box -> image
            let x = Float(box.minX) + hx / Float(wd) * Float(box.width)
            let y = Float(box.minY) + hy / Float(h) * Float(box.height)
            return Keypoint(name: COCOKeypoint(rawValue: i)!, x: x, y: y, confidence: maxv[i])
        }
    }
}
