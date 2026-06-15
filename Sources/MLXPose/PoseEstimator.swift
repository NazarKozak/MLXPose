//
//  PoseEstimator.swift
//  MLXPose
//
//  Created by Nazar Kozak on 15.06.2026.
//
//  Public entry point: person detection (top-down) + ViTPose keypoints, on-device.
//

import Foundation
import CoreVideo
import MLX

public struct PoseEstimator {
    let model: ViTPose
    let detector: PersonDetector

    /// Load converted MLX weights (`weights.safetensors`) from a directory.
    public init(weightsDirectory: URL, detector: PersonDetector) throws {
        let url = weightsDirectory.appendingPathComponent("weights.safetensors")
        let weights = try loadArrays(url: url)
        self.model = ViTPose(weights: weights)
        self.detector = detector
    }

    /// Lower-level init from preloaded weights.
    public init(weights: [String: MLXArray], detector: PersonDetector) {
        self.model = ViTPose(weights: weights)
        self.detector = detector
    }

    /// Detect people, then estimate keypoints for each. Returns one `Pose` per person.
    public func estimate(_ pixelBuffer: CVPixelBuffer) async throws -> [Pose] {
        let boxes = try await detector.detect(in: pixelBuffer)
        return boxes.map { box in
            let input = Preprocess.makeInput(pixelBuffer, box: box)
            let heatmaps = model(input)
            let keypoints = HeatmapDecoder.decode(heatmaps, box: box)
            return Pose(keypoints: keypoints, box: box)
        }
    }

    /// Run the model directly on an already-preprocessed input (1,3,256,192).
    /// Useful for parity tests against the Python reference.
    public func heatmaps(forInput input: MLXArray) -> MLXArray {
        model(input)
    }
}

#if canImport(Vision)
public extension PoseEstimator {
    /// Convenience init with the default Apple Vision person detector.
    init(weightsDirectory: URL) throws {
        try self.init(weightsDirectory: weightsDirectory, detector: VisionPersonDetector())
    }
}
#endif
