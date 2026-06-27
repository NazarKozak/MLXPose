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

public struct PoseEstimator: @unchecked Sendable {
    let model: ViTPose
    let detector: PersonDetector

    /// Download the model from the Hugging Face Hub (cached) and load it.
    public init(model: Model = .vitPoseBaseSimple, detector: PersonDetector) async throws {
        let dir = try await WeightStore.shared.directory(for: model)
        try self.init(weightsDirectory: dir, detector: detector)
    }

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
        return try boxes.map { box in
            let cs = Geometry.centerScale(for: box)
            let input = try Preprocess.makeInput(pixelBuffer, cs: cs)
            let heatmaps = model(input)
            let keypoints = HeatmapDecoder.decode(heatmaps, cs: cs)
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

    /// Download from HF Hub (cached) and use the default Apple Vision person detector.
    init(model: Model = .vitPoseBaseSimple) async throws {
        try await self.init(model: model, detector: VisionPersonDetector())
    }
}
#endif
