//
//  DownloadTests.swift
//  MLXPoseTests
//
//  Created by Nazar Kozak on 26.06.2026.
//
//  Verifies the Hugging Face auto-download path: fetch weights, load, run a forward.
//  Network test — skips gracefully if the Hub is unreachable.
//

import Foundation
import CoreGraphics
import Testing
import MLX
@testable import MLXPose

@Suite("Hub download")
struct DownloadTests {
    @Test("Downloads weights from HF and runs a forward")
    func downloadAndRun() async throws {
        Device.setDefault(device: Device(.cpu))

        let dir: URL
        do {
            dir = try await WeightStore.shared.directory(for: .vitPoseBaseSimple)
        } catch {
            print("hub unreachable (\(error)) — skipping"); return
        }

        let weightsURL = dir.appendingPathComponent("weights.safetensors")
        #expect(FileManager.default.fileExists(atPath: weightsURL.path))

        let weights = try loadArrays(url: weightsURL)
        let model = ViTPose(weights: weights)
        let input = MLXArray.zeros([1, 3, 256, 192])
        let heatmaps = model(input)
        #expect(heatmaps.shape == [1, 17, 64, 48])

        let cs = Geometry.centerScale(for: CGRect(x: 0, y: 0, width: 192, height: 256))
        let kps = HeatmapDecoder.decode(heatmaps, cs: cs)
        #expect(kps.count == 17)
        print("HF download OK — model produced \(kps.count) keypoints")
    }
}
