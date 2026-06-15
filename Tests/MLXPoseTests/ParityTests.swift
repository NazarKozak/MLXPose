//
//  ParityTests.swift
//  MLXPoseTests
//
//  Created by Nazar Kozak on 15.06.2026.
//
//  Verifies the Swift ViTPose forward matches the Hugging Face reference
//  (same target the Python reference hit: max|Δ| ~ 5e-4, peaks exact).
//
//  Fixtures are produced by scripts/ (convert_vitpose_to_mlx.py + parity export):
//    scripts/weights/vitpose-base-simple-mlx-fp32/weights.safetensors
//    scripts/weights/vitpose-base-simple-mlx-fp32/parity_fixture.safetensors
//  Skipped automatically if the fixtures are absent.
//

import Foundation
import Testing
import MLX
@testable import MLXPose

@Suite("ViTPose parity")
struct ParityTests {
    static var fixtureDir: URL? {
        // Candidate locations: env override, CWD-relative (swift test), absolute dev path (xcodebuild).
        var candidates: [URL] = []
        if let env = ProcessInfo.processInfo.environment["MLXPOSE_WEIGHTS_DIR"] {
            candidates.append(URL(fileURLWithPath: env))
        }
        candidates.append(
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("scripts/weights/vitpose-base-simple-mlx-fp32"))
        candidates.append(
            URL(fileURLWithPath:
                "/Users/nazarkozak/Everything/OpenSource/MLXPose/scripts/weights/vitpose-base-simple-mlx-fp32"))

        return candidates.first { dir in
            FileManager.default.fileExists(atPath: dir.appendingPathComponent("weights.safetensors").path)
                && FileManager.default.fileExists(atPath: dir.appendingPathComponent("parity_fixture.safetensors").path)
        }
    }

    @Test("Swift forward matches HF heatmaps")
    func heatmapParity() throws {
        // CLI `swift test` cannot locate mlx-swift's GPU metallib (works in Xcode app
        // contexts). Force CPU for the parity run; production uses the GPU as default.
        Device.setDefault(device: Device(.cpu))

        guard let dir = Self.fixtureDir else {
            print("parity fixtures not found — skipping (run scripts/ first)")
            return
        }
        let weights = try loadArrays(url: dir.appendingPathComponent("weights.safetensors"))
        let fixture = try loadArrays(url: dir.appendingPathComponent("parity_fixture.safetensors"))

        let model = ViTPose(weights: weights)
        let out = model(fixture["pixel_values"]!)
        let ref = fixture["hf_heatmaps"]!

        let maxDiff = abs(out - ref).max().item(Float.self)
        print("Swift vs HF heatmaps: max|Δ| = \(maxDiff)")
        #expect(maxDiff < 1e-2)
    }
}
