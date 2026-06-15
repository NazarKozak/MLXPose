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
import CoreGraphics
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

    @Test("Decoded keypoints match HF (image coords)")
    func keypointParity() throws {
        Device.setDefault(device: Device(.cpu))
        guard let dir = Self.fixtureDir else {
            print("parity fixtures not found — skipping")
            return
        }
        let kpURL = dir.appendingPathComponent("parity_kp_fixture.safetensors")
        guard FileManager.default.fileExists(atPath: kpURL.path) else {
            print("kp fixture not found — skipping"); return
        }
        let weights = try loadArrays(url: dir.appendingPathComponent("weights.safetensors"))
        let f = try loadArrays(url: kpURL)

        let model = ViTPose(weights: weights)
        let heatmaps = model(f["pixel_values"]!)

        let box = f["box"]!.asArray(Float.self)   // x, y, w, h
        let cs = Geometry.centerScale(for: CGRect(x: CGFloat(box[0]), y: CGFloat(box[1]),
                                                  width: CGFloat(box[2]), height: CGFloat(box[3])))
        let kps = HeatmapDecoder.decode(heatmaps, cs: cs)

        let hf = f["hf_keypoints"]!.asArray(Float.self)   // (17,3): x,y,score
        var maxErr: Float = 0
        for i in 0 ..< kps.count {
            let dx = abs(kps[i].x - hf[i * 3 + 0])
            let dy = abs(kps[i].y - hf[i * 3 + 1])
            maxErr = max(maxErr, max(dx, dy))
        }
        print("Swift vs HF keypoints: max pixel error = \(maxErr)")
        #expect(maxErr < 2.0)
    }
}
