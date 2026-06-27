//
//  BenchmarkTests.swift
//  MLXPoseTests
//
//  Created by Nazar Kozak on 26.06.2026.
//
//  Measures ViTPose forward throughput on the Apple Silicon GPU.
//

import Foundation
import Testing
import MLX
@testable import MLXPose

@Suite("Benchmark")
struct BenchmarkTests {
    @Test("ViTPose forward FPS (GPU)")
    func forwardFPS() async throws {
        Device.setDefault(device: Device(.gpu))

        let dir: URL
        do { dir = try await WeightStore.shared.directory(for: .vitPoseBaseSimple) }
        catch { print("hub unreachable — skipping benchmark"); return }

        let model = ViTPose(weights: try loadArrays(url: dir.appendingPathComponent("weights.safetensors")))
        let input = MLXArray.zeros([1, 3, 256, 192])

        for _ in 0 ..< 5 { eval(model(input)) }   // warmup

        let iters = 30
        let t0 = Date()
        for _ in 0 ..< iters { eval(model(input)) }
        let elapsed = Date().timeIntervalSince(t0)

        let msPerFrame = elapsed / Double(iters) * 1000
        let fps = Double(iters) / elapsed
        print(String(format: "ViTPose-base forward: %.2f ms/frame, %.1f FPS (GPU, batch 1, 256x192)", msPerFrame, fps))
        #expect(fps > 0)
    }
}
