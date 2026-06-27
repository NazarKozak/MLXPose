//
//  VideoTests.swift
//  MLXPoseTests
//
//  Created by Nazar Kozak on 26.06.2026.
//
//  Verifies the video pipeline: synthesize a short clip from a real image,
//  annotate every frame, and confirm an output video is produced.
//

import Foundation
import AVFoundation
import CoreGraphics
import CoreVideo
import ImageIO
import Testing
import MLX
@testable import MLXPose

@Suite("Video")
struct VideoTests {
    @Test("Annotate a synthesized clip end-to-end")
    func annotateVideo() async throws {
        Device.setDefault(device: Device(.cpu))
        guard let dir = ParityTests.fixtureDir else { print("weights not found — skipping"); return }

        // Real image -> even-sized CGImage.
        let url = URL(string: "http://images.cocodataset.org/val2017/000000000785.jpg")!
        guard let data = try? Data(contentsOf: url),
              let isrc = CGImageSourceCreateWithData(data as CFData, nil),
              var cg = CGImageSourceCreateImageAtIndex(isrc, 0, nil) else {
            print("image download failed — skipping"); return
        }
        let ew = cg.width & ~1, eh = cg.height & ~1
        cg = cg.cropping(to: CGRect(x: 0, y: 0, width: ew, height: eh)) ?? cg

        // Synthesize a ~16-frame source clip.
        let tmp = FileManager.default.temporaryDirectory
        let srcURL = tmp.appendingPathComponent("mlxpose-src-\(UUID().uuidString).mov")
        try await makeClip(cg: cg, frames: 16, fps: 24, url: srcURL)

        // Annotate it.
        let outURL = tmp.appendingPathComponent("mlxpose-out-\(UUID().uuidString).mov")
        let estimator = try PoseEstimator(weightsDirectory: dir, detector: VisionPersonDetector())
        let count = try await VideoPoseAnnotator.annotate(input: srcURL, output: outURL, with: estimator)

        print("annotated \(count) frames -> \(outURL.lastPathComponent)")
        #expect(count >= 12)
        #expect(FileManager.default.fileExists(atPath: outURL.path))
        let size = (try FileManager.default.attributesOfItem(atPath: outURL.path)[.size] as? Int) ?? 0
        #expect(size > 0)

        // The output must decode to at least one frame.
        let outAsset = AVURLAsset(url: outURL)
        let gen = AVAssetImageGenerator(asset: outAsset)
        let frame = try? await gen.image(at: CMTime(value: 1, timescale: 24)).image
        #expect(frame != nil)
    }

    private func makeClip(cg: CGImage, frames: Int, fps: Int32, url: URL) async throws {
        if FileManager.default.fileExists(atPath: url.path) { try? FileManager.default.removeItem(at: url) }
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: cg.width, AVVideoHeightKey: cg.height])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: cg.width, kCVPixelBufferHeightKey as String: cg.height])
        writer.add(input)
        #expect(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        let pb = PoseRenderer.pixelBuffer(from: cg)!
        for i in 0 ..< frames {
            while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(2)) }
            adaptor.append(pb, withPresentationTime: CMTime(value: Int64(i), timescale: fps))
        }
        input.markAsFinished()
        await withCheckedContinuation { c in writer.finishWriting { c.resume() } }
    }
}
