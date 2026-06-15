//
//  RenderTests.swift
//  MLXPoseTests
//
//  Created by Nazar Kozak on 15.06.2026.
//
//  End-to-end visual check: real image -> BGRA CVPixelBuffer -> full PoseEstimator
//  (affine warp + model + DARK decode) -> skeleton drawn -> annotated PNG.
//  Exercises Preprocess on a real pixel buffer (the last piece verified only in numpy).
//

import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import Testing
import MLX
@testable import MLXPose

struct FixedBoxDetector: PersonDetector {
    let box: CGRect
    func detect(in pixelBuffer: CVPixelBuffer) async throws -> [CGRect] { [box] }
}

@Suite("Render")
struct RenderTests {
    @Test("Annotate a real image end-to-end")
    func annotate() async throws {
        Device.setDefault(device: Device(.cpu))
        guard let dir = ParityTests.fixtureDir else { print("weights not found — skipping"); return }

        // Load a real image (COCO sample with a person).
        let url = URL(string: "http://images.cocodataset.org/val2017/000000000785.jpg")!
        guard let data = try? Data(contentsOf: url),
              let srcImg = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(srcImg, 0, nil) else {
            print("image download failed — skipping"); return
        }
        let w = cg.width, h = cg.height

        // CGImage -> top-left-origin BGRA CVPixelBuffer.
        var pbOpt: CVPixelBuffer?
        CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferCGImageCompatibilityKey: true,
                             kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary, &pbOpt)
        let pb = try #require(pbOpt)
        CVPixelBufferLockBaseAddress(pb, [])
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h,
                            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)!
        ctx.translateBy(x: 0, y: CGFloat(h)); ctx.scaleBy(x: 1, y: -1)   // top-left origin
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        CVPixelBufferUnlockBaseAddress(pb, [])

        // Run the full pipeline.
        let estimator = try PoseEstimator(weightsDirectory: dir,
                                          detector: FixedBoxDetector(box: CGRect(x: 180, y: 40, width: 260, height: 360)))
        let poses = try await estimator.estimate(pb)
        #expect(!poses.isEmpty)
        let visible = poses.first!.keypoints.filter { $0.confidence >= 0.3 }.count
        print("detected \(poses.count) pose(s), \(visible)/17 keypoints above threshold")
        #expect(visible >= 10)

        // Draw skeleton over the image (top-left origin).
        let out = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        out.translateBy(x: 0, y: CGFloat(h)); out.scaleBy(x: 1, y: -1)
        out.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        out.setLineWidth(3); out.setStrokeColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))
        for pose in poses {
            for (a, b) in COCOKeypoint.skeleton {
                let ka = pose.keypoint(a), kb = pose.keypoint(b)
                guard ka.confidence >= 0.3, kb.confidence >= 0.3 else { continue }
                out.move(to: CGPoint(x: CGFloat(ka.x), y: CGFloat(ka.y)))
                out.addLine(to: CGPoint(x: CGFloat(kb.x), y: CGFloat(kb.y)))
                out.strokePath()
            }
            out.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
            for k in pose.keypoints where k.confidence >= 0.3 {
                out.fillEllipse(in: CGRect(x: CGFloat(k.x) - 4, y: CGFloat(k.y) - 4, width: 8, height: 8))
            }
        }
        let annotated = try #require(out.makeImage())

        // Write PNG into the repo (Examples/output) for visual confirmation.
        // Resolve the package root from this source file, not the CWD.
        let outDir = URL(fileURLWithPath: #filePath)          // .../Tests/MLXPoseTests/RenderTests.swift
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Examples/output")
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let pngURL = outDir.appendingPathComponent("annotated.png")
        let dest = try #require(CGImageDestinationCreateWithURL(pngURL as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(dest, annotated, nil)
        #expect(CGImageDestinationFinalize(dest))
        print("wrote \(pngURL.path)")
    }
}
