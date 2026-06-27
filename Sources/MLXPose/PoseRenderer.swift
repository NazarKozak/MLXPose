//
//  PoseRenderer.swift
//  MLXPose
//
//  Created by Nazar Kozak on 26.06.2026.
//
//  Helpers to run pose on a CGImage and draw the skeleton — shared by the
//  image/video demos and tests.
//

import Foundation
import CoreGraphics
import CoreVideo

public enum PoseRenderer {
    /// Build a top-left-origin BGRA CVPixelBuffer from a CGImage (matches `Preprocess`).
    public static func pixelBuffer(from cg: CGImage) -> CVPixelBuffer? {
        let w = cg.width, h = cg.height
        var pbOpt: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs, &pbOpt)
        guard let pb = pbOpt else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.translateBy(x: 0, y: CGFloat(h)); ctx.scaleBy(x: 1, y: -1)   // top-left origin
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return pb
    }

    /// Draw skeletons over a CGImage and return the annotated image.
    public static func draw(_ poses: [Pose], on cg: CGImage, threshold: Float = 0.3,
                            lineWidth: CGFloat = 3, pointRadius: CGFloat = 4) -> CGImage {
        let w = cg.width, h = cg.height
        let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.translateBy(x: 0, y: CGFloat(h)); ctx.scaleBy(x: 1, y: -1)
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        ctx.setLineWidth(lineWidth); ctx.setLineCap(.round)
        for pose in poses {
            ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))
            for (a, b) in COCOKeypoint.skeleton {
                let ka = pose.keypoint(a), kb = pose.keypoint(b)
                guard ka.confidence >= threshold, kb.confidence >= threshold else { continue }
                ctx.move(to: CGPoint(x: CGFloat(ka.x), y: CGFloat(ka.y)))
                ctx.addLine(to: CGPoint(x: CGFloat(kb.x), y: CGFloat(kb.y)))
                ctx.strokePath()
            }
            ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
            for k in pose.keypoints where k.confidence >= threshold {
                ctx.fillEllipse(in: CGRect(x: CGFloat(k.x) - pointRadius, y: CGFloat(k.y) - pointRadius,
                                           width: 2 * pointRadius, height: 2 * pointRadius))
            }
        }
        return ctx.makeImage()!
    }

    /// Detect + estimate on a CGImage and return the annotated image.
    public static func annotate(_ cg: CGImage, with estimator: PoseEstimator,
                                threshold: Float = 0.3) async throws -> CGImage {
        guard let pb = pixelBuffer(from: cg) else { return cg }
        let poses = try await estimator.estimate(pb)
        return draw(poses, on: cg, threshold: threshold)
    }
}
