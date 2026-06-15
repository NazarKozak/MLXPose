//
//  PersonDetector.swift
//  MLXPose
//
//  Created by Nazar Kozak on 15.06.2026.
//
//  ViTPose is top-down: it needs person bounding boxes first.
//  Default detector uses Apple Vision (on-device, free). Swappable via the protocol.
//

import Foundation
import CoreVideo

/// Supplies person bounding boxes (image pixels: x, y, width, height) for a frame.
public protocol PersonDetector: Sendable {
    func detect(in pixelBuffer: CVPixelBuffer) async throws -> [CGRect]
}

#if canImport(Vision)
import Vision

/// On-device person detector backed by Apple Vision.
public struct VisionPersonDetector: PersonDetector {
    public var minimumConfidence: Float

    public init(minimumConfidence: Float = 0.3) {
        self.minimumConfidence = minimumConfidence
    }

    public func detect(in pixelBuffer: CVPixelBuffer) async throws -> [CGRect] {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let request = VNDetectHumanRectanglesRequest()
        request.upperBodyOnly = false

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        try handler.perform([request])

        let observations = (request.results ?? [])
            .filter { $0.confidence >= minimumConfidence }

        // Vision boxes are normalized with bottom-left origin; convert to top-left pixels.
        return observations.map { obs in
            let bb = obs.boundingBox
            return CGRect(
                x: bb.minX * CGFloat(width),
                y: (1 - bb.maxY) * CGFloat(height),
                width: bb.width * CGFloat(width),
                height: bb.height * CGFloat(height)
            )
        }
    }
}
#endif
