//
//  VideoPoseAnnotator.swift
//  MLXPose
//
//  Created by Nazar Kozak on 26.06.2026.
//
//  Reads a video, runs ViTPose on every frame, and writes an output video with
//  the skeleton burned in. Honors the track's preferred transform (orientation).
//

import Foundation
import AVFoundation
import CoreImage
import CoreGraphics
import CoreVideo

public enum VideoPoseAnnotator {
    public enum Error: Swift.Error { case noVideoTrack, pipelineInit, encodeFailed }

    /// Annotate `input` -> `output` (.mov, H.264). Returns the number of frames written.
    @discardableResult
    public static func annotate(input: URL, output: URL, with estimator: PoseEstimator,
                                progress: (@Sendable (Double) -> Void)? = nil) async throws -> Int {
        let asset = AVURLAsset(url: input)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw Error.noVideoTrack }
        let transform = try await track.load(.preferredTransform)
        let duration = CMTimeGetSeconds(try await asset.load(.duration))

        let reader = try AVAssetReader(asset: asset)
        let readerOutput = AVAssetReaderTrackOutput(
            track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        readerOutput.alwaysCopiesSampleData = false
        reader.add(readerOutput)
        guard reader.startReading() else { throw reader.error ?? Error.pipelineInit }

        if FileManager.default.fileExists(atPath: output.path) { try? FileManager.default.removeItem(at: output) }
        let writer = try AVAssetWriter(outputURL: output, fileType: .mov)
        let ciContext = CIContext()

        var writerInput: AVAssetWriterInput?
        var adaptor: AVAssetWriterInputPixelBufferAdaptor?
        var count = 0
        let totalSec = max(duration, 0.0001)

        while let sample = readerOutput.copyNextSampleBuffer() {
            guard let src = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)

            // Orient to an upright, even-sized, top-left CGImage.
            var ci = CIImage(cvPixelBuffer: src).transformed(by: transform)
            ci = ci.transformed(by: CGAffineTransform(translationX: -ci.extent.minX, y: -ci.extent.minY))
            guard var cg = ciContext.createCGImage(ci, from: CGRect(origin: .zero, size: ci.extent.size))
            else { continue }
            let ew = cg.width & ~1, eh = cg.height & ~1
            if ew != cg.width || eh != cg.height {
                cg = cg.cropping(to: CGRect(x: 0, y: 0, width: ew, height: eh)) ?? cg
            }

            // Lazily configure the writer once the frame size is known.
            if writerInput == nil {
                let wi = AVAssetWriterInput(mediaType: .video, outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.h264,
                    AVVideoWidthKey: cg.width, AVVideoHeightKey: cg.height])
                wi.expectsMediaDataInRealTime = false
                let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: wi, sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: cg.width, kCVPixelBufferHeightKey as String: cg.height])
                writer.add(wi)
                guard writer.startWriting() else { throw writer.error ?? Error.pipelineInit }
                writer.startSession(atSourceTime: .zero)
                writerInput = wi; adaptor = ad
            }

            let annotated = try await PoseRenderer.annotate(cg, with: estimator)
            guard let outPB = PoseRenderer.pixelBuffer(from: annotated) else { continue }
            while !(writerInput?.isReadyForMoreMediaData ?? false) { try await Task.sleep(for: .milliseconds(2)) }
            adaptor?.append(outPB, withPresentationTime: pts)
            count += 1
            progress?(min(CMTimeGetSeconds(pts) / totalSec, 1))
        }

        writerInput?.markAsFinished()
        await withCheckedContinuation { cont in writer.finishWriting { cont.resume() } }
        if writer.status == .failed { throw writer.error ?? Error.encodeFailed }
        progress?(1)
        return count
    }
}
