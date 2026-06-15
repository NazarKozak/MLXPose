//
//  CameraController.swift
//  MLXPoseDemo
//
//  Created by Nazar Kozak on 15.06.2026.
//
//  Streams camera frames (BGRA), runs MLXPose, and publishes the latest poses.
//

import Foundation
import AVFoundation
import CoreVideo
import Observation
import MLXPose

@Observable
final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var poses: [Pose] = []
    var imageSize: CGSize = .zero

    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "mlxpose.camera")
    private var estimator: PoseEstimator?
    private var inFlight = false

    func start() {
        if estimator == nil { estimator = Self.loadEstimator() }
        queue.async { [weak self] in
            guard let self else { return }
            self.configureIfNeeded()
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    func stop() { queue.async { [weak self] in self?.session.stopRunning() } }

    private static func loadEstimator() -> PoseEstimator? {
        // Add the converted `weights.safetensors` to the app bundle (Phase 0 output).
        guard let url = Bundle.main.url(forResource: "weights", withExtension: "safetensors") else {
            assertionFailure("weights.safetensors missing from the app bundle")
            return nil
        }
        return try? PoseEstimator(weightsDirectory: url.deletingLastPathComponent())
    }

    private func configureIfNeeded() {
        guard session.inputs.isEmpty else { return }
        session.beginConfiguration()
        session.sessionPreset = .high
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
           let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) {
            session.addInput(input)
        }
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard !inFlight, let estimator,
              let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        inFlight = true
        let size = CGSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        Task { @MainActor in
            defer { self.inFlight = false }
            if let result = try? await estimator.estimate(pb) {
                self.poses = result
                self.imageSize = size
            }
        }
    }
}
