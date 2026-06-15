//
//  ContentView.swift
//  MLXPoseDemo
//
//  Created by Nazar Kozak on 15.06.2026.
//

import SwiftUI
import AVFoundation
import MLXPose

struct ContentView: View {
    @State private var camera = CameraController()

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()
            PoseOverlay(poses: camera.poses, imageSize: camera.imageSize)
                .ignoresSafeArea()
        }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
    }
}

#if os(iOS)
import UIKit

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
#endif
