//
//  MLXPoseDemoApp.swift
//  MLXPoseDemo (macOS)
//
//  Created by Nazar Kozak on 26.06.2026.
//
//  Open a video -> annotate every frame with ViTPose keypoints (MLX, on-device)
//  -> loop the result. Perfect for screen-recording a demo. No camera needed.
//

import SwiftUI

@main
struct MLXPoseDemoApp: App {
    var body: some Scene {
        WindowGroup("MLXPose Demo") {
            ContentView()
        }
        .windowResizability(.contentSize)
    }
}
