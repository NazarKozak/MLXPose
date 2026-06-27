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
import AppKit

@main
struct MLXPoseDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("MLXPose Demo") {
            ContentView()
        }
        .windowResizability(.contentSize)
    }
}

/// Runs the SPM executable as a normal foreground GUI app (Dock icon, menu, focused
/// window) — otherwise a package executable can launch without showing its window.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
