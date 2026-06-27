//
//  ContentView.swift
//  MLXPoseDemo (macOS)
//
//  Created by Nazar Kozak on 26.06.2026.
//

import SwiftUI
import AVKit
import AppKit
import UniformTypeIdentifiers
import MLXPose

@MainActor
@Observable
final class DemoModel {
    enum Stage {
        case idle
        case preparing
        case processing(Double)
        case ready(URL)
        case failed(String)
    }

    var stage: Stage = .idle
    private var estimator: PoseEstimator?

    func pickVideo() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { process(url) }
    }

    func process(_ input: URL) {
        stage = .preparing
        Task {
            do {
                if estimator == nil { estimator = try await PoseEstimator(model: .vitPoseBaseSimple) }
                stage = .processing(0)
                let out = FileManager.default.temporaryDirectory
                    .appendingPathComponent("mlxpose-annotated-\(UUID().uuidString).mov")
                try await VideoPoseAnnotator.annotate(input: input, output: out, with: estimator!) { p in
                    Task { @MainActor in self.stage = .processing(p) }
                }
                stage = .ready(out)
            } catch {
                stage = .failed("\(error)")
            }
        }
    }
}

struct ContentView: View {
    @State private var model = DemoModel()

    var body: some View {
        VStack(spacing: 16) {
            content
        }
        .padding()
        .frame(width: 720, height: 540)
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            model.process(url)
            return true
        }
    }

    @ViewBuilder
    private var content: some View {
        switch model.stage {
        case .idle:
            dropZone(title: "Drop a video here", subtitle: "or pick one — every frame gets a skeleton")
            Button("Open Video…") { model.pickVideo() }
                .keyboardShortcut("o")

        case .preparing:
            ProgressView("Loading model…")

        case .processing(let p):
            ProgressView(value: p) { Text("Annotating… \(Int(p * 100))%") }
                .frame(maxWidth: 360)

        case .ready(let url):
            LoopingPlayer(url: url)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .cornerRadius(8)
            HStack {
                Button("Open another…") { model.pickVideo() }
                Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            }

        case .failed(let message):
            Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange)
            Text(message).font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Try another…") { model.pickVideo() }
        }
    }

    private func dropZone(title: String, subtitle: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: "figure.walk.motion").font(.system(size: 48)).foregroundStyle(.tint)
            Text(title).font(.headline)
            Text(subtitle).font(.subheadline).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 12).strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8])))
        .foregroundStyle(.secondary)
    }
}

/// Plays a local video on an endless loop.
struct LoopingPlayer: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> AVPlayerView {
        let item = AVPlayerItem(url: url)
        let queue = AVQueuePlayer()
        context.coordinator.looper = AVPlayerLooper(player: queue, templateItem: item)
        let view = AVPlayerView()
        view.player = queue
        view.controlsStyle = .floating
        view.videoGravity = .resizeAspect
        queue.play()
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var looper: AVPlayerLooper? }
}
