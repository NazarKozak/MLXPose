# MLXPoseDemo (macOS)

A SwiftUI macOS app: **open a video → every frame gets a ViTPose skeleton (MLX, on-device) → the annotated result loops**. Drag-and-drop a clip or use the open panel. Ideal for screen-recording a demo — no camera or permissions needed.

It is a Swift Package **executable target**, so there is nothing to assemble:

```bash
open Package.swift          # opens the package in Xcode
# choose the "MLXPoseDemo" scheme → Run (⌘R) → "Open Video…"
```

> First run downloads the MLX weights from the Hugging Face Hub (cached afterwards),
> and Xcode may prompt once to install the **Metal Toolchain** (needed to build
> mlx-swift from source on the command line / first app build).

## How it works

- `VideoPoseAnnotator.annotate(input:output:with:)` reads the clip (honoring the
  track's orientation), runs `PoseEstimator` per frame, and writes an H.264 `.mov`
  with the skeleton burned in (`PoseRenderer`).
- The app then loops the annotated file with `AVPlayer`.

## Use the annotator directly

```swift
import MLXPose

let estimator = try await PoseEstimator(model: .vitPoseBaseSimple)
let frames = try await VideoPoseAnnotator.annotate(
    input: inputURL, output: outputURL, with: estimator
) { progress in print("\(Int(progress * 100))%") }
```

## Live camera (optional)

For a live-camera build, add `PoseEstimator.estimate(_:)` to an `AVCaptureVideoDataOutput`
delegate and overlay `PoseOverlay`. Camera requires a real app bundle with
`NSCameraUsageDescription` — the video-file flow above avoids that entirely.
