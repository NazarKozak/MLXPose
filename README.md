# MLXPose

**State-of-the-art human pose estimation, fully on-device, on Apple Silicon — in MLX Swift.**

MLXPose runs [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) (a vision-transformer keypoint model) natively via [MLX Swift](https://github.com/ml-explore/mlx-swift), with Apple Vision supplying person bounding boxes. No PyTorch, no cloud, no server — just keypoints on your Mac, iPhone, iPad, or Vision Pro.

> **Status: working & verified.** The full Swift pipeline (affine preprocessing → ViTPose →
> DARK sub-pixel decode) is numerically matched against the Hugging Face reference:
> heatmaps `max|Δ| = 1.5e-6`, decoded keypoints `max error = 3e-5 px`. Packaging polish
> (HF weights upload, ViTPose++, docs) still in progress.

---

## Why

The MLX-Swift ecosystem has LLMs, VLMs, diffusion, and audio — but **no pose-estimation model**. ViTPose is Apache-2.0 and SOTA (80.9 AP on COCO), yet there is **no MLX or MLX-Swift port**. MLXPose fills that gap: the ViT backbone already exists in MLX (it powers VLM vision encoders); MLXPose adds the lightweight decoder, verified heatmap decoding, weight conversion, and a clean on-device pipeline.

## Features (target)

- 🧠 Native **ViTPose** inference in MLX Swift — Apple Silicon GPU via Metal.
- 📦 Pre-converted **MLX weights** on Hugging Face — no conversion needed at runtime.
- 🍎 **Apple Vision** person detection out of the box (top-down); swappable `PersonDetector`.
- 🦴 **COCO-17** keypoints (whole-body via ViTPose++ later); SwiftUI skeleton overlay.
- ✅ **Numerically verified** against the Hugging Face reference (parity test set).
- 🔒 100% on-device — no network at inference time.

## Quick start (target API)

```swift
import MLXPose

let estimator = try await PoseEstimator(model: .vitPoseBaseSimple)   // downloads MLX weights once

// Single image
let poses = try await estimator.estimate(image)                     // [Pose]
let leftKnee = poses.first?.keypoint(.leftKnee)                     // (x, y, confidence)

// Camera stream
for await frame in camera.frames {                                  // CVPixelBuffer
    let people = try await estimator.estimate(frame)
    overlay.update(people)                                          // draw skeletons
}
```

## Architecture

```
CVPixelBuffer ─► PersonDetector (Apple Vision) ─► crop+affine ─► ViTPose (MLX) ─► heatmaps
                                                                                    │
                                              keypoints (image coords) ◄── decode (argmax + Gaussian modulation)
```

- **Backbone:** plain non-hierarchical ViT (reuses MLX ViT building blocks).
- **Head:** `VitPoseSimpleDecoder` (deconv → heatmaps `[numKeypoints, H, W]`).
- **Decode:** argmax + Gaussian modulation (kernel 11), affine transform back to original image — verified against HF `post_process_pose_estimation`.

## Models

| Model | Source | Notes |
|---|---|---|
| `vitPoseBaseSimple` | `usyd-community/vitpose-base-simple` | Phase 1 default (no MoE) |
| `vitPosePlusBase` | `usyd-community/vitpose-plus-base` | Phase 2 — whole-body / MoE heads |
| `vitPosePlusHuge` | `usyd-community/vitpose-plus-huge` | Phase 2 — highest accuracy |

## Setup & verification

1. Convert ViTPose weights to MLX (one-time):
   ```bash
   cd scripts
   pip install -r requirements.txt   # or: mlx safetensors huggingface_hub torch transformers numpy
   python convert_vitpose_to_mlx.py --model usyd-community/vitpose-base-simple \
       --out ./weights/vitpose-base-simple-mlx --dtype float16
   ```
2. The Python reference (`scripts/mlx_vitpose.py`) reproduces the model in MLX and is
   numerically checked against Hugging Face.

### Running the tests

Use **Xcode's build system** (not `swift test` — the CLI SPM build doesn't bundle
mlx-swift's GPU metallib):

```bash
xcodebuild test -scheme MLXPose -destination 'platform=macOS'
```

Tests cover: heatmap parity (`max|Δ|=1.5e-6`), decoded-keypoint parity
(`max 3e-5 px`), and an end-to-end render that draws a skeleton on a real image
(`Examples/output/annotated.png`).

## Demo

See [`Examples/MLXPoseDemo`](Examples/MLXPoseDemo) — a SwiftUI app streaming the camera
through MLXPose with a live skeleton overlay (`PoseOverlay`).

## Licensing

MLXPose code is **Apache-2.0**. ViTPose model code is Apache-2.0. Pretrained weights are derived from models trained on COCO/MPII/AIC — review the respective dataset terms for your use case. You can bring your own weights via the conversion script.

## Acknowledgements

- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) (Apache-2.0) and the Hugging Face `transformers` implementation.
- [MLX](https://github.com/ml-explore/mlx) and [MLX Swift](https://github.com/ml-explore/mlx-swift) by Apple's ml-explore.
- [`usyd-community`](https://huggingface.co/collections/usyd-community/vitpose-677fcfd0a0b2b5c8f79c4335) ViTPose checkpoints.
