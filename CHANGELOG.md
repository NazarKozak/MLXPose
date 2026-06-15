# Changelog

## [Unreleased]

### Added
- Native MLX Swift port of **ViTPose** (`vitpose-base-simple`): ViT backbone + simple decoder.
- `PoseEstimator` public API with top-down pipeline (Apple Vision person detection → ViTPose keypoints).
- COCO-17 keypoint types and argmax heatmap decoding.
- Python tooling: `convert_vitpose_to_mlx.py` (Hugging Face → MLX safetensors) and
  `mlx_vitpose.py` (MLX reference implementation).
- Parity test against the Hugging Face reference: **max|Δ| = 1.5e-6**.

### Known limitations / TODO
- Heatmap decoding is argmax only; Gaussian modulation (kernel 11) for sub-pixel accuracy pending.
- Preprocessing uses a plain resize; affine warp matching `VitPoseImageProcessor` pending.
- ViTPose++ (MoE / whole-body) and quantized weights pending.
- Run tests with `xcodebuild` (not `swift test`) due to mlx-swift GPU metallib bundling on CLI.
