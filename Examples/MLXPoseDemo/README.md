# MLXPoseDemo

A minimal SwiftUI app: live camera → ViTPose keypoints (on-device, MLX) → skeleton overlay.

> These are reference sources for an **iOS app target** (camera requires a real app bundle).
> They are intentionally not part of the Swift package's build.

## Setup

1. Create a new iOS App in Xcode (SwiftUI lifecycle).
2. Add the MLXPose package dependency:
   - File → Add Package Dependencies → `https://github.com/NazarKozak/MLXPose` → add the `MLXPose` library.
   - This transitively pulls `mlx-swift` (which bundles the Metal library for the app).
3. Add the four files in this folder to the app target
   (`MLXPoseDemoApp.swift`, `ContentView.swift`, `CameraController.swift`).
   Remove the template `App` file so there is a single `@main`.
4. Generate the weights and add them to the app target:
   ```bash
   cd ../../scripts
   python convert_vitpose_to_mlx.py \
       --model usyd-community/vitpose-base-simple \
       --out ./weights/vitpose-base-simple-mlx --dtype float16
   ```
   Drag `weights.safetensors` into the app (Target Membership ✓).
5. Add `NSCameraUsageDescription` to Info.plist.
6. Run on a physical device (camera + Apple Silicon GPU).

## Notes

- Frames are BGRA (`kCVPixelFormatType_32BGRA`), matching `Preprocess`.
- `CameraController` drops frames while one is in flight to keep the UI responsive.
- Default model is `vitPoseBaseSimple` (COCO-17). Swap weights for ViTPose++ later.
