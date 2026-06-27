// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLXPose",
    platforms: [
        .macOS(.v14),
        .iOS(.v16),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "MLXPose", targets: ["MLXPose"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
    ],
    targets: [
        .target(
            name: "MLXPose",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/MLXPose"
        ),
        .executableTarget(
            name: "MLXPoseDemo",
            dependencies: ["MLXPose"],
            path: "Examples/MLXPoseDemo"
        ),
        .testTarget(
            name: "MLXPoseTests",
            dependencies: ["MLXPose"],
            path: "Tests/MLXPoseTests"
        ),
    ]
)
