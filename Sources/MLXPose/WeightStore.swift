//
//  WeightStore.swift
//  MLXPose
//
//  Created by Nazar Kozak on 26.06.2026.
//
//  Downloads converted MLX weights from the Hugging Face Hub on first use and
//  caches them locally, so the package works out of the box (no manual conversion).
//

import Foundation

/// A ViTPose model published as MLX weights on the Hugging Face Hub.
public enum Model: Sendable {
    case vitPoseBaseSimple
    /// Bring your own Hub repo (must contain `weights.safetensors` + `config.json`).
    case custom(repoID: String)

    var repoID: String {
        switch self {
        case .vitPoseBaseSimple: return "nazarkozak/vitpose-base-simple-mlx"
        case .custom(let id): return id
        }
    }
}

/// Resolves a model's local weights directory, downloading from HF Hub if needed.
public actor WeightStore {
    public static let shared = WeightStore()

    private static let files = ["weights.safetensors", "config.json"]
    private var inFlight: [String: Task<URL, Swift.Error>] = [:]

    public init() {}

    /// Returns a local directory containing the model's files, downloading if absent.
    public func directory(for model: Model) async throws -> URL {
        let repo = model.repoID
        if let task = inFlight[repo] { return try await task.value }
        let task = Task<URL, Swift.Error> { try await Self.fetch(repo: repo) }
        inFlight[repo] = task
        defer { inFlight[repo] = nil }
        return try await task.value
    }

    private static func fetch(repo: String) async throws -> URL {
        let dir = cacheRoot().appendingPathComponent(repo.replacingOccurrences(of: "/", with: "_"))
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        for file in files {
            let dest = dir.appendingPathComponent(file)
            if FileManager.default.fileExists(atPath: dest.path) { continue }
            let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(file)?download=true")!
            let (tmp, response) = try await URLSession.shared.download(from: url)
            guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
                throw Error.downloadFailed(file, (response as? HTTPURLResponse)?.statusCode ?? -1)
            }
            // Move into place atomically.
            if FileManager.default.fileExists(atPath: dest.path) { try? FileManager.default.removeItem(at: dest) }
            try FileManager.default.moveItem(at: tmp, to: dest)
        }
        return dir
    }

    private static func cacheRoot() -> URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("MLXPose", isDirectory: true)
    }

    public enum Error: Swift.Error { case downloadFailed(String, Int) }
}
