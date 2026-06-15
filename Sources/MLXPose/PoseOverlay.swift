//
//  PoseOverlay.swift
//  MLXPose
//
//  Created by Nazar Kozak on 15.06.2026.
//
//  COCO-17 skeleton topology + a SwiftUI overlay for drawing poses.
//

import Foundation
import CoreGraphics

public extension COCOKeypoint {
    /// COCO-17 skeleton bone connections.
    static let skeleton: [(COCOKeypoint, COCOKeypoint)] = [
        (.nose, .leftEye), (.nose, .rightEye), (.leftEye, .leftEar), (.rightEye, .rightEar),
        (.leftShoulder, .rightShoulder),
        (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
        (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
        (.leftShoulder, .leftHip), (.rightShoulder, .rightHip), (.leftHip, .rightHip),
        (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
        (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
    ]
}

#if canImport(SwiftUI)
import SwiftUI

/// Draws pose skeletons over a view, scaling from image space to the view's size.
public struct PoseOverlay: View {
    public var poses: [Pose]
    public var imageSize: CGSize
    public var confidenceThreshold: Float
    public var pointRadius: CGFloat
    public var lineWidth: CGFloat

    public init(poses: [Pose], imageSize: CGSize,
                confidenceThreshold: Float = 0.3,
                pointRadius: CGFloat = 4, lineWidth: CGFloat = 2) {
        self.poses = poses
        self.imageSize = imageSize
        self.confidenceThreshold = confidenceThreshold
        self.pointRadius = pointRadius
        self.lineWidth = lineWidth
    }

    public var body: some View {
        Canvas { ctx, size in
            let sx = size.width / max(imageSize.width, 1)
            let sy = size.height / max(imageSize.height, 1)
            func pt(_ k: Keypoint) -> CGPoint { CGPoint(x: CGFloat(k.x) * sx, y: CGFloat(k.y) * sy) }

            for pose in poses {
                for (a, b) in COCOKeypoint.skeleton {
                    let ka = pose.keypoint(a), kb = pose.keypoint(b)
                    guard ka.confidence >= confidenceThreshold, kb.confidence >= confidenceThreshold else { continue }
                    var path = Path()
                    path.move(to: pt(ka)); path.addLine(to: pt(kb))
                    ctx.stroke(path, with: .color(.green), lineWidth: lineWidth)
                }
                for k in pose.keypoints where k.confidence >= confidenceThreshold {
                    let p = pt(k)
                    let r = pointRadius
                    ctx.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)),
                             with: .color(.red))
                }
            }
        }
    }
}
#endif
