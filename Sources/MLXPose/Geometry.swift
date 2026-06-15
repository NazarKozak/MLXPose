//
//  Geometry.swift
//  MLXPose
//
//  Created by Nazar Kozak on 15.06.2026.
//
//  Top-down geometry mirroring HF VitPoseImageProcessor (verified numerically):
//  box -> (center, scale), the UDP affine warp, and transform_preds (heatmap -> image).
//

import Foundation
import CoreGraphics

struct CenterScale: Sendable {
    var centerX: Float, centerY: Float
    var scaleW: Float, scaleH: Float
}

enum Geometry {
    static let inputWidth = 192
    static let inputHeight = 256
    static let normalizeFactor: Float = 200
    static let paddingFactor: Float = 1.25

    /// COCO box (x, y, w, h) -> center/scale (HF `box_to_center_and_scale`).
    static func centerScale(for box: CGRect) -> CenterScale {
        let tlx = Float(box.minX), tly = Float(box.minY)
        var w = Float(box.width), h = Float(box.height)
        let aspect = Float(inputWidth) / Float(inputHeight)
        let cx = tlx + w * 0.5, cy = tly + h * 0.5
        if w > aspect * h { h = w / aspect } else if w < aspect * h { w = h * aspect }
        return CenterScale(centerX: cx, centerY: cy,
                           scaleW: w / normalizeFactor * paddingFactor,
                           scaleH: h / normalizeFactor * paddingFactor)
    }

    /// Forward UDP warp coefficients (theta = 0): dst = scale*src + translate.
    /// Inverse-sample with src = (dst - translate) / scale.
    static func warpCoefficients(_ cs: CenterScale) -> (sx: Float, sy: Float, tx: Float, ty: Float) {
        let sx = Float(inputWidth - 1) / (cs.scaleW * normalizeFactor)
        let sy = Float(inputHeight - 1) / (cs.scaleH * normalizeFactor)
        let tx = sx * (-cs.centerX + 0.5 * cs.scaleW * normalizeFactor)
        let ty = sy * (-cs.centerY + 0.5 * cs.scaleH * normalizeFactor)
        return (sx, sy, tx, ty)
    }

    /// Map a heatmap-space coordinate to image pixels (HF `transform_preds`).
    /// Heatmap output size is (height 64, width 48).
    static func transformPred(x: Float, y: Float, cs: CenterScale,
                              heatmapW: Int, heatmapH: Int) -> (Float, Float) {
        let scaleW200 = cs.scaleW * normalizeFactor
        let scaleH200 = cs.scaleH * normalizeFactor
        let scaleX = scaleW200 / Float(heatmapW - 1)
        let scaleY = scaleH200 / Float(heatmapH - 1)
        let ix = x * scaleX + cs.centerX - scaleW200 * 0.5
        let iy = y * scaleY + cs.centerY - scaleH200 * 0.5
        return (ix, iy)
    }
}
