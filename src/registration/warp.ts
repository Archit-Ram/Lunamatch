/**
 * LunaMatch - Part 16: Image Warping
 * 
 * Performs inverse spatial coordinate warping with bilinear interpolation
 * and mask propagation to eliminate holes and align source to target coordinate frame.
 */

import { ImageData, TransformModel } from '../types';
import { applyHomographyToPoint, invert3x3 } from '../generator/synthetic';

export class ImageWarper {
  /**
   * Warps source image into reference image coordinates using estimated TransformModel
   */
  static warpImage(source: ImageData, transform: TransformModel, targetWidth?: number, targetHeight?: number): ImageData {
    const outW = targetWidth || source.width;
    const outH = targetHeight || source.height;
    const totalOut = outW * outH;

    const warpedPixels = new Float32Array(totalOut);
    const warpedMask = new Uint8Array(totalOut);

    const H_forward = transform.matrix || [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    // Compute inverse transform for hole-free backward mapping: p_src = H^-1 * p_tgt
    let H_inv: number[][];
    try {
      H_inv = invert3x3(H_forward);
    } catch {
      H_inv = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ];
    }

    const srcPixels = source.pixels;
    const srcMask = source.mask;
    const srcW = source.width;
    const srcH = source.height;

    for (let ty = 0; ty < outH; ty++) {
      for (let tx = 0; tx < outW; tx++) {
        const tgtPt = { x: tx, y: ty };
        const srcPt = applyHomographyToPoint(H_inv, tgtPt);
        const outIdx = ty * outW + tx;

        if (srcPt.x >= 0 && srcPt.x < srcW - 1 && srcPt.y >= 0 && srcPt.y < srcH - 1) {
          const x0 = Math.floor(srcPt.x);
          const y0 = Math.floor(srcPt.y);
          const x1 = x0 + 1;
          const y1 = y0 + 1;
          const fx = srcPt.x - x0;
          const fy = srcPt.y - y0;

          const idx00 = y0 * srcW + x0;
          const idx10 = y0 * srcW + x1;
          const idx01 = y1 * srcW + x0;
          const idx11 = y1 * srcW + x1;

          const m00 = srcMask ? srcMask[idx00] : 1;
          const m10 = srcMask ? srcMask[idx10] : 1;
          const m01 = srcMask ? srcMask[idx01] : 1;
          const m11 = srcMask ? srcMask[idx11] : 1;

          if (m00 > 0 && m10 > 0 && m01 > 0 && m11 > 0) {
            const v00 = srcPixels[idx00];
            const v10 = srcPixels[idx10];
            const v01 = srcPixels[idx01];
            const v11 = srcPixels[idx11];

            const val = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11;
            warpedPixels[outIdx] = val;
            warpedMask[outIdx] = 1;
          } else {
            warpedPixels[outIdx] = 0;
            warpedMask[outIdx] = 0;
          }
        } else {
          warpedPixels[outIdx] = 0;
          warpedMask[outIdx] = 0;
        }
      }
    }

    return {
      id: `${source.id}_warped`,
      pixels: warpedPixels,
      width: outW,
      height: outH,
      channels: source.channels,
      dtype: 'float32',
      sensorId: source.sensorId,
      metadata: {
        ...source.metadata,
        productId: `${source.metadata.productId}_WARPED`,
      },
      mask: warpedMask,
    };
  }
}
