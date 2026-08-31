/**
 * LunaMatch - Part 17: Sub-Pixel Refinement
 * 
 * Refines candidate point correspondences to continuous sub-pixel precision
 * using Normalized Cross-Correlation (NCC) local patch correlation surfaces
 * and analytical 2D quadratic surface peak fitting:
 * C(x,y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
 */

import { ImageData, Match, MatchSet, Point2D } from '../types';
import { SubPixelRefiner } from '../core/interfaces';

export interface SubPixelOptions {
  patchRadius?: number; // e.g. 5 for 11x11 patch
  searchRadius?: number; // search offset in target image (+/- 2 px)
  minPeakConfidence?: number;
  maxSubPixelShiftPx?: number; // Cap at e.g. 1.2 px
}

export class QuadraticSubPixelRefiner implements SubPixelRefiner {
  private options: Required<SubPixelOptions>;

  constructor(options: SubPixelOptions = {}) {
    this.options = {
      patchRadius: 5,
      searchRadius: 2,
      minPeakConfidence: 0.35,
      maxSubPixelShiftPx: 1.5,
      ...options,
    };
  }

  refine(matches: MatchSet, source: ImageData, target: ImageData): MatchSet {
    const { patchRadius, searchRadius, maxSubPixelShiftPx } = this.options;
    const refined: Match[] = [];

    for (const m of matches.matches) {
      if (!m.isInlier) {
        refined.push(m);
        continue;
      }

      const sx = Math.round(m.sourcePoint.x);
      const sy = Math.round(m.sourcePoint.y);
      const tx = Math.round(m.targetPoint.x);
      const ty = Math.round(m.targetPoint.y);

      // Boundary check for patch extraction
      if (
        sx - patchRadius < 0 ||
        sx + patchRadius >= source.width ||
        sy - patchRadius < 0 ||
        sy + patchRadius >= source.height ||
        tx - patchRadius - searchRadius < 0 ||
        tx + patchRadius + searchRadius >= target.width ||
        ty - patchRadius - searchRadius < 0 ||
        ty + patchRadius + searchRadius >= target.height
      ) {
        refined.push(m);
        continue;
      }

      // 1. Extract source patch & compute its mean and std
      const patchSize = patchRadius * 2 + 1;
      const srcPatch = new Float32Array(patchSize * patchSize);
      let sMean = 0;
      let sStd = 0;

      let pIdx = 0;
      for (let dy = -patchRadius; dy <= patchRadius; dy++) {
        for (let dx = -patchRadius; dx <= patchRadius; dx++) {
          const v = source.pixels[(sy + dy) * source.width + (sx + dx)];
          srcPatch[pIdx++] = v;
          sMean += v;
        }
      }
      sMean /= patchSize * patchSize;
      for (let i = 0; i < srcPatch.length; i++) {
        const diff = srcPatch[i] - sMean;
        sStd += diff * diff;
      }
      sStd = Math.sqrt(sStd);
      if (sStd < 1e-4) {
        refined.push(m);
        continue; // Flat patch, cannot refine
      }

      // 2. Compute 3x3 or 5x5 NCC correlation surface around candidate target point
      const gridSpan = searchRadius;
      const gridSize = gridSpan * 2 + 1;
      const corrSurface: number[][] = Array.from({ length: gridSize }, () => new Array(gridSize).fill(0));

      let bestCorr = -1;
      let bestDx = 0;
      let bestDy = 0;

      for (let oy = -gridSpan; oy <= gridSpan; oy++) {
        for (let ox = -gridSpan; ox <= gridSpan; ox++) {
          const curTx = tx + ox;
          const curTy = ty + oy;

          // Target patch mean & std
          let tMean = 0;
          let tStd = 0;
          const tgtPatch = new Float32Array(patchSize * patchSize);
          let tpIdx = 0;

          for (let dy = -patchRadius; dy <= patchRadius; dy++) {
            for (let dx = -patchRadius; dx <= patchRadius; dx++) {
              const v = target.pixels[(curTy + dy) * target.width + (curTx + dx)];
              tgtPatch[tpIdx++] = v;
              tMean += v;
            }
          }
          tMean /= patchSize * patchSize;
          for (let i = 0; i < tgtPatch.length; i++) {
            const diff = tgtPatch[i] - tMean;
            tStd += diff * diff;
          }
          tStd = Math.sqrt(tStd);

          // NCC numerator
          let cross = 0;
          for (let i = 0; i < patchSize * patchSize; i++) {
            cross += (srcPatch[i] - sMean) * (tgtPatch[i] - tMean);
          }

          const ncc = tStd > 1e-4 ? cross / (sStd * tStd) : 0;
          corrSurface[oy + gridSpan][ox + gridSpan] = ncc;

          if (ncc > bestCorr) {
            bestCorr = ncc;
            bestDx = ox;
            bestDy = oy;
          }
        }
      }

      // 3. 2D Quadratic Surface Peak Fitting around best discrete offset
      // Using 3x3 neighborhood centered on best discrete peak:
      // C(u, v) = a*u^2 + b*v^2 + c*u*v + d*u + e*v + f
      const peakRow = bestDy + gridSpan;
      const peakCol = bestDx + gridSpan;

      if (
        peakRow > 0 &&
        peakRow < gridSize - 1 &&
        peakCol > 0 &&
        peakCol < gridSize - 1
      ) {
        const C00 = corrSurface[peakRow - 1][peakCol - 1];
        const C10 = corrSurface[peakRow - 1][peakCol];
        const C20 = corrSurface[peakRow - 1][peakCol + 1];
        const C01 = corrSurface[peakRow][peakCol - 1];
        const C11 = corrSurface[peakRow][peakCol];
        const C21 = corrSurface[peakRow][peakCol + 1];
        const C02 = corrSurface[peakRow + 1][peakCol - 1];
        const C12 = corrSurface[peakRow + 1][peakCol];
        const C22 = corrSurface[peakRow + 1][peakCol + 1];

        // Discrete finite differences for quadratic parameters
        const d = (C21 - C01) / 2.0;
        const e = (C12 - C10) / 2.0;
        const a = (C21 + C01 - 2 * C11) / 2.0;
        const b = (C12 + C10 - 2 * C11) / 2.0;
        const c = (C22 + C00 - C20 - C02) / 4.0;

        // Solve Hessian matrix for extremum: [u, v]^T = - [2a c; c 2b]^-1 * [d; e]
        const detHessian = 4 * a * b - c * c;

        if (detHessian > 1e-5 && a < 0 && b < 0 && bestCorr >= this.options.minPeakConfidence) {
          // Negative-definite Hessian guarantees local maximum
          const subShiftX = -(2 * b * d - c * e) / detHessian;
          const subShiftY = -(2 * a * e - c * d) / detHessian;

          if (
            Math.abs(subShiftX) <= maxSubPixelShiftPx &&
            Math.abs(subShiftY) <= maxSubPixelShiftPx
          ) {
            const finalSubX = tx + bestDx + subShiftX;
            const finalSubY = ty + bestDy + subShiftY;

            refined.push({
              ...m,
              targetPoint: { x: finalSubX, y: finalSubY },
              confidence: Math.min(1.0, Math.max(m.confidence, bestCorr)),
              uncertaintyPx: Math.max(0.08, 0.5 / (Math.abs(a) + Math.abs(b) + 0.1)),
            });
            continue;
          }
        }
      }

      // Preserve existing continuous coordinates if NCC peak is insufficient or poorly conditioned
      refined.push(m);
    }

    return {
      ...matches,
      matches: refined,
    };
  }
}
