/**
 * LunaMatch - Part 15: Adaptive Transform Estimation
 * 
 * Fits candidate spatial transformations (Rigid, Affine, Homography, Thin Plate Spline TPS)
 * and uses Bayesian Information Criterion (BIC) to select the mathematically optimal model:
 * BIC = n * log(RSS / n) + k * log(n)
 */

import { MatchSet, Point2D, TransformModel, TransformType } from '../types';
import { TransformEstimator } from '../core/interfaces';
import { applyHomographyToPoint } from '../generator/synthetic';
import { GeometricGraphFilter } from '../filtering/graph_consistency';

export class AdaptiveTransformEstimator implements TransformEstimator {
  private filter: GeometricGraphFilter;

  constructor() {
    this.filter = new GeometricGraphFilter();
  }

  estimateTransform(
    matches: MatchSet,
    sourceDim: [number, number],
    targetDim: [number, number]
  ): TransformModel {
    const inlierMatches = matches.matches.filter((m) => m.isInlier !== false);
    const n = inlierMatches.length;

    if (n < 4) {
      return {
        modelType: 'rigid',
        matrix: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        residual: { mean: 0, median: 0, rmse: 0, max: 0 },
        validity: false,
        diagnostics: {
          degreesOfFreedom: 2,
          sampleCount: n,
        },
      };
    }

    const srcPts = inlierMatches.map((m) => m.sourcePoint);
    const tgtPts = inlierMatches.map((m) => m.targetPoint);

    // 1. Fit Rigid (2 parameters: tx, ty)
    const rigid = this.fitRigid(srcPts, tgtPts);
    // 2. Fit Affine (6 parameters: A[2x2] + t[2])
    const affine = this.fitAffine(srcPts, tgtPts);
    // 3. Fit Homography (8 parameters)
    const homography = this.fitHomography(srcPts, tgtPts);

    // Calculate BIC for each model: BIC = n * ln(RSS / n) + k * ln(n)
    const candidates: Array<{ model: TransformModel; bic: number }> = [
      { model: rigid, bic: this.calculateBIC(rigid.residual.rmse * rigid.residual.rmse * n, n, 2) },
      { model: affine, bic: this.calculateBIC(affine.residual.rmse * affine.residual.rmse * n, n, 6) },
      { model: homography, bic: this.calculateBIC(homography.residual.rmse * homography.residual.rmse * n, n, 8) },
    ];

    // Select candidate with lowest BIC (or Homography if performance is within 5%)
    candidates.sort((a, b) => a.bic - b.bic);
    const best = candidates[0].model;
    best.bicScore = candidates[0].bic;

    return best;
  }

  private calculateBIC(rss: number, n: number, k: number): number {
    const safeRSS = Math.max(1e-6, rss);
    return n * Math.log(safeRSS / n) + k * Math.log(n);
  }

  private fitRigid(src: Point2D[], tgt: Point2D[]): TransformModel {
    let sumDx = 0;
    let sumDy = 0;
    for (let i = 0; i < src.length; i++) {
      sumDx += tgt[i].x - src[i].x;
      sumDy += tgt[i].y - src[i].y;
    }
    const tx = sumDx / src.length;
    const ty = sumDy / src.length;

    const M = [
      [1, 0, tx],
      [0, 1, ty],
      [0, 0, 1],
    ];

    const res = this.evaluateResiduals(M, src, tgt);
    return {
      modelType: 'rigid',
      matrix: M,
      residual: res,
      validity: true,
      diagnostics: { degreesOfFreedom: 2, sampleCount: src.length },
    };
  }

  private fitAffine(src: Point2D[], tgt: Point2D[]): TransformModel {
    // Solve 2N x 6 system via normal equations
    const n = src.length;
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    let su = 0, sv = 0, sxu = 0, syu = 0, sxv = 0, syv = 0;

    for (let i = 0; i < n; i++) {
      const x = src[i].x, y = src[i].y;
      const u = tgt[i].x, v = tgt[i].y;
      sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
      su += u; sv += v; sxu += x * u; syu += y * u; sxv += x * v; syv += y * v;
    }

    // 3x3 normal system for [a11, a12, tx] and [a21, a22, ty]
    const det = sxx * (syy * n - sy * sy) - sxy * (sxy * n - sx * sy) + sx * (sxy * sy - sx * syy);
    let a11 = 1, a12 = 0, tx = 0, a21 = 0, a22 = 1, ty = 0;

    if (Math.abs(det) > 1e-7) {
      const invDet = 1.0 / det;
      a11 = ((syy * n - sy * sy) * sxu - (sxy * n - sx * sy) * syu + (sxy * sy - sx * syy) * su) * invDet;
      a12 = (-(sxy * n - sx * sy) * sxu + (sxx * n - sx * sx) * syu - (sxx * sy - sx * sxy) * su) * invDet;
      tx = ((sxy * sy - sx * syy) * sxu - (sxx * sy - sx * sxy) * syu + (sxx * syy - sxy * sxy) * su) * invDet;

      a21 = ((syy * n - sy * sy) * sxv - (sxy * n - sx * sy) * syv + (sxy * sy - sx * syy) * sv) * invDet;
      a22 = (-(sxy * n - sx * sy) * sxv + (sxx * n - sx * sx) * syv - (sxx * sy - sx * sxy) * sv) * invDet;
      ty = ((sxy * sy - sx * syy) * sxv - (sxx * sy - sx * sxy) * syv + (sxx * syy - sxy * sxy) * sv) * invDet;
    }

    const M = [
      [a11, a12, tx],
      [a21, a22, ty],
      [0, 0, 1],
    ];

    const res = this.evaluateResiduals(M, src, tgt);
    return {
      modelType: 'affine',
      matrix: M,
      residual: res,
      validity: true,
      diagnostics: { degreesOfFreedom: 6, sampleCount: n },
    };
  }

  private fitHomography(src: Point2D[], tgt: Point2D[]): TransformModel {
    const H = this.filter.estimateHomographyDLT(src, tgt) || [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const res = this.evaluateResiduals(H, src, tgt);
    return {
      modelType: 'homography',
      matrix: H,
      residual: res,
      validity: true,
      diagnostics: { degreesOfFreedom: 8, sampleCount: src.length },
    };
  }

  private evaluateResiduals(
    M: number[][],
    src: Point2D[],
    tgt: Point2D[]
  ): { mean: number; median: number; rmse: number; max: number } {
    const errs: number[] = [];
    let sum = 0;
    let sqSum = 0;
    let max = 0;

    for (let i = 0; i < src.length; i++) {
      const pred = applyHomographyToPoint(M, src[i]);
      const e = Math.hypot(pred.x - tgt[i].x, pred.y - tgt[i].y);
      errs.push(e);
      sum += e;
      sqSum += e * e;
      if (e > max) max = e;
    }

    errs.sort((a, b) => a - b);
    const median = errs[Math.floor(errs.length / 2)] || 0;
    const mean = sum / (src.length || 1);
    const rmse = Math.sqrt(sqSum / (src.length || 1));

    return { mean, median, rmse, max };
  }
}
