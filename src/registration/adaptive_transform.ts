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
import { fitPushbroomModel, PushbroomOptions } from './pushbroom';

/** Minimum inlier matches before the pushbroom candidate is even considered (needed for meaningful CV). */
const PUSHBROOM_MIN_MATCHES = 20;
/** Pushbroom must beat the best global model's CV error by this factor to be selected. */
const PUSHBROOM_CV_MARGIN = 0.95;

export class AdaptiveTransformEstimator implements TransformEstimator {
  private filter: GeometricGraphFilter;

  constructor() {
    this.filter = new GeometricGraphFilter();
  }

  estimateTransform(
    matches: MatchSet,
    sourceDim: [number, number],
    targetDim: [number, number],
    estimatorOptions?: { pushbroom?: PushbroomOptions }
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

    // 4. Fit TPS if >= 6 inlier matches
    if (n >= 6) {
      const tps = this.fitTPS(srcPts, tgtPts);
      if (tps.validity) {
        candidates.push({ model: tps, bic: this.calculateBIC(tps.residual.rmse * tps.residual.rmse * n, n, 2 * n + 6) });
      }
    }

    // Select candidate with lowest BIC (or Homography if performance is within 5%)
    candidates.sort((a, b) => a.bic - b.bic);
    const best = candidates[0].model;
    best.bicScore = candidates[0].bic;

    // 5. Per-strip pushbroom model (Part 24). BIC cannot judge it fairly against an
    // interpolating TPS (whose training residual is ~0 by construction), so the choice
    // between the BIC winner and the pushbroom model is made on k-fold cross-validated
    // prediction error instead, and pushbroom must win by a clear margin.
    if (estimatorOptions?.pushbroom && n >= PUSHBROOM_MIN_MATCHES) {
      const pb = this.fitPushbroom(inlierMatches, srcPts, tgtPts, sourceDim, estimatorOptions.pushbroom);
      if (pb) {
        const cv = this.crossValidate(inlierMatches, best.modelType, sourceDim, estimatorOptions.pushbroom);
        if (cv && cv.pushbroom < cv.global * PUSHBROOM_CV_MARGIN && pb.model.pushbroomSegments) {
          pb.model.pushbroomSegments.cvRmsePx = cv.pushbroom;
          pb.model.pushbroomSegments.cvRmseGlobalPx = cv.global;
          pb.model.bicScore = this.calculateBIC(pb.model.residual.rmse * pb.model.residual.rmse * n, n, pb.k);
          return pb.model;
        }
      }
    }

    return best;
  }

  /**
   * Fits the along-track segmented model and re-expresses it as an ordinary TPS
   * TransformModel (refit on a dense grid of the piecewise field) so ImageWarper
   * and the rest of the pipeline consume it unchanged. Residuals are measured
   * against the ORIGINAL matches, not the grid, so the score reflects real fit.
   */
  private fitPushbroom(
    matches: MatchSet['matches'],
    src: Point2D[],
    tgt: Point2D[],
    sourceDim: [number, number],
    options: PushbroomOptions
  ): { model: TransformModel; k: number } | null {
    const fit = fitPushbroomModel(matches, sourceDim[0], sourceDim[1], options);
    if (!fit) return null;

    const gridN = 10;
    const gs: Point2D[] = [];
    const gt: Point2D[] = [];
    for (let gy = 0; gy < gridN; gy++) {
      for (let gx = 0; gx < gridN; gx++) {
        const p = { x: ((gx + 0.5) / gridN) * sourceDim[0], y: ((gy + 0.5) / gridN) * sourceDim[1] };
        gs.push(p);
        gt.push(fit.apply(p));
      }
    }
    const tps = this.fitTPS(gs, gt, 0.01);
    if (!tps.validity || !tps.tpsControlPoints) return null;

    const residual = this.evaluateTPS(tps.tpsControlPoints, src, tgt);
    return {
      k: fit.effectiveParameters,
      model: {
        ...tps,
        residual,
        diagnostics: { degreesOfFreedom: fit.effectiveParameters, sampleCount: src.length },
        pushbroomSegments: {
          axis: fit.axis,
          segments: fit.segments.map((s) => ({
            index: s.index,
            start: s.start,
            end: s.end,
            matchCount: s.matchCount,
            fallbackToGlobal: s.fallbackToGlobal,
            params: s.params,
          })),
        },
      },
    };
  }

  /**
   * 5-fold cross-validated RMSE (px) of the currently-best global model family vs the
   * pushbroom model. Folds are deterministic (index mod k).
   */
  private crossValidate(
    matches: MatchSet['matches'],
    globalType: TransformType,
    sourceDim: [number, number],
    pbOptions: PushbroomOptions
  ): { global: number; pushbroom: number } | null {
    const K = 5;
    let gSq = 0;
    let pSq = 0;
    let count = 0;

    for (let f = 0; f < K; f++) {
      const train = matches.filter((_, i) => i % K !== f);
      const test = matches.filter((_, i) => i % K === f);
      if (train.length < 8 || test.length === 0) continue;

      const trS = train.map((m) => m.sourcePoint);
      const trT = train.map((m) => m.targetPoint);

      let globalModel: TransformModel;
      switch (globalType) {
        case 'rigid': globalModel = this.fitRigid(trS, trT); break;
        case 'affine': globalModel = this.fitAffine(trS, trT); break;
        case 'homography': globalModel = this.fitHomography(trS, trT); break;
        default: globalModel = this.fitTPS(trS, trT); break;
      }
      const pbFit = fitPushbroomModel(train, sourceDim[0], sourceDim[1], pbOptions);
      if (!pbFit || !globalModel.validity) continue;

      for (const m of test) {
        const gp = globalModel.modelType === 'tps' && globalModel.tpsControlPoints
          ? AdaptiveTransformEstimator.applyTPSTransform(globalModel.tpsControlPoints, m.sourcePoint)
          : applyHomographyToPoint(globalModel.matrix!, m.sourcePoint);
        const pp = pbFit.apply(m.sourcePoint);
        gSq += (gp.x - m.targetPoint.x) ** 2 + (gp.y - m.targetPoint.y) ** 2;
        pSq += (pp.x - m.targetPoint.x) ** 2 + (pp.y - m.targetPoint.y) ** 2;
        count++;
      }
    }

    if (count === 0) return null;
    return { global: Math.sqrt(gSq / count), pushbroom: Math.sqrt(pSq / count) };
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

  public fitTPS(src: Point2D[], tgt: Point2D[], lambda: number = 0.1): TransformModel {
    const n = src.length;
    const U = (r: number) => r === 0 ? 0 : r * r * Math.log(r);

    const L: number[][] = Array.from({ length: n + 3 }, () => new Array(n + 3).fill(0));
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const dx = src[i].x - src[j].x;
        const dy = src[i].y - src[j].y;
        const r = Math.hypot(dx, dy);
        L[i][j] = U(r);
        if (i === j) {
          L[i][j] += lambda;
        }
      }
      L[i][n] = 1;
      L[i][n + 1] = src[i].x;
      L[i][n + 2] = src[i].y;
      
      L[n][i] = 1;
      L[n + 1][i] = src[i].x;
      L[n + 2][i] = src[i].y;
    }

    const b_x = new Array(n + 3).fill(0);
    const b_y = new Array(n + 3).fill(0);
    for (let i = 0; i < n; i++) {
      b_x[i] = tgt[i].x;
      b_y[i] = tgt[i].y;
    }

    const w_x = this.solveLinearSystem(L, b_x);
    const w_y = this.solveLinearSystem(L, b_y);

    if (!w_x || !w_y) {
      return {
        modelType: 'tps',
        residual: { mean: 0, median: 0, rmse: Infinity, max: 0 },
        validity: false,
        diagnostics: { degreesOfFreedom: 2 * n + 6, sampleCount: n },
      };
    }

    const weights: number[][] = [];
    for (let i = 0; i < n; i++) {
      weights.push([w_x[i], w_y[i]]);
    }
    const affineParams: number[][] = [
      [w_x[n], w_y[n]],
      [w_x[n + 1], w_y[n + 1]],
      [w_x[n + 2], w_y[n + 2]]
    ];

    const tpsControlPoints = {
      sourceKnots: src,
      targetKnots: tgt,
      weights,
      affineParams
    };

    const residuals = this.evaluateTPS(tpsControlPoints, src, tgt);

    return {
      modelType: 'tps',
      tpsControlPoints,
      residual: residuals,
      validity: true,
      diagnostics: { degreesOfFreedom: 2 * n + 6, sampleCount: n },
    };
  }

  private solveLinearSystem(A: number[][], b: number[]): number[] | null {
    const n = b.length;
    const M: number[][] = A.map(row => [...row]);
    const vec = [...b];

    for (let i = 0; i < n; i++) {
      let maxEl = Math.abs(M[i][i]);
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(M[k][i]) > maxEl) {
          maxEl = Math.abs(M[k][i]);
          maxRow = k;
        }
      }

      if (maxEl < 1e-10) return null;

      if (maxRow !== i) {
        const tmpRow = M[i];
        M[i] = M[maxRow];
        M[maxRow] = tmpRow;
        
        const tmpVal = vec[i];
        vec[i] = vec[maxRow];
        vec[maxRow] = tmpVal;
      }

      for (let k = i + 1; k < n; k++) {
        const c = -M[k][i] / M[i][i];
        for (let j = i; j < n; j++) {
          if (i === j) {
            M[k][j] = 0;
          } else {
            M[k][j] += c * M[i][j];
          }
        }
        vec[k] += c * vec[i];
      }
    }

    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      x[i] = vec[i] / M[i][i];
      for (let k = i - 1; k >= 0; k--) {
        vec[k] -= M[k][i] * x[i];
      }
    }

    return x;
  }

  static applyTPSTransform(tps: NonNullable<TransformModel['tpsControlPoints']>, point: Point2D): Point2D {
    const U = (r: number) => r === 0 ? 0 : r * r * Math.log(r);
    let x = tps.affineParams[0][0] + tps.affineParams[1][0] * point.x + tps.affineParams[2][0] * point.y;
    let y = tps.affineParams[0][1] + tps.affineParams[1][1] * point.x + tps.affineParams[2][1] * point.y;

    for (let i = 0; i < tps.sourceKnots.length; i++) {
      const knot = tps.sourceKnots[i];
      const dx = point.x - knot.x;
      const dy = point.y - knot.y;
      const r = Math.hypot(dx, dy);
      const u = U(r);
      x += tps.weights[i][0] * u;
      y += tps.weights[i][1] * u;
    }

    return { x, y };
  }

  private evaluateTPS(
    tps: NonNullable<TransformModel['tpsControlPoints']>,
    src: Point2D[],
    tgt: Point2D[]
  ): { mean: number; median: number; rmse: number; max: number } {
    const errs: number[] = [];
    let sum = 0;
    let sqSum = 0;
    let max = 0;

    for (let i = 0; i < src.length; i++) {
      const pred = AdaptiveTransformEstimator.applyTPSTransform(tps, src[i]);
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
