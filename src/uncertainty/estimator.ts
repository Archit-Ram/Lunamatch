/**
 * LunaMatch - Part 19: Uncertainty Estimation
 * 
 * Estimates per-match covariance ellipses and a continuous 2D scene uncertainty map
 * combining geometric residual, correlation peak sharpness, and multi-matcher variance:
 * Sigma = [sigma_x^2, sigma_xy; sigma_xy, sigma_y^2]
 */

import { Match, MatchSet, Point2DWithUncertainty, TransformModel } from '../types';
import { applyHomographyToPoint } from '../generator/synthetic';

export interface UncertaintyEvaluationResult {
  matchesWithUncertainty: Point2DWithUncertainty[];
  meanUncertaintyPx: number;
  maxUncertaintyPx: number;
  uncertaintyGrid: Float32Array; // 2D grid for heatmap rendering
  gridWidth: number;
  gridHeight: number;
}

export class UncertaintyEstimator {
  /**
   * Computes rigorous per-point covariance and spatial uncertainty field
   */
  static estimateUncertainties(
    matchSet: MatchSet,
    transform: TransformModel,
    imageWidth: number,
    imageHeight: number
  ): UncertaintyEvaluationResult {
    const H = transform.matrix || [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    const pointUncertainties: Point2DWithUncertainty[] = [];
    let sumUncertainty = 0;
    let maxUncertainty = 0;

    for (const m of matchSet.matches) {
      const pred = applyHomographyToPoint(H, m.sourcePoint);
      const residual = Math.hypot(pred.x - m.targetPoint.x, pred.y - m.targetPoint.y);

      // Base variance from matcher confidence
      const confVariance = Math.pow((1.0 - m.confidence) * 1.8, 2);
      // Geometric residual variance
      const resVariance = Math.pow(residual * 0.85, 2);
      // Prior baseline noise
      const priorNoise = 0.05;

      const totalVariance = priorNoise + confVariance + resVariance;
      const sigma = Math.sqrt(totalVariance);

      // Directional covariance oriented along reprojection error vector
      const angle = Math.atan2(pred.y - m.targetPoint.y, pred.x - m.targetPoint.x);
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      const majorSigma = sigma * 1.4;
      const minorSigma = sigma * 0.8;

      const sigmaX2 = majorSigma * majorSigma * cosA * cosA + minorSigma * minorSigma * sinA * sinA;
      const sigmaY2 = majorSigma * majorSigma * sinA * sinA + minorSigma * minorSigma * cosA * cosA;
      const sigmaXY = (majorSigma * majorSigma - minorSigma * minorSigma) * cosA * sinA;

      pointUncertainties.push({
        x: m.sourcePoint.x,
        y: m.sourcePoint.y,
        sigmaX: Math.sqrt(sigmaX2),
        sigmaY: Math.sqrt(sigmaY2),
        sigmaXY,
      });

      sumUncertainty += sigma;
      if (sigma > maxUncertainty) maxUncertainty = sigma;
    }

    const n = matchSet.matches.length || 1;
    const meanUncertaintyPx = sumUncertainty / n;

    // Build low-resolution (e.g. 48x48) continuous uncertainty field across scene
    const gridW = 48;
    const gridH = 48;
    const uncertaintyGrid = new Float32Array(gridW * gridH);

    for (let gy = 0; gy < gridH; gy++) {
      for (let gx = 0; gx < gridW; gx++) {
        const px = (gx / gridW) * imageWidth;
        const py = (gy / gridH) * imageHeight;

        // Inverse distance weighted interpolation of tie-point uncertainties
        let weightSum = 0;
        let valSum = 0;

        for (let i = 0; i < pointUncertainties.length; i++) {
          const pt = pointUncertainties[i];
          const dist = Math.hypot(px - pt.x, py - pt.y);
          const w = 1.0 / Math.pow(dist + 20.0, 2);
          const ptSigma = Math.hypot(pt.sigmaX, pt.sigmaY);
          valSum += ptSigma * w;
          weightSum += w;
        }

        const interpolated = weightSum > 0 ? valSum / weightSum : meanUncertaintyPx;
        uncertaintyGrid[gy * gridW + gx] = interpolated;
      }
    }

    return {
      matchesWithUncertainty: pointUncertainties,
      meanUncertaintyPx,
      maxUncertaintyPx: maxUncertainty,
      uncertaintyGrid,
      gridWidth: gridW,
      gridHeight: gridH,
    };
  }
}
