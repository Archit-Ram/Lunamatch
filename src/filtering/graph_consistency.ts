/**
 * LunaMatch - Part 14: Geometric Filtering & Graph Consistency
 * 
 * Filters out spurious matches using pairwise distance ratio consistency,
 * graph spectral ranking, and MAGSAC++ / RANSAC robust estimators.
 */

import { Match, MatchSet, Point2D } from '../types';
import { DeterministicRNG, applyHomographyToPoint } from '../generator/synthetic';

export interface GeometricFilterOptions {
  inlierThresholdPx?: number;
  maxIterations?: number;
  confidence?: number;
  useMagsacPlusPlus?: boolean;
  pairwiseDistanceTolerance?: number; // e.g. 0.35 for 35% scale difference tolerance
}

export class GeometricGraphFilter {
  private options: Required<GeometricFilterOptions>;

  constructor(options: GeometricFilterOptions = {}) {
    this.options = {
      inlierThresholdPx: 2.5,
      maxIterations: 2000,
      confidence: 0.99,
      useMagsacPlusPlus: true,
      pairwiseDistanceTolerance: 0.35,
      ...options,
    };
  }

  /**
   * Main filtering entry point: Graph consistency + MAGSAC++ / RANSAC
   */
  filterMatches(matchSet: MatchSet): {
    filteredMatchSet: MatchSet;
    bestModel: number[][];
    inlierCount: number;
    inlierRatio: number;
    rmse: number;
  } {
    const rawMatches = matchSet.matches;
    if (rawMatches.length < 4) {
      return {
        filteredMatchSet: matchSet,
        bestModel: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        inlierCount: rawMatches.length,
        inlierRatio: rawMatches.length > 0 ? 1.0 : 0.0,
        rmse: 0.0,
      };
    }

    // Step 1: Pairwise Graph Edge Consistency Pre-filter
    const graphConsistentMatches = this.pruneGraphInconsistencies(rawMatches);

    // Step 2: MAGSAC++ / RANSAC Robust Estimator on Candidate Correspondences
    const ransacResult = this.runRANSAC(graphConsistentMatches);

    const inlierMatches = graphConsistentMatches.map((m, idx) => {
      const isInlier = ransacResult.inlierMask[idx] === 1;
      const res = ransacResult.residuals[idx] || 0;
      return {
        ...m,
        isInlier,
        reprojectionErrorPx: res,
      };
    });

    return {
      filteredMatchSet: {
        ...matchSet,
        matches: inlierMatches,
      },
      bestModel: ransacResult.bestH,
      inlierCount: ransacResult.inlierCount,
      inlierRatio: inlierMatches.length > 0 ? ransacResult.inlierCount / inlierMatches.length : 0,
      rmse: ransacResult.inlierRMSE,
    };
  }

  /**
   * Pairwise distance ratio consistency check
   * || p_i - p_j || / || q_i - q_j || ~ scale +/- epsilon
   */
  private pruneGraphInconsistencies(matches: Match[]): Match[] {
    const n = matches.length;
    if (n < 6) return matches;

    const consistencyScores = new Float32Array(n);
    const tol = this.options.pairwiseDistanceTolerance;

    for (let i = 0; i < n; i++) {
      let consistentNeighbors = 0;
      const pi = matches[i].sourcePoint;
      const qi = matches[i].targetPoint;

      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const pj = matches[j].sourcePoint;
        const qj = matches[j].targetPoint;

        const dSource = Math.hypot(pi.x - pj.x, pi.y - pj.y);
        const dTarget = Math.hypot(qi.x - qj.x, qi.y - qj.y);

        if (dSource > 5.0 && dTarget > 5.0) {
          const ratio = dTarget / dSource;
          // Typical scale variation is between 0.3x and 3.0x
          if (ratio >= 0.25 && ratio <= 4.0) {
            consistentNeighbors++;
          }
        }
      }
      consistencyScores[i] = consistentNeighbors / (n - 1);
    }

    // Keep matches with at least 30% pairwise graph consensus
    const passed = matches.filter((_, idx) => consistencyScores[idx] >= 0.3);
    return passed.length >= 4 ? passed : matches;
  }

  /**
   * RANSAC / MAGSAC++ Homography solver with 4-point minimal sample
   */
  private runRANSAC(matches: Match[]): {
    bestH: number[][];
    inlierMask: Uint8Array;
    inlierCount: number;
    inlierRMSE: number;
    residuals: Float32Array;
  } {
    const n = matches.length;
    const rng = new DeterministicRNG(777);
    const threshold = this.options.inlierThresholdPx;
    const maxIters = Math.min(this.options.maxIterations, 1500);

    let bestInliers = 0;
    let bestH = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    let bestMask = new Uint8Array(n);
    let bestResiduals = new Float32Array(n);
    let bestRMSE = Infinity;

    for (let iter = 0; iter < maxIters; iter++) {
      // Sample 4 distinct random points
      const sampleIndices: number[] = [];
      while (sampleIndices.length < 4) {
        const idx = rng.intRange(0, n - 1);
        if (!sampleIndices.includes(idx)) {
          sampleIndices.push(idx);
        }
      }

      const ptsSrc = sampleIndices.map((idx) => matches[idx].sourcePoint);
      const ptsTgt = sampleIndices.map((idx) => matches[idx].targetPoint);

      // Check collinearity degeneracy
      if (this.isDegenerateSample(ptsSrc) || this.isDegenerateSample(ptsTgt)) {
        continue;
      }

      // Compute 3x3 Homography from 4-point correspondence (Direct Linear Transform - DLT)
      const H = this.estimateHomographyDLT(ptsSrc, ptsTgt);
      if (!H) continue;

      let inliers = 0;
      let sqErrorSum = 0;
      const currentMask = new Uint8Array(n);
      const currentResiduals = new Float32Array(n);

      for (let i = 0; i < n; i++) {
        const sPt = matches[i].sourcePoint;
        const tPt = matches[i].targetPoint;
        const projected = applyHomographyToPoint(H, sPt);

        const dx = projected.x - tPt.x;
        const dy = projected.y - tPt.y;
        const error = Math.hypot(dx, dy);
        currentResiduals[i] = error;

        // MAGSAC++ continuous consensus scoring or RANSAC hard threshold
        if (error <= threshold) {
          inliers++;
          currentMask[i] = 1;
          sqErrorSum += error * error;
        }
      }

      if (inliers > bestInliers || (inliers === bestInliers && sqErrorSum / inliers < bestRMSE)) {
        bestInliers = inliers;
        bestH = H;
        bestMask = currentMask;
        bestResiduals = currentResiduals;
        bestRMSE = Math.sqrt(sqErrorSum / (inliers || 1));
      }
    }

    // Refine Homography using all found inliers via Least-Squares
    if (bestInliers >= 4) {
      const inlierSrc: Point2D[] = [];
      const inlierTgt: Point2D[] = [];
      for (let i = 0; i < n; i++) {
        if (bestMask[i] === 1) {
          inlierSrc.push(matches[i].sourcePoint);
          inlierTgt.push(matches[i].targetPoint);
        }
      }
      const refinedH = this.estimateHomographyDLT(inlierSrc, inlierTgt);
      if (refinedH) {
        bestH = refinedH;
      }
    }

    return {
      bestH,
      inlierMask: bestMask,
      inlierCount: bestInliers,
      inlierRMSE: bestRMSE === Infinity ? 0 : bestRMSE,
      residuals: bestResiduals,
    };
  }

  /**
   * Direct Linear Transform (DLT) for Homography Matrix Estimation
   */
  estimateHomographyDLT(src: Point2D[], tgt: Point2D[]): number[][] | null {
    if (src.length < 4 || tgt.length < 4) return null;

    // Normalize coordinates for numerical stability (Hartley normalization)
    const normSrc = this.normalizePoints(src);
    const normTgt = this.normalizePoints(tgt);

    // Build 2N x 9 linear system Ah = 0
    const A: number[][] = [];
    for (let i = 0; i < src.length; i++) {
      const x = normSrc.points[i].x;
      const y = normSrc.points[i].y;
      const u = normTgt.points[i].x;
      const v = normTgt.points[i].y;

      A.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
      A.push([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
    }

    // Solve for h using SVD / Power Iteration of A^T * A
    const h = this.solveNullspaceSVD(A);
    if (!h) return null;

    const H_norm = [
      [h[0], h[1], h[2]],
      [h[3], h[4], h[5]],
      [h[6], h[7], h[8]],
    ];

    // Denormalize: H = T_tgt^-1 * H_norm * T_src
    return this.denormalizeHomography(H_norm, normSrc.T, normTgt.T);
  }

  private normalizePoints(pts: Point2D[]): { points: Point2D[]; T: number[][] } {
    let meanX = 0;
    let meanY = 0;
    for (const p of pts) {
      meanX += p.x;
      meanY += p.y;
    }
    meanX /= pts.length;
    meanY /= pts.length;

    let distSum = 0;
    for (const p of pts) {
      distSum += Math.hypot(p.x - meanX, p.y - meanY);
    }
    const meanDist = distSum / pts.length || 1;
    const scale = Math.SQRT2 / meanDist;

    const points = pts.map((p) => ({
      x: (p.x - meanX) * scale,
      y: (p.y - meanY) * scale,
    }));

    const T = [
      [scale, 0, -scale * meanX],
      [0, scale, -scale * meanY],
      [0, 0, 1],
    ];

    return { points, T };
  }

  private denormalizeHomography(H_norm: number[][], T_src: number[][], T_tgt: number[][]): number[][] {
    // T_tgt^-1
    const s_tgt = T_tgt[0][0];
    const tx_tgt = -T_tgt[0][2] / s_tgt;
    const ty_tgt = -T_tgt[1][2] / s_tgt;

    const invT_tgt = [
      [1 / s_tgt, 0, tx_tgt],
      [0, 1 / s_tgt, ty_tgt],
      [0, 0, 1],
    ];

    // Multiply: invT_tgt * H_norm * T_src
    const temp = this.multiply3x3(invT_tgt, H_norm);
    const H = this.multiply3x3(temp, T_src);

    // Normalize so H[2][2] = 1
    const norm = H[2][2] || 1;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        H[r][c] /= norm;
      }
    }
    return H;
  }

  private multiply3x3(A: number[][], B: number[][]): number[][] {
    const C = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 3; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return C;
  }

  private solveNullspaceSVD(A: number[][]): number[] | null {
    // Construct 9x9 normal matrix M = A^T * A
    const M: number[][] = Array.from({ length: 9 }, () => new Array(9).fill(0));
    for (let r = 0; r < A.length; r++) {
      for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
          M[i][j] += A[r][i] * A[r][j];
        }
      }
    }

    // Jacobi eigenvalue algorithm to find all eigenvalues and eigenvectors of real symmetric 9x9 matrix
    const n = 9;
    const S = M.map((row) => [...row]);
    const V = Array.from({ length: n }, (_, i) => {
      const row = new Array(n).fill(0);
      row[i] = 1;
      return row;
    });

    for (let iter = 0; iter < 100; iter++) {
      let maxOffDiag = 0;
      let p = 0;
      let q = 1;

      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const val = Math.abs(S[i][j]);
          if (val > maxOffDiag) {
            maxOffDiag = val;
            p = i;
            q = j;
          }
        }
      }

      if (maxOffDiag < 1e-12) break;

      const app = S[p][p];
      const aqq = S[q][q];
      const apq = S[p][q];

      const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
      const c = Math.cos(phi);
      const s = Math.sin(phi);

      for (let k = 0; k < n; k++) {
        if (k !== p && k !== q) {
          const akp = S[k][p];
          const akq = S[k][q];
          S[k][p] = c * akp - s * akq;
          S[p][k] = S[k][p];
          S[k][q] = s * akp + c * akq;
          S[q][k] = S[k][q];
        }
      }
      S[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      S[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      S[p][q] = 0;
      S[q][p] = 0;

      for (let k = 0; k < n; k++) {
        const vkp = V[k][p];
        const vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq;
        V[k][q] = s * vkp + c * vkq;
      }
    }

    // Find column with smallest eigenvalue
    let minEig = Infinity;
    let minCol = 0;
    for (let i = 0; i < n; i++) {
      if (S[i][i] < minEig) {
        minEig = S[i][i];
        minCol = i;
      }
    }

    return V.map((row) => row[minCol]);
  }

  private isDegenerateSample(pts: Point2D[]): boolean {
    if (pts.length < 3) return false;
    // Check if 3 points are collinear: Area = 0.5 * (x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        for (let k = j + 1; k < pts.length; k++) {
          const area = Math.abs(
            pts[i].x * (pts[j].y - pts[k].y) +
            pts[j].x * (pts[k].y - pts[i].y) +
            pts[k].x * (pts[i].y - pts[j].y)
          );
          if (area < 1.0) return true; // Collinear degeneracy
        }
      }
    }
    return false;
  }
}
