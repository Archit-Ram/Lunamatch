/**
 * LunaMatch - Part 28: 2x2 Covariance Toolkit
 *
 * Small, dependency-free helpers for the per-match statistical covariance
 * (target-space, px^2) that the decorrelated ensemble and the hierarchical
 * chain produce and propagate:
 *   Sigma = [ xx  xy ]
 *           [ xy  yy ]
 * Image convention: x = column (right), y = row (down). Ellipse angles are
 * measured from +x toward +y, in degrees.
 */

import { Point2D } from '../types';

export interface Cov2 {
  xx: number;
  xy: number;
  yy: number;
}

export interface ErrorEllipse {
  semiMajorPx: number;
  semiMinorPx: number;
  angleDeg: number;
}

export const isoCov = (sigmaPx: number): Cov2 => ({ xx: sigmaPx * sigmaPx, xy: 0, yy: sigmaPx * sigmaPx });

export const addCov = (a: Cov2, b: Cov2): Cov2 => ({ xx: a.xx + b.xx, xy: a.xy + b.xy, yy: a.yy + b.yy });

export const scaleCov = (a: Cov2, k: number): Cov2 => ({ xx: a.xx * k, xy: a.xy * k, yy: a.yy * k });

/** Mean per-axis 1-sigma, sqrt(trace/2). This is what Match.uncertaintyPx reports when a covariance exists. */
export const meanSigmaPx = (c: Cov2): number => Math.sqrt(Math.max(0, (c.xx + c.yy) / 2));

/** Eigen-decomposition of a symmetric 2x2: l1 >= l2 with the major-axis angle (rad, from +x toward +y). */
export function eigen2(c: Cov2): { l1: number; l2: number; angleRad: number } {
  const half = (c.xx + c.yy) / 2;
  const disc = Math.sqrt(Math.max(0, ((c.xx - c.yy) / 2) ** 2 + c.xy * c.xy));
  return { l1: half + disc, l2: half - disc, angleRad: 0.5 * Math.atan2(2 * c.xy, c.xx - c.yy) };
}

/** k-sigma error ellipse. k = 1 is the 1-sigma ellipse (39% coverage for 2 DOF); k = 2.4477 gives 95%. */
export function errorEllipse(c: Cov2, k = 1): ErrorEllipse {
  const { l1, l2, angleRad } = eigen2(c);
  return {
    semiMajorPx: k * Math.sqrt(Math.max(0, l1)),
    semiMinorPx: k * Math.sqrt(Math.max(0, l2)),
    angleDeg: (angleRad * 180) / Math.PI,
  };
}

/** Rebuilds a covariance from eigenvalues/angle. */
export function covFromEigen(l1: number, l2: number, angleRad: number): Cov2 {
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  return { xx: l1 * c * c + l2 * s * s, xy: (l1 - l2) * c * s, yy: l1 * s * s + l2 * c * c };
}

/** Clips negative eigenvalues to zero (nearest positive semi-definite matrix). */
export function positiveSemiDefinitePart(c: Cov2): Cov2 {
  const { l1, l2, angleRad } = eigen2(c);
  return covFromEigen(Math.max(0, l1), Math.max(0, l2), angleRad);
}

/** J * C * J^T for a 2x2 Jacobian J = [[a, b], [c, d]]. */
export function propagateCov(c: Cov2, J: number[][]): Cov2 {
  const [[a, b], [cc, d]] = J;
  const xx = a * a * c.xx + 2 * a * b * c.xy + b * b * c.yy;
  const yy = cc * cc * c.xx + 2 * cc * d * c.xy + d * d * c.yy;
  const xy = a * cc * c.xx + (a * d + b * cc) * c.xy + b * d * c.yy;
  return { xx, xy, yy };
}

/** Central-difference Jacobian of a 2D point map at p. */
export function numericJacobian(fn: (p: Point2D) => Point2D, p: Point2D, h = 0.5): number[][] {
  const xp = fn({ x: p.x + h, y: p.y });
  const xm = fn({ x: p.x - h, y: p.y });
  const yp = fn({ x: p.x, y: p.y + h });
  const ym = fn({ x: p.x, y: p.y - h });
  return [
    [(xp.x - xm.x) / (2 * h), (yp.x - ym.x) / (2 * h)],
    [(xp.y - xm.y) / (2 * h), (yp.y - ym.y) / (2 * h)],
  ];
}

/** Squared Mahalanobis distance of an error vector (dx, dy) under c (tiny ridge for singular covariances). */
export function mahalanobisSq(c: Cov2, dx: number, dy: number): number {
  const ridge = 1e-12;
  const xx = c.xx + ridge;
  const yy = c.yy + ridge;
  const det = xx * yy - c.xy * c.xy;
  if (det <= 0) return Infinity;
  return (yy * dx * dx - 2 * c.xy * dx * dy + xx * dy * dy) / det;
}

/** Solves A x = b (small dense system) by Gaussian elimination with partial pivoting; null if singular. */
export function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[piv][i])) piv = k;
    if (Math.abs(M[piv][i]) < 1e-14) return null;
    [M[i], M[piv]] = [M[piv], M[i]];
    for (let k = i + 1; k < n; k++) {
      const f = M[k][i] / M[i][i];
      for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}
