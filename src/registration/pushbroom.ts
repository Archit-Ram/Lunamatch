/**
 * LunaMatch - Part 24: Per-Strip Pushbroom Transforms
 *
 * OHRC, TMC-2 and IIRS are pushbroom sensors: every image line is exposed at
 * a different instant, so spacecraft attitude/velocity drift shows up as a
 * transform that varies smoothly ALONG-TRACK. A single global affine cannot
 * represent that, while a global TPS spends 2n+6 parameters chasing noise.
 *
 * This module splits the image into overlapping along-track segments, fits a
 * local affine per segment (ridge-regularized toward the global affine so
 * thin or near-collinear segments stay well conditioned), smooths the
 * parameters across neighbouring segments, and evaluates the result by
 * linearly interpolating parameters between segment centers - so the
 * transform is continuous with no seams at segment boundaries.
 */

import { Match, Point2D } from '../types';

export interface PushbroomOptions {
  /** Along-track axis: 'y' = image rows (default), 'x' = image columns. */
  alongTrackAxis?: 'x' | 'y';
  /** Number of segments. Default: auto (~one per 48 px of along-track extent, clamped to [2, 8]). */
  numSegments?: number;
  /** Fraction of a segment's length that neighbouring segments overlap when collecting matches. */
  overlapFraction?: number;
  /** Segments with fewer matches than this fall back to the global affine. */
  minMatchesPerSegment?: number;
  /** Gaussian smoothing sigma across segment index (0 disables smoothing). */
  smoothingSigma?: number;
  /** Ridge strength pulling a segment toward the global affine, in "virtual matches". */
  ridgeMatches?: number;
}

export interface PushbroomSegment {
  index: number;
  start: number; // along-track start (px)
  end: number; // along-track end (px)
  center: number; // along-track center (px)
  matchCount: number;
  fallbackToGlobal: boolean;
  /** Smoothed affine params in pixel space: [a11, a12, tx, a21, a22, ty]. */
  params: number[];
}

export interface PushbroomFit {
  axis: 'x' | 'y';
  segments: PushbroomSegment[];
  globalParams: number[];
  /** Effective parameter count for model selection (6 per segment that used local data, 6 for the global fallback). */
  effectiveParameters: number;
  apply(point: Point2D): Point2D;
}

const DEFAULTS = {
  alongTrackAxis: 'y' as const,
  overlapFraction: 0.25,
  minMatchesPerSegment: 6,
  smoothingSigma: 0.8,
  ridgeMatches: 3,
};

export function defaultSegmentCount(alongTrackExtent: number): number {
  return Math.max(2, Math.min(8, Math.round(alongTrackExtent / 48)));
}

/** Solves the n x n system A x = b in place via Gaussian elimination with partial pivoting; null if singular. */
function solveSmall(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[piv][i])) piv = k;
    if (Math.abs(M[piv][i]) < 1e-12) return null;
    [M[i], M[piv]] = [M[piv], M[i]];
    for (let k = i + 1; k < n; k++) {
      const c = M[k][i] / M[i][i];
      for (let j = i; j <= n; j++) M[k][j] -= c * M[i][j];
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

/**
 * Ridge-regularized affine least squares in normalized source coordinates.
 * Returns pixel-space params [a11, a12, tx, a21, a22, ty]. `prior` (pixel-space
 * params) receives `ridgeWeight` virtual observations worth of pull; with no
 * prior the fit is plain least squares (tiny ridge for conditioning).
 */
function fitAffineParams(
  src: Point2D[],
  tgt: Point2D[],
  width: number,
  height: number,
  prior?: number[],
  ridgeWeight = 0
): number[] | null {
  // Normalized coords: u = x/W, v = y/H. Model: X = A11 u + A12 v + T1 (and likewise for Y).
  const AtA = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const AtX = [0, 0, 0];
  const AtY = [0, 0, 0];
  for (let i = 0; i < src.length; i++) {
    const r = [src[i].x / width, src[i].y / height, 1];
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) AtA[a][b] += r[a] * r[b];
      AtX[a] += r[a] * tgt[i].x;
      AtY[a] += r[a] * tgt[i].y;
    }
  }

  const priorN = prior
    ? { x: [prior[0] * width, prior[1] * height, prior[2]], y: [prior[3] * width, prior[4] * height, prior[5]] }
    : undefined;
  const lam = prior ? Math.max(ridgeWeight, 1e-6) : 1e-9;
  for (let a = 0; a < 3; a++) {
    AtA[a][a] += lam;
    if (priorN) {
      AtX[a] += lam * priorN.x[a];
      AtY[a] += lam * priorN.y[a];
    }
  }

  const px = solveSmall(AtA, AtX);
  const py = solveSmall(AtA, AtY);
  if (!px || !py) return null;
  return [px[0] / width, px[1] / height, px[2], py[0] / width, py[1] / height, py[2]];
}

function applyParams(p: number[], pt: Point2D): Point2D {
  return { x: p[0] * pt.x + p[1] * pt.y + p[2], y: p[3] * pt.x + p[4] * pt.y + p[5] };
}

/**
 * Fits the segmented pushbroom model to a set of matches.
 * Returns null if there is not enough data for even a global affine.
 */
export function fitPushbroomModel(
  matches: Match[],
  imageWidth: number,
  imageHeight: number,
  options: PushbroomOptions = {}
): PushbroomFit | null {
  const opts = { ...DEFAULTS, ...options };
  const axis = opts.alongTrackAxis;
  const extent = axis === 'y' ? imageHeight : imageWidth;
  const numSegments = Math.max(1, Math.floor(options.numSegments ?? defaultSegmentCount(extent)));

  const src = matches.map((m) => m.sourcePoint);
  const tgt = matches.map((m) => m.targetPoint);
  if (src.length < 4) return null;

  const globalParams = fitAffineParams(src, tgt, imageWidth, imageHeight);
  if (!globalParams) return null;

  const segLen = extent / numSegments;
  const overlap = segLen * opts.overlapFraction;
  const coord = (p: Point2D) => (axis === 'y' ? p.y : p.x);

  const raw: PushbroomSegment[] = [];
  for (let k = 0; k < numSegments; k++) {
    const start = k * segLen;
    const end = (k + 1) * segLen;
    const idx: number[] = [];
    for (let i = 0; i < src.length; i++) {
      const c = coord(src[i]);
      if (c >= start - overlap && c < end + overlap) idx.push(i);
    }

    let params = globalParams;
    let fallback = true;
    if (idx.length >= opts.minMatchesPerSegment) {
      const local = fitAffineParams(
        idx.map((i) => src[i]),
        idx.map((i) => tgt[i]),
        imageWidth,
        imageHeight,
        globalParams,
        opts.ridgeMatches
      );
      if (local) {
        params = local;
        fallback = false;
      }
    }
    raw.push({ index: k, start, end, center: (start + end) / 2, matchCount: idx.length, fallbackToGlobal: fallback, params });
  }

  // Gaussian smoothing of parameters across segment index, weighted by data support.
  const sigma = opts.smoothingSigma;
  const smoothed = raw.map((seg, k) => {
    if (sigma <= 0 || numSegments === 1) return seg.params;
    const out = new Array(6).fill(0);
    let wSum = 0;
    for (let j = 0; j < numSegments; j++) {
      const d = j - k;
      const support = raw[j].fallbackToGlobal ? 0.25 : 1;
      const w = Math.exp(-(d * d) / (2 * sigma * sigma)) * support;
      for (let q = 0; q < 6; q++) out[q] += w * raw[j].params[q];
      wSum += w;
    }
    return out.map((v) => v / wSum);
  });

  const segments = raw.map((seg, k) => ({ ...seg, params: smoothed[k] }));

  const apply = (pt: Point2D): Point2D => {
    const s = coord(pt);
    const first = segments[0];
    const last = segments[segments.length - 1];
    if (segments.length === 1 || s <= first.center) return applyParams(first.params, pt);
    if (s >= last.center) return applyParams(last.params, pt);
    let k = 0;
    while (k < segments.length - 2 && s >= segments[k + 1].center) k++;
    const a = segments[k];
    const b = segments[k + 1];
    const t = (s - a.center) / (b.center - a.center);
    const blended = a.params.map((v, q) => (1 - t) * v + t * b.params[q]);
    return applyParams(blended, pt);
  };

  const localSegments = segments.filter((s) => !s.fallbackToGlobal).length;
  return {
    axis,
    segments,
    globalParams,
    effectiveParameters: 6 * Math.max(1, localSegments),
    apply,
  };
}
