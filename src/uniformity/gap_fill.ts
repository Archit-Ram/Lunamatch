/**
 * LunaMatch - Part 27: Explicit Uniformity Policy (coverage feedback loop)
 *
 * Reporting "Coverage %" is not enough: a registration anchored on one crater field is fragile even when its
 * RMSE looks great. This stage measures which cells of the source grid have no accepted tie-point, then hunts
 * for correspondences to fill them, accepting lower-confidence evidence ONLY above strict floors:
 *
 *   Round 1 (candidate pool): matches that were dropped earlier (low fusion confidence, RANSAC-rejected, etc.)
 *     are reconsidered for an empty cell if confidence >= confidenceFloor AND they agree with the fitted
 *     transform to within maxReprojectionPx (stricter than RANSAC's threshold). The best one per cell wins.
 *   Round 2 (targeted probe): cells still empty get one NCC area-correlation probe at the transform-predicted
 *     location, accepted only if NCC >= nccFloor and the found offset stays within maxShiftPx of the prediction.
 *
 * Nothing below the floors is ever accepted, and every accepted match is tagged acceptedForCoverage so it can
 * be audited or excluded downstream. The fitted transform is NOT refit on filled points: they are verified
 * against it, not used to move it.
 */

import { ImageData, Match, Point2D, TransformModel } from '../types';
import { applyTransformModel } from '../registration/transform_utils';
import { AreaCorrelationMatcher } from '../matching/area_correlation_matcher';

export interface GapFillOptions {
  enabled: boolean;
  gridDimension: number;
  confidenceFloor: number;
  maxReprojectionPx: number;
  nccFloor: number;
  maxShiftPx: number;
  probeSearchRadiusPx: number;
}

export interface GapFillStats {
  gridCells: number;
  coverageBefore: number; // occupied cells / total cells
  coverageAfter: number;
  cellsFilledFromPool: number;
  cellsFilledByProbe: number;
  emptyCellsRemaining: number;
  confidenceFloor: number;
  nccFloor: number;
}

export class UniformityGapFiller {
  static fill(
    accepted: Match[],
    pool: Match[],
    transform: TransformModel,
    source: ImageData,
    target: ImageData,
    options: GapFillOptions
  ): { accepted: Match[]; added: Match[]; stats: GapFillStats } {
    const dim = options.gridDimension;
    const cells = dim * dim;
    const cw = source.width / dim;
    const ch = source.height / dim;
    const cellOf = (p: Point2D) =>
      Math.min(dim - 1, Math.max(0, Math.floor(p.y / ch))) * dim + Math.min(dim - 1, Math.max(0, Math.floor(p.x / cw)));

    const occupied = new Uint8Array(cells);
    for (const m of accepted) if (m.isInlier !== false) occupied[cellOf(m.sourcePoint)] = 1;
    const count = () => occupied.reduce((s, v) => s + v, 0);
    const before = count() / cells;

    const stats: GapFillStats = {
      gridCells: cells,
      coverageBefore: before,
      coverageAfter: before,
      cellsFilledFromPool: 0,
      cellsFilledByProbe: 0,
      emptyCellsRemaining: cells - count(),
      confidenceFloor: options.confidenceFloor,
      nccFloor: options.nccFloor,
    };
    if (!options.enabled || !transform.validity || stats.emptyCellsRemaining === 0) return { accepted, added: [], stats };

    const added: Match[] = [];
    const errOf = (m: Match) => {
      const p = applyTransformModel(transform, m.sourcePoint);
      return Math.hypot(p.x - m.targetPoint.x, p.y - m.targetPoint.y);
    };

    // Round 1: reconsider dropped matches for empty cells.
    const used = new Set(accepted.map((m) => `${m.sourcePoint.x.toFixed(2)},${m.sourcePoint.y.toFixed(2)}`));
    const bestPerCell = new Map<number, { m: Match; err: number }>();
    for (const m of pool) {
      const key = `${m.sourcePoint.x.toFixed(2)},${m.sourcePoint.y.toFixed(2)}`;
      if (used.has(key)) continue;
      const cell = cellOf(m.sourcePoint);
      if (occupied[cell]) continue;
      if (m.confidence < options.confidenceFloor) continue;
      const err = errOf(m);
      if (err > options.maxReprojectionPx) continue;
      const cur = bestPerCell.get(cell);
      if (!cur || m.confidence > cur.m.confidence || (m.confidence === cur.m.confidence && err < cur.err)) bestPerCell.set(cell, { m, err });
    }
    for (const [cell, { m, err }] of bestPerCell) {
      occupied[cell] = 1;
      stats.cellsFilledFromPool++;
      added.push({ ...m, isInlier: true, reprojectionErrorPx: err, acceptedForCoverage: true });
    }

    // Round 2: targeted NCC probe at the transform-predicted location of each still-empty cell.
    const matcher = new AreaCorrelationMatcher({
      patchRadius: 7,
      searchRadiusPx: options.probeSearchRadiusPx,
      minCorrelation: options.nccFloor,
    });
    for (let cell = 0; cell < cells; cell++) {
      if (occupied[cell]) continue;
      const cx = (cell % dim) * cw + cw / 2;
      const cy = Math.floor(cell / dim) * ch + ch / 2;
      const predicted = applyTransformModel(transform, { x: cx, y: cy });
      const probe = matcher.probe(source, target, { x: cx, y: cy }, predicted);
      if (!probe || probe.score < options.nccFloor) continue;
      const shift = Math.hypot(probe.point.x - predicted.x, probe.point.y - predicted.y);
      if (shift > options.maxShiftPx) continue;
      occupied[cell] = 1;
      stats.cellsFilledByProbe++;
      added.push({
        id: `gapfill_probe_${cell}`,
        sourcePoint: { x: cx, y: cy },
        targetPoint: probe.point,
        confidence: Math.min(1, Math.max(0, probe.score)),
        method: 'AreaCorrelation',
        methodProvenance: { photometricConsistency: probe.score },
        isInlier: true,
        reprojectionErrorPx: shift,
        acceptedForCoverage: true,
      });
    }

    stats.coverageAfter = count() / cells;
    stats.emptyCellsRemaining = cells - count();
    return { accepted: [...accepted, ...added], added, stats };
  }
}
