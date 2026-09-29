/**
 * LunaMatch - Part 29: Ground-truth-free Registration Quality Assessment
 *
 * `status: 'success'` only means the pipeline ran to completion. On real data there is no ground truth, so a
 * registration built from a handful of mutually-consistent but wrong matches looks identical to a good one.
 * This assessment uses only signals available at run time:
 *   - inlierCount / inlierRatio : RANSAC support for the fitted geometry
 *   - residualRmsePx            : how well the fitted transform explains the accepted matches
 *   - coverage                  : fraction of grid cells holding at least one accepted match
 * The level is the WORST of the per-signal levels, and each signal that fell short of 'high' is listed in
 * `reasons`. Thresholds are heuristics (configurable), not calibrated probabilities.
 */

export type QualityLevel = 'high' | 'medium' | 'low';

export interface QualityThresholds {
  minInliersHigh: number;
  minInliersMedium: number;
  minInlierRatioHigh: number;
  minInlierRatioMedium: number;
  maxResidualRmseHighPx: number;
  maxResidualRmseMediumPx: number;
  minCoverageHigh: number;
  minCoverageMedium: number;
}

export const DEFAULT_QUALITY_THRESHOLDS: QualityThresholds = {
  minInliersHigh: 20,
  minInliersMedium: 8,
  minInlierRatioHigh: 0.6,
  minInlierRatioMedium: 0.3,
  maxResidualRmseHighPx: 1.5,
  maxResidualRmseMediumPx: 3.0,
  minCoverageHigh: 0.5,
  minCoverageMedium: 0.25,
};

export interface QualityInput {
  inlierCount: number;
  inlierRatio: number;
  residualRmsePx: number;
  coverage: number;
}

export interface QualityAssessment {
  level: QualityLevel;
  reasons: string[];
}

const RANK: Record<QualityLevel, number> = { high: 0, medium: 1, low: 2 };

export function assessRegistrationQuality(
  input: QualityInput,
  t: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS
): QualityAssessment {
  let level: QualityLevel = 'high';
  const reasons: string[] = [];
  const note = (l: QualityLevel, reason: string) => {
    reasons.push(reason);
    if (RANK[l] > RANK[level]) level = l;
  };

  if (input.inlierCount < t.minInliersMedium) note('low', `only ${input.inlierCount} geometrically consistent matches (need >= ${t.minInliersMedium})`);
  else if (input.inlierCount < t.minInliersHigh) note('medium', `${input.inlierCount} consistent matches (< ${t.minInliersHigh} for high confidence)`);

  if (input.inlierRatio < t.minInlierRatioMedium) note('low', `inlier ratio ${(input.inlierRatio * 100).toFixed(0)}% (need >= ${(t.minInlierRatioMedium * 100).toFixed(0)}%)`);
  else if (input.inlierRatio < t.minInlierRatioHigh) note('medium', `inlier ratio ${(input.inlierRatio * 100).toFixed(0)}% (< ${(t.minInlierRatioHigh * 100).toFixed(0)}% for high confidence)`);

  if (input.residualRmsePx > t.maxResidualRmseMediumPx) note('low', `transform residual ${input.residualRmsePx.toFixed(2)} px (> ${t.maxResidualRmseMediumPx} px)`);
  else if (input.residualRmsePx > t.maxResidualRmseHighPx) note('medium', `transform residual ${input.residualRmsePx.toFixed(2)} px (> ${t.maxResidualRmseHighPx} px for high confidence)`);

  if (input.coverage < t.minCoverageMedium) note('low', `matches cover only ${(input.coverage * 100).toFixed(0)}% of the scene (need >= ${(t.minCoverageMedium * 100).toFixed(0)}%)`);
  else if (input.coverage < t.minCoverageHigh) note('medium', `matches cover ${(input.coverage * 100).toFixed(0)}% of the scene (< ${(t.minCoverageHigh * 100).toFixed(0)}% for high confidence)`);

  return { level, reasons };
}
