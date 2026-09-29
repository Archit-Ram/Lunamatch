/**
 * LunaMatch - Part 13 / Part 28: Decorrelated Matcher Fusion Engine
 *
 * Combines candidate correspondences from independent matching experts
 * (LoFTR, RIFT, LightGlue, area correlation) using spatial association,
 * multi-model consensus and physics-informed confidence fusion:
 *   C = w_L * C_L + w_R * C_R + w_G * C_G + w_P * C_P
 *
 * v2 additions (this file):
 *  1. Decorrelated ensemble. Experts do not fail independently: LoFTR and
 *     SuperPoint+LightGlue are both learned and tend to fail together on
 *     repetitive or featureless terrain, while RIFT (phase congruency) fails
 *     differently. Agreement between correlated experts therefore counts for
 *     less. Each cluster's expert errors are modelled with a correlation
 *     matrix R; target positions are combined by generalized least squares
 *     (BLUE), and the effective number of independent experts
 *         n_eff = k^2 / sum_ij R_ij
 *     (1 <= n_eff <= k) scales the consensus bonus.
 *  2. Propagated covariance. Every fused match carries a real 2x2 target
 *     covariance: the GLS model variance w^T S w (isotropic), plus the
 *     PSD-clipped excess of the experts' empirical scatter over their stated
 *     noise, divided by n_eff. Excess scatter is what gives the ellipse a
 *     direction: experts disagreeing along x widen the ellipse along x.
 *     Match.uncertaintyPx is sqrt(trace/2) of that covariance.
 */

import { Match, MatchSet, Point2D } from '../types';
import { GeometryProvider } from '../core/interfaces';
import {
  Cov2,
  meanSigmaPx,
  positiveSemiDefinitePart,
  solveLinear,
} from '../uncertainty/covariance';

export interface FusionWeights {
  loftr: number;
  rift: number;
  lightglue: number;
  geometryPrior: number;
}

export type ExpertFamily = 'loftr' | 'rift' | 'lightglue' | 'area' | 'other';

export interface ExpertCorrelationModel {
  /** Correlation between two matches from the SAME expert family inside one cluster (near-duplicates). */
  sameFamily: number;
  /** Correlation between different families; keys are 'a|b' with families sorted alphabetically. */
  pairs: Record<string, number>;
  /** Fallback for pairs not listed. */
  fallback: number;
}

export const DEFAULT_EXPERT_CORRELATION: ExpertCorrelationModel = {
  sameFamily: 0.9,
  pairs: {
    'lightglue|loftr': 0.5, // both learned; shared failure on repetitive / featureless terrain
    'loftr|rift': 0.15,
    'lightglue|rift': 0.15,
    'area|loftr': 0.1,
    'area|lightglue': 0.1,
    'area|rift': 0.1,
  },
  fallback: 0.2,
};

export interface FusionOptions {
  associationRadiusPx?: number; // Distance in source space to cluster matches
  disagreementThresholdPx?: number; // Target discrepancy triggering outlier penalty
  weights?: FusionWeights;
  correlation?: ExpertCorrelationModel;
  /** Per-expert 1-sigma floor (px) so an over-confident expert cannot dominate the GLS combination. */
  minExpertSigmaPx?: number;
}

export interface FusionRuntime {
  /** Experts that could not run at all (e.g. model file missing). Their weight is redistributed, not counted as disagreement. */
  unavailableExperts?: ExpertFamily[];
}

export function expertFamilyOf(method: string): ExpertFamily {
  if (method === 'LoFTR' || method === 'SimulatedLoFTR') return 'loftr';
  if (method === 'RIFT' || method === 'SimulatedRIFT') return 'rift';
  if (method === 'LightGlue' || method === 'SimulatedLightGlue') return 'lightglue';
  if (method === 'AreaCorrelation') return 'area';
  return 'other';
}

export function pairCorrelation(a: ExpertFamily, b: ExpertFamily, model: ExpertCorrelationModel): number {
  if (a === b) return model.sameFamily;
  const key = [a, b].sort().join('|');
  return model.pairs[key] ?? model.fallback;
}

/** Effective number of independent experts for a correlation matrix R (k x k): k^2 / sum_ij R_ij. */
export function effectiveIndependentCount(R: number[][]): number {
  const k = R.length;
  let sum = 0;
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) sum += R[i][j];
  return sum > 0 ? Math.min(k, Math.max(1, (k * k) / sum)) : k;
}

/**
 * Generalized least squares (BLUE) weights for a common-mean estimate from correlated estimators.
 * Falls back to inverse-variance weights if the covariance is singular or the BLUE has negative weights
 * (which strong correlation can produce and which would extrapolate beyond the experts' estimates).
 */
export function glsWeights(Sigma: number[][]): number[] {
  const k = Sigma.length;
  const invVar = Sigma.map((row, i) => 1 / Math.max(1e-12, row[i]));
  const fallback = () => {
    const s = invVar.reduce((a, b) => a + b, 0);
    return invVar.map((v) => v / s);
  };
  if (k === 1) return [1];
  const x = solveLinear(Sigma, new Array(k).fill(1));
  if (!x) return fallback();
  const s = x.reduce((a, b) => a + b, 0);
  if (!(s > 0)) return fallback();
  const w = x.map((v) => v / s);
  return w.some((v) => v < 0) ? fallback() : w;
}

export class MatchFusionEngine {
  private options: Required<FusionOptions>;

  constructor(options: FusionOptions = {}) {
    this.options = {
      associationRadiusPx: 4.5,
      disagreementThresholdPx: 5.0,
      minExpertSigmaPx: 0.15,
      correlation: DEFAULT_EXPERT_CORRELATION,
      ...options,
      weights: {
        loftr: 0.35,
        rift: 0.35,
        lightglue: 0.2,
        geometryPrior: 0.1,
        ...options.weights,
      },
    };
  }

  fuse(matchSets: MatchSet[], geometryProvider?: GeometryProvider, runtime: FusionRuntime = {}): MatchSet {
    const empty = (): MatchSet => ({
      matches: [],
      sourceImageId: matchSets[0]?.sourceImageId ?? 'unknown',
      targetImageId: matchSets[0]?.targetImageId ?? 'unknown',
      coordinateConvention: 'x=column, y=row',
    });
    if (matchSets.length === 0) return empty();

    const allMatches: Match[] = [];
    for (const ms of matchSets) allMatches.push(...ms.matches);
    if (allMatches.length === 0) return empty();

    const { associationRadiusPx, disagreementThresholdPx, correlation, minExpertSigmaPx } = this.options;

    // Redistribute the weight of experts that could not run instead of silently capping every fused score.
    const unavailable = new Set(runtime.unavailableExperts ?? []);
    const w0 = this.options.weights;
    const raw = {
      loftr: unavailable.has('loftr') ? 0 : w0.loftr,
      rift: unavailable.has('rift') ? 0 : w0.rift,
      lightglue: unavailable.has('lightglue') ? 0 : w0.lightglue,
      geometryPrior: w0.geometryPrior,
    };
    const rawSum = raw.loftr + raw.rift + raw.lightglue + raw.geometryPrior;
    const originalSum = w0.loftr + w0.rift + w0.lightglue + w0.geometryPrior;
    const rescale = rawSum > 0 ? originalSum / rawSum : 1;
    const weights = {
      loftr: raw.loftr * rescale,
      rift: raw.rift * rescale,
      lightglue: raw.lightglue * rescale,
      geometryPrior: raw.geometryPrior * rescale,
    };

    const visited = new Uint8Array(allMatches.length);
    const fusedMatches: Match[] = [];

    for (let i = 0; i < allMatches.length; i++) {
      if (visited[i]) continue;
      visited[i] = 1;

      const cluster: Match[] = [allMatches[i]];
      const baseSrc = allMatches[i].sourcePoint;
      for (let j = i + 1; j < allMatches.length; j++) {
        if (visited[j]) continue;
        const c = allMatches[j].sourcePoint;
        if (Math.hypot(baseSrc.x - c.x, baseSrc.y - c.y) <= associationRadiusPx) {
          visited[j] = 1;
          cluster.push(allMatches[j]);
        }
      }
      const k = cluster.length;

      // Per-expert confidences (max within family).
      let loftrConf = 0;
      let riftConf = 0;
      let lightglueConf = 0;
      const families: ExpertFamily[] = cluster.map((m) => expertFamilyOf(m.method));
      cluster.forEach((m, idx) => {
        if (families[idx] === 'loftr') loftrConf = Math.max(loftrConf, m.confidence);
        else if (families[idx] === 'rift') riftConf = Math.max(riftConf, m.confidence);
        else if (families[idx] === 'lightglue') lightglueConf = Math.max(lightglueConf, m.confidence);
      });

      // Error model: per-expert sigma, correlation matrix R, covariance Sigma = D R D.
      const sigmas = cluster.map((m) =>
        Math.max(minExpertSigmaPx, m.uncertaintyPx ?? 0.4 + (1 - Math.min(1, Math.max(0, m.confidence))) * 2.0)
      );
      const R: number[][] = cluster.map((_, a) =>
        cluster.map((__, b) => (a === b ? 1 : pairCorrelation(families[a], families[b], correlation)))
      );
      const Sigma = R.map((row, a) => row.map((r, b) => r * sigmas[a] * sigmas[b]));

      const w = glsWeights(Sigma);
      const nEff = effectiveIndependentCount(R);

      let meanSrc: Point2D = { x: 0, y: 0 };
      let meanTgt: Point2D = { x: 0, y: 0 };
      for (let a = 0; a < k; a++) {
        meanSrc.x += w[a] * cluster[a].sourcePoint.x;
        meanSrc.y += w[a] * cluster[a].sourcePoint.y;
        meanTgt.x += w[a] * cluster[a].targetPoint.x;
        meanTgt.y += w[a] * cluster[a].targetPoint.y;
      }

      // Model variance of the combined estimate (exact for any weight vector): w^T Sigma w.
      let modelVar = 0;
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) modelVar += w[a] * w[b] * Sigma[a][b];

      // Empirical scatter of the experts' estimates beyond what their stated noise explains.
      let covariance: Cov2 = { xx: modelVar, xy: 0, yy: modelVar };
      let maxTgtDiscrepancy = 0;
      for (const m of cluster) {
        maxTgtDiscrepancy = Math.max(maxTgtDiscrepancy, Math.hypot(m.targetPoint.x - meanTgt.x, m.targetPoint.y - meanTgt.y));
      }
      if (k >= 2) {
        let ux = 0;
        let uy = 0;
        for (const m of cluster) {
          ux += m.targetPoint.x / k;
          uy += m.targetPoint.y / k;
        }
        let sxx = 0;
        let sxy = 0;
        let syy = 0;
        for (const m of cluster) {
          const dx = m.targetPoint.x - ux;
          const dy = m.targetPoint.y - uy;
          sxx += dx * dx;
          sxy += dx * dy;
          syy += dy * dy;
        }
        const scale = 1 / (k - 1);
        const meanStated = sigmas.reduce((s, v) => s + v * v, 0) / k;
        const excess = positiveSemiDefinitePart({
          xx: sxx * scale - meanStated,
          xy: sxy * scale,
          yy: syy * scale - meanStated,
        });
        covariance = {
          xx: covariance.xx + excess.xx / nEff,
          xy: covariance.xy + excess.xy / nEff,
          yy: covariance.yy + excess.yy / nEff,
        };
      }

      let geomScore = 0.8;
      if (geometryProvider) geomScore = geometryProvider.computeGeometricPriorScore(meanSrc, meanTgt);

      let fusedScore =
        weights.loftr * loftrConf +
        weights.rift * riftConf +
        weights.lightglue * lightglueConf +
        weights.geometryPrior * geomScore;

      // Consensus bonus only for INDEPENDENT agreement (n_eff in [1, 2] maps to a 0..15% bonus).
      const bonus = 1 + 0.15 * Math.min(1, Math.max(0, nEff - 1));
      fusedScore = Math.min(1.0, fusedScore * bonus);

      if (maxTgtDiscrepancy > disagreementThresholdPx) {
        fusedScore *= Math.exp(-maxTgtDiscrepancy / (2 * disagreementThresholdPx));
      }

      fusedMatches.push({
        id: `fused_${fusedMatches.length}`,
        sourcePoint: meanSrc,
        targetPoint: meanTgt,
        confidence: Math.min(1.0, Math.max(0.01, fusedScore)),
        method: 'Fused',
        methodProvenance: {
          loftrConfidence: loftrConf,
          riftConfidence: riftConf,
          lightglueConfidence: lightglueConf,
          geometryConsistency: geomScore,
          agreementCount: k,
          effectiveIndependentExperts: nEff,
        },
        covariance,
        uncertaintyPx: meanSigmaPx(covariance),
        isInlier: maxTgtDiscrepancy <= disagreementThresholdPx,
      });
    }

    return {
      matches: fusedMatches,
      sourceImageId: matchSets[0].sourceImageId,
      targetImageId: matchSets[0].targetImageId,
      coordinateConvention: 'x=column, y=row',
      metadata: {
        totalInputMatches: allMatches.length,
        fusedClusterCount: fusedMatches.length,
        multiExpertAgreedCount: fusedMatches.filter((m) => (m.methodProvenance?.agreementCount || 0) >= 2).length,
        independentlyConfirmedCount: fusedMatches.filter((m) => (m.methodProvenance?.effectiveIndependentExperts || 1) >= 1.5).length,
        unavailableExperts: [...unavailable],
      },
    };
  }
}
