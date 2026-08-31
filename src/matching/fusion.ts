/**
 * LunaMatch - Part 13: Matcher Fusion Engine
 * 
 * Combines candidate correspondences from multiple independent matching experts
 * (LoFTR, RIFT, LightGlue) using spatial association, multi-model consensus,
 * and physics-informed confidence fusion:
 * C = w_L * C_L + w_R * C_R + w_G * C_G + w_P * C_P
 */

import { Match, MatchSet, Point2D } from '../types';
import { GeometryProvider } from '../core/interfaces';

export interface FusionWeights {
  loftr: number;
  rift: number;
  lightglue: number;
  geometryPrior: number;
}

export interface FusionOptions {
  associationRadiusPx?: number; // Distance in source space to cluster matches
  disagreementThresholdPx?: number; // Target discrepancy triggering outlier penalty
  weights?: FusionWeights;
}

export class MatchFusionEngine {
  private options: Required<FusionOptions>;

  constructor(options: FusionOptions = {}) {
    this.options = {
      associationRadiusPx: 4.5,
      disagreementThresholdPx: 5.0,
      weights: {
        loftr: 0.35,
        rift: 0.35,
        lightglue: 0.20,
        geometryPrior: 0.10,
        ...options.weights,
      },
      ...options,
    };
  }

  fuse(
    matchSets: MatchSet[],
    geometryProvider?: GeometryProvider
  ): MatchSet {
    if (matchSets.length === 0) {
      return {
        matches: [],
        sourceImageId: 'unknown',
        targetImageId: 'unknown',
        coordinateConvention: 'x=column, y=row',
      };
    }

    const allMatches: Match[] = [];
    for (const ms of matchSets) {
      allMatches.push(...ms.matches);
    }

    if (allMatches.length === 0) {
      return {
        matches: [],
        sourceImageId: matchSets[0].sourceImageId,
        targetImageId: matchSets[0].targetImageId,
        coordinateConvention: 'x=column, y=row',
      };
    }

    const { associationRadiusPx, disagreementThresholdPx, weights } = this.options;
    const visited = new Uint8Array(allMatches.length);
    const fusedMatches: Match[] = [];

    for (let i = 0; i < allMatches.length; i++) {
      if (visited[i]) continue;
      visited[i] = 1;

      const cluster: Match[] = [allMatches[i]];
      const baseSrc = allMatches[i].sourcePoint;

      // Find all neighbor matches within association radius in source space
      for (let j = i + 1; j < allMatches.length; j++) {
        if (visited[j]) continue;
        const candSrc = allMatches[j].sourcePoint;
        const d = Math.hypot(baseSrc.x - candSrc.x, baseSrc.y - candSrc.y);

        if (d <= associationRadiusPx) {
          visited[j] = 1;
          cluster.push(allMatches[j]);
        }
      }

      // Aggregate cluster across methods
      let sumSrcX = 0;
      let sumSrcY = 0;
      let sumTgtX = 0;
      let sumTgtY = 0;
      let totalWeight = 0;

      let loftrConf = 0;
      let riftConf = 0;
      let lightglueConf = 0;
      let agreeingMethodsCount = 0;

      for (const m of cluster) {
        const w = Math.max(0.1, m.confidence);
        sumSrcX += m.sourcePoint.x * w;
        sumSrcY += m.sourcePoint.y * w;
        sumTgtX += m.targetPoint.x * w;
        sumTgtY += m.targetPoint.y * w;
        totalWeight += w;

        if (m.method === 'LoFTR') {
          loftrConf = Math.max(loftrConf, m.confidence);
          agreeingMethodsCount++;
        } else if (m.method === 'RIFT') {
          riftConf = Math.max(riftConf, m.confidence);
          agreeingMethodsCount++;
        } else if (m.method === 'LightGlue') {
          lightglueConf = Math.max(lightglueConf, m.confidence);
          agreeingMethodsCount++;
        } else {
          agreeingMethodsCount++;
        }
      }

      const meanSrc: Point2D = {
        x: sumSrcX / totalWeight,
        y: sumSrcY / totalWeight,
      };
      const meanTgt: Point2D = {
        x: sumTgtX / totalWeight,
        y: sumTgtY / totalWeight,
      };

      // Check for target disagreement within the cluster
      let maxTgtDiscrepancy = 0;
      for (const m of cluster) {
        const dist = Math.hypot(m.targetPoint.x - meanTgt.x, m.targetPoint.y - meanTgt.y);
        if (dist > maxTgtDiscrepancy) maxTgtDiscrepancy = dist;
      }

      // Compute geometric prior score
      let geomScore = 0.8;
      if (geometryProvider) {
        geomScore = geometryProvider.computeGeometricPriorScore(meanSrc, meanTgt);
      }

      // Combined confidence score
      let fusedScore =
        weights.loftr * loftrConf +
        weights.rift * riftConf +
        weights.lightglue * lightglueConf +
        weights.geometryPrior * geomScore;

      // Multi-expert consensus bonus (reward when 2+ models agree)
      if (cluster.length >= 2) {
        fusedScore = Math.min(1.0, fusedScore * 1.15);
      }

      // Disagreement penalty
      if (maxTgtDiscrepancy > disagreementThresholdPx) {
        fusedScore *= Math.exp(-maxTgtDiscrepancy / (2 * disagreementThresholdPx));
      }

      const uncertainty = Math.max(
        0.15,
        maxTgtDiscrepancy * 0.7 + (1.0 - fusedScore) * 1.5 + (3.0 / Math.max(1, cluster.length))
      );

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
          agreementCount: cluster.length,
        },
        uncertaintyPx: uncertainty,
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
      },
    };
  }
}
