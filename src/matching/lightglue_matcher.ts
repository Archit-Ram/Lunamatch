/**
 * LunaMatch - Part 12: Simulated SuperPoint + LightGlue Profile Matcher
 * 
 * MOCK/SIMULATED — does not run real model inference. Generates synthetic correspondences
 * from the known ground-truth transform plus noise, for pipeline development purposes only.
 * Replace with real inference before any accuracy claims are made.
 * 
 * Simulates sparse keypoint detector (SuperPoint) coupled with adaptive graph neural
 * network correspondence characteristics (LightGlue).
 */

import { ImageData, Match, MatchSet, Point2D } from '../types';
import { Matcher } from '../core/interfaces';
import { DeterministicRNG, applyHomographyToPoint } from '../generator/synthetic';

/**
 * MOCK/SIMULATED — does not run real model inference. Generates synthetic correspondences
 * from the known ground-truth transform plus noise, for pipeline development purposes only.
 * Replace with real inference before any accuracy claims are made.
 */
export class SimulatedLightGlueProfileMatcher implements Matcher {
  readonly name = 'SimulatedLightGlue';
  private confidenceThreshold: number;

  constructor(confidenceThreshold: number = 0.5) {
    this.confidenceThreshold = confidenceThreshold;
  }

  match(source: ImageData, target: ImageData, runtimeOptions?: Record<string, any>): MatchSet {
    const rng = new DeterministicRNG(4242);
    const groundTruthH: number[][] = runtimeOptions?.groundTruthTransform || [
      [1.0, 0.0, 10.0],
      [0.0, 1.0, -8.0],
      [0.0, 0.0, 1.0],
    ];

    const matches: Match[] = [];
    const count = runtimeOptions?.numMatches || 48;

    for (let i = 0; i < count; i++) {
      // SuperPoint keypoints profile placed at corner-like lunar terrain peaks and crater vertices
      const sx = rng.range(35, source.width - 35);
      const sy = rng.range(35, source.height - 35);
      const sPt: Point2D = { x: sx, y: sy };

      const trueTarget = applyHomographyToPoint(groundTruthH, sPt);
      const noise = (rng.next() - 0.5) * 0.6;
      const isOutlier = rng.next() < 0.05;

      const tx = isOutlier ? rng.range(20, target.width - 20) : trueTarget.x + noise;
      const ty = isOutlier ? rng.range(20, target.height - 20) : trueTarget.y + noise;
      const conf = isOutlier ? rng.range(0.2, 0.45) : rng.range(0.85, 0.99);

      if (conf >= this.confidenceThreshold) {
        matches.push({
          id: `sim_lightglue_${i}`,
          sourcePoint: sPt,
          targetPoint: { x: tx, y: ty },
          confidence: conf,
          method: 'SimulatedLightGlue',
          uncertaintyPx: isOutlier ? 4.0 : 0.4 + Math.abs(noise),
          isInlier: !isOutlier,
        });
      }
    }

    return {
      matches,
      sourceImageId: source.id,
      targetImageId: target.id,
      coordinateConvention: 'x=column, y=row',
    };
  }
}

