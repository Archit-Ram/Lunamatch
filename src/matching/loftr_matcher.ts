/**
 * LunaMatch - Part 10: Simulated LoFTR Profile Matcher
 * 
 * MOCK/SIMULATED — does not run real model inference. Generates synthetic correspondences
 * from the known ground-truth transform plus noise, for pipeline development purposes only.
 * Replace with real inference before any accuracy claims are made.
 * 
 * Simulates coarse-to-fine self- and cross-attention feature correlation characteristics.
 */

import { ImageData, Match, MatchSet, Point2D } from '../types';
import { Matcher } from '../core/interfaces';
import { DeterministicRNG, applyHomographyToPoint } from '../generator/synthetic';

/**
 * MOCK/SIMULATED — does not run real model inference. Generates synthetic correspondences
 * from the known ground-truth transform plus noise, for pipeline development purposes only.
 * Replace with real inference before any accuracy claims are made.
 */
export class SimulatedLoFTRProfileMatcher implements Matcher {
  readonly name = 'SimulatedLoFTR';
  private confidenceThreshold: number;

  constructor(confidenceThreshold: number = 0.5) {
    this.confidenceThreshold = confidenceThreshold;
  }

  match(source: ImageData, target: ImageData, runtimeOptions?: Record<string, any>): MatchSet {
    const rng = new DeterministicRNG(1337);
    const groundTruthH: number[][] = runtimeOptions?.groundTruthTransform || [
      [1.0, 0.0, 10.0],
      [0.0, 1.0, -8.0],
      [0.0, 0.0, 1.0],
    ];

    const matches: Match[] = [];
    const count = runtimeOptions?.numMatches || 72;

    for (let i = 0; i < count; i++) {
      const sx = rng.range(25, source.width - 25);
      const sy = rng.range(25, source.height - 25);
      const sPt: Point2D = { x: sx, y: sy };

      const trueTarget = applyHomographyToPoint(groundTruthH, sPt);
      // LoFTR profile simulates high dense coverage, fine sub-pixel accuracy (~0.4px noise)
      const noise = (rng.next() - 0.5) * 0.7;
      const isOutlier = rng.next() < 0.06;

      const tx = isOutlier ? rng.range(20, target.width - 20) : trueTarget.x + noise;
      const ty = isOutlier ? rng.range(20, target.height - 20) : trueTarget.y + noise;
      const conf = isOutlier ? rng.range(0.3, 0.55) : rng.range(0.82, 0.99);

      if (conf >= this.confidenceThreshold) {
        matches.push({
          id: `sim_loftr_${i}`,
          sourcePoint: sPt,
          targetPoint: { x: tx, y: ty },
          confidence: conf,
          method: 'SimulatedLoFTR',
          uncertaintyPx: isOutlier ? 4.5 : 0.35 + Math.abs(noise),
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

