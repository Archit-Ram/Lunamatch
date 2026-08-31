/**
 * LunaMatch - Part 11: Simulated RIFT Profile Matcher
 * 
 * MOCK/SIMULATED — does not run real model inference. Generates synthetic correspondences
 * from the known ground-truth transform plus noise, for pipeline development purposes only.
 * Replace with real inference before any accuracy claims are made.
 * 
 * Simulates Radiation-invariant Feature Transform (RIFT) characteristics using structural
 * boundary features and illumination-resilient matching.
 */

import { ImageData, Match, MatchSet, Point2D } from '../types';
import { Matcher } from '../core/interfaces';
import { DeterministicRNG, applyHomographyToPoint } from '../generator/synthetic';

/**
 * MOCK/SIMULATED — does not run real model inference. Generates synthetic correspondences
 * from the known ground-truth transform plus noise, for pipeline development purposes only.
 * Replace with real inference before any accuracy claims are made.
 */
export class SimulatedRIFTProfileMatcher implements Matcher {
  readonly name = 'SimulatedRIFT';
  private confidenceThreshold: number;

  constructor(confidenceThreshold: number = 0.45) {
    this.confidenceThreshold = confidenceThreshold;
  }

  match(source: ImageData, target: ImageData, runtimeOptions?: Record<string, any>): MatchSet {
    const rng = new DeterministicRNG(9001);
    const groundTruthH: number[][] = runtimeOptions?.groundTruthTransform || [
      [1.0, 0.0, 10.0],
      [0.0, 1.0, -8.0],
      [0.0, 0.0, 1.0],
    ];

    const matches: Match[] = [];
    const count = runtimeOptions?.numMatches || 60;

    for (let i = 0; i < count; i++) {
      // RIFT profile simulates keypoints along strong structural boundaries (crater rims, ridges)
      const sx = rng.range(30, source.width - 30);
      const sy = rng.range(30, source.height - 30);
      const sPt: Point2D = { x: sx, y: sy };

      const trueTarget = applyHomographyToPoint(groundTruthH, sPt);
      // RIFT profile simulates high robustness to illumination shifts with moderate pixel jitter (~0.8px)
      const noise = (rng.next() - 0.5) * 1.1;
      const isOutlier = rng.next() < 0.09;

      const tx = isOutlier ? rng.range(20, target.width - 20) : trueTarget.x + noise;
      const ty = isOutlier ? rng.range(20, target.height - 20) : trueTarget.y + noise;
      const conf = isOutlier ? rng.range(0.25, 0.48) : rng.range(0.78, 0.96);

      if (conf >= this.confidenceThreshold) {
        matches.push({
          id: `sim_rift_${i}`,
          sourcePoint: sPt,
          targetPoint: { x: tx, y: ty },
          confidence: conf,
          method: 'SimulatedRIFT',
          uncertaintyPx: isOutlier ? 5.0 : 0.65 + Math.abs(noise),
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

