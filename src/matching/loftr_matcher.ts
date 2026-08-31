/**
 * LunaMatch - Part 10: LoFTR Matcher Adapter
 * 
 * Dense learned transformer matcher adapter implementing the standard Matcher protocol.
 * Simulates coarse-to-fine self- and cross-attention feature correlation on lunar surface.
 */

import { ImageData, Match, MatchSet, Point2D } from '../types';
import { Matcher } from '../core/interfaces';
import { DeterministicRNG, applyHomographyToPoint } from '../generator/synthetic';

export class LoFTRMatcher implements Matcher {
  readonly name = 'LoFTR';
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
      // LoFTR has high dense coverage, fine sub-pixel accuracy (~0.4px noise)
      const noise = (rng.next() - 0.5) * 0.7;
      const isOutlier = rng.next() < 0.06;

      const tx = isOutlier ? rng.range(20, target.width - 20) : trueTarget.x + noise;
      const ty = isOutlier ? rng.range(20, target.height - 20) : trueTarget.y + noise;
      const conf = isOutlier ? rng.range(0.3, 0.55) : rng.range(0.82, 0.99);

      if (conf >= this.confidenceThreshold) {
        matches.push({
          id: `loftr_${i}`,
          sourcePoint: sPt,
          targetPoint: { x: tx, y: ty },
          confidence: conf,
          method: 'LoFTR',
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
