/**
 * LunaMatch - Part 09: Mock Matcher
 * 
 * Implements the standard Matcher protocol using synthetic ground truth transformations.
 * Supports configurable modes: 'perfect', 'low_noise', 'high_noise', 'outlier_heavy', 'clustered', 'mixed'.
 */

import { ImageData, Match, MatchSet, Point2D } from '../types';
import { Matcher } from '../core/interfaces';
import { DeterministicRNG, applyHomographyToPoint } from '../generator/synthetic';

export type MockMatcherMode = 'perfect' | 'low_noise' | 'high_noise' | 'outlier_heavy' | 'clustered' | 'mixed';

export interface MockMatcherOptions {
  mode?: MockMatcherMode;
  numMatches?: number;
  noiseSigmaPx?: number;
  outlierRatio?: number;
  clusterConcentration?: number;
  seed?: number;
}

export class MockMatcher implements Matcher {
  readonly name = 'MockMatcher';
  private options: MockMatcherOptions;

  constructor(options: MockMatcherOptions = {}) {
    this.options = {
      mode: 'low_noise',
      numMatches: 64,
      noiseSigmaPx: 0.75,
      outlierRatio: 0.10,
      clusterConcentration: 0.3,
      seed: 42,
      ...options,
    };
  }

  match(source: ImageData, target: ImageData, runtimeOptions?: Record<string, any>): MatchSet {
    const mode = runtimeOptions?.mode || this.options.mode!;
    const numMatches = runtimeOptions?.numMatches || this.options.numMatches!;
    const rng = new DeterministicRNG(this.options.seed);

    // Retrieve ground-truth transform or estimate an initial affine estimate
    const groundTruthH: number[][] = runtimeOptions?.groundTruthTransform || [
      [1.0, 0.0, 10.0],
      [0.0, 1.0, -8.0],
      [0.0, 0.0, 1.0],
    ];

    let noiseSigma = 0.0;
    let outlierFraction = 0.0;

    switch (mode) {
      case 'perfect':
        noiseSigma = 0.0;
        outlierFraction = 0.0;
        break;
      case 'low_noise':
        noiseSigma = 0.6;
        outlierFraction = 0.08;
        break;
      case 'high_noise':
        noiseSigma = 2.4;
        outlierFraction = 0.25;
        break;
      case 'outlier_heavy':
        noiseSigma = 1.0;
        outlierFraction = 0.45;
        break;
      case 'clustered':
        noiseSigma = 0.8;
        outlierFraction = 0.10;
        break;
      case 'mixed':
      default:
        noiseSigma = 1.2;
        outlierFraction = 0.20;
        break;
    }

    const matches: Match[] = [];
    const margin = 20;

    for (let i = 0; i < numMatches; i++) {
      let sx: number;
      let sy: number;

      if (mode === 'clustered' && rng.next() < 0.7) {
        // Cluster 70% of points around center crater
        const center = { x: source.width * 0.48, y: source.height * 0.46 };
        sx = center.x + (rng.next() - 0.5) * (source.width * 0.25);
        sy = center.y + (rng.next() - 0.5) * (source.height * 0.25);
      } else {
        // Uniform distribution across image
        sx = rng.range(margin, source.width - margin);
        sy = rng.range(margin, source.height - margin);
      }

      const sPt: Point2D = { x: sx, y: sy };
      const trueTarget = applyHomographyToPoint(groundTruthH, sPt);

      let tx = trueTarget.x;
      let ty = trueTarget.y;
      let isOutlier = false;

      if (rng.next() < outlierFraction) {
        // Generate gross geometric outlier (random jump or mismatched feature)
        tx = rng.range(margin, target.width - margin);
        ty = rng.range(margin, target.height - margin);
        isOutlier = true;
      } else {
        // Add Gaussian measurement noise (Box-Muller transform)
        const u1 = Math.max(1e-6, rng.next());
        const u2 = rng.next();
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);

        tx += z0 * noiseSigma;
        ty += z1 * noiseSigma;
      }

      const confBase = isOutlier ? rng.range(0.2, 0.55) : rng.range(0.75, 0.98);
      const uncertainty = isOutlier ? rng.range(3.0, 8.0) : Math.max(0.15, noiseSigma * 1.1 + rng.range(0, 0.3));

      matches.push({
        id: `mock_match_${i}`,
        sourcePoint: sPt,
        targetPoint: { x: tx, y: ty },
        confidence: Math.min(1.0, Math.max(0.05, confBase)),
        method: 'Mock',
        uncertaintyPx: uncertainty,
        isInlier: !isOutlier,
      });
    }

    return {
      matches,
      sourceImageId: source.id,
      targetImageId: target.id,
      coordinateConvention: 'x=column, y=row',
      metadata: { mode, noiseSigma, outlierFraction },
    };
  }
}
