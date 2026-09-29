/**
 * LunaMatch - Part 22: Hierarchical Chaining Orchestrator
 *
 * Bridges large GSD-scale-gap sensor pairs (e.g. OHRC 0.25m <-> IIRS 80m,
 * a ~300x gap) by matching only across adjacent, comparable-scale hops on
 * the canonical sensor chain (OHRC -> TMC-2 -> IIRS) and composing the
 * per-hop transforms into a single equivalent transform, instead of
 * attempting one direct cross-sensor match.
 *
 * Composition strategy: rather than inventing a new "chained" TransformModel
 * type that every downstream consumer (warp, UI, evaluation) would need to
 * special-case, each hop's fitted transform is applied in sequence to a
 * dense grid of points spanning the source image. The resulting
 * source -> final-target point correspondences are then re-fit with the
 * existing AdaptiveTransformEstimator (rigid/affine/homography/TPS + BIC
 * selection), producing a single, ordinary TransformModel that composes the
 * whole chain and is a drop-in replacement anywhere a direct-match
 * TransformModel is used (ImageWarper, evaluation, UI).
 */

import {
  GroundTruthData,
  HierarchicalRegistrationResult,
  ImageData,
  Match,
  MatchSet,
  Point2D,
  RegistrationResult,
  SensorType,
} from '../types';
import { LunaMatchPipeline } from './lunamatch';
import { PipelineConfig } from '../core/config';
import { AdaptiveTransformEstimator } from '../registration/adaptive_transform';
import { applyTransformModel } from '../registration/transform_utils';
import { ImageWarper } from '../registration/warp';
import { LunaMatchEvaluator } from '../evaluation/benchmark';
import {
  DEFAULT_CHAIN_SCALE_THRESHOLD,
  describeChainPath,
  getChainPath,
  getScaleRatio,
  requiresHierarchicalChaining,
} from '../geometry/sensor_chain';

export interface HierarchicalRegisterOptions {
  matcherChoice?: 'fusion' | 'loftr' | 'rift' | 'lightglue' | 'mock';
  mockMode?: 'perfect' | 'low_noise' | 'high_noise' | 'outlier_heavy' | 'clustered' | 'mixed';
  /** Grid resolution (per axis) used to sample chained correspondences for the composed refit. */
  compositionGridSize?: number;
  /** Scale ratio above which chaining is preferred over a direct match. */
  scaleThreshold?: number;
}

/**
 * Pure composition primitive: propagates a uniform grid of points across
 * `sourceWidth`x`sourceHeight` through a sequence of TransformModels in
 * order, keeping only points that stay within `targetWidth`x`targetHeight`
 * at the final hop. Exported standalone (independent of ImageData/hop
 * bookkeeping) so the composition math can be unit-tested directly against
 * known transform chains.
 */
export function composeTransformChainToMatchSet(
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
  transforms: import('../types').TransformModel[],
  gridN: number = 14,
  sourceImageId: string = 'chain_source',
  targetImageId: string = 'chain_target'
): MatchSet {
  const matches: Match[] = [];
  const marginX = sourceWidth / (gridN + 1);
  const marginY = sourceHeight / (gridN + 1);

  let idCounter = 0;
  for (let gy = 1; gy <= gridN; gy++) {
    for (let gx = 1; gx <= gridN; gx++) {
      const startPt: Point2D = { x: gx * marginX, y: gy * marginY };
      let pt = startPt;

      for (const t of transforms) {
        pt = applyTransformModel(t, pt);
      }

      if (pt.x >= 0 && pt.x < targetWidth && pt.y >= 0 && pt.y < targetHeight) {
        matches.push({
          id: `chained_${idCounter++}`,
          sourcePoint: startPt,
          targetPoint: pt,
          confidence: 1.0,
          method: 'Fused',
          isInlier: true,
        });
      }
    }
  }

  return {
    matches,
    sourceImageId,
    targetImageId,
    coordinateConvention: 'x=column, y=row',
    metadata: { composedFromChain: true },
  };
}

export class HierarchicalChainOrchestrator {
  private pipeline: LunaMatchPipeline;
  private transformEstimator: AdaptiveTransformEstimator;

  constructor(config: Partial<PipelineConfig> = {}) {
    this.pipeline = new LunaMatchPipeline(config);
    this.transformEstimator = new AdaptiveTransformEstimator();
  }

  /**
   * Registers `sourceSensor` against `targetSensor`, bridging through
   * intermediate sensors when the GSD gap between them is large.
   *
   * @param images Map of sensor -> available ImageData. Must contain every
   *   sensor on the chain path for bridging to occur; if an intermediate
   *   sensor's image is missing, falls back to a direct match with a warning.
   * @param groundTruth Optional ground truth for the *direct* source->target
   *   transform (used only for metrics when chaining is not needed, or to
   *   score the final composed result).
   */
  async registerHierarchical(
    images: Partial<Record<SensorType, ImageData>>,
    sourceSensor: SensorType,
    targetSensor: SensorType,
    groundTruth?: GroundTruthData,
    options: HierarchicalRegisterOptions = {}
  ): Promise<HierarchicalRegistrationResult> {
    const scaleThreshold = options.scaleThreshold ?? DEFAULT_CHAIN_SCALE_THRESHOLD;
    const chainPath = getChainPath(sourceSensor, targetSensor);
    const overrideOptions = { matcherChoice: options.matcherChoice, mockMode: options.mockMode };

    const sourceImg = images[sourceSensor];
    const targetImg = images[targetSensor];
    if (!sourceImg || !targetImg) {
      throw new Error(
        `registerHierarchical requires images for both endpoints (${sourceSensor}, ${targetSensor})`
      );
    }

    const needsChain =
      chainPath.length > 2 && requiresHierarchicalChaining(sourceSensor, targetSensor, scaleThreshold);

    if (!needsChain) {
      const direct = await this.pipeline.registerImages(sourceImg, targetImg, groundTruth, overrideOptions);
      return { ...direct, chainPath: [sourceSensor, targetSensor], hopResults: [direct], usedDirectMatch: true };
    }

    // Verify every hop's bridge imagery is available.
    const missing = chainPath.filter((s) => !images[s]);
    if (missing.length > 0) {
      const direct = await this.pipeline.registerImages(sourceImg, targetImg, groundTruth, overrideOptions);
      direct.diagnostics.warnings.push(
        `Hierarchical chain ${describeChainPath(chainPath)} requires bridge imagery for ` +
          `[${missing.join(', ')}], which was not supplied. Fell back to a direct ` +
          `${sourceSensor}\u2192${targetSensor} match across a ${getScaleRatio(sourceSensor, targetSensor).toFixed(0)}x GSD gap; ` +
          `expect degraded accuracy.`
      );
      return { ...direct, chainPath: [sourceSensor, targetSensor], hopResults: [direct], usedDirectMatch: true };
    }

    // Register each adjacent, comparable-scale hop independently.
    const hopResults: RegistrationResult[] = [];
    for (let i = 0; i < chainPath.length - 1; i++) {
      const a = images[chainPath[i]]!;
      const b = images[chainPath[i + 1]]!;
      const hopResult = await this.pipeline.registerImages(a, b, undefined, overrideOptions);
      hopResults.push(hopResult);

      if (hopResult.status === 'failure') {
        return this.buildChainFailure(
          chainPath,
          hopResults,
          sourceSensor,
          targetSensor,
          `Hop ${chainPath[i]}\u2192${chainPath[i + 1]} failed (${hopResult.failureReason}); chain cannot be composed.`
        );
      }
    }

    // Compose: propagate a dense grid of source-image points through every
    // hop's fitted transform in sequence, then re-fit a single equivalent
    // transform from the resulting source -> final-target correspondences.
    const gridN = options.compositionGridSize ?? 14;
    const composedMatchSet = this.composeChainedMatchSet(sourceImg, targetImg, hopResults, gridN);

    if (composedMatchSet.matches.length < 4) {
      return this.buildChainFailure(
        chainPath,
        hopResults,
        sourceSensor,
        targetSensor,
        'Composed chain produced too few valid correspondences (grid points fell outside target bounds at every hop).'
      );
    }

    const composedTransform = this.transformEstimator.estimateTransform(
      composedMatchSet,
      [sourceImg.width, sourceImg.height],
      [targetImg.width, targetImg.height]
    );

    const registeredImage = ImageWarper.warpImage(
      sourceImg,
      composedTransform,
      targetImg.width,
      targetImg.height
    );

    const totalRuntimeMs = hopResults.reduce((sum, h) => sum + h.metrics.runtimeMs, 0);
    // First-order propagated error: hops are treated as independent, so
    // variances add in quadrature. A fuller decorrelated-covariance model
    // (per-matcher failure-mode correlation) is tracked separately.
    const propagatedRmse = Math.sqrt(hopResults.reduce((s, h) => s + h.metrics.rmsePx ** 2, 0));

    const srcPoints = composedMatchSet.matches.map((m) => m.sourcePoint);
    const tgtPoints = composedMatchSet.matches.map((m) => m.targetPoint);

    const metrics = groundTruth
      ? LunaMatchEvaluator.computeMetrics(
          srcPoints,
          tgtPoints,
          groundTruth.groundTruthTransform,
          sourceImg.width,
          sourceImg.height,
          totalRuntimeMs
        )
      : {
          rmsePx: Number(composedTransform.residual.rmse.toFixed(3)),
          medianErrorPx: Number(composedTransform.residual.median.toFixed(3)),
          percentile90ErrorPx: Number((composedTransform.residual.rmse * 1.64).toFixed(3)),
          percentile95ErrorPx: Number((composedTransform.residual.rmse * 1.96).toFixed(3)),
          inlierCount: composedMatchSet.matches.length,
          inlierRatio: 1.0,
          uniformityScore: 1.0,
          runtimeMs: Number(totalRuntimeMs.toFixed(1)),
        };

    const warnings: string[] = [];
    for (const h of hopResults) warnings.push(...h.diagnostics.warnings);
    warnings.push(
      `Composed via chain ${describeChainPath(chainPath)}: propagated 1st-order RMSE ${propagatedRmse.toFixed(2)}px across ${hopResults.length} hop(s).`
    );

    return {
      status: 'success',
      sourceSensor,
      referenceSensor: targetSensor,
      transform: composedTransform,
      rawMatches: composedMatchSet,
      fusedMatches: composedMatchSet,
      filteredMatches: composedMatchSet,
      refinedMatches: composedMatchSet,
      uniformMatches: composedMatchSet,
      registeredImage,
      metrics,
      diagnostics: {
        matcherAgreement: hopResults.reduce((s, h) => s + h.diagnostics.matcherAgreement, 0) / hopResults.length,
        geometryConsistency:
          hopResults.reduce((s, h) => s + h.diagnostics.geometryConsistency, 0) / hopResults.length,
        uncertaintyMeanPx: Number(propagatedRmse.toFixed(3)),
        timingBreakdownMs: hopResults[0].diagnostics.timingBreakdownMs,
        warnings,
      },
      chainPath,
      hopResults,
      usedDirectMatch: false,
    };
  }

  /**
   * Samples a uniform grid of points across the source image and propagates
   * each through every hop's fitted transform in order, keeping only points
   * that remain inside the final target image's bounds. Confidence is
   * discounted by each hop's geometric consistency score, so a shaky
   * intermediate hop lowers trust in the composed correspondences that
   * pass through it.
   */
  private composeChainedMatchSet(
    sourceImg: ImageData,
    targetImg: ImageData,
    hopResults: RegistrationResult[],
    gridN: number
  ): MatchSet {
    const transforms = hopResults.map((h) => h.transform);
    const matchSet = composeTransformChainToMatchSet(
      sourceImg.width,
      sourceImg.height,
      targetImg.width,
      targetImg.height,
      transforms,
      gridN,
      sourceImg.id,
      targetImg.id
    );

    const combinedConfidence = hopResults.reduce(
      (c, h) => c * (h.diagnostics.geometryConsistency || 0.9),
      1.0
    );
    for (const m of matchSet.matches) m.confidence = combinedConfidence;

    return matchSet;
  }

  private buildChainFailure(
    chainPath: SensorType[],
    hopResults: RegistrationResult[],
    sourceSensor: SensorType,
    targetSensor: SensorType,
    reason: string
  ): HierarchicalRegistrationResult {
    const lastHop = hopResults[hopResults.length - 1];
    return {
      ...lastHop,
      status: 'failure',
      failureReason: 'HIERARCHICAL_CHAIN_BROKEN',
      sourceSensor,
      referenceSensor: targetSensor,
      diagnostics: {
        ...lastHop.diagnostics,
        warnings: [...lastHop.diagnostics.warnings, reason],
      },
      chainPath,
      hopResults,
      usedDirectMatch: false,
    };
  }
}
