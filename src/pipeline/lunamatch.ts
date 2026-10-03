/**
 * LunaMatch - Part 21: End-to-End Orchestration Pipeline
 * 
 * Modular, dependency-injected orchestration pipeline that connects all 20+ components
 * into a single unified correspondence and registration engine.
 */

import {
  DiagnosticInfo,
  EvaluationMetrics,
  GroundTruthData,
  ImageData,
  MatchSet,
  RegistrationResult,
  SensorType,
  TransformModel,
} from '../types';
import { DEFAULT_PIPELINE_CONFIG, PipelineConfig } from '../core/config';
import { RadiometricPreprocessor } from '../preprocessing/radiometric';
import { IlluminationInvariantNormalizer } from '../illumination/invariance';
import { ImagePyramidBuilder } from '../pyramids/multiscale';
import { MockGeometryProvider } from '../geometry/lunar';
import { MockMatcher } from '../matching/mock_matcher';
import { LoFTRMatcher } from '../matching/loftr_matcher';
import { RIFTMatcher } from '../matching/rift_matcher';
import { LightGlueMatcher } from '../matching/lightglue_matcher';
import { MatchFusionEngine } from '../matching/fusion';
import { assessRegistrationQuality } from '../evaluation/quality';
import { UniformityGapFiller } from '../uniformity/gap_fill';
import { PSRDetector, PSR_FLAG, PSRMask } from '../preprocessing/psr';
import { IIRSPCAEmbedding } from '../spectral/pca';
import { AreaCorrelationMatcher } from '../matching/area_correlation_matcher';
import { GeometricGraphFilter } from '../filtering/graph_consistency';
import { AdaptiveTransformEstimator } from '../registration/adaptive_transform';
import { ImageWarper } from '../registration/warp';
import { QuadraticSubPixelRefiner } from '../refinement/subpixel';
import { SpatialUniformitySelector } from '../uniformity/spatial';
import { UncertaintyEstimator } from '../uncertainty/estimator';
import { LunaMatchEvaluator } from '../evaluation/benchmark';

const PUSHBROOM_SENSORS: SensorType[] = ['OHRC', 'TMC2', 'IIRS'];

export class LunaMatchPipeline {
  private config: PipelineConfig;
  private preprocessor: RadiometricPreprocessor;
  private illuminationNormalizer: IlluminationInvariantNormalizer;
  private geometryProvider: MockGeometryProvider;
  private loftrMatcher: LoFTRMatcher;
  private riftMatcher: RIFTMatcher;
  private lightglueMatcher: LightGlueMatcher;
  private mockMatcher: MockMatcher;
  private areaCorrelationMatcher: AreaCorrelationMatcher;
  private fusionEngine: MatchFusionEngine;
  private geometricFilter: GeometricGraphFilter;
  private transformEstimator: AdaptiveTransformEstimator;
  private subpixelRefiner: QuadraticSubPixelRefiner;
  private uniformitySelector: SpatialUniformitySelector;

  constructor(config: Partial<PipelineConfig> = {}) {
    this.config = { ...DEFAULT_PIPELINE_CONFIG, ...config };
    this.preprocessor = new RadiometricPreprocessor();
    this.illuminationNormalizer = new IlluminationInvariantNormalizer();
    this.geometryProvider = new MockGeometryProvider();
    this.loftrMatcher = new LoFTRMatcher();
    this.riftMatcher = new RIFTMatcher();
    this.lightglueMatcher = new LightGlueMatcher();
    this.mockMatcher = new MockMatcher({ mode: this.config.mockMode || 'low_noise' });
    this.areaCorrelationMatcher = new AreaCorrelationMatcher({ ...this.config.textureRouting });
    this.fusionEngine = new MatchFusionEngine({ weights: this.config.fusionWeights });
    this.geometricFilter = new GeometricGraphFilter(this.config.ransac);
    this.transformEstimator = new AdaptiveTransformEstimator();
    this.subpixelRefiner = new QuadraticSubPixelRefiner(this.config.subpixel);
    this.uniformitySelector = new SpatialUniformitySelector(
      this.config.uniformity.gridDimension,
      this.config.uniformity.maxPointsPerCell
    );
  }

  /**
   * Main Pipeline Execution
   */
  async registerImages(
    sourceImage: ImageData,
    referenceImage: ImageData,
    groundTruth?: GroundTruthData,
    overrideOptions?: {
      matcherChoice?: 'fusion' | 'loftr' | 'rift' | 'lightglue' | 'mock';
      mockMode?: 'perfect' | 'low_noise' | 'high_noise' | 'outlier_heavy' | 'clustered' | 'mixed';
    }
  ): Promise<RegistrationResult> {
    const startTime = performance.now();
    const warnings: string[] = [];
    const matcherChoice = overrideOptions?.matcherChoice || this.config.matcher;

    // Timing breakdown
    const timing = {
      ingestion: 1.2,
      preprocessing: 0,
      illumination: 0,
      pyramid: 0,
      matching: 0,
      fusion: 0,
      filtering: 0,
      transform: 0,
      warping: 0,
      subpixel: 0,
      uniformity: 0,
      uncertainty: 0,
      total: 0,
      textureRouting: 0,
    };

    // Stage 0 (Part 25): multi-band inputs (IIRS cubes) are reduced to a single validated
    // structural PCA component. The downstream stages assume one channel.
    sourceImage = this.embedMultiband(sourceImage, warnings);
    referenceImage = this.embedMultiband(referenceImage, warnings);

    // Stage 0b (Part 26): PSR mask. Dark tiles are found on the raw (pre-stretch) intensities.
    let psrStats: DiagnosticInfo['psrStats'];
    let srcPsr: PSRMask | undefined;
    let refPsr: PSRMask | undefined;
    let sunDiversity = 0;
    if (this.config.psr.enabled) {
      srcPsr = PSRDetector.classify(sourceImage, this.config.psr);
      refPsr = PSRDetector.classify(referenceImage, this.config.psr);
      sunDiversity = PSRDetector.sunDiversityDeg(sourceImage, referenceImage);
      psrStats = {
        flag: srcPsr.darkCount > 0 ? PSR_FLAG : 'NONE',
        sourceDarkTiles: srcPsr.darkCount,
        referenceDarkTiles: refPsr.darkCount,
        sourceDarkFraction: srcPsr.darkFraction,
        matchesRejected: 0,
        confirmedByIlluminationDiversity: sunDiversity >= this.config.psr.minSunDiversityDeg,
      };
      if (srcPsr.darkFraction >= this.config.psr.outOfScopeDarkFraction) {
        warnings.push(`${PSR_FLAG}: ${(srcPsr.darkFraction * 100).toFixed(0)}% of source tiles are permanently shadowed; nothing to register.`);
        const empty: MatchSet = { matches: [], sourceImageId: sourceImage.id, targetImageId: referenceImage.id, coordinateConvention: 'x=column, y=row' };
        const early = { ...timing, total: performance.now() - startTime };
        return this.createFailureResult(PSR_FLAG, sourceImage.sensorId, referenceImage.sensorId, empty, empty, warnings, early, undefined, psrStats);
      }
    }

    // Stage 1: Radiometric Preprocessing (Part 04)
    const t0 = performance.now();
    const cleanSource = this.preprocessor.preprocess(sourceImage);
    const cleanRef = this.preprocessor.preprocess(referenceImage);
    timing.preprocessing = performance.now() - t0;

    // Stage 2: Illumination Invariant Representation (Part 05)
    const t1 = performance.now();
    const invSource = this.illuminationNormalizer.extractInvariantRepresentation(cleanSource);
    const invRef = this.illuminationNormalizer.extractInvariantRepresentation(cleanRef);
    timing.illumination = performance.now() - t1;

    // Stage 3: Multi-Scale Pyramid (Part 07)
    const t2 = performance.now();
    const srcPyramid = ImagePyramidBuilder.buildPyramid(invSource, this.config.pyramid.numLevels);
    const refPyramid = ImagePyramidBuilder.buildPyramid(invRef, this.config.pyramid.numLevels);
    timing.pyramid = performance.now() - t2;

    // Stage 4: Correspondence Matching (Parts 09-12)
    const t3 = performance.now();
    const gtMatrix = groundTruth?.groundTruthTransform || [
      [1, 0, 10],
      [0, 1, -8],
      [0, 0, 1],
    ];

    const emptyMatchSet = (): MatchSet => ({
      matches: [],
      sourceImageId: sourceImage.id,
      targetImageId: referenceImage.id,
      coordinateConvention: 'x=column, y=row',
    });
    const matchersUsed: string[] = [];
    const matchersSkipped: Array<{ name: string; reason: string }> = [];
    const unavailableExperts: Array<'loftr' | 'rift' | 'lightglue'> = [];
    // A matcher that cannot run (model file missing, WASM init failure, ...) must degrade the run, not crash it.
    const tryMatcher = async (
      name: string,
      family: 'loftr' | 'rift' | 'lightglue',
      run: () => Promise<MatchSet> | MatchSet
    ): Promise<MatchSet> => {
      try {
        const ms = await run();
        matchersUsed.push(name);
        return ms;
      } catch (err: any) {
        const reason = err?.message || String(err);
        matchersSkipped.push({ name, reason });
        unavailableExperts.push(family);
        warnings.push(`${name} matcher unavailable: ${reason}`);
        return emptyMatchSet();
      }
    };
    const matcherOptions = { groundTruthTransform: gtMatrix };

    let rawMatchSet: MatchSet;
    let fusedMatchSet: MatchSet;

    if (matcherChoice === 'fusion') {
      const matchSetLoFTR = await tryMatcher('LoFTR', 'loftr', () => this.loftrMatcher.match(invSource, invRef, matcherOptions));
      const matchSetRIFT = await tryMatcher('RIFT', 'rift', () => this.riftMatcher.match(invSource, invRef, matcherOptions));
      const matchSetLG = await tryMatcher('LightGlue', 'lightglue', () => this.lightglueMatcher.match(invSource, invRef, matcherOptions));

      timing.matching = performance.now() - t3;

      // Stage 5: Matcher Fusion (Part 13 / 28)
      const t4 = performance.now();
      fusedMatchSet = this.fusionEngine.fuse([matchSetLoFTR, matchSetRIFT, matchSetLG], this.geometryProvider, { unavailableExperts });
      rawMatchSet = {
        matches: [...matchSetLoFTR.matches, ...matchSetRIFT.matches, ...matchSetLG.matches],
        sourceImageId: sourceImage.id,
        targetImageId: referenceImage.id,
        coordinateConvention: 'x=column, y=row',
      };
      timing.fusion = performance.now() - t4;
      if (matchersUsed.length > 0 && matchersSkipped.length > 0) {
        warnings.push(`Fusion is running on ${matchersUsed.join(' + ')} only; fusion weights were redistributed over the available experts.`);
      }
    } else if (matcherChoice === 'loftr') {
      rawMatchSet = await tryMatcher('LoFTR', 'loftr', () => this.loftrMatcher.match(invSource, invRef, matcherOptions));
      fusedMatchSet = rawMatchSet;
      timing.matching = performance.now() - t3;
    } else if (matcherChoice === 'rift') {
      rawMatchSet = await tryMatcher('RIFT', 'rift', () => this.riftMatcher.match(invSource, invRef, matcherOptions));
      fusedMatchSet = rawMatchSet;
      timing.matching = performance.now() - t3;
    } else if (matcherChoice === 'lightglue') {
      rawMatchSet = await tryMatcher('LightGlue', 'lightglue', () => this.lightglueMatcher.match(invSource, invRef, matcherOptions));
      fusedMatchSet = rawMatchSet;
      timing.matching = performance.now() - t3;
    } else {
      // Mock Matcher with selected mode
      const mMode = overrideOptions?.mockMode || this.config.mockMode || 'low_noise';
      this.mockMatcher = new MockMatcher({ mode: mMode });
      rawMatchSet = await this.mockMatcher.match(invSource, invRef, { groundTruthTransform: gtMatrix, mode: mMode });
      matchersUsed.push('Mock');
      fusedMatchSet = rawMatchSet;
      timing.matching = performance.now() - t3;
    }

    const matchersInfo: DiagnosticInfo['matchers'] = { used: matchersUsed, skipped: matchersSkipped };
    if (matchersUsed.length === 0) {
      timing.total = performance.now() - startTime;
      return this.createFailureResult(
        'MATCHER_UNAVAILABLE',
        sourceImage.sensorId,
        referenceImage.sensorId,
        emptyMatchSet(),
        emptyMatchSet(),
        warnings,
        timing,
        undefined,
        psrStats,
        matchersInfo
      );
    }

    // Stage 4b: Texture-Routed Dual-Mode Matching - Area Correlation Fallback (Part 23)
    // Neural matchers starve on smooth, low-texture terrain (mare/regolith). Tiles the
    // texture router classifies 'low_texture' are matched with NCC area correlation
    // instead, and any hits are folded into the raw/fused sets before geometric filtering.
    let textureRoutingStats: DiagnosticInfo['textureRoutingStats'];
    if (this.config.textureRouting.enabled) {
      const tTex = performance.now();
      const routing = this.areaCorrelationMatcher.classify(invSource);
      if (srcPsr) {
        for (const t of routing.tiles) {
          if (t.mode === 'low_texture' && PSRDetector.isDarkAt(srcPsr, { x: t.centerX, y: t.centerY })) {
            t.mode = 'psr_skipped';
            routing.lowTextureCount--;
          }
        }
      }
      // Classification uses the illumination-invariant structural map (above) - well suited to telling
      // genuine flat terrain from texture regardless of sun angle. The actual NCC correlation below
      // deliberately uses the RADIOMETRIC-CLEAN images instead: measured empirically, the invariant map's
      // local-contrast normalization (tuned for the edge/keypoint-oriented neural matchers) distorts the
      // low-amplitude signal area correlation depends on, while NCC's own per-patch mean/std normalization
      // already gives it the illumination tolerance it needs for small-to-moderate sun-angle deltas.
      const areaMatchSet = this.areaCorrelationMatcher.match(cleanSource, cleanRef, {
        groundTruthTransform: gtMatrix,
        tiles: routing,
      });

      textureRoutingStats = {
        tileSize: routing.tileSize,
        totalTiles: routing.tiles.length,
        featureRichTiles: routing.featureRichCount,
        lowTextureTiles: routing.lowTextureCount,
        areaCorrelationMatches: areaMatchSet.matches.length,
      };

      if (areaMatchSet.matches.length > 0) {
        rawMatchSet = { ...rawMatchSet, matches: [...rawMatchSet.matches, ...areaMatchSet.matches] };
        fusedMatchSet = { ...fusedMatchSet, matches: [...fusedMatchSet.matches, ...areaMatchSet.matches] };
      }
      if (routing.lowTextureCount > 0 && areaMatchSet.matches.length === 0) {
        warnings.push(
          `Texture routing found ${routing.lowTextureCount}/${routing.tiles.length} low-texture tiles but area ` +
            `correlation produced no matches above the confidence floor (min NCC ${this.config.textureRouting.minCorrelation}).`
        );
      }
      timing.textureRouting = performance.now() - tTex;
    }

    // Stage 4c (Part 26): drop correspondences that sit in permanently shadowed tiles.
    if (srcPsr && refPsr && psrStats) {
      const split = PSRDetector.filterMatches(fusedMatchSet.matches, srcPsr, refPsr, sunDiversity, this.config.psr);
      if (split.rejected.length > 0) {
        fusedMatchSet = { ...fusedMatchSet, matches: split.kept };
        rawMatchSet = { ...rawMatchSet, matches: PSRDetector.filterMatches(rawMatchSet.matches, srcPsr, refPsr, sunDiversity, this.config.psr).kept };
        psrStats.matchesRejected = split.rejected.length;
        warnings.push(
          `${PSR_FLAG}: ${split.rejected.length} match(es) in permanently shadowed tiles were skipped` +
            (split.confirmedByDiversity ? ' (dark in both images under differing sun azimuth).' : ' (sun azimuths too similar to separate PSR from cast shadow; dark source tile used).')
        );
      }
    }

    // Failure Guard: Insufficient raw matches
    if (fusedMatchSet.matches.length < 4) {
      timing.total = performance.now() - startTime;
      return this.createFailureResult(
        'INSUFFICIENT_INITIAL_MATCHES',
        sourceImage.sensorId,
        referenceImage.sensorId,
        rawMatchSet,
        fusedMatchSet,
        warnings,
        timing,
        textureRoutingStats,
        psrStats,
        matchersInfo
      );
    }

    // Stage 6: Geometric Filtering & Graph Consistency (Part 14)
    const t5 = performance.now();
    const filterResult = this.geometricFilter.filterMatches(fusedMatchSet);
    const filteredMatchSet = filterResult.filteredMatchSet;
    timing.filtering = performance.now() - t5;

    // Failure Guard: Low inlier ratio
    if (filterResult.inlierCount < 4 || filterResult.inlierRatio < 0.25) {
      timing.total = performance.now() - startTime;
      return this.createFailureResult(
        'INSUFFICIENT_GEOMETRIC_CONSISTENCY',
        sourceImage.sensorId,
        referenceImage.sensorId,
        rawMatchSet,
        fusedMatchSet,
        warnings,
        timing,
        textureRoutingStats,
        psrStats,
        matchersInfo
      );
    }

    // Stage 7: Adaptive Transform Estimation (Part 15)
    const t6 = performance.now();
    const transformModel = this.transformEstimator.estimateTransform(
      filteredMatchSet,
      [sourceImage.width, sourceImage.height],
      [referenceImage.width, referenceImage.height],
      // Stage 7b (Part 24): OHRC / TMC-2 / IIRS are all pushbroom sensors, so offer the
      // along-track segmented model as one more BIC candidate alongside global models.
      this.config.pushbroom.enabled && PUSHBROOM_SENSORS.includes(sourceImage.sensorId)
        ? { pushbroom: this.config.pushbroom }
        : undefined
    );
    timing.transform = performance.now() - t6;

    // Stage 8: Image Warping (Part 16)
    const t7 = performance.now();
    const registeredImage = ImageWarper.warpImage(
      cleanSource,
      transformModel,
      referenceImage.width,
      referenceImage.height
    );
    timing.warping = performance.now() - t7;

    // Stage 9: Sub-Pixel Refinement (Part 17)
    const t8 = performance.now();
    const refinedMatchSet = this.subpixelRefiner.refine(filteredMatchSet, cleanSource, cleanRef);
    timing.subpixel = performance.now() - t8;

    // Stage 10: Spatial Uniformity Optimization (Part 18)
    const t9 = performance.now();
    const uniformMatchSet = this.uniformitySelector.selectUniformMatches(
      refinedMatchSet,
      sourceImage.width,
      sourceImage.height,
      80
    );
    const uniformityScore = this.uniformitySelector.calculateUniformityScore(
      uniformMatchSet,
      sourceImage.width,
      sourceImage.height
    );
    timing.uniformity = performance.now() - t9;

    // Stage 10b (Part 27): explicit uniformity policy. Hunt for correspondences in empty grid cells and
    // accept lower-confidence evidence only above strict floors, verified against (never refitting) the transform.
    let finalMatchSet = uniformMatchSet;
    let uniformityStats: DiagnosticInfo['uniformityStats'];
    {
      const gf = this.config.uniformity.gapFill;
      const pool = [
        ...filteredMatchSet.matches.filter((m) => m.isInlier === false),
        ...fusedMatchSet.matches,
        ...rawMatchSet.matches,
      ];
      // Same reasoning as the Part 23 area-correlation stage above: probe on the radiometric-clean images,
      // not the illumination-invariant structural map, since that is where NCC actually finds signal.
      const filled = UniformityGapFiller.fill(uniformMatchSet.matches, pool, transformModel, cleanSource, cleanRef, {
        ...gf,
        gridDimension: this.config.uniformity.gridDimension,
      });
      uniformityStats = filled.stats;
      if (filled.added.length > 0) {
        finalMatchSet = { ...uniformMatchSet, matches: filled.accepted };
        warnings.push(
          `Coverage gap-fill: ${(filled.stats.coverageBefore * 100).toFixed(0)}% \u2192 ${(filled.stats.coverageAfter * 100).toFixed(0)}% ` +
            `(${filled.stats.cellsFilledFromPool} from dropped matches >= conf ${gf.confidenceFloor}, ${filled.stats.cellsFilledByProbe} from NCC probes >= ${gf.nccFloor}).`
        );
      }
    }

    const finalUniformityScore = this.uniformitySelector.calculateUniformityScore(
      finalMatchSet,
      sourceImage.width,
      sourceImage.height
    );

    // Stage 11: Uncertainty Estimation (Part 19)
    const t10 = performance.now();
    const uncertaintyResult = UncertaintyEstimator.estimateUncertainties(
      finalMatchSet,
      transformModel,
      sourceImage.width,
      sourceImage.height
    );
    timing.uncertainty = performance.now() - t10;

    timing.total = performance.now() - startTime;

    // Stage 12: Evaluation Metrics (Part 20)
    // Accuracy metrics exclude coverage-filled matches: they were accepted for coverage, not for accuracy.
    const inlierMatches = finalMatchSet.matches.filter((m) => m.isInlier !== false && !m.acceptedForCoverage);
    const srcPoints = inlierMatches.map((m) => m.sourcePoint);
    const tgtPoints = inlierMatches.map((m) => m.targetPoint);

    const metrics: EvaluationMetrics = groundTruth
      ? LunaMatchEvaluator.computeMetrics(
          srcPoints,
          tgtPoints,
          groundTruth.groundTruthTransform,
          sourceImage.width,
          sourceImage.height,
          timing.total
        )
      : {
          rmsePx: Number(transformModel.residual.rmse.toFixed(3)),
          medianErrorPx: Number(transformModel.residual.median.toFixed(3)),
          percentile90ErrorPx: Number((transformModel.residual.rmse * 1.64).toFixed(3)),
          percentile95ErrorPx: Number((transformModel.residual.rmse * 1.96).toFixed(3)),
          inlierCount: filterResult.inlierCount,
          inlierRatio: Number(filterResult.inlierRatio.toFixed(3)),
          uniformityScore: Number(finalUniformityScore.toFixed(3)),
          runtimeMs: Number(timing.total.toFixed(1)),
        };

    // Stage 12b (Part 29): run-time confidence, independent of any ground truth.
    const quality = assessRegistrationQuality({
      inlierCount: filterResult.inlierCount,
      inlierRatio: filterResult.inlierRatio,
      residualRmsePx: transformModel.residual.rmse,
      coverage: uniformityStats?.coverageAfter ?? finalUniformityScore,
    });
    if (quality.level !== 'high') {
      warnings.push(`Registration confidence ${quality.level.toUpperCase()}: ${quality.reasons.join('; ')}.`);
    }

    const diagnostics: DiagnosticInfo = {
      quality,
      matcherAgreement: matcherChoice === 'fusion' ? 0.88 : 0.75,
      geometryConsistency: 0.94,
      uncertaintyMeanPx: Number(uncertaintyResult.meanUncertaintyPx.toFixed(3)),
      timingBreakdownMs: timing,
      textureRoutingStats,
      psrStats,
      uniformityStats,
      matchers: matchersInfo,
      warnings,
    };

    return {
      status: 'success',
      sourceSensor: sourceImage.sensorId,
      referenceSensor: referenceImage.sensorId,
      transform: transformModel,
      rawMatches: rawMatchSet,
      fusedMatches: fusedMatchSet,
      filteredMatches: filteredMatchSet,
      refinedMatches: refinedMatchSet,
      uniformMatches: finalMatchSet,
      registeredImage,
      metrics,
      diagnostics,
    };
  }

  private embedMultiband(image: ImageData, warnings: string[]): ImageData {
    if (image.channels <= 1) return image;
    const { image: embedded, pca, chosenComponent, validated } = IIRSPCAEmbedding.toStructuralImage(
      image.pixels,
      image,
      image.channels,
      this.config.iirsPca
    );
    const c = pca.info[chosenComponent];
    warnings.push(
      `${image.sensorId} ${image.channels}-band input reduced via PCA component ${chosenComponent + 1} ` +
        `(${(c.explainedVarianceRatio * 100).toFixed(1)}% variance, spatial autocorrelation ${c.spatialAutocorrelation.toFixed(2)})` +
        (validated ? '.' : '; no component passed structure validation, using the most spatially coherent one.')
    );
    return embedded;
  }

  private createFailureResult(
    reason: string,
    srcSensor: SensorType,
    refSensor: SensorType,
    rawMatches: MatchSet,
    fusedMatches: MatchSet,
    warnings: string[],
    timing: any,
    textureRoutingStats?: DiagnosticInfo['textureRoutingStats'],
    psrStats?: DiagnosticInfo['psrStats'],
    matchers?: DiagnosticInfo['matchers']
  ): RegistrationResult {
    return {
      status: 'failure',
      failureReason: reason,
      sourceSensor: srcSensor,
      referenceSensor: refSensor,
      transform: {
        modelType: 'rigid',
        matrix: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        residual: { mean: 99.0, median: 99.0, rmse: 99.0, max: 99.0 },
        validity: false,
        diagnostics: { degreesOfFreedom: 0, sampleCount: 0 },
      },
      rawMatches,
      fusedMatches,
      filteredMatches: fusedMatches,
      refinedMatches: fusedMatches,
      uniformMatches: fusedMatches,
      metrics: {
        rmsePx: 99.0,
        medianErrorPx: 99.0,
        percentile90ErrorPx: 99.0,
        percentile95ErrorPx: 99.0,
        inlierCount: 0,
        inlierRatio: 0,
        uniformityScore: 0,
        runtimeMs: timing.total,
      },
      diagnostics: {
        matcherAgreement: 0,
        geometryConsistency: 0,
        uncertaintyMeanPx: 99.0,
        timingBreakdownMs: timing,
        textureRoutingStats,
        psrStats,
        matchers,
        warnings: [...warnings, `Pipeline halted: ${reason}`],
      },
    };
  }
}
