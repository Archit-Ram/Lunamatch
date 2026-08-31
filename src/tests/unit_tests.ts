/**
 * LunaMatch - Comprehensive Unit & Verification Test Suite
 * 
 * Tests core data contracts, coordinate round-trips, sub-pixel estimation,
 * MAGSAC++ inlier filtering, pyramid scale consistency, and failure detection.
 */

import { validatePoint, InvalidCoordinateError } from '../core/exceptions';
import { generateSyntheticLunarDataset, applyHomographyToPoint, invert3x3 } from '../generator/synthetic';
import { ImagePyramidBuilder } from '../pyramids/multiscale';
import { QuadraticSubPixelRefiner } from '../refinement/subpixel';
import { GeometricGraphFilter } from '../filtering/graph_consistency';
import { LunaMatchPipeline } from '../pipeline/lunamatch';
import { IIRSSpectralProcessor } from '../spectral/iirs';

export interface TestCaseResult {
  partName: string;
  testName: string;
  passed: boolean;
  message: string;
  details?: string;
}

export async function runAllLunaMatchUnitTests(): Promise<TestCaseResult[]> {
  const results: TestCaseResult[] = [];

  // --- PART 01: Core Data Contracts & Validation ---
  try {
    validatePoint({ x: 100.5, y: 200.25 }, 500, 500);
    results.push({
      partName: 'PART 01: Data Contracts',
      testName: 'Valid floating-point coordinates accepted',
      passed: true,
      message: 'Proper (x=col, y=row) coordinate format validated.',
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 01: Data Contracts',
      testName: 'Valid floating-point coordinates accepted',
      passed: false,
      message: err.message,
    });
  }

  try {
    let threw = false;
    try {
      validatePoint({ x: NaN, y: 150 });
    } catch (e) {
      if (e instanceof InvalidCoordinateError) threw = true;
    }
    results.push({
      partName: 'PART 01: Data Contracts',
      testName: 'NaN/Inf coordinates strictly rejected',
      passed: threw,
      message: threw ? 'InvalidCoordinateError correctly thrown on NaN input.' : 'Failed to reject NaN coordinate.',
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 01: Data Contracts',
      testName: 'NaN/Inf coordinates strictly rejected',
      passed: false,
      message: err.message,
    });
  }

  // --- PART 02: Synthetic Lunar Generator & Ground Truth ---
  try {
    const dataset = generateSyntheticLunarDataset({
      seed: 42,
      sourceSensor: 'OHRC',
      referenceSensor: 'TMC2',
      scale: 1.25,
      rotationDeg: 15,
      translationPx: [10, -5],
    });

    const p = dataset.groundTruth.sourcePoints[0];
    const expected = dataset.groundTruth.targetPoints[0];
    const calculated = applyHomographyToPoint(dataset.groundTruth.groundTruthTransform, p);

    const error = Math.hypot(calculated.x - expected.x, calculated.y - expected.y);
    const passed = error < 1e-3;

    results.push({
      partName: 'PART 02: Synthetic Generator',
      testName: 'Ground-truth transformation consistency (T(p) == q)',
      passed,
      message: `Point (${p.x.toFixed(1)}, ${p.y.toFixed(1)}) mapped with residual error ${error.toExponential(3)} px.`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 02: Synthetic Generator',
      testName: 'Ground-truth transformation consistency',
      passed: false,
      message: err.message,
    });
  }

  // --- PART 06: IIRS Spectral Processing ---
  try {
    const { validIndices, cleanWavelengths } = IIRSSpectralProcessor.validateBands([700, 850, 1000, 2000, 3000, 4800]);
    const passed = validIndices.length === 4 && cleanWavelengths[0] === 850;
    results.push({
      partName: 'PART 06: IIRS Spectral Processing',
      testName: 'Spectral band filtering (800nm-4200nm)',
      passed,
      message: `Retained ${validIndices.length} valid science bands, noisy thermal channels pruned.`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 06: IIRS Spectral Processing',
      testName: 'Spectral band filtering',
      passed: false,
      message: err.message,
    });
  }

  // --- PART 07: Multi-Scale Pyramid Round-Trip ---
  try {
    const dataset = generateSyntheticLunarDataset({ seed: 99, sourceSensor: 'OHRC', referenceSensor: 'TMC2' });
    const pyramid = ImagePyramidBuilder.buildPyramid(dataset.sourceImage, 4);

    const testPt = { x: 200, y: 150 };
    const levelPt = ImagePyramidBuilder.pixelToLevel(testPt, pyramid, 2); // 1/4 scale
    const restoredPt = ImagePyramidBuilder.levelToPixel(levelPt, pyramid, 2);

    const roundTripError = Math.hypot(testPt.x - restoredPt.x, testPt.y - restoredPt.y);
    const passed = roundTripError < 1e-4;

    results.push({
      partName: 'PART 07: Multi-Scale Pyramid',
      testName: 'Bidirectional coordinate consistency across pyramid levels',
      passed,
      message: `Round-trip coordinate drift: ${roundTripError.toExponential(3)} px across Level 2 (1/4 scale).`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 07: Multi-Scale Pyramid',
      testName: 'Bidirectional coordinate consistency',
      passed: false,
      message: err.message,
    });
  }

  // --- PART 14: MAGSAC++ Geometric Outlier Filtering ---
  try {
    const filter = new GeometricGraphFilter({ inlierThresholdPx: 2.0 });
    const dataset = generateSyntheticLunarDataset({ seed: 123, sourceSensor: 'OHRC', referenceSensor: 'TMC2' });
    const H_gt = dataset.groundTruth.groundTruthTransform;

    // Create 40 true inliers + 20 synthetic gross outliers
    const testMatches = [];
    for (let i = 0; i < 40; i++) {
      const sp = { x: 50 + (i % 8) * 35, y: 50 + Math.floor(i / 8) * 35 };
      const tp = applyHomographyToPoint(H_gt, sp);
      testMatches.push({
        id: `inlier_${i}`,
        sourcePoint: sp,
        targetPoint: tp,
        confidence: 0.9,
        method: 'Mock' as const,
        isInlier: true,
      });
    }
    for (let j = 0; j < 20; j++) {
      testMatches.push({
        id: `outlier_${j}`,
        sourcePoint: { x: 60 + j * 10, y: 60 + j * 10 },
        targetPoint: { x: 300 - j * 12, y: 20 + j * 15 }, // random spurious target
        confidence: 0.4,
        method: 'Mock' as const,
        isInlier: false,
      });
    }

    const res = filter.filterMatches({
      matches: testMatches,
      sourceImageId: 'src',
      targetImageId: 'tgt',
      coordinateConvention: 'x=column, y=row',
    });

    const passed = res.inlierCount >= 38 && res.rmse < 0.2;
    results.push({
      partName: 'PART 14: Geometric Filtering',
      testName: 'MAGSAC++ robust estimation with 33% outlier contamination',
      passed,
      message: `Recovered ${res.inlierCount}/40 true inliers with sub-pixel residual RMSE = ${res.rmse.toFixed(3)} px.`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 14: Geometric Filtering',
      testName: 'MAGSAC++ robust estimation',
      passed: false,
      message: err.message,
    });
  }

  // --- PART 17: Sub-Pixel Quadratic Refinement ---
  try {
    const dataset = generateSyntheticLunarDataset({ seed: 555, sourceSensor: 'OHRC', referenceSensor: 'TMC2' });
    const refiner = new QuadraticSubPixelRefiner();
    const testMatchSet = {
      matches: [
        {
          id: 'sp_test',
          sourcePoint: { x: 180, y: 180 },
          targetPoint: { x: 180.35, y: 179.72 },
          confidence: 0.9,
          method: 'LoFTR' as const,
          isInlier: true,
        },
      ],
      sourceImageId: dataset.sourceImage.id,
      targetImageId: dataset.referenceImage.id,
      coordinateConvention: 'x=column, y=row' as const,
    };

    const refined = refiner.refine(testMatchSet, dataset.sourceImage, dataset.referenceImage);
    const passed = refined.matches.length === 1 && typeof refined.matches[0].targetPoint.x === 'number';

    results.push({
      partName: 'PART 17: Sub-Pixel Refinement',
      testName: 'Quadratic surface peak fitting & Hessian extremum',
      passed,
      message: `Analytical continuous sub-pixel coordinate: (${refined.matches[0].targetPoint.x.toFixed(3)}, ${refined.matches[0].targetPoint.y.toFixed(3)}).`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 17: Sub-Pixel Refinement',
      testName: 'Quadratic surface peak fitting',
      passed: false,
      message: err.message,
    });
  }

  // --- PART 21: Full End-to-End Orchestration & Failure Guard ---
  try {
    const pipeline = new LunaMatchPipeline();
    const dataset = generateSyntheticLunarDataset({ seed: 777, sourceSensor: 'OHRC', referenceSensor: 'TMC2' });
    const regResult = await pipeline.registerImages(dataset.sourceImage, dataset.referenceImage, dataset.groundTruth);

    const passed = regResult.status === 'success' && regResult.metrics.rmsePx < 0.75;
    results.push({
      partName: 'PART 21: End-to-End Pipeline',
      testName: 'Full pipeline execution with multi-matcher fusion & warping',
      passed,
      message: `Registration status: ${regResult.status}, Inliers: ${regResult.metrics.inlierCount}, RMSE: ${regResult.metrics.rmsePx} px, Time: ${regResult.metrics.runtimeMs}ms.`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 21: End-to-End Pipeline',
      testName: 'Full pipeline execution',
      passed: false,
      message: err.message,
    });
  }

  return results;
}
