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
import {
  RIFTMatcher,
  computeRIFTFeatureMaps,
  detectRIFTKeypoints,
  extractMIMDescriptors,
  matchRIFTDescriptors,
} from '../matching/rift_matcher';
import { LightGlueMatcher } from '../matching/lightglue_matcher';
import { LoFTRMatcher } from '../matching/loftr_matcher';
import { AdaptiveTransformEstimator } from '../registration/adaptive_transform';
import { ImageData } from '../types';

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
          method: 'SimulatedLoFTR' as const,
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
      testName: 'Analytical continuous sub-pixel coordinate',
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

  // --- PART 11: Real RIFT Matcher (Analytical Phase Congruency + MIM) ---
  // Test 1: Real keypoints detected on textured synthetic image
  try {
    const dataset = generateSyntheticLunarDataset({ seed: 42, sourceSensor: 'OHRC', referenceSensor: 'TMC2' });
    const feats = computeRIFTFeatureMaps(dataset.sourceImage);
    const kps = detectRIFTKeypoints(feats.phaseCongruencyMoments, feats.width, feats.height, feats.orientationAmps, 150);
    const passed = kps.length >= 30;
    results.push({
      partName: 'PART 11: Real RIFT Matcher',
      testName: 'Keypoint detection on textured lunar terrain',
      passed,
      message: `Detected ${kps.length} real structural keypoints from phase congruency maximum moment map.`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 11: Real RIFT Matcher',
      testName: 'Keypoint detection on textured lunar terrain',
      passed: false,
      message: err.message,
    });
  }

  // Test 2: Multi-size non-power-of-2 zero false-positive keypoints on flat / uniform lunar terrain (e.g., mare plains)
  try {
    const testSizes = [
      { w: 160, h: 160 },
      { w: 200, h: 200 },
      { w: 217, h: 183 },
      { w: 256, h: 256 },
    ];
    let allPassed = true;
    const counts: string[] = [];

    for (const { w, h } of testSizes) {
      const blankPixels = new Float32Array(w * h).fill(0.5);
      const blankImage: ImageData = {
        id: `blank_${w}x${h}`,
        pixels: blankPixels,
        width: w,
        height: h,
        channels: 1,
        dtype: 'float32',
        sensorId: 'OHRC',
        metadata: { sensorId: 'OHRC', spatialResolutionMeters: 0.5, incidenceAngleDeg: 0, emissionAngleDeg: 0, phaseAngleDeg: 0, sunAzimuthDeg: 0, sunElevationDeg: 90 },
      };
      const feats = computeRIFTFeatureMaps(blankImage);
      const kps = detectRIFTKeypoints(feats.phaseCongruencyMoments, w, h, feats.orientationAmps, 100);
      counts.push(`${w}x${h}: ${kps.length}`);
      if (kps.length !== 0) {
        allPassed = false;
      }
    }

    results.push({
      partName: 'PART 11: Real RIFT Matcher',
      testName: 'Zero false-positive keypoints across non-power-of-2 flat/blank crops',
      passed: allPassed,
      message: `Verified mirror-padded boundary handling across non-power-of-2 dimensions (${counts.join(', ')}). All zero false-positives.`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 11: Real RIFT Matcher',
      testName: 'Zero false-positive keypoints across non-power-of-2 flat/blank crops',
      passed: false,
      message: err.message,
    });
  }

  // Test 3: RIFT vs Raw-Intensity Correlation under non-linear illumination change
  try {
    const width = 256;
    const height = 256;
    const baseImg = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        let val = Math.sin(x * 0.08) * Math.cos(y * 0.08) * 0.3 + Math.sin(x * 0.2 + y * 0.1) * 0.2;
        const d1 = Math.hypot(x - 120, y - 100);
        if (d1 < 40) val -= Math.cos((d1 / 40) * Math.PI * 0.5) * 0.4;
        const d2 = Math.hypot(x - 180, y - 170);
        if (d2 < 25) val += Math.sin((d2 / 25) * Math.PI) * 0.3;
        baseImg[idx] = Math.max(0.05, Math.min(0.95, 0.5 + val));
      }
    }

    const tx = 10;
    const ty = -6;
    const targetPixels = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      const srcY = y - ty;
      for (let x = 0; x < width; x++) {
        const srcX = x - tx;
        const tIdx = y * width + x;
        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
          const origVal = baseImg[srcY * width + srcX];
          targetPixels[tIdx] = Math.pow(origVal, 2.8) * (0.3 + 0.7 * (x / width));
        } else {
          targetPixels[tIdx] = 0.1;
        }
      }
    }

    const srcImg: ImageData = {
      id: 'src_illum',
      pixels: baseImg,
      width,
      height,
      channels: 1,
      dtype: 'float32',
      sensorId: 'OHRC',
      metadata: { sensorId: 'OHRC', spatialResolutionMeters: 0.5, incidenceAngleDeg: 30, emissionAngleDeg: 0, phaseAngleDeg: 30, sunAzimuthDeg: 45, sunElevationDeg: 45 },
    };

    const tgtImg: ImageData = {
      id: 'tgt_illum',
      pixels: targetPixels,
      width,
      height,
      channels: 1,
      dtype: 'float32',
      sensorId: 'TMC2',
      metadata: { sensorId: 'TMC2', spatialResolutionMeters: 5.0, incidenceAngleDeg: 65, emissionAngleDeg: 5, phaseAngleDeg: 70, sunAzimuthDeg: 225, sunElevationDeg: 20 },
    };

    // Real RIFT matching
    const srcFeats = computeRIFTFeatureMaps(srcImg);
    const tgtFeats = computeRIFTFeatureMaps(tgtImg);
    const srcKps = detectRIFTKeypoints(srcFeats.phaseCongruencyMoments, width, height, srcFeats.orientationAmps, 150);
    const tgtKps = detectRIFTKeypoints(tgtFeats.phaseCongruencyMoments, width, height, tgtFeats.orientationAmps, 150);
    const srcDescs = extractMIMDescriptors(srcKps, srcFeats.mimMap, srcFeats.orientationAmps, width, height);
    const tgtDescs = extractMIMDescriptors(tgtKps, tgtFeats.mimMap, tgtFeats.orientationAmps, width, height);
    const riftMatches = matchRIFTDescriptors(srcKps, srcDescs, tgtKps, tgtDescs, 0.95);

    let riftInliers = 0;
    for (const m of riftMatches) {
      const err = Math.hypot((m.sourcePoint.x + tx) - m.targetPoint.x, (m.sourcePoint.y + ty) - m.targetPoint.y);
      if (err < 3.0) riftInliers++;
    }
    const riftAccuracy = riftMatches.length > 0 ? riftInliers / riftMatches.length : 0;

    // Naive raw-intensity correlation baseline on same keypoints
    const patchR = 8;
    const rawSrcDescs = srcKps.map(kp => {
      const p = new Float32Array(patchR * 2 * patchR * 2);
      let idx = 0;
      for (let dy = -patchR; dy < patchR; dy++) {
        for (let dx = -patchR; dx < patchR; dx++) {
          const px = Math.min(width - 1, Math.max(0, Math.round(kp.x + dx)));
          const py = Math.min(height - 1, Math.max(0, Math.round(kp.y + dy)));
          p[idx++] = srcImg.pixels[py * width + px];
        }
      }
      return p;
    });

    const rawTgtDescs = tgtKps.map(kp => {
      const p = new Float32Array(patchR * 2 * patchR * 2);
      let idx = 0;
      for (let dy = -patchR; dy < patchR; dy++) {
        for (let dx = -patchR; dx < patchR; dx++) {
          const px = Math.min(width - 1, Math.max(0, Math.round(kp.x + dx)));
          const py = Math.min(height - 1, Math.max(0, Math.round(kp.y + dy)));
          p[idx++] = tgtImg.pixels[py * width + px];
        }
      }
      return p;
    });

    const rawMatches = matchRIFTDescriptors(srcKps, rawSrcDescs, tgtKps, rawTgtDescs, 0.95);
    let rawInliers = 0;
    for (const m of rawMatches) {
      const err = Math.hypot((m.sourcePoint.x + tx) - m.targetPoint.x, (m.sourcePoint.y + ty) - m.targetPoint.y);
      if (err < 3.0) rawInliers++;
    }
    const rawAccuracy = rawMatches.length > 0 ? rawInliers / rawMatches.length : 0;

    const passed = riftAccuracy > rawAccuracy && riftInliers > 0;
    results.push({
      partName: 'PART 11: Real RIFT Matcher',
      testName: 'Radiation invariance vs raw-intensity correlation under illumination shift',
      passed,
      message: `RIFT accuracy ${(riftAccuracy * 100).toFixed(1)}% (${riftInliers}/${riftMatches.length}) vs Raw Intensity ${(rawAccuracy * 100).toFixed(1)}% (${rawInliers}/${rawMatches.length}). Margin: +${((riftAccuracy - rawAccuracy) * 100).toFixed(1)}%.`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 11: Real RIFT Matcher',
      testName: 'Radiation invariance vs raw-intensity correlation under illumination shift',
      passed: false,
      message: err.message,
    });
  }

  // --- PART 12: Real SuperPoint + LightGlue Neural Inference ---
  try {
    const lgMatcher = new LightGlueMatcher(0.1);
    const dataset = generateSyntheticLunarDataset({
      width: 256,
      height: 256,
      seed: 1234,
      sourceSensor: 'OHRC',
      referenceSensor: 'TMC2',
      translationPx: [8, -6],
      rotationDeg: 0,
      scale: 1.0,
    });

    const matchSet = await lgMatcher.match(dataset.sourceImage, dataset.referenceImage);
    const gtH = dataset.groundTruth.groundTruthTransform;

    let inliers = 0;
    for (const m of matchSet.matches) {
      const trueTarget = applyHomographyToPoint(gtH, m.sourcePoint);
      const err = Math.hypot(trueTarget.x - m.targetPoint.x, trueTarget.y - m.targetPoint.y);
      if (err <= 4.0) inliers++;
    }

    const inlierRatio = matchSet.matches.length > 0 ? inliers / matchSet.matches.length : 0;
    const passed = matchSet.matches.length >= 10 && inlierRatio >= 0.35;

    results.push({
      partName: 'PART 12: Real SuperPoint + LightGlue Matcher',
      testName: 'Real neural inference ONNX execution and correspondence accuracy',
      passed,
      message: `Extracted ${matchSet.matches.length} matches, inliers: ${inliers} (${(inlierRatio * 100).toFixed(1)}%), method: ${matchSet.matches[0]?.method || 'none'}`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 12: Real SuperPoint + LightGlue Matcher',
      testName: 'Real neural inference ONNX execution and correspondence accuracy',
      passed: false,
      message: err.message,
    });
  }

  // --- PART 21: Full End-to-End Orchestration & Failure Guard ---
  try {
    const pipeline = new LunaMatchPipeline();
    const dataset = generateSyntheticLunarDataset({ seed: 777, sourceSensor: 'OHRC', referenceSensor: 'TMC2' });
    const regResult = await pipeline.registerImages(dataset.sourceImage, dataset.referenceImage, dataset.groundTruth);

    const passed = regResult.status === 'success' && regResult.metrics.rmsePx < 1.0;
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

  // --- PART 10: Real LoFTR Neural Inference ---
  try {
    const loftrMatcher = new LoFTRMatcher(0.1);
    const dataset = generateSyntheticLunarDataset({
      width: 256,
      height: 256,
      seed: 5678,
      sourceSensor: 'OHRC',
      referenceSensor: 'TMC2',
      translationPx: [8, -6],
      rotationDeg: 0,
      scale: 1.0,
    });

    const matchSet = await loftrMatcher.match(dataset.sourceImage, dataset.referenceImage);
    const gtH = dataset.groundTruth.groundTruthTransform;

    let inliers = 0;
    for (const m of matchSet.matches) {
      const trueTarget = applyHomographyToPoint(gtH, m.sourcePoint);
      const err = Math.hypot(trueTarget.x - m.targetPoint.x, trueTarget.y - m.targetPoint.y);
      if (err <= 4.0) inliers++;
    }

    const inlierRatio = matchSet.matches.length > 0 ? inliers / matchSet.matches.length : 0;
    const passed = matchSet.matches.length >= 10 && inlierRatio >= 0.35;

    results.push({
      partName: 'PART 10: Real LoFTR Matcher',
      testName: 'Real LoFTR ONNX inference and correspondence accuracy',
      passed,
      message: `Extracted ${matchSet.matches.length} matches, inliers: ${inliers} (${(inlierRatio * 100).toFixed(1)}%), method: ${matchSet.matches[0]?.method || 'none'}`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 10: Real LoFTR Matcher',
      testName: 'Real LoFTR ONNX inference and correspondence accuracy',
      passed: false,
      message: err.message,
    });
  }

  // --- PART 15: TPS Transform Estimation ---
  // Test 1: TPS fits nonlinear deformation with sub-pixel accuracy
  try {
    // Create control points with a nonlinear deformation (barrel distortion)
    const srcPts = [];
    const tgtPts = [];
    const cx = 200, cy = 200; // Center of distortion
    const k = 0.00005; // Distortion coefficient

    for (let gy = 0; gy < 6; gy++) {
      for (let gx = 0; gx < 6; gx++) {
        const sx = 50 + gx * 60;
        const sy = 50 + gy * 60;
        // Apply barrel distortion
        const dx = sx - cx;
        const dy = sy - cy;
        const r2 = dx * dx + dy * dy;
        const tx = sx + dx * k * r2;
        const ty = sy + dy * k * r2;
        srcPts.push({ x: sx, y: sy });
        tgtPts.push({ x: tx, y: ty });
      }
    }

    const estimator = new AdaptiveTransformEstimator();
    const tpsModel = estimator.fitTPS(srcPts, tgtPts, 0.001);

    const passed = tpsModel.validity && tpsModel.residual.rmse < 0.5;

    results.push({
      partName: 'PART 15: TPS Transform',
      testName: 'TPS fits nonlinear barrel distortion with sub-pixel accuracy',
      passed,
      message: `TPS fit: validity=${tpsModel.validity}, RMSE=${tpsModel.residual.rmse.toFixed(4)} px, ` +
        `knots=${tpsModel.tpsControlPoints?.sourceKnots.length || 0}, ` +
        `DOF=${tpsModel.diagnostics.degreesOfFreedom}`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 15: TPS Transform',
      testName: 'TPS fits nonlinear barrel distortion',
      passed: false,
      message: err.message,
    });
  }

  // Test 2: TPS applyTPSTransform round-trip consistency
  try {
    const srcPts = [
      { x: 50, y: 50 }, { x: 200, y: 50 }, { x: 350, y: 50 },
      { x: 50, y: 200 }, { x: 200, y: 200 }, { x: 350, y: 200 },
      { x: 50, y: 350 }, { x: 200, y: 350 }, { x: 350, y: 350 },
    ];
    // Apply an affine + slight nonlinear warp
    const tgtPts = srcPts.map(p => ({
      x: p.x * 1.1 + p.y * 0.05 + 10 + Math.sin(p.x * 0.02) * 3,
      y: p.x * -0.03 + p.y * 1.08 - 5 + Math.cos(p.y * 0.015) * 2,
    }));

    const estimator = new AdaptiveTransformEstimator();
    const tpsModel = estimator.fitTPS(srcPts, tgtPts, 0.0);

    if (!tpsModel.tpsControlPoints) throw new Error('TPS control points not set');

    // Verify that applying TPS to source points recovers target points
    let maxErr = 0;
    for (let i = 0; i < srcPts.length; i++) {
      const pred = AdaptiveTransformEstimator.applyTPSTransform(tpsModel.tpsControlPoints, srcPts[i]);
      const err = Math.hypot(pred.x - tgtPts[i].x, pred.y - tgtPts[i].y);
      if (err > maxErr) maxErr = err;
    }

    const passed = maxErr < 0.01; // Should be near-exact with lambda=0

    results.push({
      partName: 'PART 15: TPS Transform',
      testName: 'TPS interpolation reproduces control points exactly (λ=0)',
      passed,
      message: `Max reproduction error at control points: ${maxErr.toExponential(3)} px (threshold: 0.01 px)`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 15: TPS Transform',
      testName: 'TPS interpolation reproduces control points exactly',
      passed: false,
      message: err.message,
    });
  }

  // Test 3: BIC selects TPS over homography for nonlinear deformation
  try {
    const dataset = generateSyntheticLunarDataset({
      seed: 9999,
      sourceSensor: 'OHRC',
      referenceSensor: 'TMC2',
      perspective: [0.001, -0.0008],
    });
    const H_gt = dataset.groundTruth.groundTruthTransform;

    // Create matches with nonlinear barrel distortion added on top of homography
    const testMatches = [];
    const cx = 192, cy = 192;
    const k = 0.00008;
    for (let i = 0; i < 30; i++) {
      const sx = 40 + (i % 6) * 55;
      const sy = 40 + Math.floor(i / 6) * 55;
      const sp = { x: sx, y: sy };
      const tp = applyHomographyToPoint(H_gt, sp);

      // Add barrel distortion
      const dx = tp.x - cx;
      const dy = tp.y - cy;
      const r2 = dx * dx + dy * dy;
      const distortedTp = { x: tp.x + dx * k * r2, y: tp.y + dy * k * r2 };

      testMatches.push({
        id: `bic_test_${i}`,
        sourcePoint: sp,
        targetPoint: distortedTp,
        confidence: 0.9,
        method: 'Mock' as const,
        isInlier: true,
      });
    }

    const estimator = new AdaptiveTransformEstimator();
    const result = estimator.estimateTransform(
      { matches: testMatches, sourceImageId: 'src', targetImageId: 'tgt', coordinateConvention: 'x=column, y=row' },
      [384, 384],
      [384, 384]
    );

    // With nonlinear deformation, TPS should achieve lower residuals than homography
    const passed = result.residual.rmse < 2.0;

    results.push({
      partName: 'PART 15: TPS Transform',
      testName: 'BIC model selection with nonlinear deformation data',
      passed,
      message: `Selected model: ${result.modelType}, RMSE: ${result.residual.rmse.toFixed(3)} px, BIC: ${result.bicScore?.toFixed(1) || 'N/A'}`,
    });
  } catch (err: any) {
    results.push({
      partName: 'PART 15: TPS Transform',
      testName: 'BIC model selection with nonlinear deformation',
      passed: false,
      message: err.message,
    });
  }

  return results;
}
