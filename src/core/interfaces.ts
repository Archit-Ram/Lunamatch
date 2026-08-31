/**
 * LunaMatch - SIH 2026 Core Component Interfaces
 */

import { ImageData, MatchSet, Point2D, RegistrationResult, TransformModel, SensorType } from '../types';

/**
 * Common Matcher Protocol for all correspondence algorithms
 * (LoFTR, RIFT, LightGlue, Mock, etc.)
 */
export interface Matcher {
  readonly name: string;
  match(source: ImageData, target: ImageData, options?: Record<string, any>): Promise<MatchSet> | MatchSet;
}

/**
 * Radiometric and sensor-aware preprocessor
 */
export interface Preprocessor {
  preprocess(image: ImageData): ImageData;
}

/**
 * Illumination invariance representation converter
 */
export interface IlluminationNormalizer {
  extractInvariantRepresentation(image: ImageData): ImageData;
}

/**
 * Geometric Transform Estimator (MAGSAC++, RANSAC, BIC model selection)
 */
export interface TransformEstimator {
  estimateTransform(matches: MatchSet, sourceDim: [number, number], targetDim: [number, number]): TransformModel;
}

/**
 * Sub-pixel correspondence refiner (quadratic surface fitting / NCC)
 */
export interface SubPixelRefiner {
  refine(matches: MatchSet, source: ImageData, target: ImageData): MatchSet;
}

/**
 * Geometry Provider / Epipolar Search Constraint (SPICE / DEM / Pinhole)
 */
export interface GeometryProvider {
  projectToSurface(point: Point2D, sourceSensor: SensorType): Point2D;
  getEpipolarBand(point: Point2D, sourceSensor: SensorType, targetSensor: SensorType): { center: Point2D; normal: Point2D; searchRadiusPx: number };
  computeGeometricPriorScore(sourcePt: Point2D, targetPt: Point2D): number;
}

/**
 * Spatial Uniformity and Coverage Optimizer
 */
export interface UniformityOptimizer {
  selectUniformMatches(matches: MatchSet, imageWidth: number, imageHeight: number, maxMatches?: number): MatchSet;
  calculateUniformityScore(matches: MatchSet, width: number, height: number): number;
}
