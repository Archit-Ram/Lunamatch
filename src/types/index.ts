/**
 * LunaMatch - SIH 2026 (SIH26166)
 * Core Data Contracts and Type Definitions
 * 
 * Defines standard contracts for multi-modal lunar image correspondence,
 * sensor metadata, transformations, and registration evaluation metrics.
 */

export type SensorType = 'OHRC' | 'TMC2' | 'IIRS' | 'LRO_NAC' | 'SELENE';

export interface SensorMetadata {
  sensorId: SensorType;
  productId?: string;
  spatialResolutionMeters: number; // Ground Sampling Distance (GSD), e.g., OHRC: 0.25m, TMC-2: 5m, IIRS: 80m
  incidenceAngleDeg: number; // Sun zenith angle relative to surface normal
  emissionAngleDeg: number;  // Spacecraft viewing angle
  phaseAngleDeg: number;     // Angle between illumination vector and viewing vector
  sunAzimuthDeg: number;     // Solar azimuth angle (0-360 deg)
  sunElevationDeg: number;   // Solar elevation angle above horizon
  spacecraftAltitudeKm?: number;
  centerLatitude?: number;
  centerLongitude?: number;
  acquisitionTimestamp?: string;
  bandsCount?: number;
  wavelengthRangeNm?: [number, number];
}

/**
 * 2D Image Coordinate.
 * Project convention strictly enforced:
 * x = column (horizontal index)
 * y = row (vertical index)
 * Coordinates can be continuous sub-pixel floating point values.
 */
export interface Point2D {
  x: number;
  y: number;
}

export interface Point2DWithUncertainty extends Point2D {
  sigmaX: number;
  sigmaY: number;
  sigmaXY: number; // Covariance term
}

/**
 * ImageData contract representing a lunar observation
 */
export interface ImageData {
  id: string;
  pixels: Float32Array; // Normalized image data in [0, 1] or raw DN values
  width: number;
  height: number;
  channels: number; // Typically 1 for monochrome/panchromatic, 3 for false-color RGB, K for hyperspectral
  bands?: number[]; // Wavelength values in nm for hyperspectral cubes
  dtype: 'float32' | 'uint8' | 'uint16';
  sensorId: SensorType;
  metadata: SensorMetadata;
  mask?: Uint8Array; // 1 = valid pixel, 0 = invalid / shadow / saturated / nodata
  rawImageDataUrl?: string; // Cache for browser rendering
}

export type MatchMethod = 'RIFT' | 'SimulatedLoFTR' | 'SimulatedRIFT' | 'SimulatedLightGlue' | 'Mock' | 'Fused' | 'GroundTruth';

/**
 * Point correspondence between source and target images
 */
export interface Match {
  id: string;
  sourcePoint: Point2D;
  targetPoint: Point2D;
  confidence: number; // [0, 1]
  method: MatchMethod;
  methodProvenance?: {
    loftrConfidence?: number;
    riftConfidence?: number;
    lightglueConfidence?: number;
    geometryConsistency?: number;
    photometricConsistency?: number;
    agreementCount?: number;
  };
  scale?: number;
  orientationDeg?: number;
  uncertaintyPx?: number; // Estimated 1-sigma uncertainty in pixels
  isInlier?: boolean;
  reprojectionErrorPx?: number;
}

export interface MatchSet {
  matches: Match[];
  sourceImageId: string;
  targetImageId: string;
  coordinateConvention: 'x=column, y=row';
  metadata?: Record<string, any>;
}

export type TransformType = 'rigid' | 'affine' | 'homography' | 'tps';

export interface TransformModel {
  modelType: TransformType;
  // Parameters matrix (3x3 for Affine/Homography) or knot coordinates for TPS
  matrix?: number[][]; // 3x3 homogeneous matrix
  tpsControlPoints?: {
    sourceKnots: Point2D[];
    targetKnots: Point2D[];
    weights: number[][];
    affineParams: number[][];
  };
  residual: {
    mean: number;
    median: number;
    rmse: number;
    max: number;
  };
  validity: boolean;
  bicScore?: number; // Bayesian Information Criterion for model selection
  inlierRatio?: number;
  diagnostics: {
    conditionNumber?: number;
    degreesOfFreedom: number;
    sampleCount: number;
    singularValues?: number[];
  };
}

export interface EvaluationMetrics {
  rmsePx: number;
  medianErrorPx: number;
  percentile90ErrorPx: number;
  percentile95ErrorPx: number;
  inlierCount: number;
  inlierRatio: number;
  uniformityScore: number; // [0, 1] spatial distribution score
  runtimeMs: number;
  reprojectionResiduals?: number[];
}

export interface DiagnosticInfo {
  matcherAgreement: number; // [0, 1]
  geometryConsistency: number; // [0, 1]
  spectralContinuity?: number;
  uncertaintyMeanPx: number;
  timingBreakdownMs: {
    ingestion: number;
    preprocessing: number;
    illumination: number;
    pyramid: number;
    matching: number;
    fusion: number;
    filtering: number;
    transform: number;
    warping: number;
    subpixel: number;
    uniformity: number;
    uncertainty: number;
    total: number;
  };
  warnings: string[];
}

export interface RegistrationResult {
  status: 'success' | 'failure';
  failureReason?: string;
  sourceSensor: SensorType;
  referenceSensor: SensorType;
  transform: TransformModel;
  rawMatches: MatchSet;
  fusedMatches: MatchSet;
  filteredMatches: MatchSet;
  refinedMatches: MatchSet;
  uniformMatches: MatchSet;
  registeredImage?: ImageData;
  metrics: EvaluationMetrics;
  diagnostics: DiagnosticInfo;
}

export interface GroundTruthData {
  sourcePoints: Point2D[];
  targetPoints: Point2D[];
  groundTruthTransform: number[][];
  transformType: TransformType;
  knownScale: number;
  knownRotationDeg: number;
  knownTranslation: [number, number];
  knownIlluminationDeltaDeg: number;
}
