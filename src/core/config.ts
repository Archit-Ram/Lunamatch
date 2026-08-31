/**
 * LunaMatch Global Configuration
 */

export interface PipelineConfig {
  matcher: 'fusion' | 'loftr' | 'rift' | 'lightglue' | 'mock';
  mockMode?: 'perfect' | 'low_noise' | 'high_noise' | 'outlier_heavy' | 'clustered' | 'mixed';
  ransac: {
    maxIterations: number;
    inlierThresholdPx: number;
    confidence: number;
    useMagsacPlusPlus: boolean;
  };
  illumination: {
    useLogTransform: boolean;
    useHighPassFilter: boolean;
    usePhaseCongruency: boolean;
    gaussianSigma: number;
  };
  pyramid: {
    numLevels: number;
    scaleFactor: number;
  };
  subpixel: {
    patchRadius: number; // e.g. 5 for 11x11 patch
    minConfidence: number;
  };
  uniformity: {
    gridDimension: number; // 8 for 8x8 grid
    maxPointsPerCell: number;
  };
  fusionWeights: {
    loftr: number;
    rift: number;
    lightglue: number;
    geometryPrior: number;
  };
  modelSelectionCriteria: 'bic' | 'aic' | 'rmse';
}

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  matcher: 'fusion',
  mockMode: 'low_noise',
  ransac: {
    maxIterations: 2000,
    inlierThresholdPx: 2.5,
    confidence: 0.99,
    useMagsacPlusPlus: true,
  },
  illumination: {
    useLogTransform: true,
    useHighPassFilter: true,
    usePhaseCongruency: true,
    gaussianSigma: 3.0,
  },
  pyramid: {
    numLevels: 4,
    scaleFactor: 0.5,
  },
  subpixel: {
    patchRadius: 5,
    minConfidence: 0.4,
  },
  uniformity: {
    gridDimension: 8,
    maxPointsPerCell: 15,
  },
  fusionWeights: {
    loftr: 0.35,
    rift: 0.35,
    lightglue: 0.20,
    geometryPrior: 0.10,
  },
  modelSelectionCriteria: 'bic',
};
