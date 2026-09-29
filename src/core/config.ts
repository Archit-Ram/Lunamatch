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
    gapFill: {
      enabled: boolean;
      confidenceFloor: number; // never accept a fill candidate below this confidence
      maxReprojectionPx: number; // fill candidates must agree with the fitted transform this closely
      nccFloor: number; // minimum NCC for a targeted area-correlation probe
      maxShiftPx: number; // probe result must stay this close to the transform prediction
      probeSearchRadiusPx: number;
    };
  };
  fusionWeights: {
    loftr: number;
    rift: number;
    lightglue: number;
    geometryPrior: number;
  };
  modelSelectionCriteria: 'bic' | 'aic' | 'rmse';
  textureRouting: {
    enabled: boolean;
    tileSize: number; // pixels per tile edge
    gradientEnergyThreshold: number; // below this AND entropy below its threshold -> low_texture
    entropyThreshold: number; // normalized [0,1] Shannon entropy of the gradient histogram
    patchRadius: number; // half-width of the NCC template patch
    searchRadiusPx: number; // half-width of the target search window
    minCorrelation: number; // reject area-correlation matches below this NCC score
  };
  psr: {
    enabled: boolean;
    tileSize: number;
    darkThreshold: number; // p95 intensity below this marks a tile dark
    minSunDiversityDeg: number; // sun azimuth difference needed to confirm PSR by dark-in-both-images
    outOfScopeDarkFraction: number; // if this fraction of source tiles is dark, the whole pair is out of scope
  };
  iirsPca: {
    numComponents: number; // components computed for validation
    minAutocorrelation: number; // lag-1 spatial autocorrelation needed to call a component structural
  };
  pushbroom: {
    enabled: boolean; // offer the per-strip along-track model as a BIC candidate for pushbroom sensors
    alongTrackAxis: 'x' | 'y';
    numSegments?: number; // undefined = auto from image extent
    smoothingSigma: number;
    minMatchesPerSegment: number;
  };
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
    gapFill: {
      enabled: true,
      confidenceFloor: 0.25,
      maxReprojectionPx: 1.5,
      nccFloor: 0.75,
      maxShiftPx: 3,
      probeSearchRadiusPx: 6,
    },
  },
  fusionWeights: {
    loftr: 0.35,
    rift: 0.35,
    lightglue: 0.20,
    geometryPrior: 0.10,
  },
  modelSelectionCriteria: 'bic',
  textureRouting: {
    enabled: true,
    tileSize: 32,
    gradientEnergyThreshold: 0.007,
    entropyThreshold: 0.55,
    patchRadius: 7,
    searchRadiusPx: 24,
    minCorrelation: 0.6,
  },
  psr: {
    enabled: true,
    tileSize: 32,
    darkThreshold: 0.06,
    minSunDiversityDeg: 30,
    outOfScopeDarkFraction: 0.9,
  },
  iirsPca: {
    numComponents: 4,
    minAutocorrelation: 0.5,
  },
  pushbroom: {
    enabled: true,
    alongTrackAxis: 'y',
    smoothingSigma: 0.8,
    minMatchesPerSegment: 6,
  },
};
