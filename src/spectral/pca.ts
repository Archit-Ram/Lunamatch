/**
 * LunaMatch - Part 25: Validated IIRS PCA Embedding
 *
 * Replaces naive band averaging / fixed sine-cosine "PCA" with a data-driven
 * principal component analysis of the hyperspectral cube (pixels interleaved
 * per band, index = p * numBands + b):
 *   1. Band-space covariance from a deterministic pixel subsample.
 *   2. Top-K eigenvectors by power iteration with deflation.
 *   3. Per-component validation: explained variance and lag-1 spatial
 *      autocorrelation of the score image. Pure sensor noise is spatially
 *      uncorrelated, real terrain structure is not, so low-autocorrelation
 *      components are rejected.
 *   4. The best-validated component becomes a single-channel structural image
 *      (normalized to [0,1], sign fixed deterministically) that the rest of
 *      the pipeline can match against panchromatic OHRC/TMC-2 imagery.
 */

import { ImageData } from '../types';

export interface PCAComponentInfo {
  index: number;
  eigenvalue: number;
  explainedVarianceRatio: number;
  spatialAutocorrelation: number; // lag-1, in [-1, 1]
  isStructural: boolean;
}

export interface PCAResult {
  numBands: number;
  mean: Float32Array;
  components: Float32Array[]; // K vectors of length numBands (unit norm)
  info: PCAComponentInfo[];
  scores: Float32Array[]; // K score images, each width*height
}

export interface PCAOptions {
  numComponents?: number; // default 4
  maxCovariancePixels?: number; // default 4096
  powerIterations?: number; // default 80
  minAutocorrelation?: number; // structural threshold, default 0.5
}

export class IIRSPCAEmbedding {
  static computePCA(
    cube: Float32Array,
    width: number,
    height: number,
    numBands: number,
    options: PCAOptions = {}
  ): PCAResult {
    const K = Math.max(1, Math.min(options.numComponents ?? 4, numBands));
    const maxPix = options.maxCovariancePixels ?? 4096;
    const iters = options.powerIterations ?? 80;
    const minAC = options.minAutocorrelation ?? 0.5;
    const N = width * height;

    // Mean spectrum over ALL pixels.
    const mean = new Float32Array(numBands);
    for (let p = 0; p < N; p++) for (let b = 0; b < numBands; b++) mean[b] += cube[p * numBands + b];
    for (let b = 0; b < numBands; b++) mean[b] /= N || 1;

    // Covariance from a deterministic stride subsample.
    const stride = Math.max(1, Math.floor(N / maxPix));
    const cov = Array.from({ length: numBands }, () => new Float64Array(numBands));
    let used = 0;
    const centered = new Float64Array(numBands);
    for (let p = 0; p < N; p += stride) {
      for (let b = 0; b < numBands; b++) centered[b] = cube[p * numBands + b] - mean[b];
      for (let i = 0; i < numBands; i++) {
        const ci = centered[i];
        const row = cov[i];
        for (let j = i; j < numBands; j++) row[j] += ci * centered[j];
      }
      used++;
    }
    const denom = Math.max(1, used - 1);
    for (let i = 0; i < numBands; i++)
      for (let j = i; j < numBands; j++) {
        cov[i][j] /= denom;
        cov[j][i] = cov[i][j];
      }

    let totalVariance = 0;
    for (let i = 0; i < numBands; i++) totalVariance += cov[i][i];
    totalVariance = totalVariance || 1e-12;

    // Top-K eigenpairs via power iteration + deflation (deterministic start vector).
    const components: Float32Array[] = [];
    const eigenvalues: number[] = [];
    for (let k = 0; k < K; k++) {
      let v = new Float64Array(numBands);
      for (let i = 0; i < numBands; i++) v[i] = 1 + 0.01 * Math.sin((i + 1) * (k + 1)) + 0.05 * Math.cos((i + 1) * (k + 2) * 1.7);
      const orthonormalize = (vec: Float64Array) => {
        // Gram-Schmidt against already-found components keeps the basis orthonormal even when the
        // deflated covariance is (numerically) zero, e.g. on rank-deficient cubes.
        for (const prev of components) {
          let d = 0;
          for (let i = 0; i < numBands; i++) d += vec[i] * prev[i];
          for (let i = 0; i < numBands; i++) vec[i] -= d * prev[i];
        }
        let n = 0;
        for (let i = 0; i < numBands; i++) n += vec[i] * vec[i];
        n = Math.sqrt(n) || 1;
        for (let i = 0; i < numBands; i++) vec[i] /= n;
      };
      orthonormalize(v);
      let lambda = 0;
      for (let it = 0; it < iters; it++) {
        const w = new Float64Array(numBands);
        for (let i = 0; i < numBands; i++) {
          let s = 0;
          const row = cov[i];
          for (let j = 0; j < numBands; j++) s += row[j] * v[j];
          w[i] = s;
        }
        let norm = 0;
        for (let i = 0; i < numBands; i++) norm += w[i] * w[i];
        norm = Math.sqrt(norm);
        if (norm < 1e-14) { lambda = 0; break; }
        lambda = norm;
        orthonormalize(w);
        v = w;
      }
      eigenvalues.push(lambda);
      components.push(Float32Array.from(v));
      // Deflate: C <- C - lambda v v^T
      for (let i = 0; i < numBands; i++) for (let j = 0; j < numBands; j++) cov[i][j] -= lambda * v[i] * v[j];
    }

    // Score images + validation.
    const scores: Float32Array[] = [];
    const info: PCAComponentInfo[] = [];
    for (let k = 0; k < K; k++) {
      const comp = components[k];
      const sc = new Float32Array(N);
      for (let p = 0; p < N; p++) {
        let s = 0;
        for (let b = 0; b < numBands; b++) s += (cube[p * numBands + b] - mean[b]) * comp[b];
        sc[p] = s;
      }
      scores.push(sc);
      const ac = IIRSPCAEmbedding.lag1Autocorrelation(sc, width, height);
      info.push({
        index: k,
        eigenvalue: eigenvalues[k],
        explainedVarianceRatio: eigenvalues[k] / totalVariance,
        spatialAutocorrelation: ac,
        isStructural: ac >= minAC,
      });
    }

    return { numBands, mean, components, info, scores };
  }

  /** Mean of horizontal and vertical lag-1 Pearson autocorrelation. */
  static lag1Autocorrelation(img: Float32Array, width: number, height: number): number {
    const n = width * height;
    let m = 0;
    for (let i = 0; i < n; i++) m += img[i];
    m /= n || 1;
    let varSum = 0;
    for (let i = 0; i < n; i++) varSum += (img[i] - m) ** 2;
    if (varSum < 1e-18) return 0;

    let cH = 0, cV = 0, cntH = 0, cntV = 0;
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const v = img[y * width + x] - m;
        if (x + 1 < width) { cH += v * (img[y * width + x + 1] - m); cntH++; }
        if (y + 1 < height) { cV += v * (img[(y + 1) * width + x] - m); cntV++; }
      }
    const rH = cntH ? cH / cntH : 0;
    const rV = cntV ? cV / cntV : 0;
    return (0.5 * (rH + rV)) / (varSum / n);
  }

  /**
   * Builds a single-channel structural ImageData from the highest-variance
   * component that passes spatial validation. Falls back to the component with
   * the highest autocorrelation (with a warning flag) if none pass.
   */
  static toStructuralImage(
    cube: Float32Array,
    source: Pick<ImageData, 'id' | 'width' | 'height' | 'metadata' | 'mask'>,
    numBands: number,
    options: PCAOptions = {}
  ): { image: ImageData; pca: PCAResult; chosenComponent: number; validated: boolean } {
    const { width, height } = source;
    const pca = IIRSPCAEmbedding.computePCA(cube, width, height, numBands, options);

    let chosen = pca.info.find((c) => c.isStructural)?.index;
    const validated = chosen !== undefined;
    if (chosen === undefined) {
      chosen = pca.info.reduce((best, c) => (c.spatialAutocorrelation > pca.info[best].spatialAutocorrelation ? c.index : best), 0);
    }

    const N = width * height;
    const sc = Float32Array.from(pca.scores[chosen]);

    // Deterministic sign: orient positively with mean band brightness when correlated, else by largest loading.
    let corrNum = 0, dS = 0, dB = 0, meanS = 0, meanB = 0;
    const bright = new Float32Array(N);
    for (let p = 0; p < N; p++) {
      let s = 0;
      for (let b = 0; b < numBands; b++) s += cube[p * numBands + b];
      bright[p] = s / numBands;
      meanS += sc[p];
      meanB += bright[p];
    }
    meanS /= N; meanB /= N;
    for (let p = 0; p < N; p++) {
      const a = sc[p] - meanS, b = bright[p] - meanB;
      corrNum += a * b; dS += a * a; dB += b * b;
    }
    const corr = dS > 0 && dB > 0 ? corrNum / Math.sqrt(dS * dB) : 0;
    let flip = false;
    if (Math.abs(corr) > 0.1) flip = corr < 0;
    else {
      const comp = pca.components[chosen];
      let li = 0;
      for (let b = 1; b < numBands; b++) if (Math.abs(comp[b]) > Math.abs(comp[li])) li = b;
      flip = comp[li] < 0;
    }
    if (flip) for (let p = 0; p < N; p++) sc[p] = -sc[p];

    // Robust normalization to [0,1] (1st-99th percentile).
    const sorted = Float32Array.from(sc).sort();
    const lo = sorted[Math.floor(0.01 * (N - 1))];
    const hi = sorted[Math.floor(0.99 * (N - 1))];
    const range = hi - lo || 1;
    const pixels = new Float32Array(N);
    for (let p = 0; p < N; p++) pixels[p] = Math.min(1, Math.max(0, (sc[p] - lo) / range));

    const image: ImageData = {
      id: `${source.id}_pca${chosen}`,
      pixels,
      width,
      height,
      channels: 1,
      dtype: 'float32',
      sensorId: 'IIRS',
      metadata: { ...source.metadata, bandsCount: 1 },
      mask: source.mask,
    };
    return { image, pca, chosenComponent: chosen, validated };
  }
}
