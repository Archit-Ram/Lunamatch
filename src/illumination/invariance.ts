/**
 * LunaMatch - Part 05: Sun-Angle / Illumination Invariant Representation
 * 
 * Extracts illumination-invariant structural representations from lunar imagery.
 * Removes low-frequency lighting gradients, normalizes local contrast, and
 * extracts orientation-invariant phase / gradient structure.
 */

import { ImageData } from '../types';
import { IlluminationNormalizer } from '../core/interfaces';

export interface IlluminationOptions {
  epsilon?: number; // small positive constant for log transform
  gaussianSigma?: number; // radius for local illumination background estimation
  usePhaseCongruency?: boolean;
  normalizeStdDev?: boolean;
}

export class IlluminationInvariantNormalizer implements IlluminationNormalizer {
  private options: IlluminationOptions;

  constructor(options: IlluminationOptions = {}) {
    this.options = {
      epsilon: 0.005,
      gaussianSigma: 4.0,
      usePhaseCongruency: true,
      normalizeStdDev: true,
      ...options,
    };
  }

  extractInvariantRepresentation(image: ImageData): ImageData {
    const { width, height, pixels, mask } = image;
    const totalPixels = width * height;
    const epsilon = this.options.epsilon!;

    // Step 1: Log-intensity representation R(x) = log(I(x) + epsilon)
    // Homomorphic separation of Illumination L(x) and Reflectance R(x):
    // I(x) = L(x) * R(x) => log(I) = log(L) + log(R)
    const logImg = new Float32Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      const val = Math.max(0.0, pixels[i]);
      logImg[i] = Math.log(val + epsilon);
    }

    // Step 2: Low-frequency background estimation via separable Gaussian filter
    const lowFreq = this.separableGaussianBlur(logImg, width, height, this.options.gaussianSigma!);

    // Step 3: High-pass reflectance extraction: R'(x) = log(I) - G_sigma * log(I)
    const highPass = new Float32Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      highPass[i] = logImg[i] - lowFreq[i];
    }

    // Step 4: Local standard deviation normalization (Local Contrast Normalization)
    let invariantPixels = highPass;
    if (this.options.normalizeStdDev) {
      invariantPixels = this.normalizeLocalContrast(highPass, width, height, 5.0);
    }

    // Step 5: Phase Congruency / Maximum Moment Structural Map
    if (this.options.usePhaseCongruency) {
      invariantPixels = this.computeStructuralMoments(invariantPixels, width, height);
    }

    // Normalize final invariant output to [0, 1] range
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < totalPixels; i++) {
      if (!mask || mask[i] > 0) {
        if (invariantPixels[i] < minVal) minVal = invariantPixels[i];
        if (invariantPixels[i] > maxVal) maxVal = invariantPixels[i];
      }
    }
    const range = maxVal - minVal || 1.0;
    const normalizedOutput = new Float32Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      if (!mask || mask[i] > 0) {
        normalizedOutput[i] = (invariantPixels[i] - minVal) / range;
      } else {
        normalizedOutput[i] = 0;
      }
    }

    return {
      ...image,
      id: `${image.id}_illum_invariant`,
      pixels: normalizedOutput,
      mask: mask ? new Uint8Array(mask) : undefined,
    };
  }

  /**
   * Fast Separable Gaussian Blur (O(K * N) instead of O(K^2 * N))
   */
  private separableGaussianBlur(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
    const radius = Math.max(1, Math.ceil(sigma * 2.5));
    const size = radius * 2 + 1;
    const kernel = new Float32Array(size);
    let kSum = 0;
    for (let i = -radius; i <= radius; i++) {
      const v = Math.exp(-(i * i) / (2 * sigma * sigma));
      kernel[i + radius] = v;
      kSum += v;
    }
    for (let i = 0; i < size; i++) kernel[i] /= kSum;

    // Horizontal pass
    const temp = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {
          const sampleX = Math.min(w - 1, Math.max(0, x + k));
          sum += src[y * w + sampleX] * kernel[k + radius];
        }
        temp[y * w + x] = sum;
      }
    }

    // Vertical pass
    const dst = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {
          const sampleY = Math.min(h - 1, Math.max(0, y + k));
          sum += temp[sampleY * w + x] * kernel[k + radius];
        }
        dst[y * w + x] = sum;
      }
    }

    return dst;
  }

  /**
   * Local Standard Deviation Normalization
   */
  private normalizeLocalContrast(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
    const squared = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      squared[i] = src[i] * src[i];
    }

    const localVariance = this.separableGaussianBlur(squared, w, h, sigma);
    const dst = new Float32Array(w * h);
    const eps = 1e-4;

    for (let i = 0; i < w * h; i++) {
      const std = Math.sqrt(Math.max(0, localVariance[i])) + eps;
      dst[i] = src[i] / std;
    }

    return dst;
  }

  /**
   * Structural phase energy / maximum moment of phase congruency proxy
   */
  private computeStructuralMoments(src: Float32Array, w: number, h: number): Float32Array {
    const dst = new Float32Array(w * h);
    // 2-scale difference of Gaussians as frequency-selective structural band
    const g1 = this.separableGaussianBlur(src, w, h, 1.2);
    const g2 = this.separableGaussianBlur(src, w, h, 3.2);

    for (let i = 0; i < w * h; i++) {
      const bandpass = Math.abs(g1[i] - g2[i]);
      dst[i] = bandpass;
    }

    return dst;
  }
}
