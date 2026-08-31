/**
 * LunaMatch - Part 04: Radiometric & Sensor-Aware Preprocessing
 * 
 * Cleans raw lunar sensor data: invalid-value masking, robust percentile clipping,
 * gradient calculation, and noise reduction while preserving physical signal.
 */

import { ImageData } from '../types';
import { Preprocessor } from '../core/interfaces';

export interface PreprocessingOptions {
  clipPercentiles?: [number, number]; // e.g. [1, 99]
  enableDenoising?: boolean;
  gaussianSigma?: number;
  computeGradients?: boolean;
  minValidThreshold?: number;
  maxValidThreshold?: number;
}

export class RadiometricPreprocessor implements Preprocessor {
  private options: PreprocessingOptions;

  constructor(options: PreprocessingOptions = {}) {
    this.options = {
      clipPercentiles: [1, 99],
      enableDenoising: false,
      gaussianSigma: 1.0,
      computeGradients: false,
      minValidThreshold: 0.001,
      maxValidThreshold: 0.999,
      ...options,
    };
  }

  preprocess(image: ImageData): ImageData {
    const { width, height, pixels, mask: inMask } = image;
    const totalPixels = width * height;
    const processedPixels = new Float32Array(totalPixels);
    const processedMask = new Uint8Array(totalPixels);

    // 1. Filter out NaNs, Infs, and create initial validity mask
    const validValues: number[] = [];
    for (let i = 0; i < totalPixels; i++) {
      const val = pixels[i];
      const isValidNumber = !Number.isNaN(val) && Number.isFinite(val);
      const isMasked = inMask ? inMask[i] > 0 : true;

      if (isValidNumber && isMasked && val >= 0.0) {
        processedMask[i] = 1;
        validValues.push(val);
      } else {
        processedMask[i] = 0;
        processedPixels[i] = 0.0;
      }
    }

    if (validValues.length === 0) {
      return {
        ...image,
        pixels: processedPixels,
        mask: processedMask,
      };
    }

    // 2. Compute percentile clipping bounds for robust contrast normalization
    validValues.sort((a, b) => a - b);
    const pLowIdx = Math.floor((this.options.clipPercentiles![0] / 100) * (validValues.length - 1));
    const pHighIdx = Math.floor((this.options.clipPercentiles![1] / 100) * (validValues.length - 1));
    const lowBound = validValues[pLowIdx];
    const highBound = validValues[pHighIdx];
    const dynamicRange = highBound - lowBound || 1.0;

    for (let i = 0; i < totalPixels; i++) {
      if (processedMask[i] === 1) {
        const raw = pixels[i];
        const clamped = Math.max(lowBound, Math.min(highBound, raw));
        processedPixels[i] = (clamped - lowBound) / dynamicRange;
      }
    }

    // 3. Optional 3x3 Gaussian Denoising Filter
    let outputPixels = processedPixels;
    if (this.options.enableDenoising) {
      outputPixels = this.applyGaussianFilter(processedPixels, width, height, processedMask);
    }

    return {
      ...image,
      id: `${image.id}_preprocessed`,
      pixels: outputPixels,
      mask: processedMask,
    };
  }

  /**
   * Safe 3x3 Gaussian smoothing respecting validity mask
   */
  private applyGaussianFilter(src: Float32Array, w: number, h: number, mask: Uint8Array): Float32Array {
    const dst = new Float32Array(w * h);
    const kernel = [1 / 16, 2 / 16, 1 / 16, 2 / 16, 4 / 16, 2 / 16, 1 / 16, 2 / 16, 1 / 16];

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        if (mask[idx] === 0) {
          dst[idx] = 0;
          continue;
        }

        let sum = 0;
        let weightSum = 0;
        let k = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nIdx = (y + dy) * w + (x + dx);
            if (mask[nIdx] === 1) {
              sum += src[nIdx] * kernel[k];
              weightSum += kernel[k];
            }
            k++;
          }
        }
        dst[idx] = weightSum > 0 ? sum / weightSum : src[idx];
      }
    }
    return dst;
  }

  /**
   * Computes Sobel Gradient Magnitude and Orientation
   */
  static computeSobelGradients(
    image: Float32Array,
    w: number,
    h: number
  ): { magnitude: Float32Array; orientation: Float32Array } {
    const magnitude = new Float32Array(w * h);
    const orientation = new Float32Array(w * h);

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;

        // Sobel X: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        const gx =
          -1 * image[(y - 1) * w + (x - 1)] +
          1 * image[(y - 1) * w + (x + 1)] +
          -2 * image[y * w + (x - 1)] +
          2 * image[y * w + (x + 1)] +
          -1 * image[(y + 1) * w + (x - 1)] +
          1 * image[(y + 1) * w + (x + 1)];

        // Sobel Y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        const gy =
          -1 * image[(y - 1) * w + (x - 1)] +
          -2 * image[(y - 1) * w + x] +
          -1 * image[(y - 1) * w + (x + 1)] +
          1 * image[(y + 1) * w + (x - 1)] +
          2 * image[(y + 1) * w + x] +
          1 * image[(y + 1) * w + (x + 1)];

        const mag = Math.hypot(gx, gy);
        magnitude[idx] = mag;
        orientation[idx] = Math.atan2(gy, gx);
      }
    }

    return { magnitude, orientation };
  }
}
