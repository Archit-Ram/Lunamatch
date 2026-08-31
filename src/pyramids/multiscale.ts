/**
 * LunaMatch - Part 07: Multi-Scale Image Pyramid
 * 
 * Creates dyadic or arbitrary scale image pyramids with anti-aliasing filters.
 * Provides exact bidirectional coordinate mapping functions for coarse-to-fine matching.
 */

import { ImageData, Point2D } from '../types';

export interface PyramidLevel {
  levelIndex: number;
  scaleFactor: number; // e.g. 1.0, 0.5, 0.25, 0.125
  width: number;
  height: number;
  pixels: Float32Array;
  mask?: Uint8Array;
}

export interface MultiScalePyramid {
  baseWidth: number;
  baseHeight: number;
  levels: PyramidLevel[];
}

export class ImagePyramidBuilder {
  /**
   * Builds an anti-aliased image pyramid
   */
  static buildPyramid(image: ImageData, numLevels: number = 4): MultiScalePyramid {
    const levels: PyramidLevel[] = [];
    let currentPixels = image.pixels;
    let currentMask = image.mask;
    let currentWidth = image.width;
    let currentHeight = image.height;
    let currentScale = 1.0;

    // Level 0: Full Resolution
    levels.push({
      levelIndex: 0,
      scaleFactor: 1.0,
      width: currentWidth,
      height: currentHeight,
      pixels: new Float32Array(currentPixels),
      mask: currentMask ? new Uint8Array(currentMask) : undefined,
    });

    for (let lvl = 1; lvl < numLevels; lvl++) {
      const nextWidth = Math.max(8, Math.floor(currentWidth / 2));
      const nextHeight = Math.max(8, Math.floor(currentHeight / 2));
      const actualScaleFactor = nextWidth / image.width;

      // 1. Anti-aliasing 3x3 filter
      const smoothed = this.applyAntiAliasingFilter(currentPixels, currentWidth, currentHeight);

      // 2. Subsample by factor of 2
      const downsampledPixels = new Float32Array(nextWidth * nextHeight);
      const downsampledMask = currentMask ? new Uint8Array(nextWidth * nextHeight) : undefined;

      for (let ny = 0; ny < nextHeight; ny++) {
        for (let nx = 0; nx < nextWidth; nx++) {
          const origX = nx * 2;
          const origY = ny * 2;
          const origIdx = origY * currentWidth + origX;
          const newIdx = ny * nextWidth + nx;

          downsampledPixels[newIdx] = smoothed[origIdx];
          if (downsampledMask && currentMask) {
            downsampledMask[newIdx] = currentMask[origIdx];
          }
        }
      }

      levels.push({
        levelIndex: lvl,
        scaleFactor: actualScaleFactor,
        width: nextWidth,
        height: nextHeight,
        pixels: downsampledPixels,
        mask: downsampledMask,
      });

      currentPixels = downsampledPixels;
      currentMask = downsampledMask;
      currentWidth = nextWidth;
      currentHeight = nextHeight;
      currentScale = actualScaleFactor;
    }

    return {
      baseWidth: image.width,
      baseHeight: image.height,
      levels,
    };
  }

  /**
   * Maps 2D coordinate from full image space to pyramid level space
   */
  static pixelToLevel(point: Point2D, pyramid: MultiScalePyramid, levelIndex: number): Point2D {
    const level = pyramid.levels[levelIndex];
    if (!level) {
      throw new Error(`Invalid pyramid level index: ${levelIndex}`);
    }
    return {
      x: point.x * level.scaleFactor,
      y: point.y * level.scaleFactor,
    };
  }

  /**
   * Maps 2D coordinate from pyramid level space back to full image space
   */
  static levelToPixel(point: Point2D, pyramid: MultiScalePyramid, levelIndex: number): Point2D {
    const level = pyramid.levels[levelIndex];
    if (!level || level.scaleFactor === 0) {
      throw new Error(`Invalid pyramid level scale: ${levelIndex}`);
    }
    return {
      x: point.x / level.scaleFactor,
      y: point.y / level.scaleFactor,
    };
  }

  private static applyAntiAliasingFilter(src: Float32Array, w: number, h: number): Float32Array {
    const dst = new Float32Array(w * h);
    // Binomial 3x3 filter: [1 2 1] / 4
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const val =
          1 * src[(y - 1) * w + (x - 1)] +
          2 * src[(y - 1) * w + x] +
          1 * src[(y - 1) * w + (x + 1)] +
          2 * src[y * w + (x - 1)] +
          4 * src[y * w + x] +
          2 * src[y * w + (x + 1)] +
          1 * src[(y + 1) * w + (x - 1)] +
          2 * src[(y + 1) * w + x] +
          1 * src[(y + 1) * w + (x + 1)];
        dst[y * w + x] = val / 16;
      }
    }
    return dst;
  }
}
