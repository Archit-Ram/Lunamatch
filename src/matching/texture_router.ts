/**
 * LunaMatch - Part 23: Texture-Routed Dual-Mode Matching (Router)
 *
 * Neural matchers (LoFTR/RIFT/LightGlue) are keypoint/feature based and
 * starve on smooth, low-texture lunar terrain (mare basalt plains, fine
 * regolith, deep shadow interiors) where there is nothing distinctive to
 * key off. This router segments an image into a tile grid, scores each
 * tile's texture content via Sobel gradient energy + Shannon entropy of the
 * gradient-magnitude histogram, and classifies each tile as:
 *  - 'feature_rich' -> left to the existing neural matchers (Part 09-12)
 *  - 'low_texture'  -> routed to area-based correlation (see
 *    area_correlation_matcher.ts) as a fallback so smooth terrain still
 *    contributes correspondences instead of being silently dropped.
 */

import { ImageData } from '../types';

export type TileTextureMode = 'feature_rich' | 'low_texture' | 'psr_skipped';

export interface TextureTile {
  row: number;
  col: number;
  x0: number;
  y0: number;
  x1: number; // exclusive
  y1: number; // exclusive
  centerX: number;
  centerY: number;
  gradientEnergy: number; // mean squared gradient magnitude, normalized to [0,1]-ish
  entropy: number; // Shannon entropy of the gradient-magnitude histogram, normalized [0,1]
  mode: TileTextureMode;
}

export interface TextureRoutingOptions {
  tileSize?: number; // pixels per tile edge, default 32
  histogramBins?: number; // bins for the gradient-magnitude entropy histogram
  gradientEnergyThreshold?: number; // below this AND entropy below its threshold -> low_texture
  entropyThreshold?: number; // normalized [0,1]; below this -> low_texture candidate
}

export interface TextureRoutingResult {
  tiles: TextureTile[];
  tileSize: number;
  gridCols: number;
  gridRows: number;
  featureRichCount: number;
  lowTextureCount: number;
}

const DEFAULTS: Required<TextureRoutingOptions> = {
  tileSize: 32,
  histogramBins: 16,
  gradientEnergyThreshold: 0.007,
  entropyThreshold: 0.55,
};

export class TextureRouter {
  /**
   * Segments `image` into a tileSize x tileSize grid and classifies each
   * tile's texture content.
   */
  static classifyImage(image: ImageData, options: TextureRoutingOptions = {}): TextureRoutingResult {
    const opts = { ...DEFAULTS, ...options };
    const { width, height, pixels } = image;

    // Precompute Sobel gradient magnitude for the whole image once.
    const gradMag = TextureRouter.computeGradientMagnitude(pixels, width, height);

    // Histogram binning uses a single image-wide reference scale (not each
    // tile's own local max) so entropy stays comparable across tiles: a
    // tile whose gradients are uniformly tiny should collapse into the
    // lowest histogram bin (low entropy), not spread across many bins just
    // because its own small range got rescaled to fill them.
    let globalMax = 1e-6;
    for (let i = 0; i < gradMag.length; i++) {
      if (gradMag[i] > globalMax) globalMax = gradMag[i];
    }

    const gridCols = Math.max(1, Math.ceil(width / opts.tileSize));
    const gridRows = Math.max(1, Math.ceil(height / opts.tileSize));
    const tiles: TextureTile[] = [];

    let featureRichCount = 0;
    let lowTextureCount = 0;

    for (let row = 0; row < gridRows; row++) {
      for (let col = 0; col < gridCols; col++) {
        const x0 = col * opts.tileSize;
        const y0 = row * opts.tileSize;
        const x1 = Math.min(width, x0 + opts.tileSize);
        const y1 = Math.min(height, y0 + opts.tileSize);

        const { gradientEnergy, entropy } = TextureRouter.scoreTile(
          gradMag,
          width,
          x0,
          y0,
          x1,
          y1,
          opts.histogramBins,
          globalMax
        );

        // Gradient energy (mean squared Sobel magnitude) is the primary texture
        // signal: it's exactly what feature/keypoint matchers need to find
        // something to key off. Entropy of the gradient-magnitude histogram is
        // reported alongside as a secondary diversity metric (useful for tuning
        // and diagnostics, and to catch highly periodic/aliased tiles whose
        // energy alone wouldn't flag them), and only overrides a borderline
        // "feature_rich" energy reading when the tile is otherwise almost
        // perfectly uniform in gradient structure.
        const mode: TileTextureMode =
          gradientEnergy < opts.gradientEnergyThreshold ||
          (gradientEnergy < opts.gradientEnergyThreshold * 2 && entropy < opts.entropyThreshold * 0.5)
            ? 'low_texture'
            : 'feature_rich';

        if (mode === 'low_texture') lowTextureCount++;
        else featureRichCount++;

        tiles.push({
          row,
          col,
          x0,
          y0,
          x1,
          y1,
          centerX: (x0 + x1) / 2,
          centerY: (y0 + y1) / 2,
          gradientEnergy,
          entropy,
          mode,
        });
      }
    }

    return { tiles, tileSize: opts.tileSize, gridCols, gridRows, featureRichCount, lowTextureCount };
  }

  /** 3x3 Sobel gradient magnitude for every pixel, flattened row-major. */
  private static computeGradientMagnitude(pixels: Float32Array, width: number, height: number): Float32Array {
    const mag = new Float32Array(width * height);

    const at = (x: number, y: number): number => {
      const cx = Math.min(width - 1, Math.max(0, x));
      const cy = Math.min(height - 1, Math.max(0, y));
      return pixels[cy * width + cx];
    };

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const gx =
          -at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1) +
          at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1);
        const gy =
          -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1) +
          at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1);
        mag[y * width + x] = Math.hypot(gx, gy);
      }
    }

    return mag;
  }

  /** Computes normalized gradient energy + Shannon entropy of the gradient histogram for one tile. */
  private static scoreTile(
    gradMag: Float32Array,
    imageWidth: number,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    numBins: number,
    globalMax: number
  ): { gradientEnergy: number; entropy: number } {
    const n = (x1 - x0) * (y1 - y0);
    if (n <= 0) return { gradientEnergy: 0, entropy: 0 };

    let sumSq = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const g = gradMag[y * imageWidth + x];
        sumSq += g * g;
      }
    }
    // Sobel magnitude on [0,1]-normalized pixels tops out around ~4*sqrt(2);
    // divide by that theoretical ceiling so gradientEnergy sits roughly in [0,1].
    const theoreticalMax = 4 * Math.SQRT2;
    const gradientEnergy = Math.min(1, sumSq / n / (theoreticalMax * theoreticalMax));

    const hist = new Array(numBins).fill(0);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const g = gradMag[y * imageWidth + x];
        const bin = Math.min(numBins - 1, Math.floor((g / globalMax) * numBins));
        hist[bin]++;
      }
    }

    let entropyBits = 0;
    for (const count of hist) {
      if (count === 0) continue;
      const p = count / n;
      entropyBits -= p * Math.log2(p);
    }
    const maxEntropyBits = Math.log2(numBins);
    const entropy = maxEntropyBits > 0 ? entropyBits / maxEntropyBits : 0;

    return { gradientEnergy, entropy };
  }
}
