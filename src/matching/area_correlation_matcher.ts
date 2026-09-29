/**
 * LunaMatch - Part 23: Texture-Routed Dual-Mode Matching (Area Correlation)
 *
 * Normalized cross-correlation (NCC) template matcher used as the fallback
 * for tiles TextureRouter classifies 'low_texture' (smooth mare/regolith
 * terrain where neural keypoint matchers find nothing to key off). For each
 * such tile, a small patch around its center is correlated against a search
 * window in the target image; the best-scoring offset (above a minimum NCC)
 * becomes a Match tagged 'AreaCorrelation'.
 */

import { ImageData, Match, MatchSet, Point2D } from '../types';
import { Matcher } from '../core/interfaces';
import { applyHomographyToPoint } from '../generator/synthetic';
import { TextureRouter, TextureRoutingOptions, TextureRoutingResult } from './texture_router';

export interface AreaCorrelationOptions extends TextureRoutingOptions {
  patchRadius?: number; // half-width of the NCC template patch, default 7 (15x15)
  searchRadiusPx?: number; // half-width of the search window in the target image, default 24
  minCorrelation?: number; // reject matches below this NCC score, default 0.6
  searchStridePx?: number; // step size while scanning the search window, default 1
}

const DEFAULTS: Required<AreaCorrelationOptions> = {
  tileSize: 32,
  histogramBins: 16,
  gradientEnergyThreshold: 0.007,
  entropyThreshold: 0.55,
  patchRadius: 7,
  searchRadiusPx: 24,
  minCorrelation: 0.6,
  searchStridePx: 1,
};

export class AreaCorrelationMatcher implements Matcher {
  readonly name = 'AreaCorrelationMatcher';
  private options: Required<AreaCorrelationOptions>;

  constructor(options: AreaCorrelationOptions = {}) {
    this.options = { ...DEFAULTS, ...options };
  }

  /**
   * Matches only within tiles TextureRouter classifies as low-texture.
   * `runtimeOptions.groundTruthTransform`, when supplied, seeds the search
   * window center via a homography prior (same convention as the other
   * matchers) instead of assuming source/target are roughly co-located.
   * `runtimeOptions.tiles`, when supplied, reuses a TextureRoutingResult the
   * caller already computed instead of re-segmenting the source image.
   */
  match(source: ImageData, target: ImageData, runtimeOptions?: Record<string, any>): MatchSet {
    const routing: TextureRoutingResult =
      runtimeOptions?.tiles || TextureRouter.classifyImage(source, this.options);

    const priorH: number[][] | undefined = runtimeOptions?.groundTruthTransform;
    const matches: Match[] = [];
    let idCounter = 0;

    for (const tile of routing.tiles) {
      if (tile.mode !== 'low_texture') continue;

      const center: Point2D = { x: tile.centerX, y: tile.centerY };
      const priorCenter = priorH
        ? applyHomographyToPoint(priorH, center)
        : { x: center.x * (target.width / source.width), y: center.y * (target.height / source.height) };

      const best = this.searchBestOffset(source, target, center, priorCenter);
      if (best && best.score >= this.options.minCorrelation) {
        matches.push({
          id: `area_corr_${idCounter++}`,
          sourcePoint: center,
          targetPoint: best.point,
          confidence: Math.max(0, Math.min(1, best.score)),
          method: 'AreaCorrelation',
          methodProvenance: { photometricConsistency: best.score },
        });
      }
    }

    return {
      matches,
      sourceImageId: source.id,
      targetImageId: target.id,
      coordinateConvention: 'x=column, y=row',
      metadata: {
        tileSize: routing.tileSize,
        lowTextureTilesConsidered: routing.lowTextureCount,
        matchesFound: matches.length,
      },
    };
  }

  /**
   * Single NCC probe: correlates the patch around `templateCenter` (source) against a window in `target`
   * centered on `searchCenter`. Returns null if the patch or every candidate falls outside the image.
   */
  probe(
    source: ImageData,
    target: ImageData,
    templateCenter: Point2D,
    searchCenter: Point2D
  ): { point: Point2D; score: number } | null {
    return this.searchBestOffset(source, target, templateCenter, searchCenter);
  }

  /** Runs the texture classification without matching - used by the pipeline for diagnostics and to share work with `match`. */
  classify(source: ImageData): TextureRoutingResult {
    return TextureRouter.classifyImage(source, this.options);
  }

  private searchBestOffset(
    source: ImageData,
    target: ImageData,
    templateCenter: Point2D,
    searchCenter: Point2D
  ): { point: Point2D; score: number } | null {
    const r = this.options.patchRadius;
    const template = this.extractPatch(source, templateCenter, r);
    if (!template) return null;

    let bestScore = -Infinity;
    let bestPt: Point2D | null = null;

    const sr = this.options.searchRadiusPx;
    const stride = this.options.searchStridePx;

    for (let dy = -sr; dy <= sr; dy += stride) {
      for (let dx = -sr; dx <= sr; dx += stride) {
        const candidateCenter: Point2D = { x: Math.round(searchCenter.x + dx), y: Math.round(searchCenter.y + dy) };
        const candidatePatch = this.extractPatch(target, candidateCenter, r);
        if (!candidatePatch) continue;

        const score = AreaCorrelationMatcher.normalizedCrossCorrelation(template, candidatePatch);
        if (score > bestScore) {
          bestScore = score;
          bestPt = candidateCenter;
        }
      }
    }

    if (!bestPt) return null;
    return { point: bestPt, score: bestScore };
  }

  private extractPatch(image: ImageData, center: Point2D, radius: number): Float32Array | null {
    const cx = Math.round(center.x);
    const cy = Math.round(center.y);
    const size = 2 * radius + 1;

    if (cx - radius < 0 || cy - radius < 0 || cx + radius >= image.width || cy + radius >= image.height) {
      return null;
    }

    const patch = new Float32Array(size * size);
    let i = 0;
    for (let y = cy - radius; y <= cy + radius; y++) {
      for (let x = cx - radius; x <= cx + radius; x++) {
        patch[i++] = image.pixels[y * image.width + x];
      }
    }
    return patch;
  }

  /** Standard NCC in [-1, 1]; 1.0 = perfect linear match. */
  private static normalizedCrossCorrelation(a: Float32Array, b: Float32Array): number {
    const n = a.length;
    let meanA = 0, meanB = 0;
    for (let i = 0; i < n; i++) {
      meanA += a[i];
      meanB += b[i];
    }
    meanA /= n;
    meanB /= n;

    let num = 0, denA = 0, denB = 0;
    for (let i = 0; i < n; i++) {
      const da = a[i] - meanA;
      const db = b[i] - meanB;
      num += da * db;
      denA += da * da;
      denB += db * db;
    }

    const denom = Math.sqrt(denA * denB);
    if (denom < 1e-9) return 0; // both patches flat/constant - no reliable correlation
    return num / denom;
  }
}
