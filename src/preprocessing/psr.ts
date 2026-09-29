/**
 * LunaMatch - Part 26: Permanently Shadowed Region (PSR) Mask
 *
 * True PSR interiors receive no direct sunlight, so their pixels sit at the sensor noise floor and carry no
 * illumination-independent structure to register against. Matching there only produces confident-looking
 * garbage, so such tiles are detected and explicitly skipped, flagged OUT_OF_SCOPE_PSR.
 *
 * A dark tile is not automatically a PSR: cast shadows are dark too, but they move with the sun. Two rules
 * separate them:
 *  - Tile level (before matching): a tile is dark when its 95th-percentile valid-pixel intensity is below a
 *    threshold. Nodata pixels (mask = 0) are ignored and never count as shadow.
 *  - Match level (after matching): when the two images have different illumination (sun azimuth differs by at
 *    least `minSunDiversityDeg`), a correspondence is treated as PSR only if BOTH endpoints are dark, i.e. the
 *    same ground point stays dark under different sun geometry. With similar illumination that test cannot
 *    tell shadow from PSR, so the source-side dark tile alone is used and the result is flagged.
 */

import { ImageData, Match, Point2D } from '../types';

export const PSR_FLAG = 'OUT_OF_SCOPE_PSR';

export interface PSRTile {
  row: number;
  col: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  p95: number;
  validFraction: number;
  isDark: boolean;
}

export interface PSRMask {
  tileSize: number;
  gridCols: number;
  gridRows: number;
  tiles: PSRTile[];
  darkCount: number;
  darkFraction: number; // dark tiles / tiles that contain valid data
}

export interface PSROptions {
  tileSize?: number;
  darkThreshold?: number; // p95 intensity below this -> dark tile (images in [0,1])
  minValidFraction?: number; // tiles with less valid data than this are nodata, not shadow
  minSunDiversityDeg?: number;
}

const DEFAULTS: Required<PSROptions> = { tileSize: 32, darkThreshold: 0.06, minValidFraction: 0.5, minSunDiversityDeg: 30 };

export class PSRDetector {
  static classify(image: ImageData, options: PSROptions = {}): PSRMask {
    const o = { ...DEFAULTS, ...options };
    const { width, height, pixels, mask } = image;
    const gridCols = Math.max(1, Math.ceil(width / o.tileSize));
    const gridRows = Math.max(1, Math.ceil(height / o.tileSize));
    const tiles: PSRTile[] = [];
    let darkCount = 0;
    let validTiles = 0;

    for (let row = 0; row < gridRows; row++)
      for (let col = 0; col < gridCols; col++) {
        const x0 = col * o.tileSize, y0 = row * o.tileSize;
        const x1 = Math.min(width, x0 + o.tileSize), y1 = Math.min(height, y0 + o.tileSize);
        const vals: number[] = [];
        for (let y = y0; y < y1; y++)
          for (let x = x0; x < x1; x++) {
            const i = y * width + x;
            if (mask && mask[i] === 0) continue;
            vals.push(pixels[i]);
          }
        const total = (x1 - x0) * (y1 - y0);
        const validFraction = total ? vals.length / total : 0;
        let p95 = 1;
        if (vals.length) {
          vals.sort((a, b) => a - b);
          p95 = vals[Math.min(vals.length - 1, Math.floor(0.95 * (vals.length - 1)))];
        }
        const hasData = validFraction >= o.minValidFraction;
        const isDark = hasData && p95 < o.darkThreshold;
        if (hasData) validTiles++;
        if (isDark) darkCount++;
        tiles.push({ row, col, x0, y0, x1, y1, p95, validFraction, isDark });
      }

    return { tileSize: o.tileSize, gridCols, gridRows, tiles, darkCount, darkFraction: validTiles ? darkCount / validTiles : 0 };
  }

  static isDarkAt(mask: PSRMask, p: Point2D): boolean {
    const col = Math.floor(p.x / mask.tileSize);
    const row = Math.floor(p.y / mask.tileSize);
    if (col < 0 || row < 0 || col >= mask.gridCols || row >= mask.gridRows) return false;
    return mask.tiles[row * mask.gridCols + col].isDark;
  }

  /** Circular absolute difference between two sun azimuths, in degrees [0, 180]. */
  static sunDiversityDeg(a: ImageData, b: ImageData): number {
    const d = Math.abs(((a.metadata.sunAzimuthDeg - b.metadata.sunAzimuthDeg) % 360 + 360) % 360);
    return d > 180 ? 360 - d : d;
  }

  /** Splits matches into kept vs PSR-rejected using the two-level rule described above. */
  static filterMatches(
    matches: Match[],
    srcMask: PSRMask,
    refMask: PSRMask,
    diversityDeg: number,
    options: PSROptions = {}
  ): { kept: Match[]; rejected: Match[]; confirmedByDiversity: boolean } {
    const minDiv = options.minSunDiversityDeg ?? DEFAULTS.minSunDiversityDeg;
    const confirmed = diversityDeg >= minDiv;
    const kept: Match[] = [];
    const rejected: Match[] = [];
    for (const m of matches) {
      const srcDark = PSRDetector.isDarkAt(srcMask, m.sourcePoint);
      const refDark = PSRDetector.isDarkAt(refMask, m.targetPoint);
      const isPsr = confirmed ? srcDark && refDark : srcDark;
      (isPsr ? rejected : kept).push(m);
    }
    return { kept, rejected, confirmedByDiversity: confirmed };
  }
}
