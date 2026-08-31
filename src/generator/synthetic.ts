/**
 * LunaMatch - Part 02: Deterministic Synthetic Lunar Data Generator
 * 
 * Generates synthetic lunar surfaces with realistic cratering, illumination physics,
 * sensor profiles (OHRC, TMC-2, IIRS), and exact ground-truth coordinate transformations.
 */

import { GroundTruthData, ImageData, Point2D, SensorMetadata, SensorType, TransformType } from '../types';

export interface SyntheticGenerationOptions {
  seed?: number;
  width?: number;
  height?: number;
  sourceSensor: SensorType;
  referenceSensor: SensorType;
  sunAzimuthDeg?: number;
  sunElevationDeg?: number;
  referenceSunAzimuthDeg?: number;
  referenceSunElevationDeg?: number;
  scale?: number;
  rotationDeg?: number;
  translationPx?: [number, number];
  shear?: [number, number];
  perspective?: [number, number]; // [hx, hy] projective tilt
  noiseLevel?: number;
  numCraters?: number;
  terrainType?: 'highlands' | 'maria' | 'polar_crater' | 'complex_impact';
}

/**
 * Deterministic Pseudo-Random Number Generator (LCG / Mulberry32)
 */
export class DeterministicRNG {
  private s: number;

  constructor(seed: number = 42) {
    this.s = Math.floor(seed) >>> 0;
  }

  next(): number {
    this.s = (this.s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(this.s ^ (this.s >>> 15), 1 | this.s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  range(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  intRange(min: number, max: number): number {
    return Math.floor(this.range(min, max + 1));
  }
}

/**
 * Procedural Lunar Terrain Crater Definition
 */
interface LunarCrater {
  cx: number;
  cy: number;
  radius: number;
  depth: number;
  rimHeight: number;
  hasCentralPeak: boolean;
}

/**
 * Generates 2D procedural lunar heightmap (DEM)
 */
export function generateLunarDEM(
  width: number,
  height: number,
  rng: DeterministicRNG,
  craters: LunarCrater[]
): Float32Array {
  const dem = new Float32Array(width * height);

  // 1. Base fractal multi-octave noise
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      // Synthesize low-frequency lunar undulating topography
      let val = 0;
      let freq = 0.008;
      let amp = 0.25;
      for (let oct = 0; oct < 4; oct++) {
        val += Math.sin(x * freq + rng.next() * 0.1) * Math.cos(y * freq + rng.next() * 0.1) * amp;
        freq *= 2.1;
        amp *= 0.5;
      }
      dem[idx] = 0.5 + val;
    }
  }

  // 2. Superimpose craters with parabolic bowl + raised rims
  for (const c of craters) {
    const minX = Math.max(0, Math.floor(c.cx - c.radius * 2.5));
    const maxX = Math.min(width - 1, Math.ceil(c.cx + c.radius * 2.5));
    const minY = Math.max(0, Math.floor(c.cy - c.radius * 2.5));
    const maxY = Math.min(height - 1, Math.ceil(c.cy + c.radius * 2.5));

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - c.cx;
        const dy = y - c.cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const rNorm = dist / c.radius;
        const idx = y * width + x;

        if (rNorm <= 1.0) {
          // Inside bowl: parabolic depression
          const depthFactor = c.depth * (1.0 - rNorm * rNorm);
          dem[idx] -= depthFactor;

          // Central peak for large impact craters
          if (c.hasCentralPeak && rNorm < 0.25) {
            const peakFactor = (1.0 - rNorm / 0.25) * (c.depth * 0.4);
            dem[idx] += peakFactor;
          }
        } else if (rNorm <= 2.2) {
          // Raised rim and decaying ejecta blanket
          const rimDist = Math.abs(rNorm - 1.0);
          const rimElev = c.rimHeight * Math.exp(-rimDist * 3.5);
          dem[idx] += rimElev;
        }
      }
    }
  }

  // Normalize DEM to [0, 1]
  let minH = Infinity;
  let maxH = -Infinity;
  for (let i = 0; i < dem.length; i++) {
    if (dem[i] < minH) minH = dem[i];
    if (dem[i] > maxH) maxH = dem[i];
  }
  const range = maxH - minH || 1;
  for (let i = 0; i < dem.length; i++) {
    dem[i] = (dem[i] - minH) / range;
  }

  return dem;
}

/**
 * Physics-informed Lunar Photometric Rendering
 * Uses combined Lambertian + Lommel-Seeliger lunar surface reflectance
 * and directional shadow raymarching.
 */
export function renderLunarShading(
  dem: Float32Array,
  width: number,
  height: number,
  sunAzimuthDeg: number,
  sunElevationDeg: number,
  albedoMap?: Float32Array
): Float32Array {
  const output = new Float32Array(width * height);
  const azRad = (sunAzimuthDeg * Math.PI) / 180;
  const elRad = (sunElevationDeg * Math.PI) / 180;

  // Sun unit vector [Lx, Ly, Lz]
  const Lx = Math.cos(elRad) * Math.sin(azRad);
  const Ly = -Math.cos(elRad) * Math.cos(azRad); // y downwards in image convention
  const Lz = Math.sin(elRad);

  const zScale = 45.0; // Terrain vertical exaggeration for slope calculation

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;

      // Surface normal via central difference
      const dzdx = ((dem[idx + 1] - dem[idx - 1]) * zScale) / 2.0;
      const dzdy = ((dem[idx + width] - dem[idx - width]) * zScale) / 2.0;

      const Nx = -dzdx;
      const Ny = -dzdy;
      const Nz = 1.0;
      const nLen = Math.hypot(Nx, Ny, Nz);

      const normX = Nx / nLen;
      const normY = Ny / nLen;
      const normZ = Nz / nLen;

      // Cosine of incidence angle i (dot product with sun vector)
      const cosI = Math.max(0, normX * Lx + normY * Ly + normZ * Lz);
      // Emission angle e (viewer is overhead in nadir viewing: [0, 0, 1])
      const cosE = Math.max(0.001, normZ);

      // Lommel-Seeliger lunar photometric formula: I = cos(i) / (cos(i) + cos(e))
      let reflectance = 0;
      if (cosI > 0) {
        const lommelSeeliger = cosI / (cosI + cosE);
        const lambert = cosI;
        reflectance = 0.7 * lommelSeeliger + 0.3 * lambert;
      }

      // Raymarch shadow casting
      let inShadow = false;
      const stepSize = 1.0;
      const rayLen = 40;
      let currX = x;
      let currY = y;
      let currZ = dem[idx] * zScale;

      const stepX = (Lx / Math.max(0.1, Lz)) * stepSize;
      const stepY = (Ly / Math.max(0.1, Lz)) * stepSize;
      const stepZ = stepSize;

      for (let s = 1; s < rayLen; s++) {
        currX += stepX;
        currY += stepY;
        currZ += stepZ;

        const ix = Math.floor(currX);
        const iy = Math.floor(currY);
        if (ix < 0 || ix >= width || iy < 0 || iy >= height) break;

        const sampleH = dem[iy * width + ix] * zScale;
        if (sampleH > currZ + 0.2) {
          inShadow = true;
          break;
        }
      }

      if (inShadow) {
        reflectance *= 0.08; // Ambient deep lunar shadow
      }

      const albedo = albedoMap ? albedoMap[idx] : 0.85;
      output[idx] = Math.min(1.0, Math.max(0.0, reflectance * albedo));
    }
  }

  return output;
}

/**
 * Forward 3x3 Coordinate Transformation on 2D Point
 * [x', y', w]^T = H * [x, y, 1]^T
 */
export function applyHomographyToPoint(H: number[][], pt: Point2D): Point2D {
  const x = pt.x;
  const y = pt.y;
  const xp = H[0][0] * x + H[0][1] * y + H[0][2];
  const yp = H[1][0] * x + H[1][1] * y + H[1][2];
  const wp = H[2][0] * x + H[2][1] * y + H[2][2];

  if (Math.abs(wp) < 1e-9) {
    return { x: xp, y: yp };
  }
  return { x: xp / wp, y: yp / wp };
}

/**
 * Invert 3x3 Matrix
 */
export function invert3x3(M: number[][]): number[][] {
  const [
    [a, b, c],
    [d, e, f],
    [g, h, k],
  ] = M;

  const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) {
    throw new Error('Singular matrix cannot be inverted');
  }
  const invDet = 1.0 / det;

  return [
    [(e * k - f * h) * invDet, (c * h - b * k) * invDet, (b * f - c * e) * invDet],
    [(f * g - d * k) * invDet, (a * k - c * g) * invDet, (c * d - a * f) * invDet],
    [(d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet],
  ];
}

/**
 * Creates 3x3 Composite Transformation Matrix
 */
export function createCompositeMatrix(
  cx: number,
  cy: number,
  scale: number,
  rotationDeg: number,
  tx: number,
  ty: number,
  shearX: number = 0,
  shearY: number = 0,
  projX: number = 0,
  projY: number = 0
): number[][] {
  const rad = (rotationDeg * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);

  // Translation to center -> Apply affine/proj -> Translate back + offset
  // T_post * H_core * T_pre
  const a11 = scale * (cos + shearY * -sin);
  const a12 = scale * (-sin + shearX * cos);
  const a21 = scale * (sin + shearY * cos);
  const a22 = scale * (cos + shearX * sin);

  // Shift center
  const m13 = tx + cx - (a11 * cx + a12 * cy);
  const m23 = ty + cy - (a21 * cx + a22 * cy);

  return [
    [a11, a12, m13],
    [a21, a22, m23],
    [projX, projY, 1.0],
  ];
}

/**
 * Synthesizes a multi-band IIRS Hyperspectral cube (250 bands)
 */
export function generateIIRSSpectralCube(
  baseIntensity: Float32Array,
  width: number,
  height: number,
  rng: DeterministicRNG
): { pixels: Float32Array; bands: number[]; channels: number } {
  const numBands = 250;
  const startWavelength = 800; // nm
  const endWavelength = 5000; // nm
  const wavelengthStep = (endWavelength - startWavelength) / (numBands - 1);
  const bands = Array.from({ length: numBands }, (_, i) => Math.round(startWavelength + i * wavelengthStep));

  // We store 3 primary preview channels + compact 250-band spectral profile per region
  const totalPixels = width * height;
  const previewPixels = new Float32Array(totalPixels * 3);

  // Spectral mineral absorption features for lunar pyroxene (1000nm & 2000nm band depth)
  for (let i = 0; i < totalPixels; i++) {
    const intensity = baseIntensity[i];
    // False color RGB for IIRS preview: Band 20 (950nm), Band 70 (2000nm), Band 120 (3000nm)
    const band1 = Math.min(1.0, intensity * 1.1); // 950nm
    const band2 = Math.min(1.0, intensity * 0.85 + 0.05); // 2000nm pyroxene absorption
    const band3 = Math.min(1.0, intensity * 0.95); // 3000nm hydroxyl feature

    previewPixels[i * 3 + 0] = band1;
    previewPixels[i * 3 + 1] = band2;
    previewPixels[i * 3 + 2] = band3;
  }

  return { pixels: previewPixels, bands, channels: 3 };
}

/**
 * Master Synthetic Pair Generator
 */
export function generateSyntheticLunarDataset(options: SyntheticGenerationOptions): {
  sourceImage: ImageData;
  referenceImage: ImageData;
  groundTruth: GroundTruthData;
  dem: Float32Array;
} {
  const {
    seed = 101,
    width = 384,
    height = 384,
    sourceSensor = 'OHRC',
    referenceSensor = 'TMC2',
    sunAzimuthDeg = 45,
    sunElevationDeg = 30,
    referenceSunAzimuthDeg = 225, // Extreme 180 deg opposite sun angle
    referenceSunElevationDeg = 25,
    scale = 1.2,
    rotationDeg = 18,
    translationPx = [15, -12],
    shear = [0.05, -0.02],
    perspective = [0.0002, -0.0001],
    noiseLevel = 0.02,
    numCraters = 18,
  } = options;

  const rng = new DeterministicRNG(seed);

  // 1. Create realistic crater distribution
  const craters: LunarCrater[] = [];
  // Large prominent impact crater
  craters.push({
    cx: width * 0.48,
    cy: height * 0.46,
    radius: width * 0.22,
    depth: 0.35,
    rimHeight: 0.12,
    hasCentralPeak: true,
  });
  // Medium and micro craters
  for (let i = 0; i < numCraters; i++) {
    craters.push({
      cx: rng.range(width * 0.1, width * 0.9),
      cy: rng.range(height * 0.1, height * 0.9),
      radius: rng.range(width * 0.03, width * 0.12),
      depth: rng.range(0.1, 0.25),
      rimHeight: rng.range(0.03, 0.08),
      hasCentralPeak: rng.next() > 0.7,
    });
  }

  // 2. Generate Heightmap DEM
  const dem = generateLunarDEM(width, height, rng, craters);

  // 3. Render Source Image with Source Sun Angle
  const sourceRaw = renderLunarShading(dem, width, height, sunAzimuthDeg, sunElevationDeg);

  // Apply sensor-specific characteristics (OHRC: ultra crisp, TMC2: slight blur, IIRS: spectral)
  const sourcePixels = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const n = (rng.next() - 0.5) * noiseLevel;
    sourcePixels[i] = Math.min(1.0, Math.max(0.0, sourceRaw[i] + n));
  }

  // 4. Compute Ground Truth Homography Matrix T
  const H_forward = createCompositeMatrix(
    width / 2,
    height / 2,
    scale,
    rotationDeg,
    translationPx[0],
    translationPx[1],
    shear[0],
    shear[1],
    perspective[0],
    perspective[1]
  );
  const H_inverse = invert3x3(H_forward);

  // 5. Render Reference Image with Reference Sun Angle & Warped Geometry
  const refShaded = renderLunarShading(
    dem,
    width,
    height,
    referenceSunAzimuthDeg,
    referenceSunElevationDeg
  );

  const refPixels = new Float32Array(width * height);
  const refMask = new Uint8Array(width * height);

  // Warp shaded scene via inverse mapping to create target/reference image
  for (let ty = 0; ty < height; ty++) {
    for (let tx = 0; tx < width; tx++) {
      const targetPt = { x: tx, y: ty };
      const srcPt = applyHomographyToPoint(H_inverse, targetPt);

      const tidx = ty * width + tx;
      if (srcPt.x >= 0 && srcPt.x < width - 1 && srcPt.y >= 0 && srcPt.y < height - 1) {
        // Bilinear interpolation from refShaded
        const x0 = Math.floor(srcPt.x);
        const y0 = Math.floor(srcPt.y);
        const x1 = x0 + 1;
        const y1 = y0 + 1;
        const fx = srcPt.x - x0;
        const fy = srcPt.y - y0;

        const v00 = refShaded[y0 * width + x0];
        const v10 = refShaded[y0 * width + x1];
        const v01 = refShaded[y1 * width + x0];
        const v11 = refShaded[y1 * width + x1];

        const val = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11;
        const n = (rng.next() - 0.5) * (noiseLevel * 1.2);
        refPixels[tidx] = Math.min(1.0, Math.max(0.0, val + n));
        refMask[tidx] = 1;
      } else {
        refPixels[tidx] = 0.05; // Blank / border area
        refMask[tidx] = 0;
      }
    }
  }

  // 6. Generate exact Ground Truth points grid
  const sourcePoints: Point2D[] = [];
  const targetPoints: Point2D[] = [];
  const gridStep = Math.floor(width / 7);

  for (let gy = gridStep; gy < height - gridStep; gy += gridStep) {
    for (let gx = gridStep; gx < width - gridStep; gx += gridStep) {
      // Add slight jitter for realistic feature placement
      const sx = gx + rng.range(-gridStep * 0.25, gridStep * 0.25);
      const sy = gy + rng.range(-gridStep * 0.25, gridStep * 0.25);
      const sPt: Point2D = { x: sx, y: sy };
      const tPt = applyHomographyToPoint(H_forward, sPt);

      if (tPt.x >= 10 && tPt.x <= width - 10 && tPt.y >= 10 && tPt.y <= height - 10) {
        sourcePoints.push(sPt);
        targetPoints.push(tPt);
      }
    }
  }

  const sourceMeta: SensorMetadata = {
    sensorId: sourceSensor,
    productId: `CH2_${sourceSensor}_20240315_001`,
    spatialResolutionMeters: sourceSensor === 'OHRC' ? 0.25 : sourceSensor === 'TMC2' ? 5.0 : 80.0,
    incidenceAngleDeg: 90 - sunElevationDeg,
    emissionAngleDeg: 0.0,
    phaseAngleDeg: Math.abs(sunAzimuthDeg),
    sunAzimuthDeg: sunAzimuthDeg,
    sunElevationDeg: sunElevationDeg,
    spacecraftAltitudeKm: 100.0,
    centerLatitude: -15.42,
    centerLongitude: 85.12,
  };

  const refMeta: SensorMetadata = {
    sensorId: referenceSensor,
    productId: `CH2_${referenceSensor}_20240520_042`,
    spatialResolutionMeters: referenceSensor === 'OHRC' ? 0.25 : referenceSensor === 'TMC2' ? 5.0 : 80.0,
    incidenceAngleDeg: 90 - referenceSunElevationDeg,
    emissionAngleDeg: 3.5,
    phaseAngleDeg: Math.abs(referenceSunAzimuthDeg),
    sunAzimuthDeg: referenceSunAzimuthDeg,
    sunElevationDeg: referenceSunElevationDeg,
    spacecraftAltitudeKm: 100.0,
    centerLatitude: -15.42,
    centerLongitude: 85.12,
  };

  const sourceImage: ImageData = {
    id: `img_source_${sourceSensor}`,
    pixels: sourcePixels,
    width,
    height,
    channels: 1,
    dtype: 'float32',
    sensorId: sourceSensor,
    metadata: sourceMeta,
    mask: new Uint8Array(width * height).fill(1),
  };

  const referenceImage: ImageData = {
    id: `img_ref_${referenceSensor}`,
    pixels: refPixels,
    width,
    height,
    channels: 1,
    dtype: 'float32',
    sensorId: referenceSensor,
    metadata: refMeta,
    mask: refMask,
  };

  const groundTruth: GroundTruthData = {
    sourcePoints,
    targetPoints,
    groundTruthTransform: H_forward,
    transformType: 'homography',
    knownScale: scale,
    knownRotationDeg: rotationDeg,
    knownTranslation: translationPx,
    knownIlluminationDeltaDeg: Math.abs(sunAzimuthDeg - referenceSunAzimuthDeg),
  };

  return { sourceImage, referenceImage, groundTruth, dem };
}
