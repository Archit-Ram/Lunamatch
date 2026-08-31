/**
 * LunaMatch - Part 11: Real RIFT Matcher (Radiation-invariant Feature Transform)
 * 
 * Classical multi-modal feature matching algorithm based on Li et al. (IEEE TGRS 2020).
 * Completely analytical and deterministic — no neural network weights needed.
 * 
 * Pipeline:
 * 1. Multi-scale & multi-orientation Log-Gabor filter bank via 2D FFT.
 * 2. Phase Congruency energy & structural Maximum Moment map computation.
 * 3. Maximum Index Map (MIM) construction capturing dominant boundary orientations.
 * 4. Local maxima keypoint detection on Phase Congruency map with spatial NMS.
 * 5. Dominant orientation assignment and rotation-resilient MIM patch histogram descriptors.
 * 6. Mutual nearest-neighbor matching with Lowe's ratio test.
 * 
 * Operates purely on source.pixels and target.pixels with ZERO reliance on ground-truth data.
 */

import { ImageData, Match, MatchSet, Point2D } from '../types';
import { Matcher } from '../core/interfaces';

/**
 * Fast zero-allocation 2D FFT Workspace
 */
export class FFTWorkspace {
  readonly size: number;
  readonly bitRev: Int32Array;
  readonly cosTable: Float64Array;
  readonly sinTable: Float64Array;

  constructor(size: number) {
    this.size = size;
    this.bitRev = new Int32Array(size);
    let j = 0;
    for (let i = 0; i < size - 1; i++) {
      if (i < j) {
        this.bitRev[i] = j;
        this.bitRev[j] = i;
      } else if (this.bitRev[i] === 0) {
        this.bitRev[i] = i;
      }
      let k = size >> 1;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }
    this.bitRev[size - 1] = size - 1;

    this.cosTable = new Float64Array(size);
    this.sinTable = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const angle = (-2.0 * Math.PI * i) / size;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }
  }

  fft1d(real: Float64Array, imag: Float64Array, offset: number, stride: number, inverse: boolean): void {
    const n = this.size;
    for (let i = 0; i < n; i++) {
      const target = this.bitRev[i];
      if (i < target) {
        const idx1 = offset + i * stride;
        const idx2 = offset + target * stride;
        const tr = real[idx1];
        real[idx1] = real[idx2];
        real[idx2] = tr;
        const ti = imag[idx1];
        imag[idx1] = imag[idx2];
        imag[idx2] = ti;
      }
    }

    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const step = n / len;

      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const tableIdx = (k * step) % n;
          const wR = this.cosTable[tableIdx];
          const wI = inverse ? -this.sinTable[tableIdx] : this.sinTable[tableIdx];

          const pos = offset + (i + k) * stride;
          const posHalf = offset + (i + k + half) * stride;

          const uR = real[pos];
          const uI = imag[pos];
          const vR = real[posHalf] * wR - imag[posHalf] * wI;
          const vI = real[posHalf] * wI + imag[posHalf] * wR;

          real[pos] = uR + vR;
          imag[pos] = uI + vI;
          real[posHalf] = uR - vR;
          imag[posHalf] = uI - vI;
        }
      }
    }

    if (inverse) {
      for (let i = 0; i < n; i++) {
        const idx = offset + i * stride;
        real[idx] /= n;
        imag[idx] /= n;
      }
    }
  }

  fft2d(real: Float64Array, imag: Float64Array, inverse: boolean): void {
    const n = this.size;
    for (let y = 0; y < n; y++) {
      this.fft1d(real, imag, y * n, 1, inverse);
    }
    for (let x = 0; x < n; x++) {
      this.fft1d(real, imag, x, n, inverse);
    }
  }
}

export function nextPowerOf2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export interface RIFTFeatureMaps {
  phaseCongruencyMoments: Float64Array;
  mimMap: Uint8Array;
  orientationAmps: Float64Array[];
  width: number;
  height: number;
}

export interface RIFTKeypoint extends Point2D {
  angleRad: number;
  score: number;
}

/**
 * Computes Phase Congruency and Maximum Index Map (MIM) using Log-Gabor filter bank
 */
export function computeRIFTFeatureMaps(
  image: ImageData,
  numScales = 3,
  numOrientations = 6
): RIFTFeatureMaps {
  const { width: origW, height: origH, pixels } = image;
  const N = nextPowerOf2(Math.max(origW, origH));
  const ws = new FFTWorkspace(N);

  // Prepare padded spatial real image
  const imgReal = new Float64Array(N * N);
  const imgImag = new Float64Array(N * N);
  for (let y = 0; y < origH; y++) {
    for (let x = 0; x < origW; x++) {
      imgReal[y * N + x] = pixels[y * origW + x];
    }
  }

  // Forward 2D FFT
  ws.fft2d(imgReal, imgImag, false);

  // Frequency grid
  const radius = new Float64Array(N * N);
  const theta = new Float64Array(N * N);
  for (let y = 0; y < N; y++) {
    const v = y < N / 2 ? y / N : (y - N) / N;
    for (let x = 0; x < N; x++) {
      const u = x < N / 2 ? x / N : (x - N) / N;
      const r = Math.sqrt(u * u + v * v);
      radius[y * N + x] = r;
      theta[y * N + x] = Math.atan2(v, u);
    }
  }

  const minWaveLength = 3.0;
  const mult = 2.0;
  const sigmaF = 0.55;
  const sigmaTheta = (Math.PI / numOrientations) * 0.65;

  const orientationAmps: Float64Array[] = Array.from({ length: numOrientations }, () => new Float64Array(origW * origH));
  const phaseCongruencyMoments = new Float64Array(origW * origH);
  const mimMap = new Uint8Array(origW * origH);

  const sumEven = Array.from({ length: numOrientations }, () => new Float64Array(origW * origH));
  const sumOdd = Array.from({ length: numOrientations }, () => new Float64Array(origW * origH));
  const sumAmp = Array.from({ length: numOrientations }, () => new Float64Array(origW * origH));

  const filtR = new Float64Array(N * N);
  const filtI = new Float64Array(N * N);

  for (let s = 0; s < numScales; s++) {
    const wavelength = minWaveLength * Math.pow(mult, s);
    const fo = 1.0 / wavelength;

    for (let o = 0; o < numOrientations; o++) {
      const thetaO = (o * Math.PI) / numOrientations;

      for (let idx = 0; idx < N * N; idx++) {
        const r = radius[idx];
        if (r < 1e-5) {
          filtR[idx] = 0;
          filtI[idx] = 0;
          continue;
        }

        const radVal = Math.exp(-Math.pow(Math.log(r / fo), 2) / (2 * Math.pow(Math.log(sigmaF), 2)));
        
        let dTheta = Math.abs(theta[idx] - thetaO);
        while (dTheta > Math.PI) dTheta -= Math.PI;
        if (dTheta > Math.PI / 2) dTheta = Math.PI - dTheta;

        const angVal = Math.exp(-Math.pow(dTheta, 2) / (2 * Math.pow(sigmaTheta, 2)));
        const lgVal = radVal * angVal;

        filtR[idx] = imgReal[idx] * lgVal;
        filtI[idx] = imgImag[idx] * lgVal;
      }

      ws.fft2d(filtR, filtI, true);

      for (let y = 0; y < origH; y++) {
        for (let x = 0; x < origW; x++) {
          const spatialIdx = y * origW + x;
          const fftIdx = y * N + x;
          const e = filtR[fftIdx];
          const odd = filtI[fftIdx];
          const a = Math.sqrt(e * e + odd * odd);

          sumEven[o][spatialIdx] += e;
          sumOdd[o][spatialIdx] += odd;
          sumAmp[o][spatialIdx] += a;
          orientationAmps[o][spatialIdx] += a;
        }
      }
    }
  }

  // Phase Congruency & Kovesi Maximum Moment calculation
  for (let idx = 0; idx < origW * origH; idx++) {
    let a_moment = 0;
    let b_moment = 0;
    let c_moment = 0;

    let maxAmpVal = -1;
    let maxOriIdx = 0;

    for (let o = 0; o < numOrientations; o++) {
      const thetaO = (o * Math.PI) / numOrientations;
      const e = sumEven[o][idx];
      const odd = sumOdd[o][idx];
      const energy = Math.sqrt(e * e + odd * odd);
      const amp = sumAmp[o][idx];

      const pc = amp > 1e-4 ? Math.max(0, energy - 0.04 * amp) / (amp + 0.001) : 0;
      const cosT = Math.cos(thetaO);
      const sinT = Math.sin(thetaO);

      a_moment += Math.pow(pc * cosT, 2);
      b_moment += 2 * (pc * cosT) * (pc * sinT);
      c_moment += Math.pow(pc * sinT, 2);

      const totalAmpForOri = orientationAmps[o][idx];
      if (totalAmpForOri > maxAmpVal) {
        maxAmpVal = totalAmpForOri;
        maxOriIdx = o;
      }
    }

    const maxMoment = 0.5 * (a_moment + c_moment + Math.sqrt(b_moment * b_moment + Math.pow(a_moment - c_moment, 2)));
    phaseCongruencyMoments[idx] = maxMoment;
    mimMap[idx] = maxOriIdx;
  }

  return {
    phaseCongruencyMoments,
    mimMap,
    orientationAmps,
    width: origW,
    height: origH,
  };
}

/**
 * Detects keypoints from Phase Congruency Maximum Moment map with Non-Maximum Suppression
 */
export function detectRIFTKeypoints(
  moments: Float64Array,
  width: number,
  height: number,
  orientationAmps: Float64Array[],
  maxPoints = 200,
  minDistance = 6
): RIFTKeypoint[] {
  const margin = 18;
  const candidates: { x: number; y: number; score: number }[] = [];

  // Local maxima in 3x3 window
  for (let y = margin; y < height - margin; y++) {
    for (let x = margin; x < width - margin; x++) {
      const score = moments[y * width + x];
      if (score < 0.002) continue;

      let isMax = true;
      for (let dy = -1; dy <= 1 && isMax; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (moments[(y + dy) * width + (x + dx)] > score) {
            isMax = false;
            break;
          }
        }
      }

      if (isMax) {
        candidates.push({ x, y, score });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);

  const keypoints: RIFTKeypoint[] = [];
  const minSqDist = minDistance * minDistance;
  const numOri = orientationAmps.length;

  for (const cand of candidates) {
    let tooClose = false;
    for (const kp of keypoints) {
      const dx = cand.x - kp.x;
      const dy = cand.y - kp.y;
      if (dx * dx + dy * dy < minSqDist) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) {
      // Calculate dominant orientation from circular neighborhood
      let sumSin = 0;
      let sumCos = 0;
      const radius = 8;
      for (let dy = -radius; dy <= radius; dy++) {
        const py = cand.y + dy;
        if (py < 0 || py >= height) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const px = cand.x + dx;
          if (px < 0 || px >= width) continue;
          if (dx * dx + dy * dy > radius * radius) continue;

          const pIdx = py * width + px;
          for (let o = 0; o < numOri; o++) {
            const angle = (o * Math.PI) / numOri;
            const amp = orientationAmps[o][pIdx];
            sumCos += amp * Math.cos(2 * angle);
            sumSin += amp * Math.sin(2 * angle);
          }
        }
      }
      const angleRad = 0.5 * Math.atan2(sumSin, sumCos);

      keypoints.push({ x: cand.x, y: cand.y, angleRad, score: cand.score });
      if (keypoints.length >= maxPoints) break;
    }
  }

  return keypoints;
}

/**
 * Builds rotation-invariant Maximum Index Map (MIM) descriptors
 */
export function extractMIMDescriptors(
  keypoints: RIFTKeypoint[],
  mimMap: Uint8Array,
  orientationAmps: Float64Array[],
  width: number,
  height: number,
  patchRadius = 18,
  gridCells = 4,
  numOrientations = 6
): Float32Array[] {
  const descriptors: Float32Array[] = [];
  const descDim = gridCells * gridCells * numOrientations;
  const cellSize = (patchRadius * 2) / gridCells;
  const sigma = patchRadius * 0.5;

  for (const kp of keypoints) {
    const desc = new Float32Array(descDim);
    const cosA = Math.cos(kp.angleRad);
    const sinA = Math.sin(kp.angleRad);

    for (let cy = 0; cy < gridCells; cy++) {
      for (let cx = 0; cx < gridCells; cx++) {
        const cellIdx = cy * gridCells + cx;

        const cellCenterX = -patchRadius + (cx + 0.5) * cellSize;
        const cellCenterY = -patchRadius + (cy + 0.5) * cellSize;

        for (let dy = -cellSize / 2; dy <= cellSize / 2; dy += 1.5) {
          for (let dx = -cellSize / 2; dx <= cellSize / 2; dx += 1.5) {
            const localX = cellCenterX + dx;
            const localY = cellCenterY + dy;

            const imgX = Math.round(kp.x + localX * cosA - localY * sinA);
            const imgY = Math.round(kp.y + localX * sinA + localY * cosA);

            if (imgX < 0 || imgX >= width || imgY < 0 || imgY >= height) continue;

            const pIdx = imgY * width + imgX;
            const rawOri = mimMap[pIdx];
            const oriAngle = (rawOri * Math.PI) / numOrientations;
            let relAngle = oriAngle - kp.angleRad;
            while (relAngle < 0) relAngle += Math.PI;
            while (relAngle >= Math.PI) relAngle -= Math.PI;

            const relOriBin = Math.min(numOrientations - 1, Math.floor((relAngle / Math.PI) * numOrientations));

            const distSq = localX * localX + localY * localY;
            const weight = Math.exp(-distSq / (2 * sigma * sigma));
            const amp = orientationAmps[rawOri][pIdx];

            desc[cellIdx * numOrientations + relOriBin] += weight * (amp + 0.05);
          }
        }
      }
    }

    // L2 Normalize
    let norm = 0;
    for (let i = 0; i < descDim; i++) norm += desc[i] * desc[i];
    norm = Math.sqrt(norm) || 1e-6;
    for (let i = 0; i < descDim; i++) desc[i] /= norm;

    // Contrast clipping at 0.2
    let clippedNorm = 0;
    for (let i = 0; i < descDim; i++) {
      if (desc[i] > 0.2) desc[i] = 0.2;
      clippedNorm += desc[i] * desc[i];
    }
    clippedNorm = Math.sqrt(clippedNorm) || 1e-6;
    for (let i = 0; i < descDim; i++) desc[i] /= clippedNorm;

    descriptors.push(desc);
  }

  return descriptors;
}

/**
 * Matches MIM descriptors with mutual consistency check and Lowe's ratio test
 */
export function matchRIFTDescriptors(
  srcKps: RIFTKeypoint[],
  srcDescs: Float32Array[],
  tgtKps: RIFTKeypoint[],
  tgtDescs: Float32Array[],
  ratioThreshold = 0.92
): Match[] {
  if (srcDescs.length === 0 || tgtDescs.length === 0) return [];

  const dim = srcDescs[0].length;
  const forwardMatches: { tgtIdx: number; dist: number; ratio: number }[] = [];

  for (let i = 0; i < srcDescs.length; i++) {
    const sDesc = srcDescs[i];
    let bestDist = Infinity;
    let secondDist = Infinity;
    let bestIdx = -1;

    for (let j = 0; j < tgtDescs.length; j++) {
      const tDesc = tgtDescs[j];
      let d = 0;
      for (let k = 0; k < dim; k++) {
        const diff = sDesc[k] - tDesc[k];
        d += diff * diff;
      }
      d = Math.sqrt(d);

      if (d < bestDist) {
        secondDist = bestDist;
        bestDist = d;
        bestIdx = j;
      } else if (d < secondDist) {
        secondDist = d;
      }
    }

    const ratio = bestDist / (secondDist + 1e-6);
    forwardMatches.push({ tgtIdx: bestIdx, dist: bestDist, ratio });
  }

  const backwardBest = new Int32Array(tgtDescs.length).fill(-1);
  for (let j = 0; j < tgtDescs.length; j++) {
    const tDesc = tgtDescs[j];
    let bestDist = Infinity;
    let bestIdx = -1;

    for (let i = 0; i < srcDescs.length; i++) {
      const sDesc = srcDescs[i];
      let d = 0;
      for (let k = 0; k < dim; k++) {
        const diff = sDesc[k] - tDesc[k];
        d += diff * diff;
      }
      d = Math.sqrt(d);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    backwardBest[j] = bestIdx;
  }

  const matches: Match[] = [];
  for (let i = 0; i < srcDescs.length; i++) {
    const fm = forwardMatches[i];
    if (fm.ratio <= ratioThreshold && fm.tgtIdx !== -1) {
      if (backwardBest[fm.tgtIdx] === i) {
        const confidence = Math.max(0.1, Math.min(1.0, 1.0 - fm.ratio + 0.25));
        matches.push({
          id: `rift_${matches.length}`,
          sourcePoint: { x: srcKps[i].x, y: srcKps[i].y },
          targetPoint: { x: tgtKps[fm.tgtIdx].x, y: tgtKps[fm.tgtIdx].y },
          confidence,
          method: 'RIFT',
          uncertaintyPx: 0.4 + fm.ratio * 0.4,
          isInlier: true,
        });
      }
    }
  }

  return matches;
}

/**
 * Real Radiation-invariant Feature Transform (RIFT) Matcher
 */
export class RIFTMatcher implements Matcher {
  readonly name = 'RIFT';
  private confidenceThreshold: number;
  private maxPoints: number;
  private ratioThreshold: number;

  constructor(options?: { confidenceThreshold?: number; maxPoints?: number; ratioThreshold?: number }) {
    this.confidenceThreshold = options?.confidenceThreshold ?? 0.35;
    this.maxPoints = options?.maxPoints ?? 200;
    this.ratioThreshold = options?.ratioThreshold ?? 0.92;
  }

  match(source: ImageData, target: ImageData, runtimeOptions?: Record<string, any>): MatchSet {
    const maxPts = runtimeOptions?.numMatches || this.maxPoints;
    const ratioThresh = runtimeOptions?.ratioThreshold || this.ratioThreshold;

    // 1. Compute Phase Congruency & MIM Maps for both images
    const srcMaps = computeRIFTFeatureMaps(source);
    const tgtMaps = computeRIFTFeatureMaps(target);

    // 2. Detect keypoints
    const srcKps = detectRIFTKeypoints(srcMaps.phaseCongruencyMoments, srcMaps.width, srcMaps.height, srcMaps.orientationAmps, maxPts);
    const tgtKps = detectRIFTKeypoints(tgtMaps.phaseCongruencyMoments, tgtMaps.width, tgtMaps.height, tgtMaps.orientationAmps, maxPts);

    // 3. Extract MIM descriptors
    const srcDescs = extractMIMDescriptors(srcKps, srcMaps.mimMap, srcMaps.orientationAmps, srcMaps.width, srcMaps.height);
    const tgtDescs = extractMIMDescriptors(tgtKps, tgtMaps.mimMap, tgtMaps.orientationAmps, tgtMaps.width, tgtMaps.height);

    // 4. Match descriptors with ratio test & mutual consistency
    const rawMatches = matchRIFTDescriptors(srcKps, srcDescs, tgtKps, tgtDescs, ratioThresh);

    // Filter by confidence threshold
    const matches = rawMatches.filter(m => m.confidence >= this.confidenceThreshold);

    return {
      matches,
      sourceImageId: source.id,
      targetImageId: target.id,
      coordinateConvention: 'x=column, y=row',
    };
  }
}


