/**
 * LunaMatch - Part 12: Real SuperPoint + LightGlue Matcher (ONNX Web / Node)
 * 
 * Deep learning correspondence matching using SuperPoint (DeTone et al., CVPRW 2018)
 * interest point detector & descriptor + LightGlue (Lindenberger et al., ICCV 2023)
 * graph transformer matcher.
 * 
 * Performs true neural inference on raw pixel intensities without reading ground-truth matrices.
 */

import * as ort from 'onnxruntime-web/wasm';
import { ImageData, Match, MatchSet, Point2D } from '../types';
import { Matcher } from '../core/interfaces';

// Configure ONNX Web runtime WASM paths for browser compatibility (single-threaded, non-JSEP)
if (typeof window !== 'undefined') {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = '/wasm/';
}

interface SuperPointResult {
  keypointsRaw: BigInt64Array | Int32Array | Float32Array;
  keypointsNormalized: Float32Array;
  keypointsDims: readonly number[];
  descriptorsTensor: ort.Tensor;
  numKeypoints: number;
}

let spSessionCache: ort.InferenceSession | null = null;
let lgSessionCache: ort.InferenceSession | null = null;
let modelLoadPromise: Promise<{ spSession: ort.InferenceSession; lgSession: ort.InferenceSession }> | null = null;

/**
 * IndexedDB storage for ONNX model buffers to enable instant cached loads
 */
async function getCachedModel(key: string): Promise<ArrayBuffer | null> {
  if (typeof window === 'undefined' || !window.indexedDB) return null;
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open('lunamatch-models-v1', 1);
      req.onupgradeneeded = () => {
        req.result.createObjectStore('models');
      };
      req.onsuccess = () => {
        const db = req.result;
        const tx = db.transaction('models', 'readonly');
        const store = tx.objectStore('models');
        const getReq = store.get(key);
        getReq.onsuccess = () => resolve(getReq.result || null);
        getReq.onerror = () => resolve(null);
      };
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function setCachedModel(key: string, data: ArrayBuffer): Promise<void> {
  if (typeof window === 'undefined' || !window.indexedDB) return;
  try {
    const req = indexedDB.open('lunamatch-models-v1', 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore('models');
    };
    req.onsuccess = () => {
      const db = req.result;
      const tx = db.transaction('models', 'readwrite');
      const store = tx.objectStore('models');
      store.put(data, key);
    };
  } catch {
    // Ignore IDB write failures
  }
}

/**
 * Loads an ONNX model from IndexedDB, file, local server, or fallback CDNs
 */
async function loadModelBuffer(modelName: string): Promise<ArrayBuffer | Uint8Array> {
  const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
  if (isNode) {
    const fs = await import('fs');
    const path = await import('path');
    const fullPath = path.resolve(process.cwd(), 'public', 'models', modelName);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`ONNX model file not found at: ${fullPath}`);
    }
    const buf = fs.readFileSync(fullPath);
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  // In Browser: check IndexedDB cache first
  const cached = await getCachedModel(modelName);
  if (cached && cached.byteLength > 1000) {
    return cached;
  }

  // List of fallback URLs
  const candidateUrls = [
    `/models/${modelName}`,
    modelName === 'superpoint.onnx'
      ? 'https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0/superpoint.onnx'
      : 'https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.3/superpoint_lightglue.onnx',
    `https://huggingface.co/fabio-sim/LightGlue-ONNX/resolve/main/${modelName}`,
  ];

  let lastError: Error | null = null;
  for (const url of candidateUrls) {
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const buffer = await resp.arrayBuffer();
        if (buffer.byteLength > 1000) {
          // Cache in background for subsequent loads
          setCachedModel(modelName, buffer);
          return buffer;
        }
      }
    } catch (err: any) {
      lastError = err;
    }
  }

  throw new Error(
    `Failed to fetch ONNX model ${modelName} from local server and CDN mirrors: ${lastError?.message || 'Network error'}`
  );
}

/**
 * Initializes and caches SuperPoint and LightGlue inference sessions
 */
export async function getLightGlueSessions(): Promise<{
  spSession: ort.InferenceSession;
  lgSession: ort.InferenceSession;
}> {
  if (spSessionCache && lgSessionCache) {
    return { spSession: spSessionCache, lgSession: lgSessionCache };
  }

  if (modelLoadPromise) {
    return modelLoadPromise;
  }

  modelLoadPromise = (async () => {
    try {
      const [spBuffer, lgBuffer] = await Promise.all([
        loadModelBuffer('superpoint.onnx'),
        loadModelBuffer('superpoint_lightglue.onnx'),
      ]);

      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: ['wasm'],
      };

      const [spSession, lgSession] = await Promise.all([
        ort.InferenceSession.create(spBuffer, sessionOptions),
        ort.InferenceSession.create(lgBuffer, sessionOptions),
      ]);

      spSessionCache = spSession;
      lgSessionCache = lgSession;
      return { spSession, lgSession };
    } catch (err: any) {
      modelLoadPromise = null;
      throw new Error(`Failed to initialize SuperPoint + LightGlue ONNX models: ${err.message}`);
    }
  })();

  return modelLoadPromise;
}

/**
 * Runs SuperPoint feature extractor on a single grayscale image
 */
async function extractSuperPointFeatures(
  session: ort.InferenceSession,
  image: ImageData
): Promise<SuperPointResult> {
  const { width, height, pixels } = image;

  // Input shape: [1, 1, H, W], float32 normalized [0, 1]
  const inputTensor = new ort.Tensor('float32', pixels, [1, 1, height, width]);
  const outputs = await session.run({ image: inputTensor });

  const kptsRaw = outputs.keypoints.data as BigInt64Array | Int32Array | Float32Array;
  const numKeypoints = outputs.keypoints.dims[1];

  // Normalize keypoints for LightGlue input:
  // shift = [width / 2, height / 2]
  // scale = max(width, height) / 2
  // k_norm = (k_pixel - shift) / scale
  const shiftX = width / 2;
  const shiftY = height / 2;
  const scale = Math.max(width, height) / 2;

  const kptsNorm = new Float32Array(numKeypoints * 2);
  for (let i = 0; i < numKeypoints; i++) {
    const rawX = Number(kptsRaw[i * 2]);
    const rawY = Number(kptsRaw[i * 2 + 1]);
    kptsNorm[i * 2] = (rawX - shiftX) / scale;
    kptsNorm[i * 2 + 1] = (rawY - shiftY) / scale;
  }

  return {
    keypointsRaw: kptsRaw,
    keypointsNormalized: kptsNorm,
    keypointsDims: outputs.keypoints.dims,
    descriptorsTensor: outputs.descriptors,
    numKeypoints,
  };
}

/**
 * Real SuperPoint + LightGlue Matcher
 */
export class LightGlueMatcher implements Matcher {
  readonly name = 'LightGlue';
  private confidenceThreshold: number;

  constructor(confidenceThreshold: number = 0.1) {
    this.confidenceThreshold = confidenceThreshold;
  }

  /**
   * Executes real neural inference with SuperPoint + LightGlue
   */
  async match(
    source: ImageData,
    target: ImageData,
    runtimeOptions?: Record<string, any>
  ): Promise<MatchSet> {
    const threshold = runtimeOptions?.confidenceThreshold ?? this.confidenceThreshold;

    const { spSession, lgSession } = await getLightGlueSessions();

    // Extract SuperPoint keypoints and visual descriptors on both images
    const [feat0, feat1] = await Promise.all([
      extractSuperPointFeatures(spSession, source),
      extractSuperPointFeatures(spSession, target),
    ]);

    if (feat0.numKeypoints === 0 || feat1.numKeypoints === 0) {
      return {
        matches: [],
        sourceImageId: source.id,
        targetImageId: target.id,
        coordinateConvention: 'x=column, y=row',
      };
    }

    // Prepare tensors for LightGlue
    const kpts0Tensor = new ort.Tensor('float32', feat0.keypointsNormalized, feat0.keypointsDims);
    const kpts1Tensor = new ort.Tensor('float32', feat1.keypointsNormalized, feat1.keypointsDims);

    const lgOutputs = await lgSession.run({
      kpts0: kpts0Tensor,
      kpts1: kpts1Tensor,
      desc0: feat0.descriptorsTensor,
      desc1: feat1.descriptorsTensor,
    });

    const matches0 = lgOutputs.matches0.data as BigInt64Array | Int32Array;
    const mscores0 = lgOutputs.mscores0.data as Float32Array;

    const matches: Match[] = [];

    for (let i = 0; i < feat0.numKeypoints; i++) {
      const matchIdx = Number(matches0[i]);
      const score = mscores0[i];

      // Check for valid correspondence
      if (matchIdx >= 0 && matchIdx < feat1.numKeypoints && score >= threshold) {
        const sx = Number(feat0.keypointsRaw[i * 2]);
        const sy = Number(feat0.keypointsRaw[i * 2 + 1]);
        const tx = Number(feat1.keypointsRaw[matchIdx * 2]);
        const ty = Number(feat1.keypointsRaw[matchIdx * 2 + 1]);

        matches.push({
          id: `lg_${source.id}_${target.id}_${i}_${matchIdx}`,
          sourcePoint: { x: sx, y: sy },
          targetPoint: { x: tx, y: ty },
          confidence: Math.min(1.0, Math.max(0.0, score)),
          method: 'LightGlue',
          uncertaintyPx: Math.max(0.2, (1.0 - score) * 2.5),
        });
      }
    }

    return {
      matches,
      sourceImageId: source.id,
      targetImageId: target.id,
      coordinateConvention: 'x=column, y=row',
    };
  }
}

// Backward compatibility alias for transition
export const SimulatedLightGlueProfileMatcher = LightGlueMatcher;
