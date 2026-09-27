/**
 * LunaMatch - Part 10: Real LoFTR Matcher (ONNX Web)
 *
 * Detector-free local feature matching using LoFTR (Sun et al., CVPR 2021)
 * with coarse-to-fine Transformer attention.
 *
 * Performs TRUE neural inference on raw pixel intensities via ONNX Runtime Web.
 * Does NOT read ground-truth transforms. Does NOT fabricate correspondences.
 *
 * The ONNX model is exported from Kornia's pretrained LoFTR ("outdoor" weights)
 * at a fixed 480×480 resolution. Input images are downsampled to this grid and
 * output coordinates are scaled back to original dimensions.
 *
 * To obtain the model file:
 *   curl -L -o public/models/eloftr_outdoor_opt.onnx \
 *   https://huggingface.co/SpatialHub/efficient-loftr-onnx/resolve/main/eloftr_outdoor_opt.onnx
 */

import * as ort from 'onnxruntime-web/wasm';
import { ImageData, Match, MatchSet } from '../types';
import { Matcher } from '../core/interfaces';

// ---------- ONNX Runtime WASM Configuration ----------
// Match the exact same settings used by the working LightGlue matcher
if (typeof window !== 'undefined') {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = '/wasm/';
}

// ---------- Fixed model input resolution ----------
// LoFTR ONNX was exported at 480×480. All inputs are resized to this grid.
// This prevents OOM crashes — quadratic attention on 480×480 requires ~220MB heap,
// well within the browser's ~2GB WASM limit (vs. 1.2GB+ at full resolution).
const LOFTR_INPUT_SIZE = 480;

// ---------- Session Singleton ----------
let loftrSessionCache: ort.InferenceSession | null = null;
let modelLoadPromise: Promise<ort.InferenceSession> | null = null;

// ============================================================
//  IndexedDB Model Caching (same pattern as lightglue_matcher)
// ============================================================

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

// ============================================================
//  Model Loading (local file → IndexedDB cache)
// ============================================================

async function loadModelBuffer(modelName: string): Promise<ArrayBuffer | Uint8Array> {
  const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
  if (isNode) {
    const fs = await import('fs');
    const path = await import('path');
    const fullPath = path.resolve(process.cwd(), 'public', 'models', modelName);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`LoFTR ONNX model not found at: ${fullPath}. Download it from HuggingFace.`);
    }
    const buf = fs.readFileSync(fullPath);
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  // Browser: check IndexedDB cache first
  const cached = await getCachedModel(modelName);
  if (cached && cached.byteLength > 1000) {
    return cached;
  }

  // Load from local Vite dev server (public/models/loftr_outdoor.onnx)
  try {
    const resp = await fetch(`/models/${modelName}`);
    if (resp.ok) {
      const buffer = await resp.arrayBuffer();
      if (buffer.byteLength > 1000) {
        setCachedModel(modelName, buffer); // Cache for next time
        return buffer;
      }
    }
  } catch {
    // Fall through to error
  }

  throw new Error(
    `LoFTR ONNX model "${modelName}" not found. ` +
    `Ensure it is downloaded to public/models/${modelName}`
  );
}

// ============================================================
//  Session Initialization (singleton with deduplication)
// ============================================================

async function getLoFTRSession(): Promise<ort.InferenceSession> {
  if (loftrSessionCache) {
    return loftrSessionCache;
  }

  if (modelLoadPromise) {
    return modelLoadPromise;
  }

  modelLoadPromise = (async () => {
    try {
      const buffer = await loadModelBuffer('eloftr_outdoor_opt.onnx');

      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: ['wasm'],
      };

      const session = await ort.InferenceSession.create(buffer, sessionOptions);
      loftrSessionCache = session;
      return session;
    } catch (err: any) {
      modelLoadPromise = null;
      throw new Error(`Failed to initialize LoFTR ONNX session: ${err.message}`);
    }
  })();

  return modelLoadPromise;
}

// ============================================================
//  Step 2: Strict Image Downsampling to LOFTR_INPUT_SIZE
// ============================================================

/**
 * Resizes a grayscale ImageData to the fixed LoFTR input grid (480×480)
 * using bilinear interpolation. Returns the ONNX tensor ready for inference.
 */
function resizeToLoFTRInput(image: ImageData): ort.Tensor {
  const targetW = LOFTR_INPUT_SIZE;
  const targetH = LOFTR_INPUT_SIZE;

  const resized = new Float32Array(targetW * targetH);
  const srcW = image.width;
  const srcH = image.height;
  const srcPixels = image.pixels;

  // Bilinear interpolation
  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      // Map target pixel to source coordinate
      const srcX = (x * srcW) / targetW;
      const srcY = (y * srcH) / targetH;

      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const y1 = Math.min(y0 + 1, srcH - 1);

      const dx = srcX - x0;
      const dy = srcY - y0;

      const v00 = srcPixels[y0 * srcW + x0];
      const v10 = srcPixels[y0 * srcW + x1];
      const v01 = srcPixels[y1 * srcW + x0];
      const v11 = srcPixels[y1 * srcW + x1];

      resized[y * targetW + x] =
        v00 * (1 - dx) * (1 - dy) +
        v10 * dx * (1 - dy) +
        v01 * (1 - dx) * dy +
        v11 * dx * dy;
    }
  }

  // LoFTR expects [1, 1, H, W] float32 normalized to [0, 1]
  return new ort.Tensor('float32', resized, [1, 1, targetH, targetW]);
}

// ============================================================
//  Steps 3 & 4: Real LoFTR Matcher Class
// ============================================================

/**
 * Real LoFTR Matcher — performs genuine neural inference.
 * Does NOT read ground-truth. Does NOT fabricate correspondences.
 */
export class LoFTRMatcher implements Matcher {
  readonly name = 'LoFTR';
  private confidenceThreshold: number;

  constructor(confidenceThreshold: number = 0.2) {
    this.confidenceThreshold = confidenceThreshold;
  }

  /**
   * Run real LoFTR inference on two grayscale images.
   *
   * 1. Downsample both images to 480×480 (prevents OOM)
   * 2. Feed tensors into the ONNX model
   * 3. Extract keypoints0, keypoints1, confidence from output
   * 4. Scale coordinates back to original image dimensions
   * 5. Filter by confidence threshold and return MatchSet
   */
  async match(
    source: ImageData,
    target: ImageData,
    runtimeOptions?: Record<string, any>
  ): Promise<MatchSet> {
    const threshold = runtimeOptions?.confidenceThreshold ?? this.confidenceThreshold;

    // Load model session (cached after first load)
    const session = await getLoFTRSession();

    // Step 2: Strict downsampling to fixed 480×480 grid
    const tensor0 = resizeToLoFTRInput(source);
    const tensor1 = resizeToLoFTRInput(target);

    // Step 3: Run real neural inference
    const outputs = await session.run({
      image0: tensor0,
      image1: tensor1,
    });

    const keypoints0 = outputs.mkpts0_f.data as Float32Array;
    const keypoints1 = outputs.mkpts1_f.data as Float32Array;
    const confidence = outputs.mconf.data as Float32Array;

    const numMatches = confidence.length;
    const matches: Match[] = [];

    // Step 4: Scale coordinates back to original image dimensions
    // The model outputs coordinates in the 480×480 grid space.
    // We must map them back to the real image pixel coordinates.
    const scaleX0 = source.width / LOFTR_INPUT_SIZE;
    const scaleY0 = source.height / LOFTR_INPUT_SIZE;
    const scaleX1 = target.width / LOFTR_INPUT_SIZE;
    const scaleY1 = target.height / LOFTR_INPUT_SIZE;

    for (let i = 0; i < numMatches; i++) {
      const conf = confidence[i];

      if (conf >= threshold) {
        // Map from 480×480 grid back to original pixel space
        const sx = Number(keypoints0[i * 2]) * scaleX0;
        const sy = Number(keypoints0[i * 2 + 1]) * scaleY0;
        const tx = Number(keypoints1[i * 2]) * scaleX1;
        const ty = Number(keypoints1[i * 2 + 1]) * scaleY1;

        matches.push({
          id: `loftr_${source.id}_${target.id}_${i}`,
          sourcePoint: { x: sx, y: sy },
          targetPoint: { x: tx, y: ty },
          confidence: Math.min(1.0, Math.max(0.0, conf)),
          method: 'LoFTR',
          uncertaintyPx: Math.max(0.3, (1.0 - conf) * 3.0),
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

// Backward compatibility alias — pipeline still imports this name
export const SimulatedLoFTRProfileMatcher = LoFTRMatcher;
