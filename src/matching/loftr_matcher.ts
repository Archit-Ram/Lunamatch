/**
 * LunaMatch - Part 10: Real LoFTR Matcher
 * 
 * Deep learning correspondence matching using LoFTR (Sun et al., CVPR 2021)
 * Detector-free local feature matching with transformers.
 * 
 * Performs true neural inference on raw pixel intensities via ONNX runtime.
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

let loftrSessionCache: ort.InferenceSession | null = null;
let modelLoadPromise: Promise<ort.InferenceSession> | null = null;

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

  // List of fallback URLs for LoFTR
  const candidateUrls = [
    `/models/${modelName}`,
    'https://github.com/oooooha/loftr2onnx/releases/download/v1.0/loftr_outdoor.onnx',
    'https://huggingface.co/SpatialHub/efficient-loftr-onnx/resolve/main/eloftr_outdoor_opt.onnx',
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
 * Initializes and caches LoFTR inference session
 */
export async function getLoftrSession(): Promise<ort.InferenceSession> {
  if (loftrSessionCache) {
    return loftrSessionCache;
  }

  if (modelLoadPromise) {
    return modelLoadPromise;
  }

  modelLoadPromise = (async () => {
    try {
      const buffer = await loadModelBuffer('loftr_outdoor.onnx');

      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: ['wasm'],
      };

      const session = await ort.InferenceSession.create(buffer, sessionOptions);
      loftrSessionCache = session;
      return session;
    } catch (err: any) {
      modelLoadPromise = null;
      throw new Error(`Failed to initialize LoFTR ONNX model: ${err.message}`);
    }
  })();

  return modelLoadPromise;
}

/**
 * Resize and pad image for LoFTR (dimensions must be multiple of 8)
 */
function resizeImageForLoFTR(image: ImageData): { tensor: ort.Tensor; newWidth: number; newHeight: number } {
  // Calculate new dimensions that are multiples of 8
  const newWidth = Math.max(8, Math.round(image.width / 8) * 8);
  const newHeight = Math.max(8, Math.round(image.height / 8) * 8);

  const newPixels = new Float32Array(newWidth * newHeight);
  
  if (newWidth === image.width && newHeight === image.height) {
    newPixels.set(image.pixels);
  } else {
    // Bilinear interpolation
    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        const srcX = x * image.width / newWidth;
        const srcY = y * image.height / newHeight;
        
        const x0 = Math.floor(srcX);
        const y0 = Math.floor(srcY);
        const x1 = Math.min(x0 + 1, image.width - 1);
        const y1 = Math.min(y0 + 1, image.height - 1);
        
        const dx = srcX - x0;
        const dy = srcY - y0;

        const v00 = image.pixels[y0 * image.width + x0];
        const v01 = image.pixels[y0 * image.width + x1];
        const v10 = image.pixels[y1 * image.width + x0];
        const v11 = image.pixels[y1 * image.width + x1];

        newPixels[y * newWidth + x] = 
          v00 * (1 - dx) * (1 - dy) + 
          v01 * dx * (1 - dy) + 
          v10 * (1 - dx) * dy + 
          v11 * dx * dy;
      }
    }
  }

  // Input shape: [1, 1, H, W], float32 normalized [0, 1]
  const tensor = new ort.Tensor('float32', newPixels, [1, 1, newHeight, newWidth]);
  return { tensor, newWidth, newHeight };
}

/**
 * Real LoFTR Matcher using ONNX Runtime Web
 */
export class LoFTRMatcher implements Matcher {
  readonly name = 'LoFTR';
  private confidenceThreshold: number;

  constructor(confidenceThreshold: number = 0.2) {
    this.confidenceThreshold = confidenceThreshold;
  }

  /**
   * Executes real neural inference with LoFTR
   */
  async match(
    source: ImageData,
    target: ImageData,
    runtimeOptions?: Record<string, any>
  ): Promise<MatchSet> {
    const threshold = runtimeOptions?.confidenceThreshold ?? this.confidenceThreshold;

    let session: ort.InferenceSession;
    try {
      session = await getLoftrSession();
    } catch (e: any) {
      throw new Error(`ModelNotLoadedError: ${e.message}`);
    }

    // Resize images to multiple of 8
    const { tensor: tensor0, newWidth: w0, newHeight: h0 } = resizeImageForLoFTR(source);
    const { tensor: tensor1, newWidth: w1, newHeight: h1 } = resizeImageForLoFTR(target);

    // Run inference
    const outputs = await session.run({
      image0: tensor0,
      image1: tensor1
    });

    const keypoints0 = outputs.keypoints0.data as Float32Array;
    const keypoints1 = outputs.keypoints1.data as Float32Array;
    const confidence = outputs.confidence.data as Float32Array;

    const numMatches = confidence.length;
    const matches: Match[] = [];

    // Scale factors to map coordinates back to original image dimensions
    const scaleX0 = source.width / w0;
    const scaleY0 = source.height / h0;
    const scaleX1 = target.width / w1;
    const scaleY1 = target.height / h1;

    for (let i = 0; i < numMatches; i++) {
      const conf = confidence[i];
      
      if (conf >= threshold) {
        // Map keypoints back to original dimensions
        const x0 = Number(keypoints0[i * 2]) * scaleX0;
        const y0 = Number(keypoints0[i * 2 + 1]) * scaleY0;
        const x1 = Number(keypoints1[i * 2]) * scaleX1;
        const y1 = Number(keypoints1[i * 2 + 1]) * scaleY1;

        matches.push({
          id: `loftr_${source.id}_${target.id}_${i}`,
          sourcePoint: { x: x0, y: y0 },
          targetPoint: { x: x1, y: y1 },
          confidence: Math.min(1.0, Math.max(0.0, conf)),
          method: 'LoFTR',
          uncertaintyPx: Math.max(0.2, (1.0 - conf) * 3.0),
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
export const SimulatedLoFTRProfileMatcher = LoFTRMatcher;
