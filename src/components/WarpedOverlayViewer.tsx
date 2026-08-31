/**
 * LunaMatch Warped & Registered Overlay Viewer
 * 
 * Provides interactive registration verification modes:
 * - Alpha Blend / Opacity slider
 * - Difference image (revealing exact pixel alignment residuals)
 * - Checkerboard interleaving
 * - Side-by-Side comparison
 */

import React, { useRef, useEffect, useState } from 'react';
import { ImageData, TransformModel } from '../types';
import { Sliders, Grid, Split, Eye, Sparkles } from 'lucide-react';

interface WarpedOverlayViewerProps {
  referenceImage: ImageData;
  registeredSourceImage: ImageData;
  transform: TransformModel;
}

export const WarpedOverlayViewer: React.FC<WarpedOverlayViewerProps> = ({
  referenceImage,
  registeredSourceImage,
  transform,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [blendMode, setBlendMode] = useState<'alpha' | 'difference' | 'checkerboard'>('alpha');
  const [opacity, setOpacity] = useState<number>(0.5);
  const [checkerSize, setCheckerSize] = useState<number>(32);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = referenceImage.width;
    const h = referenceImage.height;
    canvas.width = w;
    canvas.height = h;

    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    const refPix = referenceImage.pixels;
    const regPix = registeredSourceImage.pixels;
    const regMask = registeredSourceImage.mask;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        const dIdx = idx * 4;

        const refV = refPix[idx];
        const regV = regPix[idx];
        const isValid = regMask ? regMask[idx] > 0 : true;

        if (blendMode === 'alpha') {
          // Alpha blend: (1 - a) * Ref + a * Reg
          const blended = isValid ? (1 - opacity) * refV + opacity * regV : refV;
          const byteVal = Math.min(255, Math.max(0, Math.floor(blended * 255)));

          data[dIdx + 0] = byteVal;
          data[dIdx + 1] = isValid ? byteVal : Math.floor(byteVal * 0.7); // slight tint if out of bounds
          data[dIdx + 2] = byteVal;
          data[dIdx + 3] = 255;
        } else if (blendMode === 'difference') {
          // Absolute difference map |Ref - Reg|
          const diff = isValid ? Math.abs(refV - regV) * 3.5 : 0; // enhanced difference
          const byteVal = Math.min(255, Math.floor(diff * 255));

          // Color map: zero error is deep black/blue, high error is bright yellow/red
          data[dIdx + 0] = byteVal;
          data[dIdx + 1] = Math.floor(byteVal * 0.8);
          data[dIdx + 2] = Math.floor(byteVal * 0.2);
          data[dIdx + 3] = 255;
        } else if (blendMode === 'checkerboard') {
          // Checkerboard interleaving pattern
          const checkX = Math.floor(x / checkerSize) % 2;
          const checkY = Math.floor(y / checkerSize) % 2;
          const useReg = (checkX ^ checkY) === 1;

          const chosenVal = useReg && isValid ? regV : refV;
          const byteVal = Math.min(255, Math.max(0, Math.floor(chosenVal * 255)));

          data[dIdx + 0] = byteVal;
          data[dIdx + 1] = byteVal;
          data[dIdx + 2] = byteVal;
          data[dIdx + 3] = 255;
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [referenceImage, registeredSourceImage, blendMode, opacity, checkerSize]);

  return (
    <div className="flex flex-col h-full bg-slate-950 p-4">
      {/* Control Bar */}
      <div className="flex items-center justify-between pb-3 text-xs border-b border-slate-800 flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setBlendMode('alpha')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium text-xs border transition ${
              blendMode === 'alpha'
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Alpha Blend</span>
          </button>

          <button
            onClick={() => setBlendMode('difference')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium text-xs border transition ${
              blendMode === 'difference'
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Difference Map</span>
          </button>

          <button
            onClick={() => setBlendMode('checkerboard')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium text-xs border transition ${
              blendMode === 'checkerboard'
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800'
            }`}
          >
            <Grid className="w-3.5 h-3.5" />
            <span>Checkerboard</span>
          </button>
        </div>

        {/* Dynamic Controls based on blend mode */}
        {blendMode === 'alpha' && (
          <div className="flex items-center gap-3 bg-slate-900 px-3 py-1 rounded-md border border-slate-800">
            <span className="text-[11px] text-slate-400">Reference</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-32 accent-blue-500 cursor-pointer"
            />
            <span className="text-[11px] text-slate-400">Registered Source ({(opacity * 100).toFixed(0)}%)</span>
          </div>
        )}

        {blendMode === 'checkerboard' && (
          <div className="flex items-center gap-3 bg-slate-900 px-3 py-1 rounded-md border border-slate-800">
            <span className="text-[11px] text-slate-400">Grid Tile Size:</span>
            <input
              type="range"
              min="12"
              max="64"
              step="4"
              value={checkerSize}
              onChange={(e) => setCheckerSize(parseInt(e.target.value))}
              className="w-24 accent-blue-500 cursor-pointer"
            />
            <span className="text-[11px] font-mono text-slate-200">{checkerSize}px</span>
          </div>
        )}
      </div>

      {/* Main Viewport */}
      <div className="flex-1 flex items-center justify-center p-4 bg-slate-900/50 rounded-lg border border-slate-800 mt-3 relative overflow-auto">
        <canvas ref={canvasRef} className="shadow-2xl rounded border border-slate-800 max-w-full" />
      </div>
    </div>
  );
};
