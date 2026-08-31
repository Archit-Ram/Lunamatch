/**
 * LunaMatch Dual Image Canvas Viewer
 * 
 * Interactive dual viewport rendering source and reference lunar images with:
 * - Direct vector tie-point correspondence lines
 * - Inlier (Green) vs Outlier (Red) distinction
 * - Confidence color gradation
 * - Residual error quiver vectors
 * - Uncertainty covariance ellipses
 * - Interactive hover inspector
 */

import React, { useRef, useEffect, useState } from 'react';
import { ImageData, Match, MatchSet, Point2D, TransformModel } from '../types';
import { ViewMode } from './ViewModeSelector';
import { applyHomographyToPoint } from '../generator/synthetic';
import { Crosshair, Info, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

interface DualImageViewerProps {
  sourceImage: ImageData;
  referenceImage: ImageData;
  normSourceImage?: ImageData;
  normRefImage?: ImageData;
  matches?: MatchSet;
  transform?: TransformModel;
  viewMode: ViewMode;
  onPointClick?: (point: Point2D, isSource: boolean) => void;
}

export const DualImageViewer: React.FC<DualImageViewerProps> = ({
  sourceImage,
  referenceImage,
  normSourceImage,
  normRefImage,
  matches,
  transform,
  viewMode,
  onPointClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredMatch, setHoveredMatch] = useState<Match | null>(null);
  const [zoom, setZoom] = useState<number>(1.0);
  const [showOnlyInliers, setShowOnlyInliers] = useState<boolean>(false);

  // Render pipeline onto composite canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const pad = 24; // Padding between source and reference viewport
    const w = sourceImage.width;
    const h = sourceImage.height;

    // Total composite dimensions
    canvas.width = w * 2 + pad;
    canvas.height = h;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Determine active image buffers based on view mode
    const isNormalized = viewMode === 'normalized';
    const srcImgToDraw = isNormalized && normSourceImage ? normSourceImage : sourceImage;
    const refImgToDraw = isNormalized && normRefImage ? normRefImage : referenceImage;

    // Draw Source Image (Left)
    drawGrayscaleToCanvas(ctx, srcImgToDraw, 0, 0);

    // Draw Reference Image (Right)
    drawGrayscaleToCanvas(ctx, refImgToDraw, w + pad, 0);

    // Draw Separator Line & Axis Labels
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w + pad / 2, 0);
    ctx.lineTo(w + pad / 2, h);
    ctx.stroke();

    // 2. Render Matches / Vectors if in appropriate view mode
    const matchItems = matches?.matches || [];
    const H = transform?.matrix;

    if (viewMode === 'matches' || viewMode === 'raw' || viewMode === 'normalized') {
      for (const m of matchItems) {
        if (showOnlyInliers && !m.isInlier) continue;

        const sx = m.sourcePoint.x;
        const sy = m.sourcePoint.y;
        const tx = m.targetPoint.x + w + pad;
        const ty = m.targetPoint.y;

        const isHovered = hoveredMatch?.id === m.id;
        const isInlier = m.isInlier !== false;

        // Draw Line connecting pair
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(tx, ty);

        if (isHovered) {
          ctx.strokeStyle = '#38bdf8'; // Cyan
          ctx.lineWidth = 2.5;
        } else if (isInlier) {
          // Green inlier line with confidence opacity
          ctx.strokeStyle = `rgba(52, 211, 153, ${Math.max(0.4, m.confidence)})`;
          ctx.lineWidth = 1.2;
        } else {
          // Red outlier line
          ctx.strokeStyle = 'rgba(248, 113, 113, 0.45)';
          ctx.lineWidth = 1.0;
        }
        ctx.stroke();

        // Draw Source Endpoint
        ctx.beginPath();
        ctx.arc(sx, sy, isHovered ? 4.5 : 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = isInlier ? '#10b981' : '#ef4444';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 0.75;
        ctx.stroke();

        // Draw Target Endpoint
        ctx.beginPath();
        ctx.arc(tx, ty, isHovered ? 4.5 : 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = isInlier ? '#10b981' : '#ef4444';
        ctx.fill();
        ctx.stroke();
      }
    }

    // 3. Render Residual Error Vectors (Quiver Plot)
    if (viewMode === 'residual' && H) {
      for (const m of matchItems) {
        if (!m.isInlier) continue;
        const pred = applyHomographyToPoint(H, m.sourcePoint);
        const tx = m.targetPoint.x + w + pad;
        const ty = m.targetPoint.y;
        const predX = pred.x + w + pad;
        const predY = pred.y;

        // Magnify residual for visual clarity
        const magFactor = 6.0;
        const dx = (predX - tx) * magFactor;
        const dy = (predY - ty) * magFactor;

        // Draw target point
        ctx.beginPath();
        ctx.arc(tx, ty, 3, 0, 2 * Math.PI);
        ctx.fillStyle = '#38bdf8';
        ctx.fill();

        // Draw error arrow vector
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(tx + dx, ty + dy);
        ctx.strokeStyle = '#f59e0b'; // Amber
        ctx.lineWidth = 1.75;
        ctx.stroke();

        // Arrow head
        ctx.beginPath();
        ctx.arc(tx + dx, ty + dy, 2, 0, 2 * Math.PI);
        ctx.fillStyle = '#ef4444';
        ctx.fill();
      }
    }

    // 4. Render Uncertainty Covariance Ellipses
    if (viewMode === 'uncertainty' && H) {
      for (const m of matchItems) {
        if (!m.isInlier) continue;
        const tx = m.targetPoint.x + w + pad;
        const ty = m.targetPoint.y;
        const unc = Math.max(1.5, (m.uncertaintyPx || 1.0) * 4.0);

        ctx.save();
        ctx.translate(tx, ty);
        ctx.beginPath();
        ctx.ellipse(0, 0, unc * 1.5, unc * 0.9, Math.PI / 4, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.18)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
        ctx.lineWidth = 1.2;
        ctx.stroke();
        ctx.restore();

        // Center dot
        ctx.beginPath();
        ctx.arc(tx, ty, 2, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    }
  }, [sourceImage, referenceImage, normSourceImage, normRefImage, matches, transform, viewMode, hoveredMatch, showOnlyInliers]);

  function drawGrayscaleToCanvas(ctx: CanvasRenderingContext2D, img: ImageData, offsetX: number, offsetY: number) {
    const { width, height, pixels, channels } = img;
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pIdx = y * width + x;
        const dIdx = pIdx * 4;

        if (channels === 3) {
          // False color RGB (IIRS)
          data[dIdx + 0] = Math.min(255, Math.floor(pixels[pIdx * 3 + 0] * 255));
          data[dIdx + 1] = Math.min(255, Math.floor(pixels[pIdx * 3 + 1] * 255));
          data[dIdx + 2] = Math.min(255, Math.floor(pixels[pIdx * 3 + 2] * 255));
          data[dIdx + 3] = 255;
        } else {
          // Monochrome
          const v = Math.min(255, Math.max(0, Math.floor(pixels[pIdx] * 255)));
          data[dIdx + 0] = v;
          data[dIdx + 1] = v;
          data[dIdx + 2] = v;
          data[dIdx + 3] = 255;
        }
      }
    }
    ctx.putImageData(imgData, offsetX, offsetY);
  }

  // Handle canvas mouse move to inspect closest tie-point match
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !matches) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    const pad = 24;
    const w = sourceImage.width;

    let closest: Match | null = null;
    let minDist = 15.0; // px threshold

    for (const m of matches.matches) {
      const dSrc = Math.hypot(m.sourcePoint.x - mouseX, m.sourcePoint.y - mouseY);
      const dRef = Math.hypot(m.targetPoint.x + w + pad - mouseX, m.targetPoint.y - mouseY);

      if (dSrc < minDist) {
        minDist = dSrc;
        closest = m;
      } else if (dRef < minDist) {
        minDist = dRef;
        closest = m;
      }
    }

    setHoveredMatch(closest);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !onPointClick) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    const pad = 24;
    const w = sourceImage.width;

    if (mouseX <= w) {
      onPointClick({ x: mouseX, y: mouseY }, true);
    } else if (mouseX >= w + pad) {
      onPointClick({ x: mouseX - (w + pad), y: mouseY }, false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 p-4 select-none">
      {/* Top Canvas Viewport Header */}
      <div className="flex items-center justify-between pb-3 text-xs text-slate-300 border-b border-slate-800/80 mb-3 flex-wrap gap-2">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
            <span className="font-semibold text-slate-200">SOURCE: {sourceImage.sensorId}</span>
            <span className="text-[11px] text-slate-400 font-mono">
              ({sourceImage.metadata.spatialResolutionMeters}m/px, Sun: {sourceImage.metadata.sunAzimuthDeg}° Az, {sourceImage.metadata.sunElevationDeg}° El)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-400"></span>
            <span className="font-semibold text-slate-200">REFERENCE: {referenceImage.sensorId}</span>
            <span className="text-[11px] text-slate-400 font-mono">
              ({referenceImage.metadata.spatialResolutionMeters}m/px, Sun: {referenceImage.metadata.sunAzimuthDeg}° Az, {referenceImage.metadata.sunElevationDeg}° El)
            </span>
          </div>
        </div>

        {/* Inlier Filter Toggle & Zoom */}
        <div className="flex items-center gap-3">
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950/80 border border-amber-500/50 text-amber-300">
            SIMULATED PROFILE
          </span>

          <label className="flex items-center gap-1.5 cursor-pointer text-[11px] text-slate-300">
            <input
              type="checkbox"
              checked={showOnlyInliers}
              onChange={(e) => setShowOnlyInliers(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0"
            />
            <span>Show Inliers Only</span>
          </label>

          <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded px-1 py-0.5">
            <button
              onClick={() => setZoom((z) => Math.min(2.0, z + 0.15))}
              className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <span className="text-[11px] font-mono px-1">{(zoom * 100).toFixed(0)}%</span>
            <button
              onClick={() => setZoom((z) => Math.max(0.7, z - 0.15))}
              className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoom(1.0)}
              className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white ml-1 border-l border-slate-800"
              title="Reset Zoom"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Canvas Scroll Area */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto flex items-center justify-center bg-slate-900/50 rounded-lg border border-slate-800 p-2 relative shadow-inner"
      >
        <div style={{ transform: `scale(${zoom})`, transformOrigin: 'center center', transition: 'transform 0.1s ease-out' }}>
          <canvas
            ref={canvasRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHoveredMatch(null)}
            onClick={handleCanvasClick}
            className="cursor-crosshair shadow-2xl rounded border border-slate-800 max-w-none"
          />
        </div>

        {/* Hover Inspector Tooltip */}
        {hoveredMatch && (
          <div className="absolute bottom-4 left-4 bg-slate-900/95 border border-cyan-500/50 text-white px-3.5 py-2.5 rounded-lg shadow-xl text-xs backdrop-blur-md font-mono z-20 pointer-events-none max-w-sm">
            <div className="flex items-center justify-between border-b border-slate-800 pb-1.5 mb-1.5">
              <span className="font-bold text-cyan-400 flex items-center gap-1">
                <Crosshair className="w-3.5 h-3.5" />
                Match #{hoveredMatch.id}
              </span>
              <span className={`px-1.5 py-0.2 rounded text-[10px] font-semibold ${hoveredMatch.isInlier ? 'bg-emerald-950 text-emerald-300 border border-emerald-700' : 'bg-red-950 text-red-300 border border-red-700'}`}>
                {hoveredMatch.isInlier ? 'INLIER' : 'OUTLIER'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
              <div>
                <span className="text-slate-400">Source: </span>
                <span className="text-slate-200 font-semibold">({hoveredMatch.sourcePoint.x.toFixed(2)}, {hoveredMatch.sourcePoint.y.toFixed(2)})</span>
              </div>
              <div>
                <span className="text-slate-400">Target: </span>
                <span className="text-slate-200 font-semibold">({hoveredMatch.targetPoint.x.toFixed(2)}, {hoveredMatch.targetPoint.y.toFixed(2)})</span>
              </div>
              <div>
                <span className="text-slate-400">Confidence: </span>
                <span className="text-emerald-400 font-bold">{(hoveredMatch.confidence * 100).toFixed(1)}%</span>
              </div>
              <div>
                <span className="text-slate-400">1-σ Uncertainty: </span>
                <span className="text-amber-300 font-semibold">{hoveredMatch.uncertaintyPx?.toFixed(2) || '0.35'} px</span>
              </div>
              <div className="col-span-2 text-slate-400 text-[10px] pt-1 border-t border-slate-800/80">
                Provenance: {hoveredMatch.method} (Simulated Profile) {hoveredMatch.methodProvenance ? `• Fusion: [LoFTR: ${(hoveredMatch.methodProvenance.loftrConfidence || 0).toFixed(2)}, RIFT: ${(hoveredMatch.methodProvenance.riftConfidence || 0).toFixed(2)}]` : ''}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
