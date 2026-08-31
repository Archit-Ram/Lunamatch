/**
 * LunaMatch - IIRS Hyperspectral Profile Inspector (Part 06)
 * 
 * Interactive 250-band reflectance spectrum explorer (800nm - 5000nm),
 * showing continuum removal, pyroxene/olivine band depths, and spectral slope.
 */

import React, { useState } from 'react';
import { IIRSSpectralProcessor, IIRSSpectralProfile } from '../spectral/iirs';
import { Activity, Layers, Sparkles } from 'lucide-react';

interface SpectralInspectorProps {
  imageWidth: number;
  imageHeight: number;
}

export const SpectralInspector: React.FC<SpectralInspectorProps> = ({ imageWidth, imageHeight }) => {
  const [selectedCoord, setSelectedCoord] = useState<{ x: number; y: number }>({
    x: Math.floor(imageWidth / 2),
    y: Math.floor(imageHeight / 2),
  });
  const [showContinuumRemoved, setShowContinuumRemoved] = useState<boolean>(false);

  const profile: IIRSSpectralProfile = IIRSSpectralProcessor.extractPixelSpectrum(
    0.65,
    selectedCoord.x,
    selectedCoord.y,
    imageWidth,
    imageHeight
  );

  // SVG Chart Dimensions
  const chartW = 600;
  const chartH = 220;
  const padL = 45;
  const padR = 25;
  const padT = 20;
  const padB = 35;

  const innerW = chartW - padL - padR;
  const innerH = chartH - padT - padB;

  const yData = showContinuumRemoved ? profile.continuumRemoved : profile.reflectance;
  const minY = Math.min(...yData) * 0.9;
  const maxY = Math.max(...yData) * 1.05;

  // Build SVG path
  const points = profile.wavelengths.map((w, idx) => {
    const x = padL + ((w - 800) / (5000 - 800)) * innerW;
    const y = padT + innerH - ((yData[idx] - minY) / (maxY - minY || 1)) * innerH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const pathD = `M ${points.join(' L ')}`;

  return (
    <div className="flex flex-col h-full bg-slate-950 p-4 select-none">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-bold text-slate-100">
            IIRS Hyperspectral 250-Band Mineral Profile
          </h2>
          <span className="text-[11px] font-mono px-2 py-0.5 bg-blue-950/80 border border-blue-800/40 text-blue-300 rounded">
            SWIR 800nm – 5000nm
          </span>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 cursor-pointer text-xs text-slate-300">
            <input
              type="checkbox"
              checked={showContinuumRemoved}
              onChange={(e) => setShowContinuumRemoved(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-blue-500 focus:ring-0"
            />
            <span>Continuum-Removed Ratio</span>
          </label>
        </div>
      </div>

      {/* Main Spectrum SVG Visualizer */}
      <div className="flex-1 flex flex-col items-center justify-center p-4 bg-slate-900/50 rounded-lg border border-slate-800 mt-3">
        <div className="w-full max-w-2xl bg-slate-950 p-4 rounded-lg border border-slate-800/80 shadow-lg">
          <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-auto">
            {/* Grid Lines */}
            {[1000, 2000, 3000, 4000].map((wl) => {
              const x = padL + ((wl - 800) / (5000 - 800)) * innerW;
              return (
                <g key={wl}>
                  <line
                    x1={x}
                    y1={padT}
                    x2={x}
                    y2={padT + innerH}
                    stroke="rgba(51, 65, 85, 0.4)"
                    strokeDasharray="3,3"
                  />
                  <text x={x} y={chartH - 12} fill="#64748b" fontSize="10" textAnchor="middle" fontFamily="monospace">
                    {wl}nm
                  </text>
                </g>
              );
            })}

            {/* Mineral Absorption Feature Highlights */}
            {/* 1um Pyroxene/Olivine band */}
            <rect
              x={padL + ((900 - 800) / 4200) * innerW}
              y={padT}
              width={((1100 - 900) / 4200) * innerW}
              height={innerH}
              fill="rgba(56, 189, 248, 0.08)"
            />
            {/* 2um Pyroxene band */}
            <rect
              x={padL + ((1800 - 800) / 4200) * innerW}
              y={padT}
              width={((2200 - 1800) / 4200) * innerW}
              height={innerH}
              fill="rgba(168, 85, 247, 0.08)"
            />

            {/* Spectrum Curve */}
            <path d={pathD} fill="none" stroke="#38bdf8" strokeWidth="2.5" />

            {/* Labels */}
            <text
              x={padL + ((1000 - 800) / 4200) * innerW}
              y={padT + 15}
              fill="#38bdf8"
              fontSize="9"
              fontWeight="bold"
              textAnchor="middle"
            >
              1.0µm Absorption (Pyroxene)
            </text>
            <text
              x={padL + ((2000 - 800) / 4200) * innerW}
              y={padT + 15}
              fill="#c084fc"
              fontSize="9"
              fontWeight="bold"
              textAnchor="middle"
            >
              2.0µm Absorption (Pyroxene)
            </text>
          </svg>

          {/* Key Mineral Diagnostics */}
          <div className="grid grid-cols-3 gap-3 mt-3 pt-3 border-t border-slate-800 text-xs">
            <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
              <span className="text-[10px] text-slate-400 block">1µm Band Depth:</span>
              <span className="font-mono font-bold text-cyan-300">
                {(profile.pyroxeneBandDepth1um * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
              <span className="text-[10px] text-slate-400 block">2µm Band Depth:</span>
              <span className="font-mono font-bold text-purple-300">
                {(profile.pyroxeneBandDepth2um * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
              <span className="text-[10px] text-slate-400 block">Spectral Slope:</span>
              <span className="font-mono font-bold text-emerald-300">
                {profile.spectralSlope > 0 ? '+Reddened' : 'Neutral'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
