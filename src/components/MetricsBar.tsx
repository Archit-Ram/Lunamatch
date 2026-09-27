/**
 * LunaMatch Scientific Metrics Bar
 * Conforming strictly to Part 23 specification (pages 55, 57)
 */

import React from 'react';
import { EvaluationMetrics, TransformModel } from '../types';
import { Target, CheckCircle2, AlertTriangle, Crosshair, Cpu, Gauge, Clock } from 'lucide-react';

interface MetricsBarProps {
  metrics?: EvaluationMetrics;
  transform?: TransformModel;
  status: 'idle' | 'success' | 'failure';
  failureReason?: string;
  sourceSensor: string;
  referenceSensor: string;
}

export const MetricsBar: React.FC<MetricsBarProps> = ({
  metrics,
  transform,
  status,
  failureReason,
  sourceSensor,
  referenceSensor,
}) => {
  const isFailed = status === 'failure';
  const rmse = metrics?.rmsePx ?? 0;
  const inliers = metrics?.inlierCount ?? 0;
  const inlierRatio = metrics ? (metrics.inlierRatio * 100).toFixed(1) : '0.0';
  const coverage = metrics ? (metrics.uniformityScore * 100).toFixed(0) : '0';
  const runtime = metrics?.runtimeMs ?? 0;
  const transformType = transform?.modelType ? transform.modelType.toUpperCase() : 'HOMOGRAPHY';

  // Determine RMSE status color badge (Sub-pixel goal: < 0.5px)
  const isSubpixel = rmse < 0.5 && status === 'success';

  return (
    <div className="bg-slate-900/95 border-y border-slate-800 px-5 py-2.5 shadow-inner">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
        {/* Status Indicator */}
        <div className="flex items-center gap-2">
          {isFailed ? (
            <div className="flex items-center gap-1.5 px-2.5 py-1 bg-red-950/80 border border-red-800/60 rounded text-xs font-semibold text-red-400">
              <AlertTriangle className="w-4 h-4 text-red-400" />
              <span>REGISTRATION FAILED: {failureReason || 'INSUFFICIENT_CONSISTENCY'}</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 flex-wrap">
              <div className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-950/80 border border-emerald-800/60 rounded text-xs font-semibold text-emerald-400">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>ALIGNED: {sourceSensor} ↔ {referenceSensor}</span>
              </div>
              {isSubpixel && (
                <span className="text-[11px] font-bold px-2 py-0.5 bg-cyan-950 border border-cyan-700/50 text-cyan-300 rounded">
                  SUB-PIXEL ACCURACY
                </span>
              )}
            </div>
          )}
        </div>

        {/* 4 Main Scientific Metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 sm:gap-6 items-center">
          {/* 1. RMSE */}
          <div className="flex flex-col">
            <span className="text-[11px] font-medium tracking-wider text-slate-400 uppercase flex items-center gap-1">
              <Crosshair className="w-3 h-3 text-cyan-400" />
              RMSE
            </span>
            <div className="flex items-baseline gap-1">
              <span className={`text-base font-mono font-bold ${isFailed ? 'text-slate-500' : isSubpixel ? 'text-cyan-300' : 'text-slate-100'}`}>
                {isFailed ? '—' : `${rmse.toFixed(2)}`}
              </span>
              <span className="text-[11px] text-slate-400 font-sans">px</span>
            </div>
          </div>

          {/* 2. Inliers */}
          <div className="flex flex-col">
            <span className="text-[11px] font-medium tracking-wider text-slate-400 uppercase flex items-center gap-1">
              <Target className="w-3 h-3 text-emerald-400" />
              INLIERS
            </span>
            <span className={`text-base font-mono font-bold ${isFailed ? 'text-slate-500' : 'text-emerald-400'}`}>
              {isFailed ? '0' : inliers}
            </span>
          </div>

          {/* 3. Inlier Ratio */}
          <div className="flex flex-col">
            <span className="text-[11px] font-medium tracking-wider text-slate-400 uppercase flex items-center gap-1">
              <Gauge className="w-3 h-3 text-indigo-400" />
              INLIER RATIO
            </span>
            <span className={`text-base font-mono font-bold ${isFailed ? 'text-slate-500' : 'text-indigo-300'}`}>
              {isFailed ? '0%' : `${inlierRatio}%`}
            </span>
          </div>

          {/* 4. Spatial Coverage / Uniformity */}
          <div className="flex flex-col">
            <span className="text-[11px] font-medium tracking-wider text-slate-400 uppercase flex items-center gap-1">
              <Cpu className="w-3 h-3 text-amber-400" />
              COVERAGE
            </span>
            <span className={`text-base font-mono font-bold ${isFailed ? 'text-slate-500' : 'text-amber-300'}`}>
              {isFailed ? '0%' : `${coverage}%`}
            </span>
          </div>

          {/* 5. Transform & Runtime */}
          <div className="flex flex-col">
            <span className="text-[11px] font-medium tracking-wider text-slate-400 uppercase flex items-center gap-1">
              <Clock className="w-3 h-3 text-purple-400" />
              MODEL / LATENCY
            </span>
            <div className="flex items-center gap-1.5 text-xs text-slate-300 font-mono">
              <span className="font-semibold text-purple-300">{transformType}</span>
              <span className="text-slate-500">•</span>
              <span>{runtime.toFixed(0)}ms</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
