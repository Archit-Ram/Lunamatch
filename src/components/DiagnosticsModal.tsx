/**
 * LunaMatch Diagnostics & Telemetry Modal
 */

import React from 'react';
import { DiagnosticInfo, TransformModel } from '../types';
import { Terminal, X, Clock, CheckCircle2, AlertCircle } from 'lucide-react';

interface DiagnosticsModalProps {
  isOpen: boolean;
  onClose: () => void;
  diagnostics?: DiagnosticInfo;
  transform?: TransformModel;
}

export const DiagnosticsModal: React.FC<DiagnosticsModalProps> = ({
  isOpen,
  onClose,
  diagnostics,
  transform,
}) => {
  if (!isOpen) return null;

  const timings = diagnostics?.timingBreakdownMs;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-700/50 flex items-center justify-center">
              <Terminal className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-sm">Pipeline Execution Telemetry & Diagnostics</h3>
              <p className="text-xs text-slate-400">Stage-by-stage timing logs and mathematical condition metrics</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-auto p-6 space-y-4 text-xs font-mono">
          {/* Key Consistency Scores */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block font-sans">Matcher Agreement:</span>
              <span className="text-sm font-bold text-emerald-400">
                {diagnostics ? `${(diagnostics.matcherAgreement * 100).toFixed(1)}%` : '—'}
              </span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block font-sans">Geometry Consistency:</span>
              <span className="text-sm font-bold text-cyan-400">
                {diagnostics ? `${(diagnostics.geometryConsistency * 100).toFixed(1)}%` : '—'}
              </span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block font-sans">Mean 1-σ Uncertainty:</span>
              <span className="text-sm font-bold text-amber-300">
                {diagnostics ? `${diagnostics.uncertaintyMeanPx.toFixed(2)} px` : '—'}
              </span>
            </div>
          </div>

          {/* Timing Breakdown Table */}
          <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
            <h4 className="font-bold text-slate-200 text-xs mb-2 flex items-center gap-1.5 font-sans">
              <Clock className="w-3.5 h-3.5 text-purple-400" />
              Stage Execution Breakdown
            </h4>
            {timings ? (
              <div className="space-y-1.5 text-slate-300">
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span>1. Radiometric Preprocessing (Part 04):</span>
                  <span className="font-bold text-purple-300">{timings.preprocessing.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span>2. Illumination Invariance (Part 05):</span>
                  <span className="font-bold text-purple-300">{timings.illumination.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span>3. Multi-Scale Pyramid (Part 07):</span>
                  <span className="font-bold text-purple-300">{timings.pyramid.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span>4. Deep Matching / LoFTR / RIFT (Parts 10-12):</span>
                  <span className="font-bold text-purple-300">{timings.matching.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span>5. Multi-Matcher Fusion (Part 13):</span>
                  <span className="font-bold text-purple-300">{timings.fusion.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span>6. MAGSAC++ Geometric Filtering (Part 14):</span>
                  <span className="font-bold text-purple-300">{timings.filtering.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span>7. Adaptive Transform & BIC (Part 15):</span>
                  <span className="font-bold text-purple-300">{timings.transform.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span>8. Inverse Image Warping (Part 16):</span>
                  <span className="font-bold text-purple-300">{timings.warping.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span>9. Sub-Pixel Quadratic Refinement (Part 17):</span>
                  <span className="font-bold text-purple-300">{timings.subpixel.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span>10. Spatial Uniformity (Part 18):</span>
                  <span className="font-bold text-purple-300">{timings.uniformity.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between py-1.5 pt-2 font-bold text-cyan-300 border-t border-slate-700">
                  <span>Total End-to-End Latency:</span>
                  <span>{timings.total.toFixed(2)} ms</span>
                </div>
              </div>
            ) : (
              <div className="text-slate-500 py-4 text-center">No registration run executed yet.</div>
            )}
          </div>

          {/* Transform Model Matrix Details */}
          {transform?.matrix && (
            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
              <h4 className="font-bold text-slate-200 text-xs mb-2 font-sans">
                Estimated 3x3 {transform.modelType.toUpperCase()} Matrix:
              </h4>
              <div className="grid grid-cols-3 gap-2 text-center text-cyan-300 bg-slate-900/80 p-2.5 rounded border border-slate-800 font-mono text-[11px]">
                {transform.matrix.map((row, r) =>
                  row.map((val, c) => (
                    <div key={`${r}-${c}`} className="p-1 bg-slate-950/60 rounded">
                      {val.toFixed(4)}
                    </div>
                  ))
                )}
              </div>
              <div className="mt-2 text-[11px] text-slate-400 flex justify-between font-sans">
                <span>Degrees of Freedom: {transform.diagnostics.degreesOfFreedom}</span>
                <span>Inlier Support Count: {transform.diagnostics.sampleCount} pts</span>
                <span>BIC Model Score: {transform.bicScore?.toFixed(1) || '—'}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
