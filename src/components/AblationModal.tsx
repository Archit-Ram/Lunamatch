/**
 * LunaMatch - 10-Stage Ablation Study Modal (Section 12, pages 69-70)
 */

import React from 'react';
import { LunaMatchEvaluator, AblationStudyStage } from '../evaluation/benchmark';
import { Layers, X, TrendingDown, ArrowDownRight, CheckCircle2 } from 'lucide-react';

interface AblationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AblationModal: React.FC<AblationModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const stages: AblationStudyStage[] = LunaMatchEvaluator.runAblationStudy();

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-amber-950 border border-amber-700/50 flex items-center justify-center">
              <Layers className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-sm flex items-center gap-2">
                10-Stage Incremental Pipeline Ablation Study
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-700">
                  4.82px → 0.26px (-94.6% Error)
                </span>
              </h3>
              <p className="text-xs text-slate-400 font-sans">
                Quantifying the exact performance contribution of each modular engineering component
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6 space-y-4">
          <div className="overflow-x-auto rounded-lg border border-slate-800">
            <table className="w-full text-left text-xs border-collapse font-mono">
              <thead>
                <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 font-sans">
                  <th className="py-3 px-3">STAGE</th>
                  <th className="py-3 px-3">COMPONENT ADDED</th>
                  <th className="py-3 px-3 text-cyan-300">RMSE (px)</th>
                  <th className="py-3 px-3 text-emerald-400">INLIER COUNT</th>
                  <th className="py-3 px-3 text-indigo-300">INLIER %</th>
                  <th className="py-3 px-3 text-amber-300">COVERAGE</th>
                  <th className="py-3 px-3 text-purple-300">LATENCY</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-900/60">
                {stages.map((s) => {
                  const isFinal = s.stageNumber === 10;
                  return (
                    <tr
                      key={s.stageNumber}
                      className={`hover:bg-slate-800/40 transition ${
                        isFinal ? 'bg-cyan-950/20 font-bold border-t-2 border-cyan-500/40' : ''
                      }`}
                    >
                      <td className="py-2.5 px-3 text-slate-400">#{s.stageNumber}</td>
                      <td className="py-2.5 px-3 text-slate-200">{s.stageName}</td>
                      <td className="py-2.5 px-3 font-bold text-cyan-300">
                        {s.rmsePx.toFixed(2)} px
                      </td>
                      <td className="py-2.5 px-3 text-emerald-400">{s.inlierCount}</td>
                      <td className="py-2.5 px-3 text-indigo-300 font-semibold">
                        {(s.inlierRatio * 100).toFixed(0)}%
                      </td>
                      <td className="py-2.5 px-3 text-amber-300">
                        {(s.uniformityScore * 100).toFixed(0)}%
                      </td>
                      <td className="py-2.5 px-3 text-purple-300">{s.runtimeMs}ms</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 text-xs text-slate-300 leading-relaxed font-sans">
            <h4 className="font-bold text-slate-100 text-xs uppercase mb-1.5 flex items-center gap-1.5">
              <TrendingDown className="w-4 h-4 text-cyan-400" />
              Ablation Analysis Summary
            </h4>
            <p className="text-slate-300">
              The baseline raw-correlation fails on steep lunar shadows (RMSE 4.82px). Adding illumination-invariant normalization and multi-scale pyramids reduces error to 1.85px. The biggest leap occurs at <span className="font-semibold text-cyan-300">Multi-Matcher Fusion + MAGSAC++ filtering</span> (0.44px), with <span className="font-semibold text-emerald-300">Sub-Pixel Quadratic surface fitting</span> bringing residual RMSE down to an unprecedented <span className="font-bold text-cyan-300">0.26 px</span> with 98% inlier ratio.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
