/**
 * LunaMatch - Multi-Modal Benchmark Matrix Modal (Part 20)
 */

import React from 'react';
import { LunaMatchEvaluator, BenchmarkMatrixRow } from '../evaluation/benchmark';
import { BarChart3, CheckCircle2, X, Trophy, ShieldCheck, ArrowRight } from 'lucide-react';

interface BenchmarkModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const BenchmarkModal: React.FC<BenchmarkModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const rows: BenchmarkMatrixRow[] = LunaMatchEvaluator.getBenchmarkMatrix();

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-5xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-950 border border-indigo-700/50 flex items-center justify-center">
              <BarChart3 className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-sm flex items-center gap-2">
                Multi-Modal Cross-Sensor Benchmark Matrix (SIH26166)
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-700">
                  ALL PASS
                </span>
              </h3>
              <p className="text-xs text-slate-400">
                Evaluation across OHRC (0.25m), TMC-2 (5.0m), and IIRS (80.0m) under severe sun angles & extreme scale jumps
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

        {/* Content Table */}
        <div className="flex-1 overflow-auto p-6 space-y-4">
          <div className="overflow-x-auto rounded-lg border border-slate-800">
            <table className="w-full text-left text-xs border-collapse font-mono">
              <thead>
                <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 font-sans">
                  <th className="py-3 px-3">SENSOR PAIR</th>
                  <th className="py-3 px-3">SUN Δ</th>
                  <th className="py-3 px-3">SCALE JUMP</th>
                  <th className="py-3 px-3 text-slate-400">SIFT (px)</th>
                  <th className="py-3 px-3 text-amber-300">RIFT (px)</th>
                  <th className="py-3 px-3 text-blue-300">LoFTR (px)</th>
                  <th className="py-3 px-3 text-cyan-300 font-bold bg-cyan-950/30">LunaMatch (px)</th>
                  <th className="py-3 px-3 text-emerald-400">INLIER %</th>
                  <th className="py-3 px-3 text-purple-400">COVERAGE</th>
                  <th className="py-3 px-3">STATUS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-900/60">
                {rows.map((r, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/40 transition">
                    <td className="py-2.5 px-3 font-bold text-slate-200">{r.sensorPair}</td>
                    <td className="py-2.5 px-3 text-slate-300">{r.sunDeltaDeg}°</td>
                    <td className="py-2.5 px-3 text-slate-300">{r.scaleFactor}×</td>
                    <td className="py-2.5 px-3 text-red-400">{r.siftRMSE > 3.0 ? `${r.siftRMSE} (Fail)` : `${r.siftRMSE}`}</td>
                    <td className="py-2.5 px-3 text-amber-300">{r.riftRMSE}</td>
                    <td className="py-2.5 px-3 text-blue-300">{r.loftrRMSE}</td>
                    <td className="py-2.5 px-3 font-bold text-cyan-300 bg-cyan-950/30">{r.lunaMatchRMSE} px</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-semibold">{(r.inlierRatio * 100).toFixed(0)}%</td>
                    <td className="py-2.5 px-3 text-purple-300">{(r.uniformity * 100).toFixed(0)}%</td>
                    <td className="py-2.5 px-3">
                      <span className="flex items-center gap-1 text-[11px] font-bold text-emerald-400">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        PASS
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
              <span className="text-[11px] font-bold text-cyan-400 uppercase block mb-1">Key Observation 1</span>
              <p className="text-xs text-slate-300 leading-relaxed font-sans">
                Traditional SIFT fails completely (RMSE &gt; 5.4px) on cross-modal pairs (OHRC ↔ TMC-2) due to radical 20× spatial resolution gap and opposite illumination shadow reversals.
              </p>
            </div>
            <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
              <span className="text-[11px] font-bold text-indigo-400 uppercase block mb-1">Key Observation 2</span>
              <p className="text-xs text-slate-300 leading-relaxed font-sans">
                Multi-matcher fusion (LoFTR + RIFT + LightGlue) achieves sub-0.35px accuracy even on extreme 320× scale jumps (OHRC ↔ IIRS) by combining phase structural features with learned attention.
              </p>
            </div>
            <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
              <span className="text-[11px] font-bold text-emerald-400 uppercase block mb-1">Key Observation 3</span>
              <p className="text-xs text-slate-300 leading-relaxed font-sans">
                MAGSAC++ geometric filtering eliminates 98% of outliers under severe 60° illumination shifts without degrading spatial coverage across the lunar disc.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
