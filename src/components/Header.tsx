/**
 * LunaMatch Header Component
 */

import React from 'react';
import { Satellite, ShieldCheck, Cpu, Terminal, Layers, BarChart3, FlaskConical, CheckCircle2 } from 'lucide-react';

interface HeaderProps {
  onOpenBenchmarks: () => void;
  onOpenAblation: () => void;
  onOpenTests: () => void;
  onOpenDiagnostics: () => void;
  activeMatcher: string;
  isProcessing: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  onOpenBenchmarks,
  onOpenAblation,
  onOpenTests,
  onOpenDiagnostics,
  activeMatcher,
  isProcessing,
}) => {
  return (
    <header className="bg-slate-900 border-b border-slate-800 text-white px-5 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30 shadow-md">
      <div className="flex items-center gap-3.5">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-tr from-cyan-600 to-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/20 border border-cyan-400/30">
          <Satellite className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-300 bg-clip-text text-transparent">
              LunaMatch
            </h1>
            <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-blue-950/80 border border-blue-600/40 text-blue-300 tracking-wider">
              SIH 2026 • SIH26166
            </span>
            <span className="text-[11px] font-medium px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-600/30 text-emerald-400 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              Chandrayaan-2 Core
            </span>
          </div>
          <p className="text-xs text-slate-400 hidden sm:block">
            Multi-Modal, Sun Angle & Scale Invariant Lunar Image Correspondence Engine (OHRC • TMC-2 • IIRS)
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <div className="flex items-center gap-1.5 bg-slate-950/80 px-2.5 py-1 rounded-md border border-amber-500/40 text-xs text-slate-300 mr-1">
          <Cpu className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-slate-400">Matcher:</span>
          <span className="font-mono font-semibold text-amber-300 capitalize">{activeMatcher}</span>
          <span className="text-[10px] bg-amber-950 px-1 py-0.2 rounded text-amber-400 border border-amber-700/50">Simulated</span>
        </div>

        <button
          onClick={onOpenTests}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 active:bg-slate-900 text-slate-200 rounded-md border border-slate-700 transition"
          title="Run Mathematical Unit & Verification Tests"
        >
          <FlaskConical className="w-3.5 h-3.5 text-emerald-400" />
          <span>Unit Tests</span>
        </button>

        <button
          onClick={onOpenAblation}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 active:bg-slate-900 text-slate-200 rounded-md border border-slate-700 transition"
          title="View 10-Stage Ablation Study"
        >
          <Layers className="w-3.5 h-3.5 text-amber-400" />
          <span>Ablation Study</span>
        </button>

        <button
          onClick={onOpenBenchmarks}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 active:bg-slate-900 text-slate-200 rounded-md border border-slate-700 transition"
          title="Multi-Modal Benchmark Matrix"
        >
          <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
          <span>Benchmark Matrix</span>
        </button>

        <button
          onClick={onOpenDiagnostics}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 active:bg-slate-900 text-slate-200 rounded-md border border-slate-700 transition"
          title="View Stage-by-Stage Diagnostics & Pipeline Timings"
        >
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <span>Diagnostics</span>
        </button>
      </div>
    </header>
  );
};
