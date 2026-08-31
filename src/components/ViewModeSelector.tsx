/**
 * LunaMatch View Mode Selector
 * Implements the exact visualization tabs required in Part 23 (pages 55-57)
 */

import React from 'react';
import { Eye, Sparkles, GitCommit, Split, ArrowUpRight, ShieldAlert, Activity } from 'lucide-react';

export type ViewMode = 'raw' | 'normalized' | 'matches' | 'warped' | 'residual' | 'uncertainty' | 'spectral';

interface ViewModeSelectorProps {
  currentMode: ViewMode;
  onSelectMode: (mode: ViewMode) => void;
  hasIIRSSensor: boolean;
}

export const ViewModeSelector: React.FC<ViewModeSelectorProps> = ({
  currentMode,
  onSelectMode,
  hasIIRSSensor,
}) => {
  const tabs: Array<{ id: ViewMode; label: string; icon: React.ReactNode; desc: string }> = [
    {
      id: 'raw',
      label: 'Raw',
      icon: <Eye className="w-3.5 h-3.5" />,
      desc: 'Original sensor radiometry and lighting',
    },
    {
      id: 'normalized',
      label: 'Normalized',
      icon: <Sparkles className="w-3.5 h-3.5" />,
      desc: 'Illumination-invariant DoG & log reflectance',
    },
    {
      id: 'matches',
      label: 'Matches',
      icon: <GitCommit className="w-3.5 h-3.5" />,
      desc: 'Fused tie-point correspondence vectors',
    },
    {
      id: 'warped',
      label: 'Registered',
      icon: <Split className="w-3.5 h-3.5" />,
      desc: 'Aligned image overlay & difference map',
    },
    {
      id: 'residual',
      label: 'Residual',
      icon: <ArrowUpRight className="w-3.5 h-3.5" />,
      desc: 'Reprojection error quiver vectors',
    },
    {
      id: 'uncertainty',
      label: 'Uncertainty',
      icon: <ShieldAlert className="w-3.5 h-3.5" />,
      desc: 'Covariance ellipses & spatial error heatmap',
    },
    {
      id: 'spectral',
      label: 'IIRS Spectrum',
      icon: <Activity className="w-3.5 h-3.5" />,
      desc: '250-band hyperspectral mineral signature',
    },
  ];

  return (
    <div className="bg-slate-950 px-4 py-2 border-b border-slate-800 flex items-center justify-between flex-wrap gap-2">
      <div className="flex items-center gap-1.5 flex-wrap">
        {tabs.map((tab) => {
          const isActive = currentMode === tab.id;
          if (tab.id === 'spectral' && !hasIIRSSensor) return null;

          return (
            <button
              key={tab.id}
              onClick={() => onSelectMode(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md transition border ${
                isActive
                  ? 'bg-blue-600 border-blue-500 text-white shadow-sm shadow-blue-500/20'
                  : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800 hover:text-slate-100'
              }`}
              title={tab.desc}
            >
              {tab.icon}
              <span>[ {tab.label} ]</span>
            </button>
          );
        })}
      </div>

      <div className="text-[11px] text-slate-400 font-mono hidden md:block">
        {tabs.find((t) => t.id === currentMode)?.desc}
      </div>
    </div>
  );
};
