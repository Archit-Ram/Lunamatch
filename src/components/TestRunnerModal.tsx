/**
 * LunaMatch - Interactive Unit & Verification Test Runner Modal
 */

import React, { useState, useEffect } from 'react';
import { runAllLunaMatchUnitTests, TestCaseResult } from '../tests/unit_tests';
import { FlaskConical, CheckCircle2, XCircle, Play, RotateCcw, X, ShieldCheck } from 'lucide-react';
import confetti from 'canvas-confetti';

interface TestRunnerModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const TestRunnerModal: React.FC<TestRunnerModalProps> = ({ isOpen, onClose }) => {
  const [results, setResults] = useState<TestCaseResult[]>([]);
  const [isRunning, setIsRunning] = useState<boolean>(false);

  const executeTests = async () => {
    setIsRunning(true);
    const testResults = await runAllLunaMatchUnitTests();
    setResults(testResults);
    setIsRunning(false);

    const allPassed = testResults.every((t) => t.passed);
    if (allPassed) {
      confetti({
        particleCount: 80,
        spread: 60,
        origin: { y: 0.6 },
      });
    }
  };

  useEffect(() => {
    if (isOpen && results.length === 0) {
      executeTests();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const passedCount = results.filter((r) => r.passed).length;
  const totalCount = results.length;
  const allPassed = totalCount > 0 && passedCount === totalCount;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-3xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-950 border border-emerald-700/50 flex items-center justify-center">
              <FlaskConical className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-sm flex items-center gap-2">
                Mathematical Unit & Verification Test Suite
                {totalCount > 0 && (
                  <span
                    className={`text-[11px] font-semibold px-2 py-0.5 rounded border ${
                      allPassed
                        ? 'bg-emerald-950 text-emerald-400 border-emerald-700'
                        : 'bg-red-950 text-red-400 border-red-700'
                    }`}
                  >
                    {passedCount}/{totalCount} PASSED
                  </span>
                )}
              </h3>
              <p className="text-xs text-slate-400 font-sans">
                Verifying data contracts, NaN rejection, sub-pixel quadratic fitting, MAGSAC++ outlier rejection & pyramids
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

        {/* Test List */}
        <div className="flex-1 overflow-auto p-6 space-y-3">
          {isRunning ? (
            <div className="py-12 flex flex-col items-center justify-center text-slate-400 gap-3">
              <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-xs font-mono">Executing rigorous mathematical test harness...</span>
            </div>
          ) : (
            results.map((r, idx) => (
              <div
                key={idx}
                className={`p-3.5 rounded-lg border text-xs font-mono transition ${
                  r.passed
                    ? 'bg-emerald-950/20 border-emerald-800/40 text-emerald-300'
                    : 'bg-red-950/20 border-red-800/40 text-red-300'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-bold text-slate-400">{r.partName}</span>
                  <span className="flex items-center gap-1 font-bold text-[11px]">
                    {r.passed ? (
                      <>
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                        PASS
                      </>
                    ) : (
                      <>
                        <XCircle className="w-3.5 h-3.5 text-red-400" />
                        FAIL
                      </>
                    )}
                  </span>
                </div>
                <div className="font-semibold text-slate-200">{r.testName}</div>
                <div className="text-slate-400 text-[11px] mt-1 font-sans">{r.message}</div>
              </div>
            ))
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-3.5 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between">
          <span className="text-xs text-slate-400 font-sans">
            Deterministic seed verification • No fabricated benchmarks
          </span>
          <button
            onClick={executeTests}
            disabled={isRunning}
            className="flex items-center gap-1.5 px-3.5 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-md text-xs font-semibold shadow-sm transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Re-Run All Tests</span>
          </button>
        </div>
      </div>
    </div>
  );
};
