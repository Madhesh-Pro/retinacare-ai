import React from 'react';
import { BarChart3 } from 'lucide-react';

interface ProbabilityChartProps {
  probabilities: Record<string, number>;
  predictedGrade: number;
}

const CLASS_GRADE_ORDER = [
  'No DR',
  'Mild DR',
  'Moderate DR',
  'Severe DR',
  'Proliferative DR',
];

export const ProbabilityChart: React.FC<ProbabilityChartProps> = ({
  probabilities,
  predictedGrade,
}) => {
  return (
    <div className="bg-slate-800/70 rounded-2xl p-6 border border-slate-700/80 shadow-xl backdrop-blur-sm space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-white flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-teal-400" />
          Multiclass Probability Distribution
        </h3>
        <span className="text-xs text-slate-400">ICDR 5-Grade Scale</span>
      </div>

      <div className="space-y-3 pt-1">
        {CLASS_GRADE_ORDER.map((clsName, idx) => {
          const prob = probabilities[clsName] ?? 0;
          const isSelected = idx === predictedGrade;
          const pct = (prob * 100).toFixed(1);

          return (
            <div key={clsName} className="space-y-1">
              <div className="flex justify-between text-xs font-medium">
                <span className={`flex items-center gap-2 ${isSelected ? 'text-teal-300 font-bold' : 'text-slate-300'}`}>
                  <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] ${
                    isSelected ? 'bg-teal-500 text-slate-950 font-extrabold' : 'bg-slate-700 text-slate-400'
                  }`}>
                    {idx}
                  </span>
                  {clsName}
                  {idx >= 2 && (
                    <span className="text-[10px] text-amber-400/80 font-normal">
                      (Referable)
                    </span>
                  )}
                </span>
                <span className={isSelected ? 'text-teal-300 font-bold' : 'text-slate-400'}>
                  {pct}%
                </span>
              </div>

              <div className="w-full bg-slate-900/80 rounded-full h-2.5 overflow-hidden border border-slate-700/60 p-0.5">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${
                    isSelected
                      ? idx >= 2
                        ? 'bg-gradient-to-r from-amber-500 to-rose-500 shadow-sm shadow-rose-500/50'
                        : 'bg-gradient-to-r from-teal-500 to-cyan-400 shadow-sm shadow-teal-500/50'
                      : 'bg-slate-700'
                  }`}
                  style={{ width: `${Math.max(prob * 100, 2)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
