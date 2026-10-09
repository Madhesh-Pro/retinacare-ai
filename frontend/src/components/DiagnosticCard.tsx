import React from 'react';
import { ShieldCheck, AlertOctagon, Activity, FileText } from 'lucide-react';
import { PredictionResponse } from '../types';

interface DiagnosticCardProps {
  result: PredictionResponse;
  onOpenReport?: () => void;
}

const GRADE_CONFIG: Record<number, {
  color: string;
  badgeBg: string;
  borderColor: string;
  textColor: string;
  severity: string;
}> = {
  0: {
    color: 'text-emerald-400',
    badgeBg: 'bg-emerald-950/80 border-emerald-800 text-emerald-300',
    borderColor: 'border-emerald-700/60',
    textColor: 'text-emerald-300',
    severity: 'Normal / No Pathology',
  },
  1: {
    color: 'text-teal-400',
    badgeBg: 'bg-teal-950/80 border-teal-800 text-teal-300',
    borderColor: 'border-teal-700/60',
    textColor: 'text-teal-300',
    severity: 'Early Stage (Mild)',
  },
  2: {
    color: 'text-amber-400',
    badgeBg: 'bg-amber-950/80 border-amber-800 text-amber-300',
    borderColor: 'border-amber-700/60',
    textColor: 'text-amber-300',
    severity: 'Intermediate Stage (Moderate)',
  },
  3: {
    color: 'text-orange-400',
    badgeBg: 'bg-orange-950/80 border-orange-800 text-orange-300',
    borderColor: 'border-orange-700/60',
    textColor: 'text-orange-300',
    severity: 'Advanced Stage (Severe)',
  },
  4: {
    color: 'text-rose-400',
    badgeBg: 'bg-rose-950/80 border-rose-800 text-rose-300',
    borderColor: 'border-rose-700/60',
    textColor: 'text-rose-300',
    severity: 'Critical Stage (Proliferative)',
  },
};

export const DiagnosticCard: React.FC<DiagnosticCardProps> = ({ result, onOpenReport }) => {
  const config = GRADE_CONFIG[result.predicted_grade] || GRADE_CONFIG[0];
  const isReferable = result.is_referable;

  return (
    <div className={`bg-slate-800/70 rounded-2xl p-6 border ${config.borderColor} shadow-xl backdrop-blur-sm space-y-5`}>
      <div className="flex items-start justify-between">
        <div>
          <span className="text-xs uppercase tracking-wider font-semibold text-slate-400 flex items-center gap-1.5 mb-1">
            <Activity className="w-4 h-4 text-teal-400" />
            AI Clinical Diagnosis
          </span>
          <h3 className="text-2xl font-extrabold text-white flex items-center gap-2">
            Grade {result.predicted_grade}: <span className={config.color}>{result.class_name}</span>
          </h3>
        </div>

        <div className={`px-3 py-1.5 rounded-xl border text-xs font-bold tracking-wide ${config.badgeBg}`}>
          {config.severity}
        </div>
      </div>

      {/* Referable DR Status Alert Banner */}
      <div
        className={`p-4 rounded-xl border flex items-start gap-3.5 ${
          isReferable
            ? 'bg-rose-950/40 border-rose-800/80 text-rose-200'
            : 'bg-emerald-950/40 border-emerald-800/80 text-emerald-200'
        }`}
      >
        {isReferable ? (
          <AlertOctagon className="w-6 h-6 text-rose-400 shrink-0 mt-0.5" />
        ) : (
          <ShieldCheck className="w-6 h-6 text-emerald-400 shrink-0 mt-0.5" />
        )}
        <div className="space-y-1">
          <div className="font-bold text-sm tracking-wide">
            {isReferable ? 'REFERABLE DIABETIC RETINOPATHY' : 'NON-REFERABLE CASE'}
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            {isReferable
              ? 'Pathological lesions detected require clinical ophthalmologist evaluation and intervention to prevent vision impairment.'
              : 'No immediate signs of referable diabetic retinopathy detected. Maintain standard routine diabetic eye screening.'}
          </p>
        </div>
      </div>

      {/* Model Confidence & Metrics Row */}
      <div className="grid grid-cols-2 gap-3 pt-1">
        <div className="bg-slate-900/60 p-3.5 rounded-xl border border-slate-700/60">
          <div className="text-xs text-slate-400 font-medium mb-1">AI Confidence</div>
          <div className="text-xl font-bold text-teal-400">
            {(result.confidence * 100).toFixed(1)}%
          </div>
        </div>

        <div className="bg-slate-900/60 p-3.5 rounded-xl border border-slate-700/60">
          <div className="text-xs text-slate-400 font-medium mb-1">Architecture</div>
          <div className="text-sm font-semibold text-slate-200 truncate">
            EfficientNet-B2
          </div>
        </div>
      </div>

      {/* Clinical Recommendation */}
      <div className="bg-slate-900/40 p-4 rounded-xl border border-slate-800 space-y-1">
        <div className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
          <FileText className="w-3.5 h-3.5 text-teal-400" />
          Clinical Management Guideline
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          {result.clinical_recommendation}
        </p>
      </div>

      {/* Action Button: Download Clinical PDF Report */}
      {onOpenReport && (
        <button
          onClick={onOpenReport}
          className="w-full py-3 px-4 bg-gradient-to-r from-teal-500/20 to-cyan-500/20 hover:from-teal-500/30 hover:to-cyan-500/30 border border-teal-500/50 hover:border-teal-400 text-teal-300 hover:text-white rounded-xl font-semibold text-sm flex items-center justify-center gap-2 shadow-lg transition-all active:scale-[0.99]"
        >
          <FileText className="w-4 h-4 text-teal-400" />
          Download / Print Full Clinical Report (PDF)
        </button>
      )}
    </div>
  );
};
