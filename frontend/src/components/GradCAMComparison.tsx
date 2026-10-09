import React, { useState } from 'react';
import { Eye, Layers, Sparkles, HelpCircle } from 'lucide-react';
import { PredictionResponse } from '../types';

interface GradCAMComparisonProps {
  result: PredictionResponse;
  originalPreview: string | null;
}

export const GradCAMComparison: React.FC<GradCAMComparisonProps> = ({
  result,
  originalPreview,
}) => {
  const [viewMode, setViewMode] = useState<'side-by-side' | 'gradcam-only' | 'original-only'>('side-by-side');

  if (!result.gradcam_overlay_base64) {
    return null;
  }

  return (
    <div className="bg-slate-800/70 rounded-2xl p-6 border border-slate-700/80 shadow-xl backdrop-blur-sm space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-teal-400" />
            Grad-CAM Visual Explainability Heatmap
          </h3>
          <p className="text-xs text-slate-400">
            Spatial attention map showing retinal regions that determined the Grade {result.predicted_grade} classification
          </p>
        </div>

        {/* View Mode Toggle */}
        <div className="flex bg-slate-900/80 p-1 rounded-xl border border-slate-700/80 self-start text-xs font-medium">
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`px-3 py-1 rounded-lg transition-all ${
              viewMode === 'side-by-side' ? 'bg-teal-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Side-by-Side
          </button>
          <button
            onClick={() => setViewMode('gradcam-only')}
            className={`px-3 py-1 rounded-lg transition-all ${
              viewMode === 'gradcam-only' ? 'bg-teal-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Heatmap Overlay
          </button>
          <button
            onClick={() => setViewMode('original-only')}
            className={`px-3 py-1 rounded-lg transition-all ${
              viewMode === 'original-only' ? 'bg-teal-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Enhanced ROI
          </button>
        </div>
      </div>

      {/* Image Display Area */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Preprocessed Fundus Image */}
        {(viewMode === 'side-by-side' || viewMode === 'original-only') && (
          <div className={`space-y-2 ${viewMode === 'original-only' ? 'md:col-span-2' : ''}`}>
            <div className="flex items-center justify-between text-xs text-slate-300 font-medium">
              <span className="flex items-center gap-1.5">
                <Eye className="w-3.5 h-3.5 text-teal-400" />
                Preprocessed Fundus ROI (CLAHE / Normalization)
              </span>
            </div>
            <div className="bg-slate-950 rounded-xl overflow-hidden border border-slate-700 flex items-center justify-center aspect-square shadow-inner">
              <img
                src={result.preprocessed_image_base64 || originalPreview || ''}
                alt="Preprocessed Retina"
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        )}

        {/* Grad-CAM Heatmap Overlay */}
        {(viewMode === 'side-by-side' || viewMode === 'gradcam-only') && (
          <div className={`space-y-2 ${viewMode === 'gradcam-only' ? 'md:col-span-2' : ''}`}>
            <div className="flex items-center justify-between text-xs text-slate-300 font-medium">
              <span className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-rose-400" />
                Grad-CAM Activation Map (Attention Zones)
              </span>
              <span className="text-[10px] text-teal-400 bg-teal-950/80 px-2 py-0.5 rounded border border-teal-800/60">
                Warm colors (Red/Yellow) = Higher Attention
              </span>
            </div>
            <div className="bg-slate-950 rounded-xl overflow-hidden border border-slate-700 flex items-center justify-center aspect-square shadow-inner relative group">
              <img
                src={result.gradcam_overlay_base64}
                alt="Grad-CAM Overlay"
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        )}
      </div>

      {/* Clinical Legend */}
      <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800 text-xs text-slate-400 flex items-start gap-2.5">
        <HelpCircle className="w-4 h-4 text-teal-400 shrink-0 mt-0.5" />
        <div>
          <span className="text-slate-200 font-semibold">How to read this explainability map: </span>
          The highlighted warm zones (red, orange, yellow) indicate convolutional feature areas where the neural network recognized critical diabetic microvascular biomarkers such as intraretinal hemorrhages, cotton wool spots, microaneurysms, or hard exudates.
        </div>
      </div>
    </div>
  );
};
