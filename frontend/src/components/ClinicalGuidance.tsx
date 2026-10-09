import React from 'react';
import { BookOpen, Info } from 'lucide-react';

export const ClinicalGuidance: React.FC = () => {
  return (
    <div className="bg-slate-800/60 rounded-2xl p-6 border border-slate-700/80 shadow-xl backdrop-blur-sm space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-white flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-teal-400" />
          ICDR Diabetic Retinopathy Clinical Scale
        </h3>
        <span className="text-xs text-slate-400">International Council of Ophthalmology</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead className="text-slate-400 border-b border-slate-700 uppercase bg-slate-900/60">
            <tr>
              <th className="py-2.5 px-3">Grade</th>
              <th className="py-2.5 px-3">Clinical Severity</th>
              <th className="py-2.5 px-3">Diagnostic Fundus Findings</th>
              <th className="py-2.5 px-3">Clinical Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800 text-slate-300">
            <tr className="hover:bg-slate-700/30">
              <td className="py-2 px-3 font-bold text-emerald-400">0</td>
              <td className="py-2 px-3 font-semibold text-white">No DR</td>
              <td className="py-2 px-3">No apparent abnormalities</td>
              <td className="py-2 px-3 text-emerald-400">Annual screening</td>
            </tr>
            <tr className="hover:bg-slate-700/30">
              <td className="py-2 px-3 font-bold text-teal-400">1</td>
              <td className="py-2 px-3 font-semibold text-white">Mild DR</td>
              <td className="py-2 px-3">Microaneurysms only</td>
              <td className="py-2 px-3 text-teal-400">Follow-up 6–12 mos</td>
            </tr>
            <tr className="hover:bg-slate-700/30 bg-amber-950/20">
              <td className="py-2 px-3 font-bold text-amber-400">2</td>
              <td className="py-2 px-3 font-semibold text-white">Moderate DR</td>
              <td className="py-2 px-3">Microaneurysms, dot-blot hemorrhages, hard exudates</td>
              <td className="py-2 px-3 text-amber-300 font-semibold">Referable (4–8 wks)</td>
            </tr>
            <tr className="hover:bg-slate-700/30 bg-orange-950/20">
              <td className="py-2 px-3 font-bold text-orange-400">3</td>
              <td className="py-2 px-3 font-semibold text-white">Severe DR</td>
              <td className="py-2 px-3">&gt;20 hemorrhages in 4 quadrants, venous beading, IRMA</td>
              <td className="py-2 px-3 text-orange-300 font-semibold">Urgent Referral (2–4 wks)</td>
            </tr>
            <tr className="hover:bg-slate-700/30 bg-rose-950/20">
              <td className="py-2 px-3 font-bold text-rose-400">4</td>
              <td className="py-2 px-3 font-semibold text-white">Proliferative DR</td>
              <td className="py-2 px-3">Neovascularization, vitreous hemorrhage</td>
              <td className="py-2 px-3 text-rose-400 font-bold">Emergency Referral</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="flex items-center gap-2 p-3 bg-slate-900/50 rounded-xl border border-slate-800 text-xs text-slate-400">
        <Info className="w-4 h-4 text-teal-400 shrink-0" />
        <span>
          <strong>Referable Diabetic Retinopathy (RDR):</strong> Defined as Moderate DR (Grade 2), Severe DR (Grade 3), or Proliferative DR (Grade 4).
        </span>
      </div>
    </div>
  );
};
