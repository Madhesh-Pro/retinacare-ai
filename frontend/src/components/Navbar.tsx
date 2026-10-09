import React from 'react';
import { Eye, Cpu } from 'lucide-react';
import { HealthResponse } from '../types';

interface NavbarProps {
  health: HealthResponse | null;
  loading: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ health }) => {
  const isOnline = health !== null;

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-gradient-to-tr from-teal-500 to-cyan-400 p-2.5 rounded-xl shadow-lg shadow-teal-500/20 text-slate-950">
            <Eye className="w-6 h-6 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-1.5">
                RetinaCare<span className="text-teal-400">AI</span>
              </h1>
              <span className="bg-teal-950 text-teal-300 text-xs font-semibold px-2 py-0.5 rounded-full border border-teal-800/60">
                SIH Prototype
              </span>
            </div>
            <p className="text-xs text-slate-400">Diabetic Retinopathy Screening & Explainability System</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="hidden sm:flex items-center space-x-2 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700/60 text-xs text-slate-300">
            <Cpu className="w-4 h-4 text-teal-400" />
            <span>Model: <strong className="text-slate-100">EfficientNet-B2</strong></span>
            {health?.device && (
              <span className="text-slate-400">({health.device.toUpperCase()})</span>
            )}
          </div>

          <div className="flex items-center space-x-2 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700/60 text-xs">
            <span className={`w-2.5 h-2.5 rounded-full ${isOnline ? 'bg-emerald-400 shadow-sm shadow-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
            <span className="text-slate-300 font-medium">
              {isOnline ? 'Backend Online' : 'Backend Offline'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
