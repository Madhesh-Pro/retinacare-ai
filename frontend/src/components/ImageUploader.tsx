import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, AlertCircle, RefreshCw, UserPlus } from 'lucide-react';

interface ImageUploaderProps {
  onAnalyze: (file: File) => void;
  isLoading: boolean;
  selectedFile: File | null;
  setSelectedFile: (file: File | null) => void;
  previewUrl: string | null;
  setPreviewUrl: (url: string | null) => void;
  onNewFileSelected?: () => void;
  onClearAll?: () => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onAnalyze,
  isLoading,
  selectedFile,
  setSelectedFile,
  previewUrl,
  setPreviewUrl,
  onNewFileSelected,
  onClearAll,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFile = (file: File) => {
    setErrorMsg(null);
    if (!file.type.startsWith('image/')) {
      setErrorMsg('Please select a valid retinal fundus image file (PNG, JPG, JPEG).');
      return;
    }
    // Automatically reset old prediction and patient fields when a new image is selected
    if (onNewFileSelected) {
      onNewFileSelected();
    }
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    if (onClearAll) {
      onClearAll();
    }
  };

  return (
    <div className="bg-slate-800/60 rounded-2xl p-6 border border-slate-700/80 shadow-xl backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <UploadCloud className="w-5 h-5 text-teal-400" />
          Retinal Fundus Image Upload
        </h2>
        {selectedFile && !isLoading && (
          <button
            onClick={handleClear}
            className="text-xs text-teal-300 hover:text-teal-200 bg-teal-950/60 hover:bg-teal-900/60 border border-teal-800/60 px-2.5 py-1 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer"
            title="Clear all fields and prepare for a new patient scan"
          >
            <UserPlus className="w-3.5 h-3.5" />
            New Patient / Clear
          </button>
        )}
      </div>

      {!previewUrl ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all ${
            dragOver
              ? 'border-teal-400 bg-teal-500/10'
              : 'border-slate-600 hover:border-teal-500/70 hover:bg-slate-700/40 bg-slate-900/40'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/jpg"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          <div className="p-4 bg-slate-800 rounded-full border border-slate-700 mb-3 shadow-inner">
            <ImageIcon className="w-8 h-8 text-teal-400" />
          </div>
          <p className="text-sm font-medium text-slate-200 mb-1">
            Drag & drop retinal photograph or <span className="text-teal-400 underline">browse</span>
          </p>
          <p className="text-xs text-slate-400 text-center max-w-xs">
            Supports standard high-resolution fundus scans (APTOS 2019 format: PNG, JPG, JPEG)
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-700 flex items-center justify-center max-h-72">
            <img
              src={previewUrl}
              alt="Retinal Fundus Preview"
              className="max-h-72 object-contain w-full"
            />
            <div className="absolute top-2 right-2 bg-slate-900/80 backdrop-blur-md px-2.5 py-1 rounded-md text-xs text-slate-300 border border-slate-700">
              {selectedFile?.name} ({(selectedFile ? selectedFile.size / 1024 : 0).toFixed(1)} KB)
            </div>
          </div>

          <div className="flex gap-2">
            <button
              disabled={isLoading || !selectedFile}
              onClick={() => selectedFile && onAnalyze(selectedFile)}
              className={`flex-1 py-3 px-4 rounded-xl font-semibold text-sm shadow-lg flex items-center justify-center gap-2 transition-all cursor-pointer ${
                isLoading
                  ? 'bg-teal-700/60 text-teal-200 cursor-not-allowed'
                  : 'bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 shadow-teal-500/25 active:scale-[0.99]'
              }`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-teal-200" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Performing Real AI Inference & Grad-CAM...
                </>
              ) : (
                'Run AI Screening & Grad-CAM Analysis'
              )}
            </button>

            <button
              type="button"
              onClick={handleClear}
              disabled={isLoading}
              className="px-3.5 py-3 rounded-xl border border-slate-700 hover:border-slate-600 bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
              title="Reset and clear for a new patient"
            >
              <RefreshCw className="w-4 h-4" />
              Reset
            </button>
          </div>
        </div>
      )}

      {errorMsg && (
        <div className="mt-3 p-3 bg-rose-950/50 border border-rose-800/80 rounded-xl flex items-center gap-2 text-rose-300 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
};
