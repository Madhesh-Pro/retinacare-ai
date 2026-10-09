import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { ImageUploader } from './components/ImageUploader';
import { DiagnosticCard } from './components/DiagnosticCard';
import { ProbabilityChart } from './components/ProbabilityChart';
import { GradCAMComparison } from './components/GradCAMComparison';
import { ClinicalGuidance } from './components/ClinicalGuidance';
import { ClinicalReportModal } from './components/ClinicalReportModal';
import { checkBackendHealth, predictFundusImage } from './services/api';
import { HealthResponse, PredictionResponse } from './types';
import { AlertCircle, Stethoscope } from 'lucide-react';

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isReportOpen, setIsReportOpen] = useState(false);

  // Patient & Clinical Form State
  const [patientName, setPatientName] = useState('Ananya Verma');
  const [patientId, setPatientId] = useState('MRN-849201');
  const [age, setAge] = useState('58');
  const [gender, setGender] = useState('Female');
  const [eyeExamined, setEyeExamined] = useState<'OD (Right Eye)' | 'OS (Left Eye)' | 'OU (Both Eyes)'>('OD (Right Eye)');
  const [hospitalName, setHospitalName] = useState('Apollo Specialty Eye Institute & Retinal Care');
  const [doctorName, setDoctorName] = useState('Dr. Priya Ramanathan, MS (Ophthalmology)');
  const [examDate, setExamDate] = useState(() => new Date().toISOString().split('T')[0]);
  const [customNotes, setCustomNotes] = useState('');

  // Poll backend health on initial load
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const h = await checkBackendHealth();
        setHealth(h);
      } catch (e) {
        setHealth(null);
      } finally {
        setHealthLoading(false);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // BUG 2 FIX: Automatically reset patient fields & previous prediction results on new file upload
  const handleNewFileSelected = () => {
    setResult(null);
    setError(null);
    setPatientName('');
    setPatientId('');
    setAge('');
    setGender('Female');
    setCustomNotes('');
  };

  const handleClearAll = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setPatientName('');
    setPatientId('');
    setAge('');
    setGender('Female');
    setCustomNotes('');
  };

  const handleAnalyze = async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await predictFundusImage(file);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Inference error occurred. Ensure backend is running on port 8000.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      <Navbar health={health} loading={healthLoading} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Hero / Intro Banner */}
        <div className="bg-gradient-to-r from-teal-900/40 via-slate-800/60 to-slate-900/40 border border-teal-800/40 rounded-3xl p-6 sm:p-8 backdrop-blur-sm relative overflow-hidden shadow-2xl">
          <div className="relative z-10 max-w-3xl space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-500/10 border border-teal-500/30 text-teal-300 text-xs font-semibold">
              <Stethoscope className="w-3.5 h-3.5" />
              Automated Clinical Screening Engine
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Diabetic Retinopathy Screening & Explainability
            </h2>
            <p className="text-sm text-slate-300 leading-relaxed">
              Upload a retinal fundus photograph for real-time 5-stage DR classification using transfer learning with <strong className="text-teal-300">EfficientNet-B2</strong> and visual explainability via <strong className="text-teal-300">Grad-CAM</strong> attention heatmaps.
            </p>
          </div>
        </div>

        {/* Global Error Banner */}
        {error && (
          <div className="p-4 bg-rose-950/60 border border-rose-800 rounded-2xl flex items-center gap-3 text-rose-200 text-sm shadow-lg">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
            <div className="flex-1">{error}</div>
          </div>
        )}

        {/* Main Grid: Upload & Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Upload */}
          <div className="lg:col-span-5 space-y-6">
            <ImageUploader
              onAnalyze={handleAnalyze}
              isLoading={isLoading}
              selectedFile={selectedFile}
              setSelectedFile={setSelectedFile}
              previewUrl={previewUrl}
              setPreviewUrl={setPreviewUrl}
              onNewFileSelected={handleNewFileSelected}
              onClearAll={handleClearAll}
            />

            <ClinicalGuidance />
          </div>

          {/* Right Column: Real-Time Results Dashboard */}
          <div className="lg:col-span-7 space-y-6">
            {result ? (
              <>
                <DiagnosticCard
                  result={result}
                  onOpenReport={() => setIsReportOpen(true)}
                />
                <ProbabilityChart
                  probabilities={result.probabilities}
                  predictedGrade={result.predicted_grade}
                />
                <GradCAMComparison
                  result={result}
                  originalPreview={previewUrl}
                />
              </>
            ) : (
              <div className="bg-slate-800/40 rounded-2xl p-12 border border-slate-800 text-center flex flex-col items-center justify-center space-y-4">
                <div className="p-4 bg-slate-800/80 rounded-2xl border border-slate-700 text-slate-400">
                  <Stethoscope className="w-10 h-10 text-teal-400/60 stroke-[1.5]" />
                </div>
                <div className="space-y-1">
                  <h3 className="text-base font-semibold text-white">No Image Analyzed Yet</h3>
                  <p className="text-xs text-slate-400 max-w-sm mx-auto">
                    Upload a retinal fundus image from the left panel to execute real EfficientNet-B2 screening and generate Grad-CAM lesion heatmaps.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Clinical Report Modal */}
      {result && (
        <ClinicalReportModal
          isOpen={isReportOpen}
          onClose={() => setIsReportOpen(false)}
          result={result}
          originalPreviewUrl={previewUrl}
          patientName={patientName}
          setPatientName={setPatientName}
          patientId={patientId}
          setPatientId={setPatientId}
          age={age}
          setAge={setAge}
          gender={gender}
          setGender={setGender}
          eyeExamined={eyeExamined}
          setEyeExamined={setEyeExamined}
          hospitalName={hospitalName}
          setHospitalName={setHospitalName}
          doctorName={doctorName}
          setDoctorName={setDoctorName}
          examDate={examDate}
          setExamDate={setExamDate}
          customNotes={customNotes}
          setCustomNotes={setCustomNotes}
        />
      )}

      <footer className="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
        <p>RetinaCare-AI · Real Deep Learning Prototype for Smart India Hackathon (SIH)</p>
      </footer>
    </div>
  );
};
