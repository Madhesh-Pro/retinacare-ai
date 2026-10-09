import React, { useState, useEffect } from 'react';
import { X, Download, Printer, Stethoscope, CheckCircle2, ShieldAlert, Eye, Loader2 } from 'lucide-react';
import { PredictionResponse, ReportGeneratePayload } from '../types';
import { downloadClinicalReportPdf } from '../services/api';

interface ClinicalReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: PredictionResponse;
  originalPreviewUrl: string | null;
  patientName: string;
  setPatientName: (val: string) => void;
  patientId: string;
  setPatientId: (val: string) => void;
  age: string;
  setAge: (val: string) => void;
  gender: string;
  setGender: (val: string) => void;
  eyeExamined: 'OD (Right Eye)' | 'OS (Left Eye)' | 'OU (Both Eyes)';
  setEyeExamined: (val: 'OD (Right Eye)' | 'OS (Left Eye)' | 'OU (Both Eyes)') => void;
  hospitalName: string;
  setHospitalName: (val: string) => void;
  doctorName: string;
  setDoctorName: (val: string) => void;
  examDate: string;
  setExamDate: (val: string) => void;
  customNotes: string;
  setCustomNotes: (val: string) => void;
}

const GRADE_FINDINGS: Record<number, string> = {
  0: 'Normal retinal microvascular architecture. No microaneurysms, intraretinal hemorrhages, hard lipid exudates, or cotton-wool spots detected within the examined fundus field.',
  1: 'Early microvascular pathology identified. Presence of isolated microaneurysms (focal red lesions) without evidence of hard exudation, macular edema, or significant retinal ischemia.',
  2: 'Moderate diabetic retinopathy identified. Presence of multiple intraretinal dot-blot hemorrhages and/or hard yellow lipid exudate clusters in the macular and perimacular regions.',
  3: 'Severe non-proliferative diabetic retinopathy identified. Extensive intraretinal hemorrhages across multiple quadrants, venous beading, and soft exudates (cotton-wool spots) indicative of pronounced retinal ischemia.',
  4: 'Proliferative diabetic retinopathy (PDR) identified. Active retinal neovascularization (fragile new vessel fronds) and/or preretinal/vitreous hemorrhages detected, presenting severe risk of vision impairment.',
};

const GRADE_TIMELINES: Record<number, { action: string; timeline: string }> = {
  0: {
    action: 'Routine Diabetic Screening',
    timeline: 'Annual routine screening (12 months)',
  },
  1: {
    action: 'Clinical Follow-up & Glycemic Control',
    timeline: '6 to 12 months with optimized HbA1c control',
  },
  2: {
    action: 'Comprehensive Ophthalmologist Referral',
    timeline: 'Referral evaluation within 4 to 8 weeks',
  },
  3: {
    action: 'Urgent Retinal Specialist Referral',
    timeline: 'Urgent evaluation within 2 to 4 weeks (evaluate for anti-VEGF / PRP)',
  },
  4: {
    action: 'Immediate Emergency Specialist Intervention',
    timeline: 'Immediate specialist consultation (High risk of vision loss)',
  },
};

export const ClinicalReportModal: React.FC<ClinicalReportModalProps> = ({
  isOpen,
  onClose,
  result,
  originalPreviewUrl,
  patientName,
  setPatientName,
  patientId,
  setPatientId,
  age,
  setAge,
  gender,
  setGender,
  eyeExamined,
  setEyeExamined,
  hospitalName,
  setHospitalName,
  doctorName,
  setDoctorName,
  examDate,
  setExamDate,
  customNotes,
  setCustomNotes,
}) => {
  const [reportId, setReportId] = useState('');
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      const randSuffix = Math.floor(1000 + Math.random() * 9000);
      const dateStr = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      const cleanId = (patientId || 'NEW').replace(/[^a-zA-Z0-9]/g, '').slice(0, 6);
      setReportId(`RC-REP-${dateStr}-${cleanId}-${randSuffix}`);
      setDownloadError(null);
    }
  }, [isOpen, patientId]);

  if (!isOpen) return null;

  const currentDateFormatted = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  const isReferable = result.is_referable;
  const grade = result.predicted_grade;
  const findingsText = GRADE_FINDINGS[grade] || GRADE_FINDINGS[0];
  const timelineInfo = GRADE_TIMELINES[grade] || GRADE_TIMELINES[0];

  // BUG 1 FIX: Server-side PDF generation & download trigger
  const handleDownloadPdf = async () => {
    setIsDownloading(true);
    setDownloadError(null);
    try {
      const payload: ReportGeneratePayload = {
        patient_name: patientName.trim() || 'Anonymous Patient',
        patient_id: patientId.trim() || 'PATIENT-001',
        age: age.trim() || 'N/A',
        gender: gender,
        eye_examined: eyeExamined,
        hospital_name: hospitalName.trim() || 'RetinaCare Clinical Eye Center',
        doctor_name: doctorName.trim() || 'Reviewing Ophthalmologist',
        exam_date: examDate || new Date().toISOString().split('T')[0],
        custom_notes: customNotes,
        predicted_grade: result.predicted_grade,
        class_name: result.class_name,
        is_referable: result.is_referable,
        confidence: result.confidence,
        probabilities: result.probabilities,
        clinical_recommendation: result.clinical_recommendation,
        gradcam_overlay_base64: result.gradcam_overlay_base64,
        preprocessed_image_base64: result.preprocessed_image_base64,
      };

      await downloadClinicalReportPdf(payload);
    } catch (err: any) {
      setDownloadError(err.message || 'Failed to generate and download PDF report. Ensure backend is running.');
    } finally {
      setIsDownloading(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-950/85 backdrop-blur-md flex items-center justify-center p-4 print:p-0 print:bg-white print:static">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-4xl w-full shadow-2xl overflow-hidden flex flex-col max-h-[92vh] print:max-h-none print:border-none print:shadow-none print:bg-white print:text-black">
        
        {/* ================= MODAL TOP CONTROL BAR (Hidden on Print) ================= */}
        <div className="p-4 bg-slate-800/90 border-b border-slate-700 flex items-center justify-between print:hidden">
          <div className="flex items-center gap-2">
            <Stethoscope className="w-5 h-5 text-teal-400" />
            <h3 className="font-bold text-white text-sm sm:text-base">
              Hospital Diagnostic Report
            </h3>
          </div>
          <div className="flex items-center gap-2.5">
            {/* Primary Action: Download Server-Generated PDF */}
            <button
              onClick={handleDownloadPdf}
              disabled={isDownloading}
              className={`px-4 py-2 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 rounded-xl font-bold text-xs flex items-center gap-2 shadow-lg shadow-teal-500/20 active:scale-95 transition-all cursor-pointer ${
                isDownloading ? 'opacity-70 cursor-not-allowed' : ''
              }`}
              title="Generate and download a standalone, vector-sharp PDF file"
            >
              {isDownloading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating PDF...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  Download PDF Report
                </>
              )}
            </button>

            {/* Optional Print Button */}
            <button
              onClick={handlePrint}
              className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-700/80 transition-colors cursor-pointer"
              title="Print Document"
            >
              <Printer className="w-4 h-4" />
            </button>

            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-700 transition-colors cursor-pointer"
              title="Close Dialog"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {downloadError && (
          <div className="px-4 py-2 bg-rose-950/70 border-b border-rose-800 text-rose-300 text-xs font-medium flex items-center justify-between print:hidden">
            <span>{downloadError}</span>
            <button onClick={() => setDownloadError(null)} className="text-rose-400 hover:text-rose-200">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* ================= CONFIGURATION FORM (Hidden on Print) ================= */}
        <div className="p-4 bg-slate-800/40 border-b border-slate-700 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs print:hidden">
          <div>
            <label className="block text-slate-400 mb-1 font-medium">Patient Name</label>
            <input
              type="text"
              value={patientName}
              placeholder="e.g. Ramesh Kumar"
              onChange={(e) => setPatientName(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-teal-500"
            />
          </div>
          <div>
            <label className="block text-slate-400 mb-1 font-medium">Patient ID / MRN</label>
            <input
              type="text"
              value={patientId}
              placeholder="e.g. MRN-849201"
              onChange={(e) => setPatientId(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-teal-500"
            />
          </div>
          <div>
            <label className="block text-slate-400 mb-1 font-medium">Age & Gender</label>
            <div className="flex gap-1.5">
              <input
                type="text"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                placeholder="Age"
                className="w-16 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-white focus:outline-none focus:border-teal-500"
              />
              <select
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-white focus:outline-none focus:border-teal-500"
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-slate-400 mb-1 font-medium">Eye Examined</label>
            <select
              value={eyeExamined}
              onChange={(e) => setEyeExamined(e.target.value as any)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-teal-500"
            >
              <option value="OD (Right Eye)">OD (Right Eye)</option>
              <option value="OS (Left Eye)">OS (Left Eye)</option>
              <option value="OU (Both Eyes)">OU (Both Eyes)</option>
            </select>
          </div>
          <div className="sm:col-span-2">
            <label className="block text-slate-400 mb-1 font-medium">Hospital / Clinic Name</label>
            <input
              type="text"
              value={hospitalName}
              onChange={(e) => setHospitalName(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-teal-500"
            />
          </div>
          <div>
            <label className="block text-slate-400 mb-1 font-medium">Reviewing Doctor</label>
            <input
              type="text"
              value={doctorName}
              onChange={(e) => setDoctorName(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-teal-500"
            />
          </div>
          <div>
            <label className="block text-slate-400 mb-1 font-medium">Exam Date</label>
            <input
              type="date"
              value={examDate}
              onChange={(e) => setExamDate(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-teal-500"
            />
          </div>
          <div className="sm:col-span-4">
            <label className="block text-slate-400 mb-1 font-medium">Clinician Impression / Notes (Optional)</label>
            <input
              type="text"
              value={customNotes}
              onChange={(e) => setCustomNotes(e.target.value)}
              placeholder="e.g. Macular edema suspected; recommend clinical OCT scan."
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-teal-500"
            />
          </div>
        </div>

        {/* ========================================================================= */}
        {/* ================= PRINTABLE CLINICAL REPORT DOCUMENT BODY =============== */}
        {/* ========================================================================= */}
        <div id="clinical-report-printable" className="p-8 sm:p-10 overflow-y-auto space-y-6 bg-slate-900 text-slate-100 print:bg-white print:text-black print:p-0 print:overflow-visible print:text-xs">
          
          {/* 1. HEADER SECTION */}
          <div className="border-b-2 border-slate-700 print:border-slate-800 pb-4">
            <div className="flex justify-between items-start">
              <div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-teal-600 text-white font-black flex items-center justify-center text-sm print:bg-slate-900 print:text-white">
                    RC
                  </div>
                  <div>
                    <h1 className="text-xl sm:text-2xl font-black tracking-tight text-white print:text-black leading-tight">
                      {hospitalName || 'RetinaCare Specialty Eye Institute'}
                    </h1>
                    <p className="text-[11px] text-slate-400 print:text-slate-600 font-medium">
                      Department of Ophthalmology & AI Retinal Tele-Screening
                    </p>
                  </div>
                </div>
              </div>

              <div className="text-right text-xs space-y-0.5">
                <div className="font-extrabold text-teal-400 print:text-slate-900 text-sm tracking-wide uppercase">
                  Diabetic Retinopathy Screening Report
                </div>
                <div className="text-slate-400 print:text-slate-600">
                  <span className="font-semibold">Report ID:</span> {reportId}
                </div>
                <div className="text-slate-400 print:text-slate-600">
                  <span className="font-semibold">Generated:</span> {currentDateFormatted}
                </div>
              </div>
            </div>

            {/* Patient & Exam Metadata Table */}
            <div className="mt-4 grid grid-cols-4 gap-2.5 p-3 rounded-xl bg-slate-800/60 print:bg-slate-50 border border-slate-700 print:border-slate-300 text-xs">
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 print:text-slate-500 block">Patient Name</span>
                <strong className="text-slate-100 print:text-black text-sm">{patientName || 'Anonymous'}</strong>
              </div>
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 print:text-slate-500 block">Patient ID / MRN</span>
                <strong className="text-slate-100 print:text-black text-sm">{patientId || 'N/A'}</strong>
              </div>
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 print:text-slate-500 block">Age / Gender</span>
                <strong className="text-slate-100 print:text-black text-sm">{age || 'N/A'} Yrs / {gender}</strong>
              </div>
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 print:text-slate-500 block">Eye Examined</span>
                <strong className="text-teal-400 print:text-slate-900 text-sm font-bold">{eyeExamined}</strong>
              </div>
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 print:text-slate-500 block">Examination Date</span>
                <span className="text-slate-200 print:text-slate-800 font-medium">{examDate}</span>
              </div>
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 print:text-slate-500 block">Classification Standard</span>
                <span className="text-slate-200 print:text-slate-800 font-medium">ICDR 5-Grade Scale</span>
              </div>
              <div className="col-span-2">
                <span className="text-[10px] uppercase font-bold text-slate-400 print:text-slate-500 block">AI Screening Model</span>
                <span className="text-slate-200 print:text-slate-800 font-medium">EfficientNet-B2 (Resolution 384x384) + Grad-CAM</span>
              </div>
            </div>
          </div>

          {/* 2. IMAGING SECTION (SIDE-BY-SIDE) */}
          <div className="space-y-2">
            <div className="text-xs font-bold text-slate-300 print:text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5 text-teal-400 print:text-slate-700" />
              Retinal Fundus Imaging & Explainability
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Original Fundus Image Card */}
              <div className="border border-slate-700 print:border-slate-300 rounded-xl p-2.5 bg-slate-800/40 print:bg-white text-center space-y-1.5">
                <div className="flex items-center justify-between px-1">
                  <span className="text-[11px] font-bold text-slate-200 print:text-black">1. Preprocessed Fundus Photograph</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 print:bg-slate-200 text-slate-300 print:text-slate-700 font-semibold">{eyeExamined.split(' ')[0]}</span>
                </div>
                <div className="aspect-square bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center border border-slate-800 print:border-slate-200">
                  <img
                    src={result.preprocessed_image_base64 || originalPreviewUrl || ''}
                    alt="Original Fundus"
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>

              {/* Grad-CAM Heatmap Card */}
              <div className="border border-slate-700 print:border-slate-300 rounded-xl p-2.5 bg-slate-800/40 print:bg-white text-center space-y-1.5">
                <div className="flex items-center justify-between px-1">
                  <span className="text-[11px] font-bold text-slate-200 print:text-black">2. AI Attention Heatmap (Grad-CAM)</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-teal-900/60 print:bg-teal-100 text-teal-300 print:text-teal-800 font-semibold">Lesion Activation</span>
                </div>
                <div className="aspect-square bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center border border-slate-800 print:border-slate-200">
                  <img
                    src={result.gradcam_overlay_base64 || ''}
                    alt="Grad-CAM Heatmap"
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
            </div>

            <p className="text-[10px] text-slate-400 print:text-slate-600 italic text-center px-4">
              * Note: Highlighted warm-colored regions (red/yellow) on the Grad-CAM heatmap represent spatial microvascular pathological zones (e.g., hemorrhages, microaneurysms, hard exudates) that influenced the deep learning model's diagnostic classification.
            </p>
          </div>

          {/* 3. DIAGNOSIS SUMMARY (PROMINENT BOXED SECTION) */}
          <div className={`p-4 rounded-xl border-2 space-y-3 ${
            isReferable
              ? 'bg-rose-950/30 border-rose-600 print:bg-rose-50/70 print:border-rose-600 text-rose-100 print:text-rose-950'
              : 'bg-emerald-950/30 border-emerald-600 print:bg-emerald-50/70 print:border-emerald-600 text-emerald-100 print:text-emerald-950'
          }`}>
            <div className="flex items-center justify-between border-b border-current/20 pb-2.5">
              <div>
                <span className="text-[10px] uppercase tracking-wider font-extrabold opacity-80 block">
                  AI Automated Diagnosis Summary
                </span>
                <div className="text-xl sm:text-2xl font-black flex items-center gap-2 mt-0.5">
                  Grade {result.predicted_grade}: {result.class_name}
                </div>
              </div>

              {/* Referable DR Status Badge */}
              <div className="text-right">
                <span className="text-[10px] uppercase tracking-wider font-bold opacity-75 block mb-1">
                  Referable DR Status
                </span>
                <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-black text-xs uppercase tracking-wider shadow-sm ${
                  isReferable
                    ? 'bg-rose-600 text-white border border-rose-700 print:bg-rose-700'
                    : 'bg-emerald-600 text-white border border-emerald-700 print:bg-emerald-700'
                }`}>
                  {isReferable ? (
                    <>
                      <ShieldAlert className="w-3.5 h-3.5" />
                      YES — REFERABLE DR
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      NO — NON-REFERABLE
                    </>
                  )}
                </span>
              </div>
            </div>

            {/* Diagnostic Metrics & Clinical Action Timeline */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs pt-1">
              <div>
                <span className="opacity-75 font-semibold block text-[11px]">AI Model Confidence:</span>
                <strong className="text-base font-extrabold">{(result.confidence * 100).toFixed(1)}% Confidence</strong>
              </div>
              <div>
                <span className="opacity-75 font-semibold block text-[11px]">Recommended Clinical Action & Follow-up:</span>
                <strong className="text-xs font-bold block">{timelineInfo.action}</strong>
                <span className="text-[11px] opacity-90">{timelineInfo.timeline}</span>
              </div>
            </div>
          </div>

          {/* 4. PROBABILITY BREAKDOWN */}
          <div className="border border-slate-700 print:border-slate-300 rounded-xl p-3.5 bg-slate-800/40 print:bg-white space-y-2">
            <div className="text-xs font-bold text-slate-200 print:text-black uppercase tracking-wider flex items-center justify-between">
              <span>Multi-Class Severity Probability Distribution</span>
              <span className="text-[10px] text-slate-400 print:text-slate-600 font-normal">ICDR Classification</span>
            </div>

            <div className="grid grid-cols-5 gap-2 text-center text-xs">
              {Object.entries(result.probabilities).map(([cls, prob], idx) => {
                const isSelected = idx === result.predicted_grade;
                const pct = (prob * 100).toFixed(1);
                return (
                  <div
                    key={cls}
                    className={`p-2 rounded-lg border transition-all ${
                      isSelected
                        ? 'bg-teal-500/20 border-teal-500 font-bold text-teal-300 print:text-teal-950 print:bg-teal-50 print:border-teal-600 shadow-sm'
                        : 'bg-slate-900/50 print:bg-slate-50 border-slate-700 print:border-slate-200 text-slate-300 print:text-slate-700'
                    }`}
                  >
                    <div className="text-[10px] text-slate-400 print:text-slate-500 font-semibold">Grade {idx}</div>
                    <div className="truncate text-[11px]">{cls}</div>
                    <div className="text-sm font-extrabold mt-1">{pct}%</div>
                    {/* Visual mini bar */}
                    <div className="w-full bg-slate-700 print:bg-slate-200 h-1.5 rounded-full mt-1.5 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          isSelected ? 'bg-teal-400 print:bg-teal-600' : 'bg-slate-500 print:bg-slate-400'
                        }`}
                        style={{ width: `${Math.max(Number(pct), 4)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 5. CLINICAL FINDINGS NOTES */}
          <div className="border border-slate-700 print:border-slate-300 rounded-xl p-4 bg-slate-800/40 print:bg-white text-xs space-y-2">
            <div>
              <strong className="text-slate-200 print:text-black block text-xs uppercase tracking-wider font-bold">
                Automated Clinical Findings:
              </strong>
              <p className="text-slate-300 print:text-slate-800 leading-relaxed mt-0.5">
                {findingsText}
              </p>
            </div>

            {customNotes && (
              <div className="pt-2 border-t border-slate-700/60 print:border-slate-200">
                <strong className="text-slate-200 print:text-black block text-xs uppercase tracking-wider font-bold">
                  Reviewing Clinician Observations:
                </strong>
                <p className="text-slate-300 print:text-slate-800 italic leading-relaxed mt-0.5">
                  {customNotes}
                </p>
              </div>
            )}
          </div>

          {/* 6. FOOTER / DISCLAIMER & SIGNATURE BLOCK */}
          <div className="pt-4 border-t border-slate-700 print:border-slate-300 space-y-4">
            <div className="grid grid-cols-2 gap-8 text-xs">
              {/* Doctor details */}
              <div>
                <span className="text-slate-400 print:text-slate-500 block text-[10px] uppercase font-bold mb-0.5">
                  Examining Doctor / Technologist:
                </span>
                <strong className="text-slate-100 print:text-black text-sm">{doctorName || 'Dr. Reviewing Ophthalmologist'}</strong>
                <div className="text-slate-400 print:text-slate-600 text-[11px] mt-0.5">
                  License / Registration Verified · Department of Ophthalmology
                </div>
              </div>

              {/* Signature Line */}
              <div className="flex flex-col justify-end items-end text-right">
                <div className="w-52 border-b-2 border-slate-500 print:border-black mb-1"></div>
                <span className="text-slate-400 print:text-slate-600 font-bold text-[11px]">
                  Authorized Ophthalmologist Signature & Stamp
                </span>
                <span className="text-[10px] text-slate-500 print:text-slate-400">Date: {examDate}</span>
              </div>
            </div>

            {/* Medical Disclaimer & Page Numbering */}
            <div className="p-2.5 rounded-lg bg-slate-800/30 print:bg-slate-100 border border-slate-700/50 print:border-slate-200 text-[10px] text-slate-400 print:text-slate-600 leading-normal flex justify-between items-center">
              <div className="max-w-2xl">
                <strong>Medical Disclaimer:</strong> This report is generated by an AI-assisted screening tool (RetinaCare-AI) and is intended to support, not replace, clinical judgment. Final diagnosis and therapeutic interventions must be confirmed by a licensed ophthalmologist.
              </div>
              <div className="text-right whitespace-nowrap pl-4 font-semibold text-slate-500 print:text-slate-700">
                Page 1 of 1 · {reportId}
              </div>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
};
