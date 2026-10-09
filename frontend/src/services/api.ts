import { HealthResponse, PredictionResponse, ReportGeneratePayload } from '../types';

const API_BASE = 'http://localhost:8000/api';

export async function checkBackendHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status: ${res.status}`);
  }
  return res.json();
}

export async function predictFundusImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('generate_gradcam', 'true');

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Network error or invalid server response' }));
    throw new Error(errorData.detail || 'Inference request failed');
  }

  return res.json();
}

export async function downloadClinicalReportPdf(payload: ReportGeneratePayload): Promise<void> {
  const res = await fetch(`${API_BASE}/generate-report`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'PDF generation failed on server' }));
    throw new Error(errorData.detail || 'Failed to generate PDF report');
  }

  const blob = await res.blob();
  const blobUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = blobUrl;
  const safeId = payload.patient_id.replace(/[^a-zA-Z0-9-_]/g, '_') || 'Patient';
  a.download = `RetinaCare_Report_${safeId}.pdf`;
  document.body.appendChild(a);
  a.click();

  setTimeout(() => {
    window.URL.revokeObjectURL(blobUrl);
    document.body.removeChild(a);
  }, 300);
}
