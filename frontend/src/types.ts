export interface PredictionResponse {
  filename: string;
  predicted_grade: number;
  class_name: string;
  is_referable: boolean;
  confidence: number;
  probabilities: Record<string, number>;
  clinical_recommendation: string;
  gradcam_overlay_base64: string | null;
  preprocessed_image_base64: string | null;
}

export interface HealthResponse {
  status: string;
  service: string;
  device: string;
  checkpoint_loaded: boolean;
  classes: string[];
}

export interface ReportGeneratePayload {
  patient_name: string;
  patient_id: string;
  age: string;
  gender: string;
  eye_examined: string;
  hospital_name: string;
  doctor_name: string;
  exam_date: string;
  custom_notes?: string;
  predicted_grade: number;
  class_name: string;
  is_referable: boolean;
  confidence: number;
  probabilities: Record<string, number>;
  clinical_recommendation: string;
  gradcam_overlay_base64: string | null;
  preprocessed_image_base64: string | null;
}
