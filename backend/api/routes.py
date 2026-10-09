"""
FastAPI Routes for RetinaCare-AI Screening Backend
===================================================
"""

import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ai.evaluation.metrics import DR_CLASS_LABELS
from backend.services.model_service import ModelService
from backend.services.pdf_service import generate_clinical_pdf

router = APIRouter(prefix="/api", tags=["Diabetic Retinopathy Screening"])

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/bmp",
    "image/webp",
    "application/octet-stream",
}


class PredictionResponse(BaseModel):
    filename: str
    predicted_grade: int
    class_name: str
    is_referable: bool
    confidence: float
    probabilities: Dict[str, float]
    clinical_recommendation: str
    gradcam_overlay_base64: str | None = None
    preprocessed_image_base64: str | None = None


class ReportGenerateRequest(BaseModel):
    patient_name: str = "Ananya Verma"
    patient_id: str = "MRN-849201"
    age: str = "58"
    gender: str = "Female"
    eye_examined: str = "OD (Right Eye)"
    hospital_name: str = "Apollo Specialty Eye Institute & Retinal Care"
    doctor_name: str = "Dr. Priya Ramanathan, MS (Ophthalmology)"
    exam_date: str | None = None
    custom_notes: str | None = ""
    predicted_grade: int = 0
    class_name: str = "No DR"
    is_referable: bool = False
    confidence: float = 0.95
    probabilities: Dict[str, float] = {}
    clinical_recommendation: str = ""
    gradcam_overlay_base64: str | None = None
    preprocessed_image_base64: str | None = None


@router.get("/health")
def health_check():
    predictor = ModelService.get_predictor()
    return {
        "status": "healthy",
        "service": "RetinaCare-AI Screening Engine",
        "device": str(predictor.device),
        "checkpoint_loaded": predictor.checkpoint_loaded,
        "classes": DR_CLASS_LABELS,
    }


@router.get("/classes")
def get_classes():
    return {
        "classes": [
            {
                "grade": 0,
                "name": "No Diabetic Retinopathy",
                "is_referable": False,
                "description": "No microvascular abnormalities.",
            },
            {
                "grade": 1,
                "name": "Mild Diabetic Retinopathy",
                "is_referable": False,
                "description": "Microaneurysms only.",
            },
            {
                "grade": 2,
                "name": "Moderate Diabetic Retinopathy",
                "is_referable": True,
                "description": "More than microaneurysms but less than Severe DR.",
            },
            {
                "grade": 3,
                "name": "Severe Diabetic Retinopathy",
                "is_referable": True,
                "description": ">20 intraretinal hemorrhages in 4 quadrants, venous beading, or IRMA.",
            },
            {
                "grade": 4,
                "name": "Proliferative Diabetic Retinopathy (PDR)",
                "is_referable": True,
                "description": "Neovascularization, vitreous/preretinal hemorrhage.",
            },
        ]
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_retina_image(
    image: UploadFile = File(..., description="Retinal fundus image (PNG, JPG, JPEG)"),
    generate_gradcam: bool = Form(True, description="Generate Grad-CAM explainability heatmap"),
):
    """
    Accepts uploaded retinal fundus photograph, runs real AI inference,
    and returns DR severity grade, probabilities, referability status, and Grad-CAM overlay.
    """
    # Validate content type / extension
    filename = image.filename or "uploaded_image.png"
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ["png", "jpg", "jpeg", "bmp", "tiff", "webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '.{ext}'. Please upload a PNG, JPG, or JPEG retinal image.",
        )

    try:
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        predictor = ModelService.get_predictor()
        res = predictor.predict(contents, generate_gradcam=generate_gradcam)

        return PredictionResponse(
            filename=filename,
            predicted_grade=res.predicted_grade,
            class_name=res.class_name,
            is_referable=res.is_referable,
            confidence=res.confidence,
            probabilities=res.probabilities,
            clinical_recommendation=res.clinical_recommendation,
            gradcam_overlay_base64=res.gradcam_overlay_base64,
            preprocessed_image_base64=res.preprocessed_image_base64,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed on image '{filename}': {str(e)}",
        )


@router.post("/generate-report")
async def generate_pdf_report(request: ReportGenerateRequest):
    """
    Generates a hospital-grade diagnostic PDF report from screening results and patient metadata.
    Returns downloadable PDF attachment.
    """
    try:
        pdf_bytes = generate_clinical_pdf(request.model_dump())
        safe_id = "".join(c for c in request.patient_id if c.isalnum() or c in "-_") or "Patient"
        filename = f"RetinaCare_Report_{safe_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate clinical PDF report: {str(e)}",
        )

