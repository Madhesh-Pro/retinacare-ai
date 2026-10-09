"""
Unit & Integration Tests for RetinaCare-AI Pipeline
===================================================
"""

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from ai.evaluation import calculate_metrics
from ai.explainability import GradCAM, overlay_gradcam
from ai.inference import DRPredictor
from ai.models import EfficientNetB2_DR
from ai.preprocessing import preprocess_fundus_image
from backend.main import app


def test_preprocessing():
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.circle(img, (400, 300), 250, (120, 80, 50), -1)
    cv2.circle(img, (350, 250), 30, (200, 180, 100), -1)
    preprocessed = preprocess_fundus_image(img, target_size=512)
    assert preprocessed.shape == (512, 512, 3)


def test_model_forward():
    model = EfficientNetB2_DR(num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.shape == (2, 5)


def test_gradcam():
    model = EfficientNetB2_DR(num_classes=5, pretrained=False)
    target_layer = model.get_last_conv_layer()
    gradcam = GradCAM(model, target_layer)
    x = torch.randn(1, 3, 512, 512)
    cam, pred_class, conf, probs = gradcam.generate_cam(x)
    gradcam.remove_hooks()
    assert cam.shape == (512, 512)
    assert 0 <= pred_class <= 4
    assert 0.0 <= conf <= 1.0


def test_predictor_and_metrics():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 200, (140, 90, 60), -1)
    predictor = DRPredictor(checkpoint_path=None)
    res = predictor.predict(img, generate_gradcam=True)

    assert 0 <= res.predicted_grade <= 4
    assert res.is_referable == (res.predicted_grade >= 2)
    assert res.gradcam_overlay_base64 is not None
    assert len(res.probabilities) == 5

    # Test metrics
    y_true = [0, 1, 2, 3, 4, 0, 2, 4]
    y_pred = [0, 1, 2, 2, 4, 0, 3, 4]
    m = calculate_metrics(y_true, y_pred)
    assert "qwk" in m
    assert "accuracy" in m
    assert "referable_dr" in m


def test_fastapi_backend():
    client = TestClient(app)
    # Test health
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "EfficientNet" in data["service"] or "RetinaCare" in data["service"]

    # Test classes
    res_cls = client.get("/api/classes")
    assert res_cls.status_code == 200
    assert len(res_cls.json()["classes"]) == 5

    # Test predict with mock image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 80, (100, 80, 50), -1)
    _, buffer = cv2.imencode(".png", img)
    files = {"image": ("test_retina.png", buffer.tobytes(), "image/png")}

    res_pred = client.post("/api/predict", files=files)
    assert res_pred.status_code == 200
    pred_data = res_pred.json()
    assert "predicted_grade" in pred_data
    assert "probabilities" in pred_data
    assert "gradcam_overlay_base64" in pred_data

    # Test PDF generation endpoint
    report_payload = {
        "patient_name": "Test Patient",
        "patient_id": "MRN-TEST-101",
        "age": "45",
        "gender": "Female",
        "eye_examined": "OD (Right Eye)",
        "hospital_name": "Test Eye Clinic",
        "doctor_name": "Dr. Test, MD",
        "predicted_grade": pred_data["predicted_grade"],
        "class_name": pred_data["class_name"],
        "is_referable": pred_data["is_referable"],
        "confidence": pred_data["confidence"],
        "probabilities": pred_data["probabilities"],
        "clinical_recommendation": pred_data["clinical_recommendation"],
        "gradcam_overlay_base64": pred_data["gradcam_overlay_base64"],
        "preprocessed_image_base64": pred_data["preprocessed_image_base64"],
    }
    res_pdf = client.post("/api/generate-report", json=report_payload)
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert len(res_pdf.content) > 1000  # Non-empty PDF bytes
