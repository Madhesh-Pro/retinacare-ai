# RetinaCare-AI: Real AI-Powered Diabetic Retinopathy Screening Prototype

A production-grade AI prototype for automated **Diabetic Retinopathy (DR)** 5-stage classification and **Grad-CAM visual explainability**, designed for the Smart India Hackathon (SIH) project.

---

## 🎯 Clinical Grading & Referable DR Standard

RetinaCare-AI classifies retinal fundus images based on the **International Clinical Diabetic Retinopathy (ICDR)** 5-class scale:

| Grade | Clinical Category | Primary Diagnostic Findings | Clinical Action |
|---|---|---|---|
| **0** | **No DR** | No apparent microvascular abnormalities | Annual routine screening |
| **1** | **Mild DR** | Microaneurysms only | Follow-up in 6–12 months |
| **2** | **Moderate DR** | Microaneurysms, dot-blot hemorrhages, hard exudates | **Referable** (Ophthalmologist in 4–8 wks) |
| **3** | **Severe DR** | >20 intraretinal hemorrhages in 4 quadrants, venous beading, IRMA | **Referable** (Urgent referral in 2–4 wks) |
| **4** | **Proliferative DR (PDR)** | Neovascularization, vitreous/preretinal hemorrhage | **Referable** (Immediate specialist intervention) |

> **Referable DR (RDR) Threshold**: Grades **2 (Moderate)**, **3 (Severe)**, and **4 (Proliferative)** require specialist referral.

---

## 🏗️ Project Architecture

```
RetinaCare-AI/
├── ai/
│   ├── datasets/           # PyTorch Dataset loaders & Albumentations augmentations
│   ├── preprocessing/      # Circle cropping, CLAHE, Ben Graham Gaussian normalization
│   ├── models/             # EfficientNet-B2 transfer learning architecture
│   ├── training/           # Loss functions, optimizers, learning rate schedulers
│   ├── evaluation/         # QWK (Quadratic Weighted Kappa), multiclass metrics, Referable DR metrics
│   ├── explainability/     # Layer-targeted Grad-CAM attention heatmap extraction
│   └── inference/          # DRPredictor runtime engine (real forward pass + base64 overlays)
│
├── backend/
│   ├── main.py             # FastAPI production server entrypoint
│   ├── api/routes.py       # REST endpoints: /api/predict, /api/health, /api/classes
│   └── services/           # Singleton model management service
│
├── frontend/               # React + TypeScript + Tailwind CSS UI (Vite)
│   ├── src/
│   │   ├── components/     # ImageUploader, DiagnosticCard, ProbabilityChart, GradCAMComparison, etc.
│   │   ├── services/api.ts # API client for FastAPI backend
│   │   ├── App.tsx         # Clinical screening dashboard
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
│
├── scripts/
│   ├── inspect_dataset.py  # Comprehensive dataset health & corruption checker
│   ├── prepare_dataset.py  # Circle crop, CLAHE normalization, and stratified splitting
│   ├── train.py            # EfficientNet-B2 training loop with QWK tracking
│   ├── evaluate.py         # Test split evaluation and confusion matrix generator
│   ├── predict.py          # Standalone CLI prediction tool
│   └── gradcam.py          # Standalone Grad-CAM visualizer
│
├── configs/
│   └── dataset_config.yaml # Configurable dataset paths & parameters
├── checkpoints/            # Directory for trained PyTorch checkpoints (best_model.pth)
├── results/                # Saved reports, training history, and Grad-CAM visualizations
├── tests/                  # Pytest test suite for end-to-end verification
├── data/
│   ├── raw/                # Raw APTOS 2019 dataset (train.csv, train_images/)
│   ├── processed/          # Preprocessed 512x512 images & split CSVs
│   └── samples/            # Test fundus photographs
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start: Running the Full Prototype

### 1. Start the FastAPI Backend
```bash
# In project root:
python backend/main.py
# Or using uvicorn:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
* Backend API will be live at: **http://localhost:8000**
* Interactive Swagger API Docs: **http://localhost:8000/docs**

### 2. Start the React Frontend
```bash
# Open a second terminal:
cd frontend
npm install
npm run dev
```
* Open your browser at: **http://localhost:3000** (or port displayed by Vite)

---

## 🧪 CLI Pipeline Guide

### Step 1: Inspect Dataset Health
```bash
python scripts/inspect_dataset.py --csv-path data/raw/train.csv --images-dir data/raw/train_images/
```

### Step 2: Preprocess & Generate Stratified Splits
Crops the retinal circle, strips black borders, applies Ben Graham illumination normalization, and creates balanced Train (70%), Val (15%), and Test (15%) splits:
```bash
python scripts/prepare_dataset.py --image-size 512 --workers 4
```

### Step 3: Train EfficientNet-B2
Trains the model with transfer learning, dynamic class weighting for imbalance, Cosine Annealing learning rate schedule, and saves the best model by validation Quadratic Weighted Kappa (QWK):
```bash
python scripts/train.py --epochs 15 --batch-size 16 --lr 3e-4 --image-size 512
```

### Step 4: Evaluate Checkpoint
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --test-csv data/processed/test_split.csv
```

### Step 5: Standalone Single-Image Inference
```bash
python scripts/predict.py --image-path data/samples/sample_fundus.png
```

### Step 6: Generate Standalone Grad-CAM Visualizations
```bash
python scripts/gradcam.py --image-path data/samples/sample_fundus.png --output results/gradcam_explanation.png
```

---

## 🔬 Testing & Verification

Run the automated test suite verifying preprocessing, model forward pass, Grad-CAM extraction, inference engine, and FastAPI endpoints:

```bash
python -m pytest tests/test_pipeline.py
```
