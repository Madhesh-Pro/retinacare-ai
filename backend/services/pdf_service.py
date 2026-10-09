"""
RetinaCare-AI - Professional Hospital Diagnostic PDF Generation Service
========================================================================
Generates a vector-sharp, pixel-consistent, clinical diagnostic PDF report
using ReportLab following ophthalmic tele-screening standards.
"""

import base64
import io
import os
from datetime import datetime
from typing import Any, Dict, Optional

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

GRADE_FINDINGS = {
    0: "Normal retinal microvascular architecture. No microaneurysms, intraretinal hemorrhages, hard lipid exudates, or cotton-wool spots detected within the examined fundus field.",
    1: "Early microvascular pathology identified. Presence of isolated microaneurysms (focal red lesions) without evidence of hard exudation, macular edema, or significant retinal ischemia.",
    2: "Moderate diabetic retinopathy identified. Presence of multiple intraretinal dot-blot hemorrhages and/or hard yellow lipid exudate clusters in the macular and perimacular regions.",
    3: "Severe non-proliferative diabetic retinopathy identified. Extensive intraretinal hemorrhages across multiple quadrants, venous beading, and soft exudates (cotton-wool spots) indicative of pronounced retinal ischemia.",
    4: "Proliferative diabetic retinopathy (PDR) identified. Active retinal neovascularization (fragile new vessel fronds) and/or preretinal/vitreous hemorrhages detected, presenting severe risk of vision loss.",
}

GRADE_TIMELINES = {
    0: ("Routine Diabetic Screening", "Annual routine screening (12 months)"),
    1: ("Clinical Follow-up & Glycemic Control", "6 to 12 months with optimized HbA1c control"),
    2: ("Comprehensive Ophthalmologist Referral", "Referral evaluation within 4 to 8 weeks"),
    3: ("Urgent Retinal Specialist Referral", "Urgent evaluation within 2 to 4 weeks (evaluate for anti-VEGF / PRP)"),
    4: ("Immediate Emergency Specialist Intervention", "Immediate specialist consultation (High risk of vision loss)"),
}


def _decode_base64_image(b64_str: Optional[str]) -> Optional[io.BytesIO]:
    """Decodes a base64 image data URL or raw string into an in-memory BytesIO stream."""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        # Verify it's a valid PIL image
        pil_img = PILImage.open(io.BytesIO(img_bytes))
        pil_img.verify()
        return io.BytesIO(img_bytes)
    except Exception:
        return None


def generate_clinical_pdf(data: Dict[str, Any]) -> bytes:
    """
    Builds a standalone, hospital-grade A4/Letter PDF report.
    Returns the generated PDF as raw bytes.
    """
    buffer = io.BytesIO()

    # Document setup: Letter size with 0.45 inch (32.4 pt) margins for dense, single-page fit
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=32,
        rightMargin=32,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography & Styles
    primary_color = colors.HexColor("#0f172a")     # Slate 900
    accent_teal = colors.HexColor("#0d9488")       # Teal 600
    text_dark = colors.HexColor("#1e293b")         # Slate 800
    text_muted = colors.HexColor("#64748b")        # Slate 500
    bg_light = colors.HexColor("#f8fafc")          # Slate 50
    border_color = colors.HexColor("#cbd5e1")      # Slate 300

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=primary_color,
    )
    hospital_style = ParagraphStyle(
        "HospitalName",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=accent_teal,
    )
    subtitle_style = ParagraphStyle(
        "HospitalSubtitle",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=text_muted,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=primary_color,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=text_dark,
    )
    body_bold = ParagraphStyle(
        "BodyDarkBold",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=text_dark,
    )
    meta_label = ParagraphStyle(
        "MetaLabel",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=text_muted,
    )
    meta_val = ParagraphStyle(
        "MetaVal",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=primary_color,
    )
    caption_style = ParagraphStyle(
        "CaptionText",
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=9.5,
        textColor=text_muted,
        alignment=1, # Center
    )

    story = []

    # Extract Data
    patient_name = data.get("patient_name") or "Ananya Verma"
    patient_id = data.get("patient_id") or "MRN-849201"
    age = str(data.get("age") or "58")
    gender = data.get("gender") or "Female"
    eye_examined = data.get("eye_examined") or "OD (Right Eye)"
    hospital_name = data.get("hospital_name") or "Apollo Specialty Eye Institute & Retinal Care"
    doctor_name = data.get("doctor_name") or "Dr. Priya Ramanathan, MS (Ophthalmology)"
    exam_date = data.get("exam_date") or datetime.now().strftime("%Y-%m-%d")
    custom_notes = data.get("custom_notes") or ""
    
    grade = int(data.get("predicted_grade", 0))
    class_name = data.get("class_name") or f"Grade {grade}"
    confidence = float(data.get("confidence", 0.95))
    is_referable = bool(data.get("is_referable", grade >= 2))
    probabilities = data.get("probabilities") or {
        "No DR": 0.0, "Mild DR": 0.0, "Moderate DR": 0.0, "Severe DR": 0.0, "Proliferative DR": 0.0
    }

    rand_id_suffix = abs(hash(f"{patient_id}_{exam_date}")) % 9000 + 1000
    date_str = datetime.now().strftime("%Y%m%d")
    report_id = f"RC-REP-{date_str}-{rand_id_suffix}"

    # ---------------------------------------------------------
    # 1. HEADER SECTION
    # ---------------------------------------------------------
    header_left = [
        Paragraph(hospital_name, hospital_style),
        Paragraph("Department of Ophthalmology & AI Retinal Diagnostics", subtitle_style),
    ]
    header_right = [
        Paragraph("DIABETIC RETINOPATHY SCREENING REPORT", ParagraphStyle("HR_Title", fontName="Helvetica-Bold", fontSize=10, leading=12, alignment=2, textColor=primary_color)),
        Paragraph(f"<b>Report ID:</b> {report_id}", ParagraphStyle("HR_Sub1", fontName="Helvetica", fontSize=8, leading=10, alignment=2, textColor=text_muted)),
        Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y')}", ParagraphStyle("HR_Sub2", fontName="Helvetica", fontSize=8, leading=10, alignment=2, textColor=text_muted)),
    ]

    header_table = Table([[header_left, header_right]], colWidths=[330, 218])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_teal, spaceAfter=6, spaceBefore=2))

    # Patient Metadata Box
    meta_data = [
        [
            Paragraph("PATIENT NAME", meta_label),
            Paragraph("PATIENT ID / MRN", meta_label),
            Paragraph("AGE / GENDER", meta_label),
            Paragraph("EYE EXAMINED", meta_label),
        ],
        [
            Paragraph(patient_name, meta_val),
            Paragraph(patient_id, meta_val),
            Paragraph(f"{age} Yrs / {gender}", meta_val),
            Paragraph(f"<b>{eye_examined}</b>", meta_val),
        ],
        [
            Paragraph("EXAMINATION DATE", meta_label),
            Paragraph("CLASSIFICATION STANDARD", meta_label),
            Paragraph("SCREENING MODEL", meta_label),
            Paragraph("CONFIDENCE LEVEL", meta_label),
        ],
        [
            Paragraph(exam_date, body_style),
            Paragraph("ICDR 5-Grade Standard", body_style),
            Paragraph("EfficientNet-B2 + Grad-CAM", body_style),
            Paragraph(f"{confidence*100:.1f}% AI Confidence", body_bold),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[137, 137, 137, 137])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_light),
        ("BOX", (0, 0), (-1, -1), 0.8, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # 2. IMAGING SECTION (SIDE-BY-SIDE)
    # ---------------------------------------------------------
    story.append(Paragraph("RETINAL FUNDUS IMAGING & EXPLAINABILITY", section_heading))

    preproc_stream = _decode_base64_image(data.get("preprocessed_image_base64"))
    gradcam_stream = _decode_base64_image(data.get("gradcam_overlay_base64"))

    img_size = 180  # points (~2.5 inches square)

    if preproc_stream:
        img_preproc = Image(preproc_stream, width=img_size, height=img_size)
    else:
        img_preproc = Paragraph("<i>Fundus image not available</i>", body_style)

    if gradcam_stream:
        img_gradcam = Image(gradcam_stream, width=img_size, height=img_size)
    else:
        img_gradcam = Paragraph("<i>Heatmap not available</i>", body_style)

    images_data = [
        [
            Paragraph(f"<b>1. Preprocessed Fundus Photograph ({eye_examined.split(' ')[0]})</b>", ParagraphStyle("ImgH1", fontName="Helvetica-Bold", fontSize=8, alignment=1, textColor=primary_color)),
            Paragraph("<b>2. AI Attention Heatmap (Grad-CAM Overlay)</b>", ParagraphStyle("ImgH2", fontName="Helvetica-Bold", fontSize=8, alignment=1, textColor=primary_color)),
        ],
        [
            img_preproc,
            img_gradcam,
        ],
    ]
    images_table = Table(images_data, colWidths=[274, 274])
    images_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#020617")),
        ("BOX", (0, 0), (0, 1), 0.8, border_color),
        ("BOX", (1, 0), (1, 1), 0.8, border_color),
    ]))
    story.append(images_table)

    story.append(Spacer(1, 3))
    caption_text = (
        "* Note: Highlighted warm-colored regions (red/yellow) on the Grad-CAM heatmap represent spatial microvascular "
        "pathological zones (e.g., hemorrhages, microaneurysms, hard exudates) that influenced the deep learning model's diagnosis."
    )
    story.append(Paragraph(caption_text, caption_style))
    story.append(Spacer(1, 7))

    # ---------------------------------------------------------
    # 3. DIAGNOSIS SUMMARY (PROMINENT BOX)
    # ---------------------------------------------------------
    diag_bg = colors.HexColor("#fff1f2") if is_referable else colors.HexColor("#f0fdf4")
    diag_border = colors.HexColor("#e11d48") if is_referable else colors.HexColor("#16a34a")
    badge_bg = colors.HexColor("#be123c") if is_referable else colors.HexColor("#15803d")

    badge_text = "YES - REFERABLE DR (Specialist Referral Required)" if is_referable else "NO - NON-REFERABLE (Routine Monitoring)"
    timeline_action, timeline_desc = GRADE_TIMELINES.get(grade, ("Clinical Referral", "Consult an ophthalmologist"))

    diag_content = [
        [
            Paragraph("<font size=7 color='#64748b'><b>AI AUTOMATED CLINICAL DIAGNOSIS:</b></font>", body_style),
            Paragraph(f"<font color='white'><b>{badge_text}</b></font>", ParagraphStyle("BadgeStyle", fontName="Helvetica-Bold", fontSize=8, alignment=2, backColor=badge_bg, borderPadding=3)),
        ],
        [
            Paragraph(f"<font size=13><b>Grade {grade}: {class_name}</b></font> ({confidence*100:.1f}% Confidence)", ParagraphStyle("DiagTitle", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=primary_color)),
            Paragraph(f"<b>Recommended Action:</b> {timeline_action}<br/><font color='#475569'>{timeline_desc}</font>", ParagraphStyle("ActionStyle", fontName="Helvetica", fontSize=8, leading=10, alignment=2, textColor=text_dark)),
        ],
    ]

    diag_table = Table(diag_content, colWidths=[310, 238])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), diag_bg),
        ("BOX", (0, 0), (-1, -1), 1.2, diag_border),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # 4. PROBABILITY BREAKDOWN
    # ---------------------------------------------------------
    story.append(Paragraph("MULTI-CLASS SEVERITY PROBABILITY BREAKDOWN", section_heading))

    prob_headers = ["Grade 0 (No DR)", "Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)", "Grade 4 (PDR)"]
    prob_keys = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    
    prob_values = []
    for idx, k in enumerate(prob_keys):
        p = probabilities.get(k, 0.0)
        p_pct = f"{p*100:.1f}%"
        if idx == grade:
            prob_values.append(Paragraph(f"<b>{p_pct}</b> (Predicted)", ParagraphStyle("PvSelected", fontName="Helvetica-Bold", fontSize=8, alignment=1, textColor=accent_teal)))
        else:
            prob_values.append(Paragraph(f"{p_pct}", ParagraphStyle("PvNormal", fontName="Helvetica", fontSize=8, alignment=1, textColor=text_muted)))

    prob_data = [
        [Paragraph(f"<b>{h}</b>", ParagraphStyle("Ph", fontName="Helvetica-Bold", fontSize=7.5, alignment=1, textColor=primary_color)) for h in prob_headers],
        prob_values,
    ]
    prob_table = Table(prob_data, colWidths=[109.6, 109.6, 109.6, 109.6, 109.6])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_light),
        ("BOX", (0, 0), (-1, -1), 0.8, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # 5. CLINICAL FINDINGS NOTES
    # ---------------------------------------------------------
    findings_text = GRADE_FINDINGS.get(grade, GRADE_FINDINGS[0])
    findings_cells = [
        [Paragraph("<b>AUTOMATED OPHTHALMOLOGIC FINDINGS:</b>", ParagraphStyle("Fh", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=primary_color))],
        [Paragraph(findings_text, body_style)],
    ]
    if custom_notes.strip():
        findings_cells.append([Paragraph(f"<b>REVIEWING CLINICIAN IMPRESSION:</b> <i>{custom_notes.strip()}</i>", body_style)])

    findings_table = Table(findings_cells, colWidths=[548])
    findings_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # 6. FOOTER / DISCLAIMER & SIGNATURE BLOCK
    # ---------------------------------------------------------
    sig_left = [
        Paragraph("<b>EXAMINING CLINICIAN / TECHNOLOGIST:</b>", meta_label),
        Paragraph(doctor_name, body_bold),
        Paragraph("Registered Medical Practitioner · Department of Ophthalmology", subtitle_style),
    ]
    sig_right = [
        Spacer(1, 10),
        Paragraph("____________________________________________", ParagraphStyle("SigL", fontName="Helvetica", fontSize=8, alignment=2, textColor=text_muted)),
        Paragraph("<b>Authorized Reviewing Ophthalmologist Signature & Stamp</b>", ParagraphStyle("SigT", fontName="Helvetica-Bold", fontSize=7.5, alignment=2, textColor=text_dark)),
        Paragraph(f"Date: {exam_date}", ParagraphStyle("SigD", fontName="Helvetica", fontSize=7, alignment=2, textColor=text_muted)),
    ]

    sig_table = Table([[sig_left, sig_right]], colWidths=[310, 238])
    sig_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 6))

    # Disclaimer Box
    disclaimer_text = (
        "<b>MEDICAL DISCLAIMER:</b> This report is generated by an artificial intelligence decision-support tool (RetinaCare-AI) "
        "and is intended to assist healthcare professionals in screening and triage. Definitive clinical diagnosis, "
        "OCT confirmation, and therapeutic interventions must be established by a licensed ophthalmologist."
    )
    disclaimer_p = Paragraph(disclaimer_text, ParagraphStyle("DiscP", fontName="Helvetica", fontSize=6.5, leading=8.5, textColor=text_muted))
    page_num_p = Paragraph(f"Page 1 of 1 · {report_id}", ParagraphStyle("PnP", fontName="Helvetica-Bold", fontSize=7, leading=9, alignment=2, textColor=text_muted))

    disc_table = Table([[disclaimer_p, page_num_p]], colWidths=[430, 118])
    disc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_light),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(disc_table)

    # Build PDF document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
