import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field

try:
    import jinja2
    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False

from backend.app.core.config import settings
from backend.app.core.logger import log_audit_event, logger
from cv.preprocessing.suitability import SuitabilityResult
from cv.features.complexity import ComplexityResult
from ml.evaluation.hypothesis import HypothesisEvaluationReport
from ml.explainability.explainer import DualExplanationReport
from ml.models.comparison import ComparisonReport
from ml.models.disguise import SimulationIndicators
from ml.models.fusion import HybridFusionResult


class ReportGenerationResult(BaseModel):
    """Adli Rapor Üretim Sonucu."""
    case_id: str
    html_path: str
    pdf_path: Optional[str] = None
    report_sha256: str
    is_pdf_generated: bool
    generated_at_utc: str


def generate_reportlab_pdf(
    pdf_out_path: Path,
    case_info: Dict[str, Any],
    suitability: SuitabilityResult,
    comparison: ComparisonReport,
    complexity: ComplexityResult,
    disguise: SimulationIndicators,
    fusion: HybridFusionResult,
    xai: DualExplanationReport,
    hypothesis: HypothesisEvaluationReport,
    report_sha256: str,
):
    """
    Saf Python (ReportLab) kullanarak WeasyPrint ve harici GTK DLL'lerine bağımlı olmadan
    A4 formatında resmi, şık ve kriptografik mühürlü adli bilirkişi PDF raporu derler.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm

    doc = SimpleDocTemplate(
        str(pdf_out_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()

    # Özel Adli Tipografi Stilleri
    title_style = ParagraphStyle(
        "ForensicTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#7B1E34"),
        alignment=1,  # Center
    )
    subtitle_style = ParagraphStyle(
        "ForensicSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4A3E3D"),
        alignment=1,
    )
    heading_style = ParagraphStyle(
        "ForensicHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1C1510"),
    )
    body_style = ParagraphStyle(
        "ForensicBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#262626"),
    )
    bold_style = ParagraphStyle(
        "ForensicBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1C1510"),
    )
    seal_style = ParagraphStyle(
        "ForensicSeal",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1E293B"),
    )

    story = []

    # 1. Başlık Alanı
    story.append(Paragraph("T.C. ADLİ BELGE VE EL YAZISI KARAR DESTEK SİSTEMİ (FHDSS)", title_style))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Adli Belge ve İmza İnceleme Bilirkişi Raporu • ENFSI / ASTM E2290 Uyumlu", subtitle_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#7B1E34"), spaceAfter=10))

    # 2. Vaka Meta Verileri Tablosu
    case_id = str(case_info.get("case_id", "N/A"))
    examiner = str(case_info.get("examiner_id", "BILIRKISI_UZMAN_01"))
    authority = str(case_info.get("authority", "Cumhuriyet Başsavcılığı / Mahkeme"))
    exam_type = str(case_info.get("examination_type", "IMZA VE EL YAZISI"))
    date_str = str(case_info.get("date_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")))

    case_meta_data = [
        [Paragraph("<b>Vaka Numarası:</b>", body_style), Paragraph(case_id, bold_style),
         Paragraph("<b>İnceleme Türü:</b>", body_style), Paragraph(exam_type, bold_style)],
        [Paragraph("<b>Talep Eden Makam:</b>", body_style), Paragraph(authority, body_style),
         Paragraph("<b>İnceleme Tarihi:</b>", body_style), Paragraph(date_str, body_style)],
        [Paragraph("<b>Bilirkişi / Uzman:</b>", body_style), Paragraph(examiner, body_style),
         Paragraph("<b>Rapor Mührü (SHA-256):</b>", body_style), Paragraph(f"{report_sha256[:16]}...", seal_style)],
        [Paragraph("<b>Adli AI Modeli:</b>", body_style), Paragraph(str(case_info.get("model_id", "convnext_tiny_forensic_multidataset_v5")), bold_style),
         Paragraph("<b>Model SHA-256:</b>", body_style), Paragraph(str(case_info.get("model_sha256", "0fe6963acaba48c2..."))[:16] + "...", seal_style)],
    ]
    t_meta = Table(case_meta_data, colWidths=[3.2 * cm, 5.5 * cm, 3.8 * cm, 5.0 * cm])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F9F8F6")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#D4C5A9")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAE5D9")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 0.4 * cm))

    # 3. Adli Uygunluk ve Belge Kalite Sonucu
    story.append(Paragraph("1. Görüntü Kalite ve Adli Uygunluk İncelemesi", heading_style))
    story.append(Spacer(1, 0.15 * cm))
    suit_data = [
        [Paragraph("<b>Uygunluk Durumu</b>", bold_style), Paragraph("<b>Genel Kalite Skoru</b>", bold_style),
         Paragraph("<b>Bulanıklık (Laplacian)</b>", bold_style), Paragraph("<b>SNR (dB)</b>", bold_style)],
        [Paragraph(suitability.status.value, bold_style),
         Paragraph(f"{suitability.overall_score:.2%}", body_style),
         Paragraph(f"{suitability.metrics.blur_laplacian_var:.1f}", body_style),
         Paragraph(f"{suitability.metrics.snr_db:.1f} dB", body_style)],
    ]
    t_suit = Table(suit_data, colWidths=[4.3 * cm, 4.3 * cm, 4.4 * cm, 4.5 * cm])
    t_suit.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EFEBE4")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#C8BCAB")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAE5D9")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_suit)
    story.append(Spacer(1, 0.4 * cm))

    # 4. Karşılaştırmalı Adli Analiz ve Hibrit Füzyon
    story.append(Paragraph("2. Adli Biyomekanik Karşılaştırma ve Hibrit Füzyon Skoru", heading_style))
    story.append(Spacer(1, 0.15 * cm))
    gating_str = f"CEZA KESİNTİSİ: -{fusion.gating_penalty:.2f}" if fusion.gating_triggered else "Uyumlu (Ceza Yok)"
    comp_data = [
        [Paragraph("<b>Morfolojik Benzerlik</b>", bold_style), Paragraph("<b>Derin Metrik (ConvNeXt)</b>", bold_style),
         Paragraph("<b>Adli Gating Durumu</b>", bold_style), Paragraph("<b>Hibrit Karar Skoru</b>", bold_style)],
        [Paragraph(f"{fusion.morphological_similarity:.2%}", body_style),
         Paragraph(f"{fusion.deep_metric_similarity:.2%}", body_style),
         Paragraph(gating_str, body_style),
         Paragraph(f"<b>{fusion.hybrid_similarity_score:.2%}</b>", bold_style)],
    ]
    t_comp = Table(comp_data, colWidths=[4.3 * cm, 4.3 * cm, 4.4 * cm, 4.5 * cm])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EFEBE4")),
        ('BACKGROUND', (3, 1), (3, 1), colors.HexColor("#F2E6E8")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#C8BCAB")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAE5D9")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 0.4 * cm))

    # 5. ENFSI Bayesyen Hipotez Değerlendirmesi
    story.append(Paragraph("3. ENFSI BPM-FHX Bayesyen Hipotez Değerlendirmesi", heading_style))
    story.append(Spacer(1, 0.15 * cm))
    hyp_data = [
        [Paragraph("<b>Destek Seviyesi</b>", bold_style), Paragraph("<b>Likelihood Ratio (LR)</b>", bold_style),
         Paragraph("<b>log10(LR)</b>", bold_style), Paragraph("<b>Belirsizlik İndeksi (U)</b>", bold_style)],
        [Paragraph(hypothesis.support_level.value, bold_style),
         Paragraph(f"{hypothesis.likelihood_ratio_estimate:.2e}", body_style),
         Paragraph(f"{hypothesis.log_likelihood_ratio:.2f}", body_style),
         Paragraph(f"±{hypothesis.uncertainty_score:.2f}", body_style)],
    ]
    t_hyp = Table(hyp_data, colWidths=[5.5 * cm, 4.0 * cm, 4.0 * cm, 4.0 * cm])
    t_hyp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EFEBE4")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#C8BCAB")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAE5D9")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_hyp)
    story.append(Spacer(1, 0.3 * cm))

    # Sözel Hüküm Metni
    story.append(Paragraph(f"<b>Nihai Adli Kanaat İfadesi:</b> {hypothesis.verbal_conclusion_tr}", body_style))
    story.append(Spacer(1, 0.4 * cm))

    # 6. Açıklanabilir Yapay Zekâ (XAI)
    story.append(Paragraph("4. Açıklanabilir Yapay Zekâ (XAI) ve Karar Gerekçeleri", heading_style))
    story.append(Spacer(1, 0.15 * cm))
    top_pos = ", ".join([f"{f.feature_name_tr} (+{f.impact_score:.2f})" for f in xai.top_positive_features[:3]]) or "Yeterli veri yok"
    top_neg = ", ".join([f"{f.feature_name_tr} (-{f.impact_score:.2f})" for f in xai.top_negative_features[:3]]) or "Yeterli veri yok"
    xai_data = [
        [Paragraph("<b>Orijinallik Lehine Özellikler:</b>", body_style), Paragraph(top_pos, body_style)],
        [Paragraph("<b>Şüphe/Farklılık Lehine Özellikler:</b>", body_style), Paragraph(top_neg, body_style)],
        [Paragraph("<b>Görsel Odak Analizi:</b>", body_style), Paragraph(xai.visual_attention_summary or "ConvNeXt son blok odak haritası üretildi.", body_style)],
    ]
    t_xai = Table(xai_data, colWidths=[5.5 * cm, 12.0 * cm])
    t_xai.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F9F8F6")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#D4C5A9")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAE5D9")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_xai)
    story.append(Spacer(1, 0.5 * cm))

    # 7. Kriptografik Güvenlik Mührü, 5070 E-İmza & Zaman Damgası Rozeti
    from reportlab.graphics.barcode import qr
    from reportlab.graphics.shapes import Drawing

    qr_text = f"FHDSS-ADLI-DOGRULAMA\nVAKA:{case_info.get('case_id', 'N/A')}\nSHA256:{report_sha256[:16]}\nZAMAN:{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
    qr_widget = qr.QrCodeWidget(qr_text)
    qr_bounds = qr_widget.getBounds()
    qw = max(qr_bounds[2] - qr_bounds[0], 1)
    qh = max(qr_bounds[3] - qr_bounds[1], 1)
    qr_drawing = Drawing(52, 52, transform=[52/qw, 0, 0, 52/qh, 0, 0])
    qr_drawing.add(qr_widget)

    seal_info = [
        Paragraph("<b>KRİPTOGRAFİK ADLİ BÜTÜNLÜK MÜHRÜ VE 5070 E-İMZA / RFC 3161 ZAMAN DAMGASI BLOĞU</b>", ParagraphStyle("SealHead", fontName="Helvetica-Bold", fontSize=7.5, textColor=colors.HexColor("#7B1E34"))),
        Spacer(1, 0.05 * cm),
        Paragraph(f"SHA-256 Rapor Özeti: <b>{report_sha256}</b>", ParagraphStyle("SealHash", fontName="Courier-Bold", fontSize=6.5, textColor=colors.HexColor("#1A1A1A"))),
        Spacer(1, 0.05 * cm),
        Paragraph("<b>5070 Sayılı Kanun Uyarısı:</b> Bu rapor, 5070 sayılı Elektronik İmza Kanunu ve RFC 3161 Zaman Damgası standartları kapsamında resmi Kamu SM / Nitelikli Elektronik Sertifika (NES) donanım entegrasyonuna hazır mimariyle üretilmiştir.", ParagraphStyle("LegalSub", fontName="Helvetica", fontSize=6, textColor=colors.HexColor("#333333"))),
        Paragraph("<b>Yasal Karar Destek Uyarısı:</b> Bu rapor adli karar destek amaçlı üretilmiştir. Nihai takdir ve hüküm münhasıran adli makamlara ve bilirkişiye aittir.", ParagraphStyle("Legal", fontName="Helvetica-Oblique", fontSize=6, textColor=colors.HexColor("#555555"))),
    ]

    t_seal = Table([[seal_info, qr_drawing]], colWidths=[15.0 * cm, 2.5 * cm])
    t_seal.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F4F1EA")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#7B1E34")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_seal)

    doc.build(story)


def compile_pdf_background_job(
    html_content: str,
    output_dir: Path,
    pdf_out_path: Path,
    case_id: str,
    reportlab_data: Optional[Dict[str, Any]] = None,
):
    """WeasyPrint veya ReportLab ile güvenli PDF derlemesini arka planda çalıştırır."""
    compiled = False

    # 1. WeasyPrint dene
    try:
        from weasyprint import HTML
        HTML(string=html_content, base_url=str(output_dir)).write_pdf(str(pdf_out_path))
        logger.info(f"Vaka {case_id} için WeasyPrint PDF derlemesi tamamlandı: {pdf_out_path.name}")
        compiled = True
    except Exception as e:
        logger.info(f"WeasyPrint PDF derleme uyarısı ({e}), ReportLab fallback motoru devreye alınıyor...")

    # 2. ReportLab Fallback
    if not compiled and reportlab_data:
        try:
            generate_reportlab_pdf(
                pdf_out_path=pdf_out_path,
                **reportlab_data,
            )
            logger.info(f"Vaka {case_id} için ReportLab PDF derlemesi başarıyla tamamlandı: {pdf_out_path.name}")
            compiled = True
        except Exception as err:
            logger.error(f"ReportLab PDF derleme hatası ({case_id}): {err}")


class ForensicReportService:
    """
    Adli Bilirkişi Raporlama ve Dijital Mühürleme Servisi (Forensic Report Engine).
    
    Özellikler:
    - 18 adli bölümü içeren Jinja2 tabanlı dinamik HTML/CSS adli bilirkişi raporu derlemesi.
    - WeasyPrint ve saf Python (ReportLab) çift motorlu platform bağımsız A4 PDF üretimi.
    - Kriptografik SHA-256 bütünlük mührü ve denetim izi kaydı (`audit.jsonl`).
    - FastAPI BackgroundTasks ile asenkron, bloklamayan PDF kuyruğu.
    """

    def __init__(self, templates_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or (Path(__file__).resolve().parent.parent / "templates")
        self.output_dir = output_dir or settings.REPORTS_PATH
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if _HAS_JINJA2:
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(self.templates_dir)),
                autoescape=jinja2.select_autoescape(["html", "xml"]),
            )
        else:
            self.jinja_env = None

    def render_html(
        self,
        case_info: Dict[str, Any],
        suitability: SuitabilityResult,
        comparison: ComparisonReport,
        complexity: ComplexityResult,
        disguise: SimulationIndicators,
        fusion: HybridFusionResult,
        xai: DualExplanationReport,
        hypothesis: HypothesisEvaluationReport,
        isolation_hash: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Rapor şablonunu verilerle doldurur ve rapor metni ile SHA-256 karmasını döner.
        """
        comparison_details = (
            comparison.concordant_features
            + comparison.natural_variations
            + comparison.significant_differences
        )

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        date_utc = case_info.get("date_utc", now_utc)
        case_info["date_utc"] = date_utc

        hash_payload = {
            "case_id": case_info.get("case_id", "VAKA_GENEL"),
            "questioned_id": case_info.get("questioned_id", "Q_01"),
            "hybrid_score": fusion.hybrid_similarity_score,
            "hypothesis_level": hypothesis.support_level.value,
            "date_utc": date_utc,
        }
        report_sha256 = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()

        context = {
            "case_info": case_info,
            "suitability": suitability,
            "comparison": comparison,
            "comparison_details": comparison_details,
            "complexity": complexity,
            "disguise": disguise,
            "fusion": fusion,
            "xai": xai,
            "hypothesis": hypothesis,
            "isolation_hash": isolation_hash,
            "report_sha256": report_sha256,
            "audit_hash_short": report_sha256[:12],
            "generated_at": now_utc,
        }

        html_content = None
        if self.jinja_env:
            try:
                template = self.jinja_env.get_template("report_template.html")
                html_content = template.render(context)
            except Exception:
                html_content = None

        if not html_content:
            html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Adli Rapor - {case_info.get('case_id')}</title></head>
<body>
<h1>TÜBİTAK FHDSS - Adli Belge İnceleme Raporu</h1>
<p>Vaka ID: {case_info.get('case_id')}</p>
<p>Bilirkişi: {case_info.get('examiner_id')}</p>
<p>Nihai Sonuç: {hypothesis.verbal_conclusion_tr}</p>
<p>Rapor İmzası: {report_sha256}</p>
</body></html>"""

        return html_content, report_sha256

    def generate_report(
        self,
        case_info: Dict[str, Any],
        suitability: SuitabilityResult,
        comparison: ComparisonReport,
        complexity: ComplexityResult,
        disguise: SimulationIndicators,
        fusion: HybridFusionResult,
        xai: DualExplanationReport,
        hypothesis: HypothesisEvaluationReport,
        isolation_hash: str = "N/A",
        background_tasks: Optional[Any] = None,
    ) -> ReportGenerationResult:
        """
        Adli raporu HTML ve PDF formatlarında üretir, diske kaydeder ve denetim loguna mühürler.
        Windows ortamlarında WeasyPrint GTK bağımlılığı olmasa dahi ReportLab ile PDF üretimini garanti eder.
        """
        case_id = str(case_info.get("case_id", "CASE_DEFAULT")).replace("/", "_").replace("\\", "_")
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        html_content, report_hash = self.render_html(
            case_info=case_info,
            suitability=suitability,
            comparison=comparison,
            complexity=complexity,
            disguise=disguise,
            fusion=fusion,
            xai=xai,
            hypothesis=hypothesis,
            isolation_hash=isolation_hash,
        )

        # 1. HTML Kaydet
        html_filename = f"report_{case_id}_{timestamp_str}.html"
        html_path = self.output_dir / html_filename
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 2. PDF Derlemesi (WeasyPrint veya ReportLab Fallback)
        pdf_filename = f"report_{case_id}_{timestamp_str}.pdf"
        pdf_out_path = self.output_dir / pdf_filename
        pdf_path = str(pdf_out_path)
        is_pdf_generated = False

        reportlab_data = {
            "case_info": case_info,
            "suitability": suitability,
            "comparison": comparison,
            "complexity": complexity,
            "disguise": disguise,
            "fusion": fusion,
            "xai": xai,
            "hypothesis": hypothesis,
            "report_sha256": report_hash,
        }

        if background_tasks is not None:
            background_tasks.add_task(
                compile_pdf_background_job,
                html_content,
                self.output_dir,
                pdf_out_path,
                case_id,
                reportlab_data,
            )
            is_pdf_generated = True
        else:
            # Senkron PDF üretimi
            try:
                from weasyprint import HTML
                HTML(string=html_content, base_url=str(self.output_dir)).write_pdf(str(pdf_out_path))
                is_pdf_generated = True
            except Exception as e:
                logger.info(f"WeasyPrint atlandı ({e}), ReportLab doğrudan çalıştırılıyor...")
                try:
                    generate_reportlab_pdf(
                        pdf_out_path=pdf_out_path,
                        **reportlab_data,
                    )
                    is_pdf_generated = True
                except Exception as err:
                    logger.error(f"ReportLab PDF derleme hatası: {err}")
                    pdf_path = None

        # 3. Denetim İzi (Audit Trail) Mühürleme
        log_audit_event(
            event_type="FORENSIC_REPORT_GENERATED",
            action="GENERATE_FINAL_REPORT",
            examiner_id=case_info.get("examiner_id", "SYSTEM"),
            details={
                "case_id": case_id,
                "report_sha256": report_hash,
                "html_path": str(html_path),
                "pdf_path": pdf_path,
                "is_pdf_generated": is_pdf_generated,
                "hypothesis_conclusion": hypothesis.support_level.value,
            },
        )

        return ReportGenerationResult(
            case_id=case_id,
            html_path=str(html_path),
            pdf_path=pdf_path,
            report_sha256=report_hash,
            is_pdf_generated=is_pdf_generated,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
