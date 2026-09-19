"use client";

import React, { useState, useEffect, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  Shield,
  Play,
  Download,
  ArrowLeft,
  Activity,
  Fingerprint,
  Eye,
  BarChart3,
  BrainCircuit,
  Scale,
  CheckCircle2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Sliders,
  Sparkles,
  User,
  LogOut,
  Archive,
} from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import {
  api,
  apiClient,
  formatApiError,
  CaseResponse,
  FullPipelineResponse,
  IsolationState,
} from "@/services/api";
import { LayerState } from "@/components/workstation/LayerControls";
import { AuditItem } from "@/components/workstation/AuditTrailTimeline";
import { MicroscopyTab } from "@/components/workstation/MicroscopyTab";
import { MorphologyTab } from "@/components/workstation/MorphologyTab";
import { XaiTab } from "@/components/workstation/XaiTab";
import { EnfsiReportTab } from "@/components/workstation/EnfsiReportTab";
import { BiometricTab } from "@/components/workstation/BiometricTab";
import { CounterfactualTab } from "@/components/workstation/CounterfactualTab";
import { useCaseContext } from "@/context/CaseContext";
import { CaseSwitchConfirmationModal } from "@/components/cases/CaseSwitchConfirmationModal";
import {
  DemoSamplePickerModal,
  DemoScenario,
  renderSignatureToDataUrl,
} from "@/components/workstation/DemoSamplePickerModal";
import { AnalysisProgressModal } from "@/components/workstation/AnalysisProgressModal";
import { ForensicArchiveModal } from "@/components/workstation/ForensicArchiveModal";

type WorkflowTab = "microscopy" | "morphology" | "xai" | "enfsi" | "biometric" | "counterfactual";

interface NavTabItem {
  id: WorkflowTab;
  stepNumber: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

const forensicNavTabs: NavTabItem[] = [
  {
    id: "microscopy",
    stepNumber: "1",
    title: "Görsel Karşılaştırma",
    description: "Karşılaştırmalı inceleme & katman kontrolleri",
    icon: Eye,
  },
  {
    id: "morphology",
    stepNumber: "2",
    title: "Yazı & İmza Özellikleri",
    description: "Morfolojik & grafik öznitelik analizi",
    icon: BarChart3,
  },
  {
    id: "xai",
    stepNumber: "3",
    title: "Model Analizi & Açıklama",
    description: "Yapay zekâ odak haritası & öznitelik etkisi",
    icon: BrainCircuit,
  },
  {
    id: "enfsi",
    stepNumber: "4",
    title: "Bilirkişi Raporu",
    description: "Adli kanıt seviyesi & resmi bilirkişi raporu",
    icon: Scale,
  },
  {
    id: "biometric",
    stepNumber: "5",
    title: "Dinamik Biyometri",
    description: "Tablet hız, ivme & DTW analizi",
    icon: Activity,
  },
  {
    id: "counterfactual",
    stepNumber: "6",
    title: "Karşıtlı Simülasyon",
    description: "'Ne Olurdu?' duyarlılık testi",
    icon: Sliders,
  },
];

export default function CaseWorkstationPage() {
  const params = useParams();
  const router = useRouter();
  const caseId = (params.id as string) || "";
  const { setActiveCaseId, hasUnsavedChanges } = useCaseContext();
  const { user, logout } = useAuth();

  // Aktif Sekme ve Sidebar Durumu
  const [activeTab, setActiveTab] = useState<WorkflowTab>("microscopy");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isExitModalOpen, setIsExitModalOpen] = useState(false);
  const [isDemoPickerOpen, setIsDemoPickerOpen] = useState<boolean>(false);
  const [demoPickerTarget, setDemoPickerTarget] = useState<"both" | "questioned" | "reference">("both");

  // Sprint 20 (Problem B): Analiz öncesi sekme durumunun korunması ve kullanıcı etkileşimi
  const analysisOriginTabRef = useRef<WorkflowTab | null>(null);
  const userInteractedTabRef = useRef<boolean>(false);

  const handleTabChange = (tabId: WorkflowTab) => {
    if (isProgressModalOpen || activeJobId || loading) {
      userInteractedTabRef.current = true;
    }
    setActiveTab(tabId);
  };

  // Durum Yönetimi
  const [caseData, setCaseData] = useState<CaseResponse | null>(null);
  const [isolationState, setIsolationState] = useState<IsolationState>("PHASE_A_INIT");
  const [pipelineResult, setPipelineResult] = useState<FullPipelineResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [loadError, setLoadError] = useState<string | null>(null);

  // Sprint 15: Asenkron İlerleme ve Arşiv Modalleri
  const [isProgressModalOpen, setIsProgressModalOpen] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [isArchiveModalOpen, setIsArchiveModalOpen] = useState(false);

  // Kanvas & Katman Kontrolleri
  const [zoomScale, setZoomScale] = useState(1.0);
  const [selectedRefIndex, setSelectedRefIndex] = useState(0);
  const [layers, setLayers] = useState<LayerState>({
    showOriginal: true,
    showSkeleton: false,
    showNodes: false,
    showGradCam: false,
    showSlantGrid: false,
    isSynchronized: true,
  });

  const [questionedImageUrl, setQuestionedImageUrl] = useState<string>("");
  const [referenceImages, setReferenceImages] = useState<{ id: string; url: string; name: string }[]>([]);
  const [auditItems, setAuditItems] = useState<AuditItem[]>([]);

  // Vaka Bilgilerini ve Geçmiş Denetim İzi (Audit Trail) Kayıtlarını Veritabanından Yükle (F5 Kalıcılığı)
  useEffect(() => {
    if (!caseId) return;
    setActiveCaseId(caseId);

    const initCase = async () => {
      try {
        const c = await api.getCase(caseId);
        setCaseData(c);
        setIsolationState(c.state);

        // Numune önizlemeleri
        if (c.questioned_sample_id) {
          setQuestionedImageUrl(api.getSampleImageUrl(caseId, c.questioned_sample_id));
        }
        if (c.reference_sample_ids && c.reference_sample_ids.length > 0) {
          const refs = c.reference_sample_ids.map((id, idx) => ({
            id,
            url: api.getSampleImageUrl(caseId, id),
            name: `Mukayese Numune #${idx + 1} (${id})`,
          }));
          setReferenceImages(refs);
        }

        // Audit Trail geçmişini veritabanından çek (F5 ile silinmeyi kökten engeller)
        try {
          const auditRes = await api.getCaseAuditLogs(caseId);
          if (auditRes?.audit_trail && auditRes.audit_trail.length > 0) {
            const mapped: AuditItem[] = auditRes.audit_trail.map((log) => ({
              timestamp: log.created_at_utc ? new Date(log.created_at_utc).toLocaleTimeString("tr-TR") : "",
              action: log.action,
              eventType: log.event_type,
              examinerId: log.examiner_id || "SİSTEM",
              sha256Hash: log.sha256_hash,
              detailsSummary: typeof log.details === "object" ? JSON.stringify(log.details) : String(log.details || ""),
            }));
            setAuditItems(mapped);
          }
        } catch (auditErr) {
          console.warn("Audit logs yüklenirken uyarı:", auditErr);
        }
      } catch (err: any) {
        setLoadError(formatApiError(err));
      }
    };

    initCase();
  }, [caseId, setActiveCaseId]);

  // Demo İmza Görseli Oluşturucu Yardımcı Fonksiyon
  const generateDemoSignatureDataUrl = (title: string, slantAngle: number, withTremor: boolean): string => {
    if (typeof window === "undefined") return "";
    const canvas = document.createElement("canvas");
    canvas.width = 600;
    canvas.height = 400;
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";

    // Arka plan
    ctx.fillStyle = "#FAF7F0";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Kılavuz çizgisi
    ctx.strokeStyle = "rgba(184, 164, 132, 0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(50, 300);
    ctx.lineTo(550, 300);
    ctx.stroke();

    // İmza vuruşları
    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate((slantAngle * Math.PI) / 180);

    ctx.strokeStyle = "#1A1A2E";
    ctx.lineWidth = 3.5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.beginPath();
    ctx.moveTo(-160, 30);
    ctx.bezierCurveTo(-130, -80, -80, -70, -60, 15);
    ctx.bezierCurveTo(-40, 60, -10, 70, 10, 10);
    ctx.bezierCurveTo(40, -100, 100, -80, 120, 20);
    ctx.bezierCurveTo(140, 60, 180, 70, 200, 40);

    if (withTremor) {
      for (let t = -150; t < 190; t += 12) {
        const offY = Math.sin(t * 0.4) * 3.5;
        ctx.lineTo(t, 20 + offY);
      }
    }
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(-140, 65);
    ctx.bezierCurveTo(-40, 85, 90, 45, 180, 65);
    ctx.stroke();

    ctx.restore();
    return canvas.toDataURL("image/png");
  };

  const dataURLtoFile = (dataurl: string, filename: string): File => {
    const arr = dataurl.split(",");
    const mime = arr[0].match(/:(.*?);/)![1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  };

  // 1. Şüpheli Belge Yükleme İşlemi
  const handleUploadQuestioned = async (file: File) => {
    setLoading(true);
    setStatusMessage("Şüpheli belge sisteme yükleniyor ve SHA-256 adli bütünlük özeti çıkarılıyor...");
    try {
      const res = await api.uploadSample(caseId, file, "questioned");
      const previewUrl = URL.createObjectURL(file);
      setQuestionedImageUrl(previewUrl);

      setCaseData((prev) =>
        prev
          ? {
              ...prev,
              questioned_sample_id: res.sample_id,
              state: "PHASE_A_INIT",
            }
          : null
      );

      setPipelineResult(null);
      addAudit(
        "NUMUNE_YUKLENDI",
        `Şüpheli belge sisteme yüklendi: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`,
        caseData?.examiner_id || "BILIRKISI",
        res.sha256_hash
      );
      setStatusMessage("Şüpheli belge başarıyla yüklendi. Şimdi inceleyip onaylayabilirsiniz.");
    } catch (err: any) {
      setStatusMessage(`Şüpheli belge yükleme hatası: ${formatApiError(err)}`);
    } finally {
      setLoading(false);
    }
  };

  // 2. Mukayese / Referans Belgeleri Yükleme İşlemi
  const handleUploadReference = async (files: FileList | File[]) => {
    setLoading(true);
    setStatusMessage("Mukayese belgeleri adli inceleme havuzuna yükleniyor...");
    try {
      const fileArray = Array.from(files);
      const newRefs: { id: string; url: string; name: string }[] = [];

      for (const file of fileArray) {
        const res = await api.uploadSample(caseId, file, "reference");
        const previewUrl = URL.createObjectURL(file);
        newRefs.push({
          id: res.sample_id,
          url: previewUrl,
          name: file.name,
        });

        addAudit(
          "NUMUNE_YUKLENDI",
          `Mukayese belgesi eklendi: ${file.name}`,
          caseData?.examiner_id || "BILIRKISI",
          res.sha256_hash
        );
      }

      setReferenceImages((prev) => [...prev, ...newRefs]);
      setPipelineResult(null);

      // Backend'den en güncel vaka durumunu senkronize et
      try {
        const freshCase = await api.getCase(caseId);
        setCaseData(freshCase);
        setIsolationState(freshCase.state);
      } catch {
        const currentRefIds = [...(caseData?.reference_sample_ids || []), ...newRefs.map((r) => r.id)];
        setCaseData((prev) => (prev ? { ...prev, reference_sample_ids: currentRefIds } : null));
      }

      setStatusMessage(`${fileArray.length} adet mukayese belgesi başarıyla eklendi.`);
    } catch (err: any) {
      setStatusMessage(`Mukayese belgesi yükleme hatası: ${formatApiError(err)}`);
    } finally {
      setLoading(false);
    }
  };

  // 3. Hazır Örnek İmza Seçim Modalı Açma
  const handleLoadDemoSample = (type: "questioned" | "reference") => {
    setDemoPickerTarget(type);
    setIsDemoPickerOpen(true);
  };

  // 4. Seçilen Senaryoyu Çalışma Alanına Uygulama
  const handleApplyDemoScenario = async (
    scenario: DemoScenario,
    target: "both" | "questioned" | "reference"
  ) => {
    setLoading(true);
    setStatusMessage(`'${scenario.title}' yükleniyor...`);

    try {
      const dataUrl = renderSignatureToDataUrl(
        scenario.config.slant,
        scenario.config.withTremor,
        scenario.config.style
      );

      // 1. Şüpheli Belge Yükleme
      if (target === "both" || target === "questioned") {
        const qFile = dataURLtoFile(dataUrl, `supheli_${scenario.id}.png`);
        await handleUploadQuestioned(qFile);
      }

      // 2. Mukayese Belgesi Yükleme (Tek 1 adet seçilen numune)
      if (target === "both" || target === "reference") {
        const refFile = dataURLtoFile(dataUrl, `mukayese_${scenario.id}.png`);
        await handleUploadReference([refFile]);
      }

      setStatusMessage(`'${scenario.title}' başarıyla yüklendi.`);
    } catch (err: any) {
      setStatusMessage(`Örnek yükleme hatası: ${formatApiError(err)}`);
    } finally {
      setLoading(false);
    }
  };

  // 5. Şüpheli Belgeyi Kaldırma / Temizleme
  const handleClearQuestioned = async () => {
    if (caseData?.questioned_sample_id && caseId) {
      try {
        await api.deleteSample(caseId, caseData.questioned_sample_id);
      } catch (e) {
        console.warn("Şüpheli numune backendden silinirken uyarı:", e);
      }
    }

    setQuestionedImageUrl("");
    setPipelineResult(null);
    setIsolationState("PHASE_A_INIT");

    try {
      const freshCase = await api.getCase(caseId);
      setCaseData(freshCase);
      setIsolationState(freshCase.state);
    } catch {
      setCaseData((prev) =>
        prev
          ? {
              ...prev,
              questioned_sample_id: undefined,
              state: "PHASE_A_INIT",
            }
          : null
      );
    }

    addAudit(
      "NUMUNE_SILINDI",
      "Şüpheli belge çalışma alanından kaldırıldı.",
      caseData?.examiner_id || "BILIRKISI",
      "SYSTEM_CLEAR"
    );
    setStatusMessage("Şüpheli belge kaldırıldı. Yeni bir belge yükleyebilirsiniz.");
  };

  // 6. Seçili Mukayese Belgesini Kaldırma
  const handleRemoveReference = async (indexToRemove: number) => {
    const targetRef = referenceImages[indexToRemove];
    const newRefs = referenceImages.filter((_, idx) => idx !== indexToRemove);

    if (targetRef?.id && caseId) {
      try {
        await api.deleteSample(caseId, targetRef.id);
      } catch (e) {
        console.warn("Mukayese numunesi backendden silinirken uyarı:", e);
      }
    }

    setReferenceImages(newRefs);
    setSelectedRefIndex((prev) => (prev >= newRefs.length ? Math.max(0, newRefs.length - 1) : prev));
    setPipelineResult(null);

    // Backend'den en güncel vaka durumunu senkronize et
    try {
      const freshCase = await api.getCase(caseId);
      setCaseData(freshCase);
      setIsolationState(freshCase.state);
    } catch {
      const newSampleIds = (caseData?.reference_sample_ids || []).filter((id) => id !== targetRef?.id);
      setCaseData((prev) =>
        prev
          ? {
              ...prev,
              reference_sample_ids: newSampleIds,
              state: prev.questioned_sample_id ? "PHASE_A_FROZEN" : "PHASE_A_INIT",
            }
          : null
      );
      setIsolationState(caseData?.questioned_sample_id ? "PHASE_A_FROZEN" : "PHASE_A_INIT");
    }

    addAudit(
      "NUMUNE_SILINDI",
      `Mukayese belgesi silindi: ${targetRef?.name || `Numune ${indexToRemove + 1}`}`,
      caseData?.examiner_id || "BILIRKISI",
      "SYSTEM_CLEAR"
    );
    setStatusMessage("Mukayese belgesi kaldırıldı.");
  };

  // 7. Tüm Mukayese Belgelerini Temizleme
  const handleClearAllReferences = async () => {
    if (caseData?.reference_sample_ids && caseId) {
      for (const refId of caseData.reference_sample_ids) {
        try {
          await api.deleteSample(caseId, refId);
        } catch (e) {
          console.warn("Mukayese silinirken hata:", e);
        }
      }
    }

    setReferenceImages([]);
    setSelectedRefIndex(0);
    setPipelineResult(null);

    try {
      const freshCase = await api.getCase(caseId);
      setCaseData(freshCase);
      setIsolationState(freshCase.state);
    } catch {
      setCaseData((prev) =>
        prev
          ? {
              ...prev,
              reference_sample_ids: [],
              state: prev.questioned_sample_id ? "PHASE_A_FROZEN" : "PHASE_A_INIT",
            }
          : null
      );
      setIsolationState(caseData?.questioned_sample_id ? "PHASE_A_FROZEN" : "PHASE_A_INIT");
    }

    setStatusMessage("Tüm mukayese belgeleri temizlendi.");
  };

  // Vaka Bilgisini Yükle ve Aktif Vaka Olarak Belirle
  useEffect(() => {
    if (!caseId) return;
    setActiveCaseId(caseId);

    async function loadCase() {
      setLoading(true);
      setLoadError(null);
      try {
        const res = await api.getCase(caseId);
        setCaseData(res);
        setIsolationState(res.state);

        // Vaka daha önce tamamlanmışsa üretilen analiz sonucunu otomatik yükle
        if (res.state === "EVALUATION_COMPLETED") {
          try {
            const savedResult = await api.getLatestAnalysisResult(caseId);
            setPipelineResult(savedResult);
          } catch (resErr) {
            console.warn("Kayıtlı analiz sonucu yüklenemedi:", resErr);
          }
        }

        addAudit(
          "VAKA_YUKLENDI",
          `'${caseId}' adli dosyası çalışma alanına yüklendi.`,
          res.examiner_id || "BILIRKISI",
          "SYSTEM_INIT"
        );
      } catch (err: any) {
        console.error("Vaka yüklenemedi:", err);
        setCaseData(null);
        setLoadError(`'${caseId}' numaralı vaka bilgileri alınamadı: ${formatApiError(err)}`);
      } finally {
        setLoading(false);
      }
    }
    loadCase();
  }, [caseId, setActiveCaseId]);

  const addAudit = (eventType: string, action: string, examinerId: string, sha256: string) => {
    const newItem: AuditItem = {
      timestamp: new Date().toLocaleTimeString("tr-TR"),
      eventType,
      action,
      examinerId,
      sha256Hash: sha256,
    };
    setAuditItems((prev: AuditItem[]) => [newItem, ...prev]);
  };

  // Şüpheli Belgeyi Onayla ve Kilitle
  const handleFreezePhaseA = async () => {
    if (!caseData?.questioned_sample_id && !questionedImageUrl) {
      setStatusMessage("Şüpheli belge henüz yüklenmediğinden kilitleme yapılamaz.");
      return;
    }
    setLoading(true);
    setStatusMessage("Şüpheli belgenin adli öznitelikleri çıkarılıyor ve kilitleniyor...");
    try {
      const qId = caseData?.questioned_sample_id || "QUESTIONED_001";
      const res = await api.isolatePhaseA(caseId, qId, "Şüpheli belge ön incelemesi tamamlandı.");
      setIsolationState("PHASE_A_FROZEN");
      setStatusMessage("Şüpheli belge analizi kilitlendi. Mukayese / Referans belgeler açıldı.");
      addAudit(
        "BELGE_KILITLENDI",
        "Şüpheli belge öznitelikleri kilitlendi.",
        caseData?.examiner_id || "BILIRKISI",
        res.integrity_hash_sha256
      );
    } catch (err: any) {
      setStatusMessage(`İzolasyon hatası: ${formatApiError(err)}`);
    } finally {
      setLoading(false);
    }
  };

  // Kapsamlı Adli Analiz Başlatma
  const handleRunFullPipeline = async () => {
    setLoadError(null);

    // Sprint 20 (Problem B): Analizin başlatıldığı kaynak sekmeyi kaydet
    analysisOriginTabRef.current = activeTab;
    userInteractedTabRef.current = false;

    let qId = caseData?.questioned_sample_id;
    let refIds = caseData?.reference_sample_ids || [];

    if (!qId || !refIds.length) {
      // Backend'den son durumu çekmeyi dene
      try {
        const fresh = await api.getCase(caseId);
        setCaseData(fresh);
        qId = fresh.questioned_sample_id;
        refIds = fresh.reference_sample_ids;
      } catch (e) {
        console.warn("Vaka senkronizasyonu hatası:", e);
      }
    }

    if (!refIds.length && referenceImages.length > 0) {
      refIds = referenceImages.map((r) => r.id).filter(Boolean);
    }

    if (!qId || !refIds.length) {
      setStatusMessage("Analiz başlatmak için hem şüpheli belge hem de en az 1 adet mukayese belgesi gereklidir.");
      return;
    }

    // Sprint 15: Asenkron Arka Plan Kuyruğu & Gerçek Zamanlı SSE İlerlemesi
    try {
      const jobRes = await apiClient.post("/jobs/submit", {
        case_id: caseId,
        questioned_sample_id: qId,
        reference_sample_ids: refIds,
        suspect_name: caseData?.name || "Şüpheli Şahıs",
        generate_pdf: true,
      });

      if (jobRes.data?.job_id) {
        setActiveJobId(jobRes.data.job_id);
        setIsProgressModalOpen(true);
        setStatusMessage("Analiz arka plan iş kuyruğuna alındı, ilerleme takip ediliyor...");
        return;
      }
    } catch (jobErr) {
      console.warn("Asenkron iş kuyruğu başlatılamadı, doğrudan analiz hattına geçiliyor:", jobErr);
    }

    setLoading(true);
    setStatusMessage("Yapay zekâ ve morfolojik karşılaştırma modelleri çalıştırılıyor...");
    try {
      const payload = {
        case_id: caseId,
        questioned_sample_id: qId,
        reference_sample_ids: refIds,
        suspect_name: caseData?.name || "Şüpheli Şahıs",
        generate_pdf: true,
      };
      const res = await api.runFullPipeline(payload);
      setPipelineResult(res);
      setIsolationState("EVALUATION_COMPLETED");
      setStatusMessage("Adli analiz başarıyla tamamlandı ve bilirkişi raporu mühürlendi.");
      addAudit(
        "ANALIZ_TAMAMLANDI",
        "Tüm karşılaştırmalı model analizleri tamamlandı ve rapor mühürlendi.",
        caseData?.examiner_id || "BILIRKISI",
        res.audit_checksum_sha256
      );

      try {
        const freshCase = await api.getCase(caseId);
        setCaseData(freshCase);
      } catch {}
    } catch (err: any) {
      setStatusMessage(`Analiz hatası: ${formatApiError(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const handleJobCompleted = async (result: any) => {
    setIsolationState("EVALUATION_COMPLETED");
    setStatusMessage("Asenkron adli analiz başarıyla tamamlandı ve rapor mühürlendi.");

    // 1. Sonuç verisini kaydet (Detaylı sonuçlar varsa doğrudan ata, yoksa API'den çek)
    if (result && result.hypothesis) {
      setPipelineResult(result);
    } else {
      try {
        const fullResult = await api.getLatestAnalysisResult(caseId);
        setPipelineResult(fullResult);
      } catch (err) {
        console.warn("Tam analiz sonucu API'den çekilemedi, mevcut sonuç atanıyor:", err);
        if (result) {
          setPipelineResult(result);
        }
      }
    }

    try {
      const freshCase = await api.getCase(caseId);
      setCaseData(freshCase);
      setIsolationState(freshCase.state);
    } catch (e) {
      console.warn("Vaka yenilenirken hata:", e);
    }

    // 2. Sprint 20 (Problem B): Analiz tamamlandıktan sonra analiz öncesi sekmeyi koru.
    // Kullanıcı analiz sürerken manuel olarak başka bir sekmeye geçtiyse kullanıcının seçimine saygı duy.
    if (userInteractedTabRef.current) {
      // Kullanıcı manuel geçiş yaptı, seçilen sekmede kal
    } else if (analysisOriginTabRef.current) {
      setActiveTab(analysisOriginTabRef.current);
    }
    analysisOriginTabRef.current = null;
    userInteractedTabRef.current = false;
  };

  // Mühürlü PDF Rapor İndir
  const handleDownloadPdf = () => {
    if (!pipelineResult) {
      setStatusMessage("Rapor henüz üretilmedi. Lütfen önce analizi başlatınız.");
      return;
    }
    const url = api.getReportPdfUrl(caseId);
    window.open(url, "_blank");
  };

  // Durum Rozet Bilgisi
  const getStatusBadge = () => {
    switch (isolationState) {
      case "PHASE_A_INIT":
      case "PHASE_A_ANALYSIS":
        return {
          label: "1. Adım: Şüpheli Belge Ön İncelemesi",
          className: "bg-[#2A2019] text-amber-400 border-amber-500/40",
        };
      case "PHASE_A_FROZEN":
        return {
          label: "Şüpheli Belge Kilitlendi (Mukayeseye Hazır)",
          className: "bg-[#2A2019] text-teal-400 border-teal-500/40",
        };
      case "PHASE_B_ACTIVE":
        return {
          label: "2. Adım: Karşılaştırmalı İnceleme Aktif",
          className: "bg-[#2A2019] text-burgundy-600 border-[#D4C5A9]/30",
        };
      case "EVALUATION_COMPLETED":
        return {
          label: "Adli İnceleme Tamamlandı",
          className: "bg-[#2A2019] text-emerald-400 border-emerald-500/40",
        };
      default:
        return {
          label: isolationState,
          className: "bg-[#2A2019] text-[#D4C5A9] border-[#D4C5A9]/20",
        };
    }
  };

  const statusBadge = getStatusBadge();
  const qualityScore = pipelineResult?.suitability?.overall_score !== undefined
    ? Math.round(pipelineResult.suitability.overall_score * 100)
    : null;

  const hybridScore = pipelineResult?.fusion?.hybrid_similarity_score !== undefined
    ? Math.round(pipelineResult.fusion.hybrid_similarity_score * 100)
    : null;

  const hasQuestionedImage = Boolean(questionedImageUrl || caseData?.questioned_sample_id);
  const hasReferenceImages = Boolean(
    referenceImages.length > 0 || (caseData?.reference_sample_ids && caseData.reference_sample_ids.length > 0)
  );
  const hasSamples = hasQuestionedImage || hasReferenceImages;
  const hasPipelineResult = Boolean(pipelineResult);

  return (
    <div className="flex flex-col min-h-screen bg-[#18120F] text-[#FAF7F0] font-sans">
      {/* ÜST BAŞLIK BARI (ESPRESSO & KREM ÇERÇEVELİ) */}
      <header className="bg-[#1C1510]/90 border-b border-[#B8A484]/25 backdrop-blur-md px-4 md:px-6 py-2.5 flex flex-wrap items-center justify-between gap-3 sticky top-0 z-30 shadow-xl">
        {/* Sol Alan: Vaka Değiştir Butonu ve AKTİF VAKA Rozeti */}
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setIsExitModalOpen(true)}
            title="Vaka Çalışma Alanına Dön / Vaka Değiştir"
            className="px-3 py-1.5 bg-[#2A2019] hover:bg-[#3D2E24] rounded-xl text-xs text-[#FAF7F0] font-semibold border border-[#B8A484]/20 transition flex items-center gap-1.5 shadow-sm"
          >
            <ArrowLeft className="w-4 h-4 text-cream-200" />
            <span className="hidden sm:inline">Vaka Değiştir</span>
          </button>

          <div>
            <div className="flex items-center gap-2 flex-wrap">
              {/* AKTİF VAKA Rozeti */}
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase bg-burgundy-900/80 text-cream-100 border border-burgundy-600 shadow-sm flex items-center gap-1">
                <Fingerprint className="w-3 h-3 text-amber-400" /> AKTİF VAKA
              </span>

              {/* Örnek Dosya Rozeti */}
              {(caseData?.name?.includes("ÖRNEK") || caseData?.name?.includes("DEMO") || caseId.startsWith("VAKA-2026-00")) && (
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm">
                  Örnek / Demo Vaka
                </span>
              )}

              <h1 className="text-sm font-bold text-[#FAF7F0] flex items-center gap-1.5">
                <span className="font-mono text-cream-200">{caseId}</span>
                {caseData?.name && <span className="text-[#D4C5A9] font-medium hidden md:inline">• {caseData.name}</span>}
              </h1>

              {caseData?.authority && (
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-[#2A2019] text-[#EFE7DA] border border-[#B8A484]/25 shadow-sm">
                  {caseData.authority}
                </span>
              )}
            </div>

            <p className="text-[11px] text-[#D4C5A9]/75 mt-0.5">
              Bilirkişi: <span className="text-[#FAF7F0] font-medium">{caseData?.examiner_id || "Belirtilmedi"}</span>
              {caseData?.description && <span className="hidden lg:inline ml-2 italic text-[#D4C5A9]/60">({caseData.description})</span>}
            </p>
          </div>
        </div>

        {/* Sağ Alan: Durum Rozetleri, Delil Özeti ve Bordo Aksiyon Butonları */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Durum Rozeti */}
          <span className={`px-2.5 py-1 rounded-full text-xs font-semibold border shadow-sm ${statusBadge.className}`}>
            {statusBadge.label}
          </span>

          {/* Görsel Kalite Skoru (Sadece analiz sonucu varsa gösterilir) */}
          {qualityScore !== null ? (
            <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#2A2019] text-emerald-400 border border-[#D4C5A9]/20 shadow-sm">
              Görsel Kalite Skoru: %{qualityScore}
            </span>
          ) : (
            <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#2A2019] text-[#D4C5A9]/60 border border-[#B8A484]/15 shadow-sm">
              Kalite Analizi Bekleniyor
            </span>
          )}

          {/* Genel Delil Özeti & Uyum Oranı */}
          {hybridScore !== null ? (
            <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#2A2019] text-teal-400 border border-teal-500/30 shadow-sm flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Genel Delil Uyumu: %{hybridScore}
            </span>
          ) : (
            <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#2A2019] text-[#D4C5A9] border border-[#D4C5A9]/20 shadow-sm">
              Delil Durumu: İnceleme Aşamasında
            </span>
          )}

          {/* Aksiyon Butonları */}
          {isolationState === "PHASE_A_INIT" || isolationState === "PHASE_A_ANALYSIS" ? (
            <button
              type="button"
              disabled={loading || !caseData?.questioned_sample_id}
              onClick={handleFreezePhaseA}
              title="Bilişsel yanlılığı önlemek için şüpheli belgeyi inceleyip onaylayın"
              className="px-3.5 py-1.5 bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] disabled:opacity-50 text-[#FAF7F0] font-semibold text-xs rounded-full transition shadow-lg flex items-center gap-1.5 border border-[#B8A484]/30"
            >
              <Shield className="w-3.5 h-3.5 text-cream-200" /> Analiz Kilidi: Şüpheli Belgeyi İncele ve Onayla
            </button>
          ) : (
            <button
              type="button"
              disabled={loading}
              onClick={handleRunFullPipeline}
              className="px-3.5 py-1.5 bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] disabled:opacity-50 text-[#FAF7F0] font-semibold text-xs rounded-full transition shadow-lg flex items-center gap-1.5 border border-[#B8A484]/30"
            >
              <Play className="w-3.5 h-3.5 text-cream-200" /> Kapsamlı Adli Analizi Başlat
            </button>
          )}

          <button
            type="button"
            disabled={!pipelineResult}
            onClick={handleDownloadPdf}
            className="px-3.5 py-1.5 bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] disabled:opacity-40 text-[#FAF7F0] font-semibold text-xs rounded-full transition shadow-lg flex items-center gap-1.5 border border-[#B8A484]/30"
          >
            <Download className="w-3.5 h-3.5 text-cream-200" /> Bilirkişi Raporu (PDF)
          </button>

          {/* Sprint 15: Adli Arşiv Paketi (.ZIP) Butonu */}
          <button
            type="button"
            onClick={() => setIsArchiveModalOpen(true)}
            className="px-3.5 py-1.5 bg-gradient-to-r from-indigo-900/90 to-slate-900/90 hover:from-indigo-800 hover:to-slate-800 text-indigo-200 font-semibold text-xs rounded-full transition shadow-lg flex items-center gap-1.5 border border-indigo-500/40"
            title="Tüm vaka delillerini, raporlarını ve denetim izini mühürlü ZIP paketi olarak indir veya doğrula"
          >
            <Archive className="w-3.5 h-3.5 text-indigo-400" /> Adli Arşiv (.ZIP)
          </button>

          {/* Oturum Bilgisi ve Çıkış Butonu */}
          {user && (
            <div className="flex items-center gap-2 pl-2 border-l border-[#B8A484]/20">
              <span className="text-[11px] text-[#D4C5A9] font-medium hidden xl:inline">
                {user.full_name || user.username}
              </span>
              <button
                type="button"
                onClick={logout}
                title="Oturumu Kapat"
                className="p-1.5 bg-[#2A2019] hover:bg-rose-950/60 text-[#D4C5A9] hover:text-rose-300 border border-[#B8A484]/20 hover:border-rose-700/40 rounded-lg text-xs font-semibold transition"
              >
                <LogOut className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      </header>

      {/* DURUM BİLGİLENDİRME BANTI */}
      {statusMessage && (
        <div className="bg-[#2A2019]/95 border-b border-[#B8A484]/20 px-6 py-2 text-xs flex items-center justify-between text-[#EFE7DA] shadow-inner">
          <div className="flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-burgundy-600 animate-pulse" />
            <span>{statusMessage}</span>
          </div>
          {loading && <span className="text-[11px] text-[#D4C5A9] font-semibold">İşleniyor...</span>}
        </div>
      )}

      {/* HATA BİLDİRİMİ */}
      {loadError && (
        <div className="bg-rose-950/90 border-b border-rose-800 px-6 py-3 text-xs text-rose-200 flex items-center justify-between shadow-inner">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{loadError}</span>
          </div>
          <button
            type="button"
            onClick={() => router.push("/")}
            className="px-3 py-1 bg-rose-900 hover:bg-rose-800 text-rose-100 rounded-lg text-[11px] font-semibold transition"
          >
            Vaka Listesine Dön
          </button>
        </div>
      )}

      {/* ANA GÖVDE: SOL AÇILIR-KAPANIR ADLİ MENÜ + İÇERİK ALANI */}
      <div className="flex flex-1 min-h-[calc(100vh-65px)]">
        {/* 1-4 DİKEY ADLİ MENÜ */}
        <aside
          className={`bg-[#1E1612]/95 border-r border-[#B8A484]/20 backdrop-blur-md flex flex-col transition-all duration-300 ease-in-out shrink-0 sticky top-[53px] h-[calc(100vh-53px)] z-20 ${
            isSidebarOpen ? "w-64 md:w-72" : "w-16 md:w-20"
          }`}
        >
          {/* Sidebar Üst Barı & Daralt/Genişlet Butonu */}
          <div
            className={`flex items-center border-b border-[#B8A484]/15 px-3 py-3 ${
              isSidebarOpen ? "justify-between" : "justify-center"
            }`}
          >
            {isSidebarOpen && (
              <div className="flex items-center gap-2 overflow-hidden">
                <div className="w-2 h-2 rounded-full bg-burgundy-600 animate-pulse shrink-0" />
                <span className="text-xs font-bold uppercase tracking-wider text-[#FAF7F0] truncate">
                  Adli Menü
                </span>
              </div>
            )}
            <button
              type="button"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              title={isSidebarOpen ? "Menüyü Daralt" : "Menüyü Genişlet"}
              className="p-1.5 rounded-lg bg-[#2A2019]/90 hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] border border-[#B8A484]/20 transition flex items-center justify-center shrink-0 shadow-sm"
            >
              {isSidebarOpen ? (
                <ChevronLeft className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </button>
          </div>

          {/* Sekme Butonları Listesi (Dikey) */}
          <nav className="flex-1 p-2 space-y-2 overflow-y-auto">
            {forensicNavTabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => handleTabChange(tab.id)}
                  title={`${tab.stepNumber}. ${tab.title}`}
                  className={`w-full flex items-center rounded-xl transition-all duration-200 group text-left ${
                    isSidebarOpen ? "px-3 py-2.5 gap-3" : "p-2.5 justify-center"
                  } ${
                    isActive
                      ? "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] text-[#FAF7F0] shadow-md border border-[#B8A484]/40 ring-1 ring-[#D4C5A9]/30"
                      : "bg-[#2A2019]/40 hover:bg-[#3D2E24]/90 text-[#D4C5A9] hover:text-[#FAF7F0] border border-transparent hover:border-[#B8A484]/25"
                  }`}
                >
                  <div
                    className={`p-2 rounded-lg shrink-0 transition ${
                      isActive
                        ? "bg-white/15 text-[#FAF7F0] shadow-inner"
                        : "bg-[#18120F] text-[#D4C5A9] group-hover:text-[#FAF7F0] group-hover:bg-[#2A2019]"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                  </div>

                  {isSidebarOpen && (
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-bold truncate text-[#FAF7F0]">
                        {tab.stepNumber}. {tab.title}
                      </div>
                      <p
                        className={`text-[10px] truncate mt-0.5 ${
                          isActive
                            ? "text-[#FAF7F0]/80"
                            : "text-[#D4C5A9]/60 group-hover:text-[#D4C5A9]/80"
                        }`}
                      >
                        {tab.description}
                      </p>
                    </div>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Sidebar Alt Bilgi Rozeti */}
          {isSidebarOpen && (
            <div className="p-3 border-t border-[#B8A484]/15 bg-[#18120F]/60">
              <div className="flex items-center justify-between text-[11px] text-[#D4C5A9]/80">
                <span className="font-medium">Adli Akış:</span>
                <span className="font-semibold text-[#FAF7F0]">
                  {isolationState === "EVALUATION_COMPLETED"
                    ? "İnceleme Tamamlandı"
                    : isolationState === "PHASE_A_FROZEN" || isolationState === "PHASE_B_ACTIVE"
                    ? "2. Mukayese Aşaması"
                    : "1. Şüpheli Belge Aşaması"}
                </span>
              </div>
            </div>
          )}
        </aside>

        {/* ANA İÇERİK ALANI */}
        <main className="flex-1 p-4 md:p-6 max-w-[1800px] w-full min-w-0 space-y-4">
          {/* ÜST YATAY SEKME NAVİGASYON ÇUBUĞU (HAP / PILL BUTONLAR) */}
          <div className="bg-[#1E1612]/90 border border-[#B8A484]/20 rounded-2xl p-2 shadow-xl backdrop-blur-md flex items-center justify-between gap-2 overflow-x-auto">
            <div className="flex items-center gap-2 flex-nowrap min-w-max">
              {forensicNavTabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => handleTabChange(tab.id)}
                    className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all duration-200 flex items-center gap-2 ${
                      isActive
                        ? "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] text-[#FAF7F0] shadow-md border border-[#B8A484]/40 ring-1 ring-[#D4C5A9]/30"
                        : "bg-[#2A2019] text-[#D4C5A9] hover:bg-[#3D2E24] hover:text-[#FAF7F0] border border-[#B8A484]/15"
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{tab.stepNumber}. {tab.title}</span>
                  </button>
                );
              })}
            </div>

            <div className="hidden lg:flex items-center gap-2 pr-3 text-xs text-[#D4C5A9]/80 font-mono">
              <span>İşlem Adımı:</span>
              <span className="text-[#FAF7F0] font-semibold">
                {activeTab === "microscopy"
                  ? "1/4 Görsel Karşılaştırma"
                  : activeTab === "morphology"
                  ? "2/4 Yazı & İmza Özellikleri"
                  : activeTab === "xai"
                  ? "3/4 Açıklanabilir YZ & Odak Haritası"
                  : "4/4 Resmi Bilirkişi Raporu"}
              </span>
            </div>
          </div>

          {activeTab === "microscopy" && (
            <MicroscopyTab
              isolationState={isolationState}
              caseData={caseData}
              layers={layers}
              setLayers={setLayers}
              zoomScale={zoomScale}
              setZoomScale={setZoomScale}
              questionedImageUrl={questionedImageUrl}
              referenceImages={referenceImages}
              selectedRefIndex={selectedRefIndex}
              setSelectedRefIndex={setSelectedRefIndex}
              handleFreezePhaseA={handleFreezePhaseA}
              loading={loading}
              onUploadQuestioned={handleUploadQuestioned}
              onUploadReference={handleUploadReference}
              onLoadDemoSample={handleLoadDemoSample}
              onClearQuestioned={handleClearQuestioned}
              onRemoveReference={handleRemoveReference}
              onClearAllReferences={handleClearAllReferences}
              pipelineResult={pipelineResult}
              onRunPipeline={handleRunFullPipeline}
              onNavigateTab={(tabId) => handleTabChange(tabId as WorkflowTab)}
            />
          )}

          {activeTab === "morphology" && (
            <MorphologyTab pipelineResult={pipelineResult} caseId={caseId} />
          )}

          {activeTab === "xai" && (
            <XaiTab pipelineResult={pipelineResult} />
          )}

          {activeTab === "enfsi" && (
            <EnfsiReportTab
              pipelineResult={pipelineResult}
              caseData={caseData}
              caseId={caseId}
              auditItems={auditItems}
              handleDownloadPdf={handleDownloadPdf}
            />
          )}

          {activeTab === "biometric" && (
            <BiometricTab
              caseId={caseId}
              hasSamples={hasSamples}
              onNavigateMicroscopy={() => handleTabChange("microscopy")}
            />
          )}

          {activeTab === "counterfactual" && (
            <CounterfactualTab
              caseId={caseId}
              hasSamples={hasSamples}
              hasPipelineResult={hasPipelineResult}
              onNavigateMicroscopy={() => setActiveTab("microscopy")}
            />
          )}
        </main>
      </div>

      {/* VAKADAN ÇIKIŞ / VAKA DEĞİŞTİRME GÜVENLİK DİYALOĞU */}
      <CaseSwitchConfirmationModal
        isOpen={isExitModalOpen}
        hasUnsavedChanges={hasUnsavedChanges}
        currentCaseId={caseId}
        onCancel={() => setIsExitModalOpen(false)}
        onConfirm={() => {
          setIsExitModalOpen(false);
          router.push("/");
        }}
      />

      {/* 🎯 HAZIR ADLİ İMZA ÖRNEKLERİ SEÇİM MODALI */}
      <DemoSamplePickerModal
        isOpen={isDemoPickerOpen}
        onClose={() => setIsDemoPickerOpen(false)}
        onSelectScenario={handleApplyDemoScenario}
        defaultTarget={demoPickerTarget}
      />

      {/* 🚀 ASENKRON ADLİ ANALİZ VE SSE İLERLEME MODALI */}
      <AnalysisProgressModal
        isOpen={isProgressModalOpen}
        jobId={activeJobId}
        onClose={() => setIsProgressModalOpen(false)}
        onCompleted={handleJobCompleted}
      />

      {/* 📦 GÜVENLİ ADLİ ARŞİV PAKETİ (.ZIP) MODALI */}
      <ForensicArchiveModal
        isOpen={isArchiveModalOpen}
        caseId={caseId}
        onClose={() => setIsArchiveModalOpen(false)}
      />
    </div>
  );
}
