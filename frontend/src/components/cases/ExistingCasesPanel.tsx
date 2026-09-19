"use client";

import React, { useState, useMemo } from "react";
import {
  FolderOpen,
  ArrowLeft,
  PlusCircle,
  Search,
  Calendar,
  Clock,
  User,
  Building2,
  CheckCircle2,
  AlertCircle,
  PlayCircle,
  FileSearch,
  Filter,
  ArrowRight,
  RefreshCw,
  Trash2,
  AlertTriangle,
  X,
  Check,
} from "lucide-react";
import { useCaseContext } from "@/context/CaseContext";
import { useAuth } from "@/context/AuthContext";
import { CaseResponse, CaseStatus, CaseStage, ExaminationType } from "@/services/api";

interface ExistingCasesPanelProps {
  onBack: () => void;
  onSelectNew: () => void;
}

export const ExistingCasesPanel: React.FC<ExistingCasesPanelProps> = ({
  onBack,
  onSelectNew,
}) => {
  const { cases, loadingCases, error, selectCaseAndNavigate, deleteCase, fetchCases, activeCaseId } = useCaseContext();
  const { hasRole } = useAuth();
  const canDeleteCase = hasRole(["ADMIN", "SENIOR_REVIEWER"]);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [caseToDelete, setCaseToDelete] = useState<CaseResponse | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteFeedback, setDeleteFeedback] = useState<{ type: "success" | "error"; message: string } | null>(null);

  const handleConfirmDelete = async () => {
    if (!caseToDelete) return;
    setIsDeleting(true);
    setDeleteFeedback(null);
    try {
      await deleteCase(caseToDelete.case_id);
      setDeleteFeedback({
        type: "success",
        message: `'${caseToDelete.name}' (${caseToDelete.case_id}) numaralı adli dosya ve ilişkili tüm veriler veritabanından kalıcı olarak silindi.`,
      });
      setCaseToDelete(null);
    } catch (err: any) {
      setDeleteFeedback({
        type: "error",
        message: `Vaka silinirken hata oluştu: ${err?.message || "Bilinmeyen hata"}`,
      });
    } finally {
      setIsDeleting(false);
    }
  };

  // Filtrelenmiş Vaka Listesi
  const filteredCases = useMemo(() => {
    return cases.filter((c) => {
      const matchesSearch =
        c.case_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        c.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (c.description && c.description.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (c.authority && c.authority.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (c.reference_number && c.reference_number.toLowerCase().includes(searchTerm.toLowerCase()));

      const matchesStatus = statusFilter === "ALL" || c.status === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [cases, searchTerm, statusFilter]);

  // İnceleme Türü Türkçe Etiketi
  const getExamTypeLabel = (type: ExaminationType) => {
    switch (type) {
      case "HANDWRITING":
        return "El Yazısı İncelemesi";
      case "SIGNATURE":
        return "İmza İncelemesi";
      case "HANDWRITING_AND_SIGNATURE":
        return "El Yazısı + İmza";
      default:
        return "Adli İnceleme";
    }
  };

  // Durum Rozeti
  const getStatusBadge = (status: CaseStatus) => {
    switch (status) {
      case "IN_PROGRESS":
        return {
          label: "Analiz Devam Ediyor",
          className: "bg-[#2A2019] text-amber-400 border border-amber-500/40",
          icon: Clock,
        };
      case "COMPLETED":
        return {
          label: "Tamamlandı",
          className: "bg-[#2A2019] text-emerald-400 border border-emerald-500/40",
          icon: CheckCircle2,
        };
      case "DRAFT":
        return {
          label: "Taslak",
          className: "bg-[#2A2019] text-cream-300 border border-[#B8A484]/30",
          icon: AlertCircle,
        };
      case "ARCHIVED":
        return {
          label: "Arşivlendi",
          className: "bg-[#1E1612] text-[#D4C5A9]/60 border border-[#B8A484]/15",
          icon: FileSearch,
        };
      default:
        return {
          label: status,
          className: "bg-[#2A2019] text-[#D4C5A9] border border-[#B8A484]/20",
          icon: AlertCircle,
        };
    }
  };

  // Aşama Türkçe Açıklaması
  const getStageLabel = (stage: CaseStage) => {
    switch (stage) {
      case "CASE_CREATED":
        return "1. Vaka Oluşturuldu (Numune Bekleniyor)";
      case "SUITABILITY":
        return "2. Görüntü Kalite ve Adli Uygunluk";
      case "PREPROCESSING":
        return "3. Ön İşleme & Hat Normalizasyonu";
      case "FEATURE_EXTRACTION":
        return "4. Öznitelik Çıkarımı (Feature Extraction)";
      case "COMPARISON":
        return "5. Karşılaştırmalı İnceleme (Comparison)";
      case "COMPLEXITY_ANALYSIS":
        return "6. Found & Rogers Karmaşıklık Analizi";
      case "SIMULATION_ANALYSIS":
        return "7. Taklit & Sahtecilik Tespiti";
      case "EVALUATION":
        return "8. Bayesyen Adli Değerlendirme";
      case "REPORT":
      case "COMPLETED":
        return "9. Resmi Bilirkişi Raporu (Final Report)";
      default:
        return "Adli Analiz Süreci";
    }
  };

  // Tarih Formatlayıcı
  const formatDate = (isoStr?: string) => {
    if (!isoStr) return "-";
    try {
      const d = new Date(isoStr);
      return d.toLocaleDateString("tr-TR", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return isoStr;
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto py-6 px-4 sm:px-6 flex flex-col flex-1 space-y-6">
      {/* ÜST BAŞLIK & NAVİGASYON BARI */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[#B8A484]/20">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="p-2 rounded-xl bg-[#2A2019] hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] border border-[#B8A484]/20 transition shadow-sm"
            title="Vaka Seçim Alanına Dön"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h2 className="text-xl sm:text-2xl font-bold text-[#FAF7F0] flex items-center gap-2">
              <FolderOpen className="w-6 h-6 text-burgundy-600" />
              Mevcut Vakalar
            </h2>
            <p className="text-xs text-[#D4C5A9]/80">
              Kayıtlı adli inceleme dosyaları, son durumları ve çalışma aşamaları
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onSelectNew}
          className="px-4 py-2 bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] font-semibold text-xs rounded-xl shadow-lg transition duration-200 flex items-center gap-2 border border-[#B8A484]/30"
        >
          <PlusCircle className="w-4 h-4 text-cream-200" />
          Yeni Vaka Oluştur
        </button>
      </div>

      {/* ARAMA VE FİLTRELEME ÇUBUĞU */}
      <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl p-4 shadow-2xl backdrop-blur-md flex flex-wrap items-center justify-between gap-3">
        {/* Arama Alanı */}
        <div className="relative flex-1 min-w-[240px]">
          <Search className="w-4 h-4 text-[#D4C5A9]/70 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Vaka No, Vaka Adı, Mahkeme veya Esas No ile ara..."
            className="w-full pl-10 pr-4 py-2 bg-[#120D0A] border border-[#B8A484]/20 rounded-xl text-xs text-[#FAF7F0] placeholder-[#D4C5A9]/40 focus:outline-none focus:border-[#B8A484]/60 focus:ring-1 focus:ring-[#B8A484]/30 shadow-inner"
          />
        </div>

        {/* Durum Filtre Butonları */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[11px] text-[#D4C5A9] font-medium mr-1 flex items-center gap-1">
            <Filter className="w-3 h-3 text-burgundy-600" /> Durum:
          </span>
          {[
            { id: "ALL", label: "Tümü" },
            { id: "IN_PROGRESS", label: "Devam Eden" },
            { id: "COMPLETED", label: "Tamamlanan" },
            { id: "DRAFT", label: "Taslak" },
          ].map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => setStatusFilter(f.id)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition ${
                statusFilter === f.id
                  ? "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] text-[#FAF7F0] border border-[#B8A484]/40 shadow-sm"
                  : "bg-[#2A2019] text-[#D4C5A9] hover:text-[#FAF7F0] hover:bg-[#3D2E24] border border-[#B8A484]/15"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* BAŞARI / BİLGİLENDİRME BİLDİRİMİ */}
      {deleteFeedback && (
        <div
          className={`p-4 rounded-2xl text-xs flex items-center justify-between gap-3 shadow-inner ${
            deleteFeedback.type === "success"
              ? "bg-emerald-950/80 border border-emerald-700/80 text-emerald-200"
              : "bg-rose-950/80 border border-rose-800/80 text-rose-200"
          }`}
        >
          <div className="flex items-center gap-2">
            {deleteFeedback.type === "success" ? (
              <Check className="w-4 h-4 text-emerald-400 shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            )}
            <span>{deleteFeedback.message}</span>
          </div>
          <button
            type="button"
            onClick={() => setDeleteFeedback(null)}
            className="p-1 hover:bg-black/30 rounded-lg transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* HATA BİLDİRİMİ & YENİDEN DENEME ÇUBUĞU */}
      {error && (
        <div className="bg-rose-950/80 border border-rose-800/80 rounded-2xl p-4 text-xs text-rose-200 flex flex-wrap items-center justify-between gap-3 shadow-inner">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => fetchCases()}
            className="px-3.5 py-1.5 bg-rose-900 hover:bg-rose-800 text-rose-100 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-sm"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Tekrar Dene</span>
          </button>
        </div>
      )}

      {/* VAKA LİSTESİ VEYA EMPTY STATE */}
      {loadingCases ? (
        <div className="py-16 text-center text-xs text-[#D4C5A9] space-y-2 bg-[#1E1612]/60 rounded-2xl border border-[#B8A484]/15">
          <div className="w-8 h-8 rounded-full border-2 border-burgundy-600 border-t-transparent animate-spin mx-auto" />
          <p>Adli vaka kayıtları yükleniyor...</p>
        </div>
      ) : filteredCases.length === 0 ? (
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-3xl p-10 shadow-2xl backdrop-blur-md text-center space-y-4 max-w-lg mx-auto my-8">
          <div className="w-14 h-14 rounded-2xl bg-[#2A2019] border border-[#B8A484]/20 flex items-center justify-center text-[#D4C5A9] mx-auto shadow-inner">
            <FileSearch className="w-7 h-7 text-[#D4C5A9]/70" />
          </div>
          <div>
            <h3 className="text-base font-bold text-[#FAF7F0]">Henüz kayıtlı vaka bulunmamaktadır</h3>
            <p className="text-xs text-[#D4C5A9]/80 mt-1 max-w-xs mx-auto">
              Arama kriterlerinize uygun dosya bulunamadı veya henüz yeni bir adli vaka oluşturulmadı.
            </p>
          </div>
          <button
            type="button"
            onClick={onSelectNew}
            className="px-5 py-2.5 bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] font-semibold text-xs rounded-xl shadow-lg transition duration-200 inline-flex items-center gap-2 border border-[#B8A484]/30"
          >
            <PlusCircle className="w-4 h-4" />
            İlk Vakayı Oluştur
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {filteredCases.map((c) => {
            const statusBadge = getStatusBadge(c.status);
            const StatusIcon = statusBadge.icon;
            const isCompleted = c.status === "COMPLETED";

            return (
              <div
                key={c.case_id}
                className="bg-[#1E1612]/90 hover:bg-[#231A15] border border-[#B8A484]/25 hover:border-[#B8A484]/45 rounded-2xl p-5 shadow-xl backdrop-blur-md transition duration-200 flex flex-col md:flex-row md:items-center justify-between gap-4"
              >
                {/* Sol Alan: Vaka Künyesi */}
                <div className="flex-1 space-y-2.5">
                  <div className="flex flex-wrap items-center gap-2.5">
                    {/* Vaka ID */}
                    <span className="font-mono text-xs font-bold text-cream-200 bg-[#120D0A] px-2.5 py-0.5 rounded-lg border border-[#B8A484]/20 shadow-inner">
                      {c.case_id}
                    </span>

                    {/* Vaka Adı */}
                    <h3 className="text-sm sm:text-base font-bold text-[#FAF7F0]">
                      {c.name}
                    </h3>

                    {/* Örnek Dosya Rozeti */}
                    {(c.name.includes("ÖRNEK") || c.name.includes("DEMO") || c.case_id.startsWith("VAKA-2026-00")) && (
                      <span className="px-2.5 py-0.5 rounded-full text-[10px] font-extrabold bg-amber-500/20 text-amber-300 border border-amber-500/40 tracking-wide uppercase">
                        Örnek / Demo Vaka
                      </span>
                    )}

                    {/* İnceleme Türü */}
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-[#2A2019] text-[#EFE7DA] border border-[#B8A484]/20">
                      {getExamTypeLabel(c.examination_type)}
                    </span>
                  </div>

                  {/* Açıklama */}
                  {c.description && (
                    <p className="text-xs text-[#D4C5A9]/80 line-clamp-2 leading-relaxed">
                      {c.description}
                    </p>
                  )}

                  {/* Alt Üstveri Çubuğu */}
                  <div className="flex flex-wrap items-center gap-y-1.5 gap-x-4 text-[11px] text-[#D4C5A9]/75 pt-1">
                    {c.authority && (
                      <div className="flex items-center gap-1">
                        <Building2 className="w-3.5 h-3.5 text-burgundy-600" />
                        <span>{c.authority}</span>
                      </div>
                    )}
                    {c.examiner_id && (
                      <div className="flex items-center gap-1">
                        <User className="w-3.5 h-3.5 text-[#D4C5A9]" />
                        <span>Uzman: {c.examiner_id}</span>
                      </div>
                    )}
                    <div className="flex items-center gap-1">
                      <Calendar className="w-3.5 h-3.5 text-teal-400" />
                      <span>Son Çalışma: {formatDate(c.updated_at_utc || c.created_at_utc)}</span>
                    </div>
                  </div>
                </div>

                {/* Sağ Alan: Durum, Son Aşama ve Aksiyon Butonları */}
                <div className="flex flex-col sm:flex-row md:flex-col items-start md:items-end justify-between gap-3 border-t md:border-t-0 md:border-l border-[#B8A484]/15 pt-3 md:pt-0 md:pl-5 shrink-0">
                  <div className="flex flex-wrap md:flex-col items-start md:items-end gap-1.5">
                    {/* Durum Rozeti */}
                    <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-semibold flex items-center gap-1 ${statusBadge.className}`}>
                      <StatusIcon className="w-3 h-3" />
                      {statusBadge.label}
                    </span>

                    {/* Son Aşama */}
                    <span className="text-[10px] text-[#D4C5A9]/85 bg-[#120D0A] px-2.5 py-0.5 rounded-full border border-[#B8A484]/15">
                      Son aşama: <strong className="text-[#FAF7F0]">{getStageLabel(c.current_stage)}</strong>
                    </span>
                  </div>

                  {/* Buton Grubu: Vaka Sil (Yetkili) + Devam Et */}
                  <div className="flex items-center gap-2">
                    {canDeleteCase && (
                      <button
                        type="button"
                        onClick={() => setCaseToDelete(c)}
                        title="Vakayı veritabanından kalıcı olarak sil"
                        className="px-3 py-2 rounded-xl bg-rose-950/40 hover:bg-rose-900/80 text-rose-300 hover:text-rose-100 border border-rose-800/40 hover:border-rose-700 transition shadow-sm flex items-center gap-1 text-xs font-medium"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">Vaka Sil</span>
                      </button>
                    )}

                    <button
                      type="button"
                      onClick={() => selectCaseAndNavigate(c.case_id)}
                      className={`px-4 py-2 rounded-xl text-xs font-semibold shadow-lg transition duration-200 flex items-center gap-1.5 border ${
                        isCompleted
                          ? "bg-[#2A2019] hover:bg-[#3D2E24] text-[#FAF7F0] border-[#B8A484]/30 hover:border-[#B8A484]/50"
                          : "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] border-[#B8A484]/30"
                      }`}
                    >
                      <PlayCircle className="w-3.5 h-3.5 text-cream-200" />
                      <span>{isCompleted ? "Vakayı Görüntüle" : "Vakaya Devam Et"}</span>
                      <ArrowRight className="w-3.5 h-3.5 text-cream-200" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* VAKA SİLME ONAY MODALI */}
      {caseToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-[#1E1612] border border-rose-800/60 rounded-3xl p-6 sm:p-7 shadow-2xl max-w-lg w-full space-y-5 text-xs text-[#FAF7F0]">
            {/* Modal Başlığı */}
            <div className="flex items-center justify-between pb-3 border-b border-[#B8A484]/20">
              <div className="flex items-center gap-2.5 text-rose-400 font-bold text-sm">
                <div className="p-2 rounded-xl bg-rose-950/80 border border-rose-800/80">
                  <Trash2 className="w-5 h-5 text-rose-400" />
                </div>
                <span>Adli Vakayı Silme Onayı</span>
              </div>
              <button
                type="button"
                disabled={isDeleting}
                onClick={() => setCaseToDelete(null)}
                className="p-1 text-[#D4C5A9] hover:text-[#FAF7F0] rounded-lg transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Açıklaması */}
            <div className="space-y-3 text-xs leading-relaxed text-[#D4C5A9]">
              <p className="text-sm font-semibold text-[#FAF7F0]">
                Bu vakayı veritabanından kalıcı olarak silmek istediğinizden emin misiniz?
              </p>
              <div className="p-3 bg-[#120D0A] rounded-xl border border-[#B8A484]/20 space-y-1">
                <div>
                  <strong>Vaka Adı:</strong> <span className="text-[#FAF7F0]">{caseToDelete.name}</span>
                </div>
                <div>
                  <strong>Vaka ID:</strong> <span className="font-mono text-cream-200">{caseToDelete.case_id}</span>
                </div>
              </div>
              <p className="text-rose-300">
                Bu işlem vakanın tüm veritabanı kayıtlarını (yüklenen numuneler, analiz sonuçları, denetim günlükleri) ve diskteki adli görsellerini kalıcı olarak silecek ve geri alınamayacaktır.
              </p>
              {activeCaseId === caseToDelete.case_id && (
                <div className="p-3 bg-amber-950/70 border border-amber-600/50 rounded-xl text-amber-200 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <span>
                    Bu vaka şu anda aktif çalışma alanınızda bulunmaktadır. Silme işlemi tamamlandığında çalışma alanı kapatılacaktır.
                  </span>
                </div>
              )}
            </div>

            {/* Modal Butonları */}
            <div className="flex items-center justify-end gap-3 pt-3 border-t border-[#B8A484]/15">
              <button
                type="button"
                disabled={isDeleting}
                onClick={() => setCaseToDelete(null)}
                className="px-4 py-2.5 rounded-xl bg-[#2A2019] hover:bg-[#3D2E24] text-[#FAF7F0] font-medium border border-[#B8A484]/25 transition"
              >
                Vazgeç
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={handleConfirmDelete}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-rose-900 to-rose-700 hover:from-rose-800 hover:to-rose-600 text-rose-100 font-semibold border border-rose-600/60 shadow-lg transition flex items-center gap-2 disabled:opacity-50"
              >
                {isDeleting ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-rose-200 border-t-transparent rounded-full animate-spin" />
                    <span>Siliniyor...</span>
                  </>
                ) : (
                  <>
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Vakayı Kalıcı Olarak Sil</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
