"use client";

import React, { useState } from "react";
import {
  Archive,
  Download,
  Upload,
  CheckCircle2,
  AlertTriangle,
  X,
  FileCode,
  ShieldCheck,
  Lock,
  Clock,
  Sparkles,
  RefreshCw,
} from "lucide-react";

import { api, BACKEND_URL, formatApiError } from "@/services/api";

interface Props {
  isOpen: boolean;
  caseId: string;
  onClose: () => void;
}

export const ForensicArchiveModal: React.FC<Props> = ({ isOpen, caseId, onClose }) => {
  const [activeSubTab, setActiveSubTab] = useState<"export" | "verify">("export");

  // Export State
  const [exporting, setExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState<any>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  // Verify State
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<any>(null);
  const [verifyError, setVerifyError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleExportArchive = async () => {
    setExporting(true);
    setExportError(null);
    setExportSuccess(null);
    try {
      const data = await api.exportArchive(caseId);
      setExportSuccess(data);

      // Otomatik indirmeyi tetikle
      if (data.download_url) {
        const downloadAnchor = document.createElement("a");
        downloadAnchor.href = `${BACKEND_URL}${data.download_url}`;
        downloadAnchor.download = data.filename;
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
      }
    } catch (err: any) {
      setExportError(formatApiError(err) || "İhraç işlemi başarısız.");
    } finally {
      setExporting(false);
    }
  };

  const handleVerifyUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setVerifying(true);
    setVerifyError(null);
    setVerifyResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const data = await api.verifyArchive(formData);
      setVerifyResult(data);
    } catch (err: any) {
      setVerifyError(formatApiError(err) || "Doğrulama işlemi başarısız.");
    } finally {
      setVerifying(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/85 backdrop-blur-md animate-fade-in">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-2xl p-6 shadow-2xl relative overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-indigo-500/10 rounded-lg border border-indigo-500/30 text-indigo-400">
              <Archive className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white tracking-wide">
                Adli Vaka Arşiv Paketi (.ZIP) Yönetimi
              </h3>
              <p className="text-xs font-mono text-slate-400 mt-0.5">
                5070 E-İmza & RFC 3161 Zaman Damgalı Kriptografik Paket
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Subtabs */}
        <div className="flex items-center gap-2 border-b border-slate-800 pt-3 pb-3">
          <button
            onClick={() => setActiveSubTab("export")}
            className={`px-4 py-2 rounded-lg text-xs font-mono font-bold flex items-center gap-2 transition ${
              activeSubTab === "export"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/20"
                : "text-slate-400 hover:text-white hover:bg-slate-800"
            }`}
          >
            <Download className="w-3.5 h-3.5" /> Adli Paket İhraç Et (.ZIP)
          </button>
          <button
            onClick={() => setActiveSubTab("verify")}
            className={`px-4 py-2 rounded-lg text-xs font-mono font-bold flex items-center gap-2 transition ${
              activeSubTab === "verify"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/20"
                : "text-slate-400 hover:text-white hover:bg-slate-800"
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5" /> Gelen Paketi Doğrula (Tamper Check)
          </button>
        </div>

        {/* Body Content */}
        <div className="overflow-y-auto flex-1 py-4 space-y-4">
          {activeSubTab === "export" ? (
            <div className="space-y-4">
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-3">
                <h4 className="text-xs font-bold font-mono text-cyan-300 uppercase tracking-wider">
                  Paket İçerik Ağacı (FHDSS Forensic Archive Standard)
                </h4>
                <div className="space-y-2 text-xs font-mono text-slate-300">
                  <div className="flex items-center gap-2">
                    <span className="text-cyan-400">📁</span>
                    <span>/evidence/ – Orijinal Sorgulanan ve Referans Delil Görselleri (SHA-256)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-cyan-400">📁</span>
                    <span>/reports/ – Mühürlü Adli Karar Destek PDF ve HTML Bilirkişi Raporları</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-cyan-400">📁</span>
                    <span>/analysis/ – Ham Öznitelik, ONNX Derin Metrik ve ENFSI Bayesyen Sonuçları</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-cyan-400">📁</span>
                    <span>/audit/ – Değiştirilemez Kriptografik Hash-Chained Denetim İzi Zinciri</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-cyan-400">📄</span>
                    <span className="text-emerald-300 font-semibold">manifest.json – 5070 E-İmza & RFC 3161 Zaman Damgalı Bütünlük Özeti</span>
                  </div>
                </div>
              </div>

              {exportError && (
                <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg flex items-center gap-2 text-xs text-rose-300">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{exportError}</span>
                </div>
              )}

              {exportSuccess && (
                <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg space-y-2 text-xs font-mono">
                  <div className="flex items-center gap-2 text-emerald-400 font-bold">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Adli Arşiv Paketi Başarıyla Üretildi!</span>
                  </div>
                  <p className="text-slate-300">Dosya Adı: <strong className="text-white">{exportSuccess.filename}</strong></p>
                  <p className="text-slate-300">Boyut: <strong className="text-white">{(exportSuccess.size_bytes / 1024).toFixed(1)} KB</strong></p>
                </div>
              )}

              <button
                onClick={handleExportArchive}
                disabled={exporting}
                className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-bold rounded-xl text-xs font-mono transition flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/30"
              >
                {exporting ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" /> Adli Paket Paketleniyor & Mühürleniyor...
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4" /> Güvenli Adli Arşivi Üret ve İndir (.ZIP)
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-8 flex flex-col items-center justify-center text-center transition cursor-pointer relative bg-slate-950/40">
                <input
                  type="file"
                  accept=".zip"
                  onChange={handleVerifyUpload}
                  disabled={verifying}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                />
                <Upload className="w-10 h-10 text-indigo-400 mb-3" />
                <p className="text-sm font-semibold text-slate-200">
                  Doğrulanacak Adli Arşiv Paketini (.ZIP) Seçin veya Sürükleyin
                </p>
                <p className="text-xs text-slate-400 font-mono mt-1">
                  manifest.json dosyasındaki SHA-256 hashleri ve mühürler doğrulanacaktır
                </p>
              </div>

              {verifying && (
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-center gap-2 text-xs font-mono text-cyan-400">
                  <RefreshCw className="w-4 h-4 animate-spin" /> Kriptografik manifest ve SHA-256 özetleri denetleniyor...
                </div>
              )}

              {verifyError && (
                <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg flex items-center gap-2 text-xs text-rose-300">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{verifyError}</span>
                </div>
              )}

              {verifyResult && (
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-slate-400">Doğrulama Sonucu:</span>
                    <span
                      className={`px-2.5 py-0.5 rounded text-xs font-bold font-mono border ${
                        verifyResult.is_valid
                          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                          : "bg-rose-500/10 text-rose-400 border-rose-500/30"
                      }`}
                    >
                      {verifyResult.is_valid ? "TAM & TAHRİF EDİLMEMİŞ" : "TAHRİFAT / BOZULMA TESPİT EDİLDİ"}
                    </span>
                  </div>

                  <p className="text-xs text-slate-300 font-semibold">{verifyResult.message}</p>

                  <div className="grid grid-cols-2 gap-2 text-xs font-mono pt-2 border-t border-slate-800">
                    <div>
                      <span className="text-slate-400 text-[10px] block">Vaka No:</span>
                      <strong className="text-white">{verifyResult.case_id || "Bilinmiyor"}</strong>
                    </div>
                    <div>
                      <span className="text-slate-400 text-[10px] block">Toplam Dosya:</span>
                      <strong className="text-white">
                        {verifyResult.verified_files_count} / {verifyResult.total_files} Doğrulandı
                      </strong>
                    </div>
                  </div>

                  {verifyResult.tampered_files?.length > 0 && (
                    <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg space-y-1">
                      <span className="text-xs font-bold text-rose-400 block">Bozuk / Değiştirilmiş Dosyalar:</span>
                      {verifyResult.tampered_files.map((t: any, i: number) => (
                        <div key={i} className="text-[11px] font-mono text-rose-300">
                          {t.path}: {t.error}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
