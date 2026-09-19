"use client";

import React, { useState, useEffect } from "react";
import {
  Shield,
  Activity,
  CheckCircle2,
  AlertCircle,
  X,
  Sparkles,
  Cpu,
  Lock,
  FileCheck,
  Scale,
  BrainCircuit,
  ArrowRight,
  Fingerprint,
} from "lucide-react";

import { api } from "@/services/api";

interface Props {
  isOpen: boolean;
  jobId: string | null;
  onClose: () => void;
  onCompleted: (result: any) => void;
}

const pipelineMilestones = [
  { id: "QUEUED", label: "İş Kuyruğuna Alındı", percent: 5, icon: Lock },
  { id: "PREPROCESSING", label: "Adli Ön İşleme & Görsel Kalite", percent: 20, icon: Cpu },
  { id: "MORPHOLOGY", label: "Morfolojik & Geometrik Özellik Çıkarımı", percent: 45, icon: Activity },
  { id: "CONVNEXT", label: "ConvNeXt-Tiny Derin Metrik Öğrenme", percent: 70, icon: BrainCircuit },
  { id: "FUSION", label: "ENFSI Bayesyen Adli Füzyon", percent: 85, icon: Scale },
  { id: "COMPLETED", label: "Rapor Derlendi & Merkle Mühürlendi", percent: 100, icon: FileCheck },
];

export const AnalysisProgressModal: React.FC<Props> = ({
  isOpen,
  jobId,
  onClose,
  onCompleted,
}) => {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState("Adli analiz boru hattı hazırlanıyor...");
  const [status, setStatus] = useState<"PENDING" | "RUNNING" | "COMPLETED" | "FAILED">("PENDING");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [finalResult, setFinalResult] = useState<any>(null);

  useEffect(() => {
    if (!isOpen || !jobId) {
      setProgress(0);
      setStatus("PENDING");
      setCurrentStep("Adli analiz boru hattı hazırlanıyor...");
      setErrorMessage(null);
      setFinalResult(null);
      return;
    }

    const eventSource = new EventSource(api.getJobStreamUrl(jobId));

    eventSource.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);
        if (payload.progress !== undefined) {
          setProgress(payload.progress);
        }
        if (payload.step) {
          setCurrentStep(payload.step);
        }
        if (payload.status) {
          setStatus(payload.status);
        }
        if (payload.error) {
          setErrorMessage(payload.error);
        }
        if (payload.status === "COMPLETED") {
          setFinalResult(payload.result);
          eventSource.close();
        }
        if (payload.status === "FAILED") {
          eventSource.close();
        }
      } catch (err) {
        console.error("SSE parse error:", err);
      }
    };

    eventSource.onerror = () => {
      console.warn("SSE bağlantı kesildi veya kapandı.");
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [isOpen, jobId]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#120D0A]/85 backdrop-blur-xl animate-fade-in">
      <div className="bg-gradient-to-b from-[#1C1510] via-[#18120F] to-[#120D0A] border border-[#B8A484]/35 rounded-2xl w-full max-w-xl p-6 shadow-2xl shadow-black/90 relative overflow-hidden">
        {/* Top Accent Gold Glow Line */}
        <div className="absolute top-0 inset-x-0 h-[2px] bg-gradient-to-r from-transparent via-[#B8A484]/60 to-transparent" />

        {/* Ambient Depth Glows - Softened, warm espresso & deep burgundy without neon green diffusion */}
        <div className="absolute -top-12 -right-12 w-64 h-64 bg-[#7B1E34]/15 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-16 -left-16 w-48 h-48 bg-[#3D2E24]/20 rounded-full blur-3xl pointer-events-none" />

        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-[#B8A484]/20 relative z-10">
          <div className="flex items-center gap-3">
            <div
              className={`p-2.5 rounded-xl border transition-all duration-300 ${
                status === "COMPLETED"
                  ? "bg-emerald-950/50 border-emerald-700/40 text-emerald-400/80 shadow-sm"
                  : status === "FAILED"
                  ? "bg-rose-950/50 border-rose-700/40 text-rose-400/80 shadow-sm"
                  : "bg-gradient-to-br from-[#7B1E34]/30 to-[#2A2019] border-[#B8A484]/30 text-[#FAF7F0]"
              }`}
            >
              {status === "COMPLETED" ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-500/90" />
              ) : status === "FAILED" ? (
                <AlertCircle className="w-5 h-5 text-rose-400" />
              ) : (
                <Sparkles className="w-5 h-5 text-amber-400 animate-spin-slow" />
              )}
            </div>

            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold text-[#FAF7F0] tracking-wide font-sans">
                  Uçtan Uca Asenkron Adli Analiz Boru Hattı
                </h3>
                {status === "RUNNING" && (
                  <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-500/15 border border-amber-400/30 text-[10px] font-mono text-amber-300 font-semibold">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping" />
                    İŞLENİYOR
                  </span>
                )}
                {status === "COMPLETED" && (
                  <span className="px-2 py-0.5 rounded-full bg-[#18120F] border border-[#B8A484]/30 text-[10px] font-mono text-[#D4C5A9] font-medium">
                    MÜHÜRLENDİ
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs font-mono text-[#D4C5A9]/70">İş Takip Kodu:</span>
                <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-md bg-[#120D0A] border border-[#B8A484]/20 text-[#FAF7F0]">
                  {jobId}
                </span>
              </div>
            </div>
          </div>

          {(status === "COMPLETED" || status === "FAILED") && (
            <button
              onClick={onClose}
              className="text-[#D4C5A9]/60 hover:text-[#FAF7F0] p-1.5 rounded-xl hover:bg-[#2A2019] border border-transparent hover:border-[#B8A484]/20 transition"
              title="Pencereyi Kapat"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Dynamic Progress Indicator & Bar - Muted and soft without neon glare */}
        <div className="mt-5 space-y-2 relative z-10">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-[#FAF7F0] font-semibold flex items-center gap-1.5 truncate max-w-[80%]">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse shrink-0" />
              <span className="truncate">{currentStep}</span>
            </span>
            <span className="text-sm font-bold font-mono px-2 py-0.5 rounded-lg bg-[#120D0A] border border-[#B8A484]/20 text-[#FAF7F0] shadow-sm shrink-0">
              %{progress}
            </span>
          </div>

          <div className="w-full bg-[#120D0A] h-3 rounded-full overflow-hidden border border-[#B8A484]/20 p-0.5 shadow-inner relative">
            <div
              className={`h-full rounded-full transition-all duration-500 relative overflow-hidden ${
                status === "FAILED"
                  ? "bg-gradient-to-r from-rose-700 to-burgundy-800"
                  : status === "COMPLETED"
                  ? "bg-gradient-to-r from-emerald-800 via-emerald-700 to-teal-700/90 border-r border-emerald-600/40"
                  : "bg-gradient-to-r from-[#7B1E34] via-[#5C1425] to-emerald-800/80"
              }`}
              style={{ width: `${progress}%` }}
            >
              {/* Subtle softened light line without aggressive white pearlescent flash */}
              {status !== "FAILED" && status !== "COMPLETED" && (
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-shimmer" />
              )}
            </div>
          </div>
        </div>

        {/* Pipeline Step Checklist - Toned down, warm and elegant */}
        <div className="mt-5 space-y-2 relative z-10">
          {pipelineMilestones.map((m, idx) => {
            const isFinished = progress >= m.percent;
            const isCurrent =
              progress < m.percent &&
              (idx === 0 || progress >= pipelineMilestones[idx - 1].percent);
            const Icon = m.icon;

            return (
              <div
                key={m.id}
                className={`flex items-center justify-between p-2.5 rounded-xl text-xs font-mono border transition-all duration-200 ${
                  isFinished
                    ? "bg-[#1E1612]/80 border-[#B8A484]/20 text-[#FAF7F0]"
                    : isCurrent
                    ? "bg-[#2A2019] border-[#B8A484]/40 text-[#FAF7F0] shadow-sm"
                    : "bg-[#120D0A]/40 border-[#B8A484]/10 text-[#D4C5A9]/35"
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <div
                    className={`p-1.5 rounded-lg border transition ${
                      isFinished
                        ? "bg-[#120D0A] border-emerald-800/40 text-emerald-500/80"
                        : isCurrent
                        ? "bg-[#18120F] border-amber-500/40 text-amber-300"
                        : "bg-[#18120F] border-[#B8A484]/10 text-[#D4C5A9]/30"
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <span className={isFinished || isCurrent ? "font-semibold text-[#FAF7F0]" : ""}>
                    {m.label}
                  </span>
                </div>

                {isFinished ? (
                  <div className="flex items-center gap-1 text-emerald-500/80">
                    <CheckCircle2 className="w-4 h-4" />
                  </div>
                ) : isCurrent ? (
                  <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-amber-500/15 border border-amber-400/30 text-amber-300 text-[10px] font-semibold">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping" />
                    İşleniyor
                  </span>
                ) : (
                  <span className="text-[10px] font-mono text-[#D4C5A9]/30">Bekliyor</span>
                )}
              </div>
            );
          })}
        </div>

        {/* Error State Card */}
        {status === "FAILED" && (
          <div className="mt-5 p-4 bg-gradient-to-br from-[#3D0D18]/90 via-[#2A0810] to-[#1C1510] border border-rose-500/50 rounded-xl flex items-start gap-3.5 text-xs text-rose-200 shadow-xl shadow-rose-950/40 relative z-10">
            <div className="p-2 rounded-lg bg-rose-500/20 border border-rose-400/40 text-rose-400 shrink-0 mt-0.5">
              <AlertCircle className="w-5 h-5" />
            </div>
            <div className="flex-1 min-w-0">
              <strong className="block font-bold text-rose-100 text-sm mb-1 font-sans">
                Analiz Hatası / Bütünlük İhlali
              </strong>
              <p className="font-mono text-[11px] text-rose-200/90 break-all bg-[#120D0A]/60 p-2 rounded border border-rose-500/20">
                {errorMessage}
              </p>
            </div>
          </div>
        )}

        {/* Completed State Card */}
        {status === "COMPLETED" && (
          <div className="mt-5 p-4 bg-gradient-to-br from-[#221913] via-[#1C1510] to-[#140E0A] border border-[#B8A484]/30 rounded-xl space-y-3.5 shadow-xl shadow-black/60 relative z-10 overflow-hidden">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-emerald-500/90 font-bold text-sm">
                <CheckCircle2 className="w-5 h-5 text-emerald-500/90" />
                <span className="text-[#FAF7F0] font-sans">Adli İnceleme Başarıyla Tamamlandı</span>
              </div>
              <span className="flex items-center gap-1 text-[11px] font-mono text-[#D4C5A9] bg-[#120D0A] px-2 py-0.5 rounded border border-[#B8A484]/30">
                <Shield className="w-3 h-3 text-amber-400" />
                Mühürlü Rapor
              </span>
            </div>

            {finalResult && (
              <div className="grid grid-cols-2 gap-3 pt-2.5 border-t border-[#B8A484]/20 text-xs font-mono">
                <div className="bg-[#120D0A]/70 p-2.5 rounded-xl border border-[#B8A484]/20">
                  <span className="text-[#D4C5A9]/70 text-[10px] block uppercase font-bold tracking-wider">
                    Benzerlik Skoru
                  </span>
                  <div className="text-xl font-black text-emerald-400/85 font-mono mt-0.5">
                    %{(finalResult.similarity_score * 100).toFixed(1)}
                  </div>
                </div>

                <div className="bg-[#120D0A]/70 p-2.5 rounded-xl border border-[#B8A484]/20">
                  <span className="text-[#D4C5A9]/70 text-[10px] block uppercase font-bold tracking-wider">
                    ENFSI Kanaati
                  </span>
                  <div className="text-xs font-semibold text-[#FAF7F0] truncate block mt-1" title={finalResult.verbal_conclusion}>
                    {finalResult.verbal_conclusion || "Adli Rapor Oluşturuldu"}
                  </div>
                </div>
              </div>
            )}

            {finalResult?.merkle_root_hash && (
              <div className="flex items-center gap-1.5 text-[10px] font-mono text-[#D4C5A9]/60 px-2 py-1 bg-[#120D0A]/50 rounded border border-[#B8A484]/10">
                <Fingerprint className="w-3 h-3 text-amber-400 shrink-0" />
                <span className="truncate">Merkle Kökü: {finalResult.merkle_root_hash}</span>
              </div>
            )}

            <button
              onClick={() => {
                onCompleted(finalResult);
                onClose();
              }}
              className="w-full mt-2 py-3 rounded-xl bg-gradient-to-r from-[#7B1E34] via-[#5C1425] to-[#7B1E34] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] font-bold text-xs tracking-wider uppercase shadow-xl shadow-[#7B1E34]/30 border border-[#B8A484]/40 hover:scale-[1.01] active:scale-[0.99] transition-all flex items-center justify-center gap-2 cursor-pointer"
            >
              <span>Sonuçları ve Raporu Görüntüle</span>
              <ArrowRight className="w-4 h-4 text-amber-400" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
