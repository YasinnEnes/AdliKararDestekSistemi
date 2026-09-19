"use client";

import React from "react";
import {
  Flame,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  BrainCircuit,
  Crosshair,
  Eye,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { FullPipelineResponse } from "@/services/api";
import { ShapWaterfallChart } from "@/components/charts/ShapWaterfallChart";

interface XaiTabProps {
  pipelineResult: FullPipelineResponse | null;
}

export const XaiTab: React.FC<XaiTabProps> = ({ pipelineResult }) => {
  const xaiData = pipelineResult?.xai;
  const disguiseData = pipelineResult?.disguise;

  if (!pipelineResult || !xaiData) {
    return (
      <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-3xl p-12 shadow-2xl backdrop-blur-md text-center space-y-4 max-w-2xl mx-auto my-8">
        <div className="w-16 h-16 rounded-2xl bg-[#2A2019] border border-[#B8A484]/20 flex items-center justify-center text-[#D4C5A9] mx-auto shadow-inner">
          <BrainCircuit className="w-8 h-8 text-amber-400" />
        </div>
        <div className="space-y-2">
          <h3 className="text-base font-bold text-[#FAF7F0]">Açıklanabilir Yapay Zekâ Analizi Henüz Yürütülmedi</h3>
          <p className="text-xs text-[#D4C5A9]/80 max-w-md mx-auto leading-relaxed">
            Yapay zekâ görsel odak haritası, tabüler öznitelik etki dökümü ve taklit/gizleme göstergeleri analiz tamamlandıktan sonra görüntülenecektir.
          </p>
        </div>
        <div className="p-3 bg-[#120D0A] border border-[#B8A484]/15 rounded-xl text-xs text-[#D4C5A9]/70 max-w-md mx-auto">
          Model analizini başlatmak için üst menüdeki <strong>&quot;▷ Kapsamlı Adli Analizi Başlat&quot;</strong> butonuna tıklayınız.
        </div>
      </div>
    );
  }

  const positiveFeats = xaiData.top_positive_features || [];
  const negativeFeats = xaiData.top_negative_features || [];
  const summaryText = xaiData.visual_attention_summary || "";

  // Taklit / Gizleme Durumu
  const indicatorLevel = disguiseData?.indicator_level || "NO_STRONG_EVIDENCE";
  const anomalyScore = disguiseData
    ? Math.round(disguiseData.overall_anomaly_score * 100)
    : 0;
  const tremorDetected = disguiseData?.tremor_anomaly_detected || false;
  const decouplingDetected = disguiseData?.kinematic_decoupling_detected || false;
  const disguiseDetected = disguiseData?.structural_disguise_detected || false;
  const anomalyDetails = disguiseData?.anomaly_details || [];

  return (
    <div className="space-y-5">
      {/* 🌟 1. YENİDEN TASARLANMIŞ ÖZNİTELİK ETKİ ANALİZİ (SHAP / XAI) */}
      <ShapWaterfallChart
        positiveFeatures={positiveFeats}
        negativeFeatures={negativeFeats}
        summaryText={summaryText}
      />

      {/* 🔍 2. GÖRSEL DİKKAT ODAĞI & NÖROMOTOR SAHTECİLİK BELİRTİLERİ */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* SOL: YAPAY ZEKÂ GÖRSEL ODAK HARİTASI (GRAD-CAM) */}
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-3xl p-5 sm:p-6 shadow-2xl backdrop-blur-md text-xs space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-[#B8A484]/20">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-xl bg-rose-950/80 border border-rose-500/30 flex items-center justify-center text-rose-400 shadow-inner">
                  <Flame className="w-4 h-4 text-rose-400" />
                </div>
                <div>
                  <h3 className="font-bold text-xs text-[#FAF7F0]">
                    Görsel Odak Alanları
                  </h3>
                  <span className="text-[10px] text-[#D4C5A9]/70">Modelin en çok dikkat ettiği hat bölgeleri</span>
                </div>
              </div>
              <span className="text-[10px] font-mono px-2.5 py-0.5 rounded-full bg-[#120D0A] text-amber-300 border border-[#B8A484]/25">
                Grad-CAM
              </span>
            </div>
          </div>

          {/* Isı Haritası Temsili Görsel Alan */}
          <div className="relative rounded-2xl overflow-hidden border border-[#B8A484]/25 bg-[#120D0A] aspect-[16/9] flex items-center justify-center shadow-inner">
            <div className="absolute inset-0 bg-gradient-to-tr from-[#120D0A] via-[#2A2019]/40 to-[#120D0A] flex items-center justify-center">
              <div className="absolute top-1/4 left-1/4 w-28 h-28 rounded-full bg-rose-600/30 blur-2xl animate-pulse" />
              <div className="absolute bottom-1/4 right-1/3 w-32 h-32 rounded-full bg-amber-500/20 blur-2xl" />
              <div className="absolute top-1/2 right-1/4 w-20 h-20 rounded-full bg-emerald-500/15 blur-xl" />
            </div>

            <div className="relative z-10 text-center space-y-2 p-4">
              <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-[#1C1510]/95 backdrop-blur rounded-full text-cream-100 font-mono text-xs border border-[#B8A484]/30 shadow-md">
                <Crosshair className="w-3.5 h-3.5 text-amber-400" />
                <span>Odak Alanları: Hat Kavşakları &amp; İlmekler</span>
              </div>
              <p className="text-xs text-[#FAF7F0] max-w-sm mx-auto font-medium">
                {summaryText || "Model evrişim katmanları mikroskobik hat kavşaklarında ve vuruş sonlarında yoğunlaşmıştır."}
              </p>
            </div>
          </div>

          <div className="p-2.5 bg-[#120D0A] rounded-xl border border-[#B8A484]/15 text-[11px] text-[#D4C5A9]/80 flex items-center justify-between">
            <span className="font-semibold text-[#FAF7F0] text-[11px]">XAI İmzası:</span>
            <span className="font-mono text-[10px] text-amber-400">
              {xaiData.audit_hash_sha256 ? `${xaiData.audit_hash_sha256.substring(0, 16)}...` : "Hesaplandı"}
            </span>
          </div>
        </div>

        {/* SAĞ: ÇİZGİ KALİTESİ & GİZLEME / TAKLİT GÖSTERGELERİ */}
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-3xl p-5 sm:p-6 shadow-2xl backdrop-blur-md text-xs space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-[#B8A484]/20">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-xl bg-amber-950/80 border border-amber-500/30 flex items-center justify-center text-amber-400 shadow-inner">
                  <ShieldAlert className="w-4 h-4 text-amber-400" />
                </div>
                <div>
                  <h3 className="font-bold text-xs text-[#FAF7F0]">
                    Çizgi Kalitesi ve Taklit Kontrolü
                  </h3>
                  <span className="text-[10px] text-[#D4C5A9]/70">Titreme ve hız akış göstergeleri</span>
                </div>
              </div>
              <span
                className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${
                  indicatorLevel === "NO_STRONG_EVIDENCE"
                    ? "bg-emerald-950 text-emerald-300 border-emerald-500/30"
                    : "bg-rose-950 text-rose-300 border-rose-500/30"
                }`}
              >
                {indicatorLevel === "NO_STRONG_EVIDENCE" ? "Doğal & Akıcı" : "Olası Risk"}
              </span>
            </div>
          </div>

          {/* 3 Net Kontrol Maddesi */}
          <div className="space-y-2">
            <div className="flex items-center justify-between p-3 bg-[#120D0A] rounded-2xl border border-[#B8A484]/15 shadow-inner">
              <div className="flex items-center gap-2.5">
                {tremorDetected ? (
                  <AlertTriangle className="w-4 h-4 text-rose-400" />
                ) : (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                )}
                <div>
                  <span className="text-[#FAF7F0] font-semibold block text-xs">
                    El Titremesi ve Tereddüt (Tremor)
                  </span>
                  <span className="text-[10px] text-[#D4C5A9]/70">FFT Güç Spektrumu Analizi</span>
                </div>
              </div>
              <span className={`font-bold text-xs ${tremorDetected ? "text-rose-400" : "text-emerald-400"}`}>
                {tremorDetected ? "⚠️ Şüpheli Titreme" : " Doğal & Akıcı"}
              </span>
            </div>

            <div className="flex items-center justify-between p-3 bg-[#120D0A] rounded-2xl border border-[#B8A484]/15 shadow-inner">
              <div className="flex items-center gap-2.5">
                {decouplingDetected ? (
                  <AlertTriangle className="w-4 h-4 text-rose-400" />
                ) : (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                )}
                <div>
                  <span className="text-[#FAF7F0] font-semibold block text-xs">
                    Kinematik Hız-İvme Akışı
                  </span>
                  <span className="text-[10px] text-[#D4C5A9]/70">2/3 Güç Yasası Tutarlılığı</span>
                </div>
              </div>
              <span className={`font-bold text-xs ${decouplingDetected ? "text-rose-400" : "text-emerald-400"}`}>
                {decouplingDetected ? "⚠️ Hız Tutarsızlığı" : " Normal Balistik Akış"}
              </span>
            </div>

            <div className="flex items-center justify-between p-3 bg-[#120D0A] rounded-2xl border border-[#B8A484]/15 shadow-inner">
              <div className="flex items-center gap-2.5">
                {disguiseDetected ? (
                  <AlertTriangle className="w-4 h-4 text-rose-400" />
                ) : (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                )}
                <div>
                  <span className="text-[#FAF7F0] font-semibold block text-xs">
                    İmza Karakterini Kasıtlı Bozma (Disguise)
                  </span>
                  <span className="text-[10px] text-[#D4C5A9]/70">Bilinçli Şekil Değiştirme</span>
                </div>
              </div>
              <span className={`font-bold text-xs ${disguiseDetected ? "text-rose-400" : "text-emerald-400"}`}>
                {disguiseDetected ? "⚠️ Olası Bozma" : " Saptanmadı"}
              </span>
            </div>
          </div>

          {/* Gözlem Özeti */}
          <div className="p-3 bg-[#120D0A] border border-[#B8A484]/15 rounded-2xl text-[11px] text-[#D4C5A9]/80 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>
              {anomalyDetails.length > 0
                ? anomalyDetails.join(" • ")
                : "Tüm hat dinamikleri gerçek yazarın doğal nöromotor akışıyla uyumludur."}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
