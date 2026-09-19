"use client";

import React, { useState } from "react";
import {
  Scale,
  AlertTriangle,
  ShieldCheck,
  Info,
  ChevronDown,
  ChevronUp,
  Fingerprint,
  Zap,
  HelpCircle,
} from "lucide-react";
import { ForensicSimilarityScaleEvaluation, HybridFusionResult } from "@/services/api";

interface ForensicSimilarityScaleCardProps {
  similarityEvaluation?: ForensicSimilarityScaleEvaluation;
  fallbackSimilarityPct?: number;
  fusionResult?: HybridFusionResult;
}

const BANDS = [
  {
    key: "COMPLETE_DISPARITY",
    label: "Tam Uyumsuzluk",
    range: "%0 - %24",
    minPct: 0,
    maxPct: 24.9,
    colorBg: "bg-rose-950/40",
    colorBorder: "border-rose-500/40",
    colorText: "text-rose-400",
    colorBar: "bg-rose-600",
    desc: "Rastgele Sahtecilik / Alakasız Şahıs (H2)",
  },
  {
    key: "LOW_SIMILARITY_NON_SOURCE",
    label: "Düşük Benzerlik",
    range: "%25 - %49",
    minPct: 25,
    maxPct: 49.9,
    colorBg: "bg-orange-950/40",
    colorBorder: "border-orange-500/40",
    colorText: "text-orange-400",
    colorBar: "bg-orange-500",
    desc: "Kaba Taklit / Farklı Kaynak (H2)",
  },
  {
    key: "INCONCLUSIVE_DISGUISE_ZONE",
    label: "Kararsız / Bozma",
    range: "%50 - %69",
    minPct: 50,
    maxPct: 69.9,
    colorBg: "bg-amber-950/40",
    colorBorder: "border-amber-500/40",
    colorText: "text-amber-400",
    colorBar: "bg-amber-500",
    desc: "Olası İstekli Bozma (Disguise) / Amatör Taklit",
  },
  {
    key: "SKILLED_SIMULATION_ZONE",
    label: "Kritik Ayrım",
    range: "%70 - %84",
    minPct: 70,
    maxPct: 84.9,
    colorBg: "bg-yellow-950/40",
    colorBorder: "border-yellow-500/40",
    colorText: "text-yellow-300",
    colorBar: "bg-yellow-400",
    desc: "Nitelikli Serbest Taklit Şüphesi / Mikro Analiz Şart",
  },
  {
    key: "NATURAL_VARIATION_MATCH",
    label: "Doğal Varyasyon",
    range: "%85 - %97",
    minPct: 85,
    maxPct: 97.9,
    colorBg: "bg-emerald-950/40",
    colorBorder: "border-emerald-500/40",
    colorText: "text-emerald-400",
    colorBar: "bg-emerald-500",
    desc: "Orijinal / Yerleşik Motor Alışkanlıklar (H1)",
  },
  {
    key: "SUPERIMPOSITION_ANOMALY",
    label: "⚠️ Tetabuk / Dolandırıcılık",
    range: "%98 - %100",
    minPct: 98,
    maxPct: 100,
    colorBg: "bg-rose-950/70",
    colorBorder: "border-rose-500/70",
    colorText: "text-rose-300",
    colorBar: "bg-rose-500 animate-pulse",
    desc: "Anormal Çakışma / İzdüşüm & Dolandırıcılık Şüphesi (H2)",
  },
];

export const ForensicSimilarityScaleCard: React.FC<ForensicSimilarityScaleCardProps> = ({
  similarityEvaluation,
  fallbackSimilarityPct = 85,
  fusionResult,
}) => {
  const [showTheory, setShowTheory] = useState<boolean>(false);

  const activePct =
    similarityEvaluation?.similarity_percentage !== undefined
      ? similarityEvaluation.similarity_percentage
      : fallbackSimilarityPct;

  // Aktif bandı belirle
  const currentBandKey =
    similarityEvaluation?.band ||
    (activePct >= 98
      ? "SUPERIMPOSITION_ANOMALY"
      : activePct >= 85
      ? "NATURAL_VARIATION_MATCH"
      : activePct >= 70
      ? "SKILLED_SIMULATION_ZONE"
      : activePct >= 50
      ? "INCONCLUSIVE_DISGUISE_ZONE"
      : activePct >= 25
      ? "LOW_SIMILARITY_NON_SOURCE"
      : "COMPLETE_DISPARITY");

  const activeBandInfo = BANDS.find((b) => b.key === currentBandKey) || BANDS[4];
  const isTetabuk = currentBandKey === "SUPERIMPOSITION_ANOMALY";
  const isSimulationZone = currentBandKey === "SKILLED_SIMULATION_ZONE";

  return (
    <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl p-5 shadow-2xl backdrop-blur-md space-y-4">
      {/* ÜST BAŞLIK */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[#B8A484]/20">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-burgundy-900/60 border border-[#B8A484]/30 flex items-center justify-center text-[#D4C5A9] shadow-inner">
            <Fingerprint className="w-4 h-4 text-cream-200" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-[#FAF7F0] flex items-center gap-2">
              Kriminalojik Benzerlik ve Tetabuk Skalası
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#2A2019] text-[#D4C5A9] border border-[#B8A484]/25 font-mono">
                ENFSI &amp; ASTM E2290
              </span>
            </h3>
            <p className="text-[11px] text-[#D4C5A9]/75">
              Yüksek benzerlik her zaman aynı kişi demek değildir; doğal varyasyon ve izdüşüm riski değerlendirilir.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Normalizasyon Rozeti */}
          <span className="text-[10px] font-semibold px-2.5 py-1 rounded-lg bg-[#140E0B] text-teal-300 border border-teal-500/30 flex items-center gap-1">
            <ShieldCheck className="w-3 h-3 text-teal-400" /> Normalizasyon Aktif
          </span>

          <button
            type="button"
            onClick={() => setShowTheory(!showTheory)}
            className="flex items-center gap-1 text-[11px] text-[#D4C5A9] hover:text-[#FAF7F0] bg-[#2A2019] border border-[#B8A484]/20 px-2.5 py-1 rounded-lg transition"
          >
            <HelpCircle className="w-3.5 h-3.5 text-amber-400" />
            <span>{showTheory ? "Adli Teoriyi Gizle" : "Adli Teori & Tetabuk İlkesi"}</span>
            {showTheory ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* ADLİ TEORİ AÇIKLAMA MODAL / AKORDİYON */}
      {showTheory && (
        <div className="p-4 bg-[#120D0A] border border-[#B8A484]/30 rounded-xl space-y-3 text-xs text-[#EFE7DA]">
          <div className="flex items-center gap-2 text-amber-400 font-bold text-xs">
            <Info className="w-4 h-4" /> Kriminalistikte Tetabuk Paradoksu ve Doğal Varyasyon İlkesi
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1 text-[11px] leading-relaxed text-[#D4C5A9]">
            <div className="p-3 bg-[#1E1612] rounded-lg border border-[#B8A484]/15 space-y-1">
              <strong className="text-emerald-300 block">1. Doğal Varyasyon (Orijinallik Kanıtı):</strong>
              İnsan nöromüsküler sistemi gereği hiçbir şahıs imzasını iki kez milimetrik olarak birebir aynı atamaz.
              Belirli bir orandaki değişkenlik (%85–%97) gerçek yazarın motor reflekslerinin doğal sonucudur.
            </div>
            <div className="p-3 bg-[#1E1612] rounded-lg border border-rose-500/30 space-y-1">
              <strong className="text-rose-300 block">2. Tetabuk / İzdüşüm (Sahtecilik Kanıtı):</strong>
              İki imzanın %98–%100 oranında geometrik olarak tam üst üste oturması (tetabuk), imzanın insan eliyle değil;
              ışıklı masa, karbon kâğıdı, asetat, kaşe veya dijital montajla aktarıldığının kesin delilidir.
            </div>
          </div>
        </div>
      )}

      {/* GÖRSEL 6 KADEMELİ BENZERLİK GÖSTERGE ÇUBUĞU */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-[#D4C5A9]/70 font-semibold">0% (Tam Ayrışma)</span>
          <span className="text-cream-100 font-bold font-mono text-xs flex items-center gap-1.5">
            <span>Ölçülen Hibrit Benzerlik:</span>
            <span className="text-sm text-amber-300 font-extrabold">%{activePct.toFixed(1)}</span>
          </span>
          <span className="text-[#D4C5A9]/70 font-semibold">100% (Birebir)</span>
        </div>

        {/* 6 Parçalı Renkli Çubuk */}
        <div className="grid grid-cols-6 gap-1 h-3 rounded-full overflow-hidden bg-[#120D0A] p-0.5 border border-[#B8A484]/20">
          {BANDS.map((b) => {
            const isThisActive = b.key === currentBandKey;
            return (
              <div
                key={b.key}
                className={`h-full rounded-sm transition-all duration-300 ${b.colorBar} ${
                  isThisActive ? "ring-2 ring-white shadow-lg opacity-100 scale-y-110" : "opacity-40"
                }`}
                title={`${b.label}: ${b.range}`}
              />
            );
          })}
        </div>

        {/* Parça Başlıkları */}
        <div className="grid grid-cols-6 gap-1 text-center">
          {BANDS.map((b) => {
            const isThisActive = b.key === currentBandKey;
            return (
              <div key={b.key} className="space-y-0.5">
                <span
                  className={`text-[9px] block font-bold truncate ${
                    isThisActive ? b.colorText : "text-[#D4C5A9]/50"
                  }`}
                >
                  {b.label}
                </span>
                <span className="text-[8px] text-[#D4C5A9]/40 block font-mono">{b.range}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* AKTİF BANT ADLİ DEĞERLENDİRME KARTI */}
      <div
        className={`p-4 rounded-xl border ${activeBandInfo.colorBorder} ${activeBandInfo.colorBg} space-y-2 transition shadow-inner`}
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            {isTetabuk ? (
              <AlertTriangle className="w-4 h-4 text-rose-400 animate-bounce" />
            ) : isSimulationZone ? (
              <Zap className="w-4 h-4 text-yellow-400" />
            ) : (
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            )}
            <span className="text-xs font-bold text-[#FAF7F0]">
              Adli Karar Bandı: <span className={activeBandInfo.colorText}>{activeBandInfo.label}</span>
            </span>
          </div>

          <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-[#120D0A] text-[#FAF7F0] border border-[#B8A484]/30">
            Aralık: {activeBandInfo.range}
          </span>
        </div>

        <p className="text-xs text-[#EFE7DA] leading-relaxed">
          {similarityEvaluation?.forensic_interpretation_tr ||
            (isTetabuk
              ? "İki imza birebir çakışmaktadır. İnsan eli aynı imzayı iki kez %100 aynı atamayacağı için kopya / izdüşüm şüphesi taşır."
              : isSimulationZone
              ? "Genel şekil benzer; ancak taklit şüphesine karşı mikro titreme analizi gerekir."
              : "İmza akışı ve alışkanlıkları yazarın doğal varyasyon sınırları içerisindedir (Orijinal).")}
        </p>

        {/* Kademeli Ceza / Kalibrasyon Rozetleri */}
        {fusionResult?.penalty_details && fusionResult.penalty_details.length > 0 && (
          <div className="mt-2 p-2.5 bg-[#140E0B] border border-amber-500/30 rounded-xl space-y-1 text-xs">
            <div className="text-[11px] font-bold text-amber-300 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" /> Kademeli Ceza ve Kalibrasyon Dökümü:
            </div>
            <ul className="text-[10px] text-[#D4C5A9]/80 list-disc list-inside space-y-0.5">
              {fusionResult.penalty_details.map((detail, idx) => (
                <li key={idx}>{detail}</li>
              ))}
            </ul>
          </div>
        )}

        {isTetabuk && (
          <div className="p-2 bg-rose-950/80 border border-rose-500/50 rounded-lg text-[11px] text-rose-200 font-semibold flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-300 shrink-0" />
            <span>
              ⚠️ İki imza %98+ birebir çakışmaktadır; orijinal bir belgeden kopyalama / izdüşüm sahteciliği şüphesi taşır.
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
