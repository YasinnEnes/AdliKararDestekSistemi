"use client";

import React from "react";
import { Scale, AlertTriangle } from "lucide-react";
import { HypothesisEvaluationReport } from "@/services/api";

interface UncertaintyGaugeProps {
  hypothesis?: HypothesisEvaluationReport;
}

export const UncertaintyGauge: React.FC<UncertaintyGaugeProps> = ({ hypothesis }) => {
  if (!hypothesis) {
    return (
      <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl p-5 shadow-2xl backdrop-blur-md text-xs text-center text-[#D4C5A9]/70">
        Analiz tamamlandığında adli kanıt seviyesi burada görüntülenecektir.
      </div>
    );
  }

  const uPct = Math.round(hypothesis.uncertainty_score * 100);

  // Destek Düzeyi Metni
  let supportLabel = "Kuvvetle Aynı Yazar Lehine (H1)";
  let badgeColor = "bg-[#2A2019] text-emerald-400 border border-emerald-500/30";

  if (hypothesis.similarity_scale_evaluation?.is_superimposition_anomaly) {
    supportLabel = "⚠️ Tetabuk / İzdüşüm ve Dolandırıcılık Şüphesi (H2)";
    badgeColor = "bg-rose-950 text-rose-300 border border-rose-500/60 animate-pulse";
  } else if (hypothesis.support_level.includes("H2")) {
    supportLabel = "Farklı Yazar Lehine (H2)";
    badgeColor = "bg-[#2A2019] text-rose-400 border border-rose-500/30";
  } else if (!hypothesis.support_level.includes("H1")) {
    supportLabel = "Yetersiz / Eşit Kanıt";
    badgeColor = "bg-[#2A2019] text-amber-400 border border-amber-500/30";
  }

  return (
    <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl p-5 shadow-2xl backdrop-blur-md text-xs space-y-4">
      <div className="flex items-center justify-between pb-2.5 border-b border-[#B8A484]/20">
        <h3 className="font-bold text-xs text-[#FAF7F0] flex items-center gap-2">
          <Scale className="w-4 h-4 text-burgundy-600" />
          Adli Kanıt Seviyesi ve Belirsizlik Değerlendirmesi
        </h3>
        <span className={`px-3 py-0.5 rounded-full text-[11px] font-bold ${badgeColor}`}>
          {supportLabel}
        </span>
      </div>

      {/* 4 Ana Metrik */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
        <div className="bg-[#120D0A] border border-[#B8A484]/15 p-3 rounded-xl text-center shadow-inner">
          <div className="text-[#D4C5A9] text-[10px] uppercase font-semibold">Kanıt Gücü (Log-LR)</div>
          <div className="text-lg font-extrabold text-[#FAF7F0] mt-0.5">
            {hypothesis.log_likelihood_ratio.toFixed(2)}
          </div>
        </div>

        <div className="bg-[#120D0A] border border-[#B8A484]/15 p-3 rounded-xl text-center shadow-inner">
          <div className="text-[#D4C5A9] text-[10px] uppercase font-semibold">Olasılık Oranı (LR)</div>
          <div className="text-lg font-extrabold text-[#EFE7DA] mt-0.5">
            {hypothesis.likelihood_ratio_estimate >= 10000
              ? `${Math.round(hypothesis.likelihood_ratio_estimate / 1000)}k Kat`
              : `${hypothesis.likelihood_ratio_estimate.toFixed(1)} Kat`}
          </div>
        </div>

        <div className="bg-[#120D0A] border border-[#B8A484]/15 p-3 rounded-xl text-center shadow-inner">
          <div className="text-[#D4C5A9] text-[10px] uppercase font-semibold">Belirsizlik Oranı (U)</div>
          <div className={`text-lg font-extrabold mt-0.5 ${uPct > 35 ? "text-amber-400" : "text-emerald-400"}`}>
            %{uPct}
          </div>
        </div>

        <div className="bg-[#120D0A] border border-[#B8A484]/15 p-3 rounded-xl text-center shadow-inner">
          <div className="text-[#D4C5A9] text-[10px] uppercase font-semibold">%95 Güven Aralığı</div>
          <div className="text-sm font-bold text-[#EFE7DA] mt-1 font-mono">
            [{hypothesis.confidence_interval_log_lr[0].toFixed(2)}, {hypothesis.confidence_interval_log_lr[1].toFixed(2)}]
          </div>
        </div>
      </div>

      {/* Sözel Kanaat */}
      <div className="p-3.5 bg-[#2A2019]/90 border-l-4 border-burgundy-600 rounded-xl text-[11px] text-[#FAF7F0] shadow-sm">
        <strong className="text-cream-200">Bilirkişi Kanaat İfadesi: </strong>
        <span>{hypothesis.verbal_conclusion_tr}</span>
      </div>

      {/* İnceleme Şerhleri */}
      {hypothesis.forensic_limitations && hypothesis.forensic_limitations.length > 0 && (
        <div className="p-3 bg-[#2A2019]/90 border border-amber-500/30 rounded-xl text-[11px] text-amber-200 space-y-1">
          <div className="flex items-center gap-1.5 font-bold text-amber-400">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" /> İnceleme Şerhleri ve Notlar:
          </div>
          <ul className="list-disc pl-4 space-y-0.5 text-[#D4C5A9]">
            {hypothesis.forensic_limitations.map((lim, idx) => (
              <li key={idx}>{lim}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
