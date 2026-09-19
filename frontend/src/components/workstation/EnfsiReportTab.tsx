"use client";

import React from "react";
import {
  Scale,
  Download,
  ShieldCheck,
  Award,
  Hash,
  FileText,
} from "lucide-react";
import { FullPipelineResponse, CaseResponse } from "@/services/api";
import { UncertaintyGauge } from "@/components/charts/UncertaintyGauge";
import { AuditTrailTimeline, AuditItem } from "./AuditTrailTimeline";
import { ForensicSimilarityScaleCard } from "./ForensicSimilarityScaleCard";

interface EnfsiReportTabProps {
  pipelineResult: FullPipelineResponse | null;
  caseData: CaseResponse | null;
  caseId: string;
  auditItems: AuditItem[];
  handleDownloadPdf: () => void;
}

export const EnfsiReportTab: React.FC<EnfsiReportTabProps> = ({
  pipelineResult,
  caseData,
  caseId,
  auditItems,
  handleDownloadPdf,
}) => {
  if (!pipelineResult || !pipelineResult.hypothesis) {
    return (
      <div className="space-y-6">
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-3xl p-12 shadow-2xl backdrop-blur-md text-center space-y-4 max-w-2xl mx-auto my-8">
          <div className="w-16 h-16 rounded-2xl bg-[#2A2019] border border-[#B8A484]/20 flex items-center justify-center text-[#D4C5A9] mx-auto shadow-inner">
            <Scale className="w-8 h-8 text-burgundy-600" />
          </div>
          <div className="space-y-2">
            <h3 className="text-base font-bold text-[#FAF7F0]">Resmi Adli Bilirkişi Raporu Henüz Hazırlanmadı</h3>
            <p className="text-xs text-[#D4C5A9]/80 max-w-md mx-auto leading-relaxed">
              Bu vaka için ENFSI Bayesyen hipotez değerlendirmesi (H1/H2), Likelihood Ratio (LR) hesaplaması ve dijital mühürlü bilirkişi raporu henüz oluşturulmamıştır.
            </p>
          </div>
          <div className="p-3 bg-[#120D0A] border border-[#B8A484]/15 rounded-xl text-xs text-[#D4C5A9]/70 max-w-md mx-auto">
            Rapor oluşturmak ve adli kanıt seviyesini hesaplamak için üst menüdeki <strong>&quot;Kapsamlı Adli Analizi Başlat&quot;</strong> adımını tamamlayınız.
          </div>
        </div>

        {/* DENETİM İZİ ZAMAN ÇİZELGESİ (Varsa işlem kayıtlarını göster) */}
        <AuditTrailTimeline items={auditItems} />
      </div>
    );
  }

  const hypothesis = pipelineResult.hypothesis;
  const reportSha256 = pipelineResult.audit_checksum_sha256 || "Mühür Bekleniyor";
  const hybridPct = pipelineResult.fusion
    ? Math.round(pipelineResult.fusion.hybrid_similarity_score * 100)
    : 85;

  return (
    <div className="space-y-4">
      {/* RAPOR ÜST BANTI */}
      <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl p-5 shadow-2xl backdrop-blur-md flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-xl bg-burgundy-700/40 border border-burgundy-600/50 flex items-center justify-center text-cream-100 shadow-inner">
            <Award className="w-6 h-6 text-cream-200" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-[#FAF7F0]">
              Resmi Adli Bilirkişi Karar ve Değerlendirme Raporu
            </h2>
            <p className="text-xs text-[#D4C5A9]/75">
              Vaka No: <span className="font-mono text-[#FAF7F0] font-semibold">{caseId}</span> • Standart: ENFSI BPM-FHX-01
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={handleDownloadPdf}
          className="px-5 py-2.5 bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] font-semibold text-xs rounded-full transition duration-200 shadow-lg flex items-center gap-2 border border-[#B8A484]/30"
        >
          <Download className="w-4 h-4 text-cream-200" /> Mühürlü Raporu İndir (PDF)
        </button>
      </div>

      {/* ADLİ KANIT SEVİYESİ & BELİRSİZLİK GÖSTERGESİ */}
      <UncertaintyGauge hypothesis={hypothesis} />

      {/* 6 KADEMELİ KRİMİNALOJİK BENZERLİK VE TETABUK SKALASI */}
      <ForensicSimilarityScaleCard
        similarityEvaluation={hypothesis.similarity_scale_evaluation}
        fallbackSimilarityPct={hybridPct}
        fusionResult={pipelineResult.fusion}
      />

      {/* MÜHÜR VE HİPOTEZLER */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* SOL: DİJİTAL MÜHÜR */}
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl p-5 shadow-2xl backdrop-blur-md text-xs space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-[#B8A484]/20">
            <h3 className="font-bold text-xs text-[#FAF7F0] flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-teal-400" />
              Adli Karar Bütünlük Mührü
            </h3>
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-[#2A2019] text-emerald-400 border border-emerald-500/30">
              Kriptografik SHA-256
            </span>
          </div>

          <div className="space-y-2.5">
            <div className="p-3 bg-[#120D0A] rounded-xl border border-[#B8A484]/15 space-y-1 shadow-inner">
              <span className="text-[10px] text-[#D4C5A9] uppercase font-semibold flex items-center gap-1">
                <Hash className="w-3.5 h-3.5 text-burgundy-600" /> Rapor SHA-256 Özeti:
              </span>
              <p className="font-mono text-xs text-[#FAF7F0] font-bold break-all select-all">{reportSha256}</p>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="bg-[#120D0A] p-2.5 rounded-xl border border-[#B8A484]/15 shadow-inner">
                <span className="text-[10px] text-[#D4C5A9] uppercase font-semibold">Bilirkişi / Uzman:</span>
                <p className="text-xs font-semibold text-[#FAF7F0] mt-0.5">
                  {caseData?.examiner_id || "Belirtilmedi"}
                </p>
              </div>

              <div className="bg-[#120D0A] p-2.5 rounded-xl border border-[#B8A484]/15 shadow-inner">
                <span className="text-[10px] text-[#D4C5A9] uppercase font-semibold">Yetkili Makam:</span>
                <p className="text-xs font-semibold text-[#FAF7F0] mt-0.5">
                  {caseData?.authority || "Belirtilmedi"}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* SAĞ: HİPOTEZLER */}
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl p-5 shadow-2xl backdrop-blur-md text-xs space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-[#B8A484]/20">
            <h3 className="font-bold text-xs text-[#FAF7F0] flex items-center gap-2">
              <Scale className="w-4 h-4 text-burgundy-600" />
              İncelenen Yarışan Hipotezler
            </h3>
            <span className="text-[11px] text-[#D4C5A9]/70">ENFSI Modeli</span>
          </div>

          <div className="space-y-2">
            <div className="p-3 bg-[#120D0A] border border-emerald-500/30 rounded-xl shadow-inner">
              <div className="text-emerald-400 font-bold text-[11px] mb-0.5">
                H1 Hipotezi (Aynı Yazar / Orijinal İddiası):
              </div>
              <p className="text-[11px] text-[#EFE7DA]">
                {hypothesis.propositions?.H1 || "Sorgulanan imza/yazı ile mukayese numuneler aynı yazarın el ürünüdür."}
              </p>
            </div>

            <div className="p-3 bg-[#120D0A] border border-rose-500/30 rounded-xl shadow-inner">
              <div className="text-rose-400 font-bold text-[11px] mb-0.5">
                H2 Hipotezi (Farklı Yazar / Sahtecilik İddiası):
              </div>
              <p className="text-[11px] text-[#EFE7DA]">
                {hypothesis.propositions?.H2 ||
                  "Sorgulanan imza/yazı mukayese numunelerden farklı bir şahsın el ürünüdür."}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* DENETİM İZİ ZAMAN ÇİZELGESİ VE MERKLE KÖKÜ */}
      <AuditTrailTimeline items={auditItems} merkleRoot={pipelineResult?.merkle_root_hash} />
    </div>
  );
};
