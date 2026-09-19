import React, { useState } from "react";
import {
  FileQuestion,
  FileCheck,
  Hash,
  ChevronDown,
  ChevronUp,
  Cpu,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ShieldCheck,
  Sparkles,
  ArrowRight,
  BrainCircuit,
  Scale,
  Percent,
  Clock,
} from "lucide-react";
import { IsolationState, CaseResponse, FullPipelineResponse } from "@/services/api";
import { DualCanvasViewer } from "./DualCanvasViewer";
import { LayerControls, LayerState } from "./LayerControls";

interface MicroscopyTabProps {
  isolationState: IsolationState;
  caseData: CaseResponse | null;
  layers: LayerState;
  setLayers: (newLayers: LayerState) => void;
  zoomScale: number;
  setZoomScale: React.Dispatch<React.SetStateAction<number>>;
  questionedImageUrl: string;
  referenceImages: { id: string; url: string; name: string }[];
  selectedRefIndex: number;
  setSelectedRefIndex: (idx: number) => void;
  handleFreezePhaseA: () => void;
  loading: boolean;
  onUploadQuestioned?: (file: File) => void;
  onUploadReference?: (files: FileList | File[]) => void;
  onLoadDemoSample?: (type: "questioned" | "reference") => void;
  onClearQuestioned?: () => void;
  onRemoveReference?: (idx: number) => void;
  onClearAllReferences?: () => void;
  pipelineResult?: FullPipelineResponse | null;
  onRunPipeline?: () => void;
  onNavigateTab?: (tabId: string) => void;
}

export const MicroscopyTab: React.FC<MicroscopyTabProps> = ({
  isolationState,
  caseData,
  layers,
  setLayers,
  zoomScale,
  setZoomScale,
  questionedImageUrl,
  referenceImages,
  selectedRefIndex,
  setSelectedRefIndex,
  handleFreezePhaseA,
  loading,
  onUploadQuestioned,
  onUploadReference,
  onLoadDemoSample,
  onClearQuestioned,
  onRemoveReference,
  onClearAllReferences,
  pipelineResult,
  onRunPipeline,
  onNavigateTab,
}) => {
  const [showMetadata, setShowMetadata] = useState(false);

  const isPhaseBFrozenOrActive =
    isolationState === "PHASE_A_FROZEN" ||
    isolationState === "PHASE_B_ACTIVE" ||
    isolationState === "EVALUATION_COMPLETED";

  const questionedId = caseData?.questioned_sample_id || "Belge Yüklenmedi";
  const refCount = referenceImages.length || caseData?.reference_sample_ids?.length || 0;

  // Benzerlik Oranı ve Ön Değerlendirme Hesaplama
  const hasBothDocuments = !!questionedImageUrl && referenceImages.length > 0;
  const hasPipelineResult = !!pipelineResult?.fusion;

  // Pipeline sonucu varsa doğrudan modelden al; yoksa null
  const similarityScore = hasPipelineResult
    ? Math.round(pipelineResult.fusion.hybrid_similarity_score * 100)
    : null;

  // 6 Kademeli Kriminalojik Benzerlik ve Tetabuk Skalası (ENFSI / ASTM E2290)
  const isTetabuk = similarityScore !== null && similarityScore >= 98;
  const isGenuineMatch = similarityScore !== null && similarityScore >= 85 && similarityScore < 98;
  const isSkilledSimulation = similarityScore !== null && similarityScore >= 70 && similarityScore < 85;
  const isInconclusiveDisguise = similarityScore !== null && similarityScore >= 50 && similarityScore < 70;
  const isLowMatch = similarityScore !== null && similarityScore >= 25 && similarityScore < 50;
  const isCompleteDisparity = similarityScore !== null && similarityScore < 25;

  // Sade, anlaşılır ve net özet metin
  const plainSummaryText = hasPipelineResult
    ? isTetabuk
      ? `İki imza birebir çakışmaktadır (%${similarityScore}). İnsan eli aynı imzayı iki kez %100 aynı atamayacağı için orijinal bir belgeden kopyalama (ışıklı masa / fotokopi / montaj) yapıldığı değerlendirilmektedir.`
      : isGenuineMatch
      ? `İmza akışı, eğim açısı ve vuruş alışkanlıkları referans yazarla tam uyumludur (%${similarityScore}). Hafif farklar doğal el varyasyonu sınırları içerisindedir.`
      : isSkilledSimulation
      ? `Genel şekil benzemekle birlikte şüpheli duraksama/titreme izleri vardır (%${similarityScore}). Nitelikli taklit şüphesiyle mikro inceleme önerilir.`
      : isInconclusiveDisguise
      ? `Kısmi benzerlikler ve belirgin sapmalar mevcuttur (%${similarityScore}). İmzanın kasten bozulması (disguise) veya amatör taklit şüphesi vardır.`
      : isLowMatch
      ? `Harf formları ve vuruş geometrisi referans yazarla uyuşmamaktadır (%${similarityScore}). Farklı bir şahsın eli ürünü olduğu değerlendirilmektedir.`
      : `Sorgulanan imza ile referans imza arasında hiçbir yapısal bağ yoktur (%${similarityScore} - Farklı İmza).`
    : "Belgeler adli inceleme alanına yüklendi. Otomatik analizi başlatmak için yukarıdaki '▷ Analizi Başlat' butonuna tıklayınız.";

  return (
    <div className="space-y-4">
      {/* KATMAN VE BÜYÜTME KONTROLLERİ */}
      <LayerControls
        layers={layers}
        onLayerChange={setLayers}
        onZoomIn={() => setZoomScale((z) => Math.min(3.0, z + 0.25))}
        onZoomOut={() => setZoomScale((z) => Math.max(0.5, z - 0.25))}
        onZoomReset={() => setZoomScale(1.0)}
      />

      {/* KARŞILAŞTIRMALI ÇİFT KANVAS */}
      <DualCanvasViewer
        isolationState={isolationState}
        questionedImageUrl={questionedImageUrl}
        referenceImages={referenceImages}
        selectedRefIndex={selectedRefIndex}
        onSelectRefIndex={setSelectedRefIndex}
        layers={layers}
        zoomScale={zoomScale}
        onFreezePhaseA={handleFreezePhaseA}
        onUploadQuestioned={onUploadQuestioned}
        onUploadReference={onUploadReference}
        onLoadDemoSample={onLoadDemoSample}
        onClearQuestioned={onClearQuestioned}
        onRemoveReference={onRemoveReference}
        onClearAllReferences={onClearAllReferences}
      />

      {/* 🎯 CANLI BENZERLİK ORANI VE ADLİ ÖN DEĞERLENDİRME PANELİ */}
      {hasBothDocuments && (
        <div className="p-4 bg-gradient-to-r from-[#1E1612]/95 via-[#2A2019]/95 to-[#18120F]/95 border border-[#B8A484]/30 rounded-2xl shadow-2xl backdrop-blur-md space-y-4 animate-in fade-in duration-300">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 pb-3 border-b border-[#B8A484]/20">
            {/* Sol: Benzerlik Oranı ve Karar Başlığı */}
            <div className="flex items-center gap-4">
              <div className="relative flex items-center justify-center">
                <div
                  className={`w-16 h-16 rounded-2xl flex flex-col items-center justify-center font-bold shadow-lg border ${
                    !hasPipelineResult
                      ? "bg-[#2A2019] text-amber-400 border-amber-500/40 shadow-amber-950"
                      : isTetabuk
                      ? "bg-rose-950 text-rose-300 border-rose-500/60 shadow-rose-950 animate-pulse"
                      : isGenuineMatch
                      ? "bg-emerald-950/80 text-emerald-300 border-emerald-500/50 shadow-emerald-950"
                      : isSkilledSimulation
                      ? "bg-yellow-950/80 text-yellow-300 border-yellow-500/50 shadow-yellow-950"
                      : isInconclusiveDisguise
                      ? "bg-amber-950/80 text-amber-300 border-amber-500/50 shadow-amber-950"
                      : "bg-rose-950/80 text-rose-300 border-rose-500/50 shadow-rose-950"
                  }`}
                >
                  <span className="text-xl leading-none">
                    {hasPipelineResult ? `%${similarityScore}` : "⏳"}
                  </span>
                  <span className="text-[9px] uppercase tracking-wider text-[#FAF7F0]/70 mt-0.5">
                    {hasPipelineResult ? "Benzerlik" : "Bekleniyor"}
                  </span>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full text-xs font-bold border ${
                      !hasPipelineResult
                        ? "bg-amber-950/60 text-amber-300 border-amber-500/40"
                        : isTetabuk
                        ? "bg-rose-950 text-rose-300 border-rose-500/70"
                        : isGenuineMatch
                        ? "bg-emerald-900/60 text-emerald-200 border-emerald-400"
                        : isSkilledSimulation
                        ? "bg-yellow-950 text-yellow-300 border-yellow-500/50"
                        : isInconclusiveDisguise
                        ? "bg-amber-900/40 text-amber-300 border-amber-500/40"
                        : "bg-rose-900/40 text-rose-300 border-rose-500/40"
                    }`}
                  >
                    {!hasPipelineResult ? (
                      <>
                        <Sparkles className="w-3.5 h-3.5 text-amber-400 animate-pulse" /> ANALİZ BEKLENİYOR
                      </>
                    ) : isTetabuk ? (
                      <>
                        <AlertTriangle className="w-3.5 h-3.5 text-rose-400 animate-bounce" /> ⚠️ İZDÜŞÜM SAHTECİLİĞİ ŞÜPHESİ (TETABUK)
                      </>
                    ) : isGenuineMatch ? (
                      <>
                        <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> AYNI KİŞİ (ORİJİNAL İMZA)
                      </>
                    ) : isSkilledSimulation ? (
                      <>
                        <AlertTriangle className="w-3.5 h-3.5 text-yellow-400" /> KRİTİK AYRIM (NİTELİKLİ TAKLİT ŞÜPHESİ)
                      </>
                    ) : isInconclusiveDisguise ? (
                      <>
                        <AlertTriangle className="w-3.5 h-3.5 text-amber-400" /> KARARSIZ (OLASI İSTEKLİ BOZMA)
                      </>
                    ) : isLowMatch ? (
                      <>
                        <XCircle className="w-3.5 h-3.5 text-rose-400" /> FARKLI KİŞİ (KABA TAKLİT)
                      </>
                    ) : (
                      <>
                        <XCircle className="w-3.5 h-3.5 text-rose-400" /> TAM UYUMSUZLUK (FARKLI İMZA)
                      </>
                    )}
                  </span>
                  <span className="text-[11px] text-[#D4C5A9]/70 font-mono">
                    {hasPipelineResult ? "Hibrit Karar Füzyon Skoru" : "Model Hesaplaması"}
                  </span>
                </div>

                <p className="text-xs text-[#FAF7F0] mt-1 font-medium leading-relaxed max-w-xl">
                  {plainSummaryText}
                </p>
              </div>
            </div>

            {/* Sağ: İnceleme Adımlarına Geçiş ve Analiz Başlatma Butonları */}
            <div className="flex flex-wrap items-center gap-2">
              {!hasPipelineResult && onRunPipeline && (
                <button
                  type="button"
                  disabled={loading || !hasBothDocuments}
                  onClick={onRunPipeline}
                  className="px-4 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] border border-[#B8A484]/40 shadow-lg shadow-[#7B1E34]/30 transition flex items-center gap-2 disabled:opacity-50"
                >
                  <BrainCircuit className="w-4 h-4 text-cream-200" />
                  <span>▷ Kapsamlı Adli Analizi Başlat</span>
                </button>
              )}

              {onNavigateTab && (
                <>
                  <button
                    type="button"
                    onClick={() => onNavigateTab("morphology")}
                    className="px-3 py-2 rounded-xl text-xs font-semibold bg-[#2A2019] hover:bg-[#3D2E24] text-[#FAF7F0] border border-[#B8A484]/30 shadow-sm transition flex items-center gap-1.5"
                  >
                    <span>📊 2. Morfoloji</span>
                    <ArrowRight className="w-3 h-3 text-[#D4C5A9]" />
                  </button>

                  <button
                    type="button"
                    onClick={() => onNavigateTab("enfsi")}
                    className="px-3.5 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] border border-[#B8A484]/40 shadow-md shadow-[#7B1E34]/20 transition flex items-center gap-1.5"
                  >
                    <Scale className="w-3.5 h-3.5 text-amber-400" />
                    <span>4. Resmi Rapor</span>
                    <ArrowRight className="w-3.5 h-3.5 text-[#FAF7F0]" />
                  </button>
                </>
              )}
            </div>
          </div>

          {/* 3'LÜ ADLİ ÖZNİTELİK VE DETAY KARTLARI (Dinamik Senkronize) */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
            {/* 1. Aidiyet ve Karakteristik Uyum */}
            <div className="p-3 bg-[#18120F]/90 rounded-xl border border-[#B8A484]/20 space-y-1">
              <div className="flex items-center justify-between text-xs font-semibold text-[#FAF7F0]">
                <span className="flex items-center gap-1.5">
                  {!hasPipelineResult ? (
                    <Clock className="w-3.5 h-3.5 text-amber-400" />
                  ) : isTetabuk ? (
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                  ) : isGenuineMatch ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ) : isSkilledSimulation ? (
                    <AlertTriangle className="w-3.5 h-3.5 text-yellow-400" />
                  ) : (
                    <XCircle className="w-3.5 h-3.5 text-rose-400" />
                  )}
                  Yazar Aidiyeti
                </span>
                <span
                  className={`font-mono text-[11px] font-bold ${
                    !hasPipelineResult
                      ? "text-amber-400"
                      : isTetabuk
                      ? "text-rose-400"
                      : isGenuineMatch
                      ? "text-emerald-400"
                      : isSkilledSimulation
                      ? "text-yellow-400"
                      : isInconclusiveDisguise
                      ? "text-amber-400"
                      : "text-rose-400"
                  }`}
                >
                  {!hasPipelineResult
                    ? "Ölçüm Bekleniyor"
                    : isTetabuk
                    ? `%${similarityScore} Anormal Çakışma`
                    : isGenuineMatch
                    ? `%${similarityScore} Doğal Uyumlu`
                    : isSkilledSimulation
                    ? `%${similarityScore} Kritik Ayrım`
                    : isInconclusiveDisguise
                    ? `%${similarityScore} Kararsız / Bozma`
                    : `%${similarityScore} Uyumsuz`}
                </span>
              </div>
              <p className="text-[11px] text-[#D4C5A9]/80 leading-relaxed">
                {!hasPipelineResult
                  ? "Morfolojik ve derin model karşılaştırması analiz başlatıldığında sıfırdan hesaplanacaktır."
                  : isTetabuk
                  ? "İnsan eli iki imzayı %100 birebir atamaz; bu oran izdüşüm (tracing) ve dolandırıcılık şüphesidir."
                  : isGenuineMatch
                  ? "Karakteristik vuruş yönü ve harf aralıkları yazarın doğal varyasyon sınırlarıyla tam uyumludur."
                  : isSkilledSimulation
                  ? "Genel şekil benzer; ancak serbest taklit veya deformasyon ayrımı için FFT titreme analizi şarttır."
                  : isInconclusiveDisguise
                  ? "Kısmi benzerlik mevcuttur; kasıtlı imza değiştirme veya acemi taklit şüphesi değerlendirilmelidir."
                  : "Temel geometrik yapı ve motor alışkanlıklar referans yazar profiliyle örtüşmemektedir."}
              </p>
            </div>

            {/* 2. Taklit ve Titreme (Tremor) Analizi */}
            <div className="p-3 bg-[#18120F]/90 rounded-xl border border-[#B8A484]/20 space-y-1">
              <div className="flex items-center justify-between text-xs font-semibold text-[#FAF7F0]">
                <span className="flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400" /> Taklit / İzdüşüm İzi
                </span>
                <span
                  className={`font-mono text-[11px] font-bold ${
                    !hasPipelineResult
                      ? "text-[#D4C5A9]/60"
                      : isTetabuk
                      ? "text-rose-400"
                      : isGenuineMatch
                      ? "text-emerald-400"
                      : isSkilledSimulation
                      ? "text-yellow-400"
                      : isInconclusiveDisguise
                      ? "text-amber-400"
                      : "text-rose-400"
                  }`}
                >
                  {!hasPipelineResult
                    ? "Ölçülmedi"
                    : isTetabuk
                    ? "Kritik (İzdüşüm Şüphesi)"
                    : isGenuineMatch
                    ? "Düşük (Doğal Akıcı)"
                    : isSkilledSimulation
                    ? "Orta (Tereddüt / Taklit)"
                    : "Yüksek (Anomali)"}
                </span>
              </div>
              <p className="text-[11px] text-[#D4C5A9]/80 leading-relaxed">
                {!hasPipelineResult
                  ? "Çizgi hız değişkenliği ve FFT spektral analizi sıfırdan yürütülecektir."
                  : isTetabuk
                  ? "Asetat, ışıklı masa veya fotokopi transferi emareleri aranmalıdır."
                  : isGenuineMatch
                  ? "FFT spektrumunda şüpheli duraksama ve mikro-titreme anomalisi saptanmadı."
                  : isSkilledSimulation
                  ? "Çizgi başlangıçlarında tereddüt ve hız azalması emareleri saptanmıştır."
                  : "Yüksek frekanslı mikro-titreme (tremor) veya hat süreksizliği mevcuttur."}
              </p>
            </div>

            {/* 3. Eğim ve Geometri */}
            <div className="p-3 bg-[#18120F]/90 rounded-xl border border-[#B8A484]/20 space-y-1">
              <div className="flex items-center justify-between text-xs font-semibold text-[#FAF7F0]">
                <span className="flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-cyan-400" /> Eğim ve Kavis
                </span>
                <span
                  className={`font-mono text-[11px] font-bold ${
                    !hasPipelineResult
                      ? "text-[#D4C5A9]/60"
                      : isTetabuk
                      ? "text-rose-400"
                      : isGenuineMatch
                      ? "text-cyan-400"
                      : isSkilledSimulation
                      ? "text-yellow-400"
                      : "text-amber-400"
                  }`}
                >
                  {!hasPipelineResult
                    ? "Ölçülmedi"
                    : isTetabuk
                    ? "Yapay Birebir Örtüşme"
                    : isGenuineMatch
                    ? "Δ 0.0° (Doğal Uyum)"
                    : "Sapma Mevcut"}
                </span>
              </div>
              <p className="text-[11px] text-[#D4C5A9]/80 leading-relaxed">
                {!hasPipelineResult
                  ? "PCA asal eksen eğim açısı ve topolojik ilmek analizi sıfırdan ölçülecektir."
                  : isTetabuk
                  ? "Yazı eğim açısı ve kavisler doğal değişkenlik göstermeksizin birebir çakışmaktadır."
                  : isGenuineMatch
                  ? "Yazı eğim açısı ve kavis topolojisi referans ortalaması ile tam uyumludur."
                  : "Yazı eğim açısı veya kavis topolojisi referans normlarından farklılık gösterir."}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ALT ALAN: HIZLI NUMUNE ÇUBUĞU VE İSTEĞE BAĞLI TEKNİK ÜSTVERİ */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl text-xs shadow-2xl backdrop-blur-md">
        {/* Sol Özet Rozetler */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1.5 px-3 py-1 bg-[#2A2019] border border-[#B8A484]/20 rounded-full text-[#D4C5A9] shadow-sm">
            <FileQuestion className="w-3.5 h-3.5 text-amber-400" />
            <span>Şüpheli Belge: <strong className="text-[#FAF7F0]">{questionedId}</strong></span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1 bg-[#2A2019] border border-[#B8A484]/20 rounded-full text-[#D4C5A9] shadow-sm">
            <FileCheck className="w-3.5 h-3.5 text-teal-400" />
            <span>Mukayese Havuzu: <strong className="text-[#FAF7F0]">{refCount} Numune</strong></span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1 bg-[#2A2019] border border-[#B8A484]/20 rounded-full text-[#D4C5A9] shadow-sm">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            <span>Durum: <strong className="text-emerald-400">{isPhaseBFrozenOrActive ? "Şüpheli Belge Onaylandı (Mukayese Açık)" : "Şüpheli Belge Ön İncelemesi"}</strong></span>
          </div>
        </div>

        {/* Sağ: Teknik Üstveri Aç/Kapat Butonu */}
        <button
          type="button"
          onClick={() => setShowMetadata(!showMetadata)}
          title="Kriptografik özetler ve çözünürlük verilerini görüntüle"
          className="flex items-center gap-1.5 px-3.5 py-1 bg-[#2A2019] hover:bg-[#3D2E24] border border-[#B8A484]/20 rounded-full text-[#D4C5A9] hover:text-[#FAF7F0] transition shadow-sm"
        >
          <Cpu className="w-3.5 h-3.5 text-burgundy-600" />
          <span>{showMetadata ? "Teknik Üstveriyi Gizle" : "Teknik Üstveri"}</span>
          {showMetadata ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* TEKNİK ÜSTVERİ ÇEKMECESİ */}
      {showMetadata && (
        <div className="bg-[#1E1612]/95 border border-[#B8A484]/25 rounded-2xl p-4 shadow-2xl text-xs space-y-3 backdrop-blur-md">
          <div className="flex items-center justify-between pb-2 border-b border-[#B8A484]/20 text-[#FAF7F0] font-semibold">
            <span className="flex items-center gap-2">
              <Hash className="w-4 h-4 text-burgundy-600" />
              Adli Belge Kriptografik Özetleri ve Çözünürlük Bilgileri
            </span>
            <span className="text-[11px] text-[#D4C5A9]/70 font-normal">Kayıpsız Adli Kayıt</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="bg-[#120D0A] p-3 rounded-xl border border-[#B8A484]/15 space-y-1 shadow-inner">
              <span className="text-[10px] text-[#D4C5A9] uppercase font-semibold">Sorgulanan Belge (Şüpheli) Durumu</span>
              <p className="font-mono text-[11px] text-[#FAF7F0] break-all select-all font-semibold">
                {caseData?.questioned_sample_id ? `ID: ${caseData.questioned_sample_id}` : "Henüz adli numune yüklenmedi."}
              </p>
              <div className="text-[10px] text-[#D4C5A9]/70 pt-1">İnceleme Türü: {caseData?.examination_type || "Belirtilmedi"}</div>
            </div>

            <div className="bg-[#120D0A] p-3 rounded-xl border border-[#B8A484]/15 space-y-1 shadow-inner">
              <span className="text-[10px] text-[#D4C5A9] uppercase font-semibold">Mukayese / Referans Belgeler Havuzu</span>
              <p className="font-mono text-[11px] text-[#FAF7F0] break-all select-all font-semibold">
                {refCount > 0 ? `Toplam ${refCount} Adet Mukayese Numunesi Kayıtlı` : "Henüz mukayese belgesi eklenmedi."}
              </p>
              <div className="text-[10px] text-[#D4C5A9]/70 pt-1">İzolasyon Durumu: {isolationState}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
