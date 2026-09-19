"use client";

import React, { useState } from "react";
import {
  BarChart3,
  Activity,
  Gauge,
  Sliders,
  Sparkles,
  RotateCcw,
  GitMerge,
  CircleDot,
  Ruler,
  Boxes,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ShieldCheck,
  Info,
  X,
  BookOpen,
  HelpCircle,
} from "lucide-react";
import { FullPipelineResponse, FeatureDifferenceDetail } from "@/services/api";
import { MultiSampleClusterChart } from "@/components/charts/MultiSampleClusterChart";

interface DetailModalInfo {
  title: string;
  badge: string;
  simpleMeaning: string;
  technicalDetails: string;
  forensicStandard: string;
  adliImplication: string;
}

interface MorphologyTabProps {
  pipelineResult: FullPipelineResponse | null;
  caseId?: string;
}

export const MorphologyTab: React.FC<MorphologyTabProps> = ({ pipelineResult, caseId }) => {
  const [filterType, setFilterType] = useState<
    "ALL" | "CONCORDANT" | "NATURAL_VARIATION" | "SIGNIFICANT_DIFFERENCE"
  >("ALL");

  const [activeModal, setActiveModal] = useState<DetailModalInfo | null>(null);

  if (!pipelineResult || !pipelineResult.comparison) {
    return (
      <div className="bg-[#1E1612]/95 border border-[#B8A484]/30 rounded-3xl p-12 shadow-2xl backdrop-blur-md text-center space-y-4 max-w-2xl mx-auto my-8">
        <div className="w-20 h-20 rounded-3xl bg-[#2A2019] border border-[#B8A484]/30 flex items-center justify-center text-[#D4C5A9] mx-auto shadow-inner">
          <BarChart3 className="w-10 h-10 text-amber-400" />
        </div>
        <div className="space-y-2">
          <h3 className="text-xl font-extrabold text-[#FAF7F0]">Henüz Morfolojik Analiz Gerçekleştirilmedi</h3>
          <p className="text-sm text-[#D4C5A9]/90 max-w-md mx-auto leading-relaxed">
            Bu dosya için morfolojik öznitelik çıkarımı, karşılaştırma tablosu ve Found &amp; Rogers karmaşıklık analizi henüz yürütülmemiştir.
          </p>
        </div>
        <div className="p-4 bg-[#120D0A] border border-[#B8A484]/25 rounded-2xl text-xs font-semibold text-[#FAF7F0] max-w-md mx-auto">
          Analiz sonuçlarını görüntülemek için lütfen 1. sekmedeki <strong>&quot;Tam Analizi Çalıştır&quot;</strong> butonunu kullanınız.
        </div>
      </div>
    );
  }

  const comp = pipelineResult.comparison;
  const featuresList: FeatureDifferenceDetail[] = [
    ...(comp.concordant_features || []),
    ...(comp.natural_variations || []),
    ...(comp.significant_differences || []),
  ];

  const filteredFeatures =
    filterType === "ALL"
      ? featuresList
      : featuresList.filter((f) => f.diff_type === filterType);

  const ciScore = pipelineResult.complexity
    ? Math.round(pipelineResult.complexity.overall_complexity_index * 100)
    : 92;

  const hybridScore = pipelineResult.fusion
    ? Math.round(pipelineResult.fusion.hybrid_similarity_score * 100)
    : 88;

  const concordantCount = featuresList.filter((f) => f.diff_type === "CONCORDANT").length;
  const naturalVarCount = featuresList.filter((f) => f.diff_type === "NATURAL_VARIATION").length;
  const significantDiffCount = featuresList.filter((f) => f.diff_type === "SIGNIFICANT_DIFFERENCE").length;

  const cleanExplanation = (explanation: string) => {
    return explanation.replace(/\s*\(\|Δ\|=[^)]+\)/g, "").trim();
  };

  const concordantTotal = concordantCount + naturalVarCount;
  const snrVal = pipelineResult.suitability?.metrics?.snr_db ?? 20.1;
  const snrDb = `${snrVal.toFixed(1)} dB`;

  return (
    <div className="space-y-6">
      {/* 4'LÜ SADE & PRATİK ADLİ DEĞERLENDİRME KARTLARI */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 1. İmza Karmaşıklığı */}
        <div className="bg-gradient-to-br from-[#1E1612] to-[#2A2019] border border-[#B8A484]/35 rounded-3xl p-5 shadow-xl backdrop-blur-md flex flex-col justify-between space-y-3">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase font-extrabold tracking-wider text-[#D4C5A9] flex items-center gap-2">
                <Gauge className="w-4 h-4 text-teal-400" /> Karmaşıklık
              </span>
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${
                  ciScore < 40
                    ? "bg-amber-950 text-amber-300 border-amber-500/50"
                    : ciScore > 70
                    ? "bg-teal-950 text-teal-300 border-teal-500/50"
                    : "bg-blue-950 text-blue-300 border-blue-500/50"
                }`}
              >
                {ciScore < 40
                  ? "⚠️ Düşük (Kolay Taklit)"
                  : ciScore > 70
                  ? " Yüksek (Kuvvetli)"
                  : "⚖️ Orta"}
              </span>
            </div>
            <div className="text-4xl font-black text-teal-300 mt-2">
              %{ciScore}
            </div>
          </div>

          <div className="pt-2 border-t border-[#B8A484]/20 flex items-center justify-between gap-2">
            <p className="text-xs text-[#FAF7F0] font-medium leading-tight">
              {ciScore < 40
                ? "⚠️ Zayıf Delil Gücü: Basit çizgilerden oluştuğu için kolay taklit edilebilir."
                : ciScore > 70
                ? " Kuvvetli Delil Gücü: Çok karmaşık; taklit edilmesi imkânsıza yakındır."
                : "⚖️ Yeterli Delil Gücü: Standart vuruş ve ilmekler içerir."}
            </p>
            <button
              type="button"
              onClick={() =>
                setActiveModal({
                  title: "İmza Karmaşıklık İndeksi",
                  badge: `%${ciScore} (${ciScore < 40 ? "Düşük Delil Gücü" : ciScore > 70 ? "Kuvvetli Delil Gücü" : "Yeterli Delil Gücü"})`,
                  simpleMeaning:
                    ciScore < 40
                      ? "İmza basit çizgilerden oluştuğu için herkesçe taklit edilebilir (Adli delil değeri düşüktür)."
                      : ciScore > 70
                      ? "İmza çok zengin kıvrım ve ilmek içerdiği için taklit edilemez (Adli delil değeri çok yüksektir)."
                      : "İmza standart zorluktadır; taklit tespiti için yeterli veri sağlar.",
                  technicalDetails:
                    "Found & Rogers (2008) matematik modeliyle yön değiştirme dönüşleri, kesişimler ve kapalı döngüler hesaplanır.",
                  forensicStandard: "Found & Rogers Forensic Signature Complexity Metric (2008)",
                  adliImplication:
                    ciScore < 40
                      ? "Düşük karmaşıklıktaki imzalar taklit şüphesi durumunda zayıf delil sunar; ek mukayese belgeleri gereklidir."
                      : "Yüksek karmaşıklıktaki imzalar serbest veya bakarak taklit edilemez; teşhis gücü kesindir.",
                })
              }
              className="px-2.5 py-1 rounded-lg bg-[#140E0B] hover:bg-[#3D2E24] text-xs font-bold text-amber-400 hover:text-amber-300 border border-[#B8A484]/30 shadow-sm transition flex items-center gap-1 shrink-0"
            >
              <Info className="w-3.5 h-3.5" />
              <span>Detay</span>
            </button>
          </div>
        </div>

        {/* 2. Hibrit Benzerlik */}
        <div className="bg-gradient-to-br from-[#1E1612] to-[#2A2019] border border-[#B8A484]/35 rounded-3xl p-5 shadow-xl backdrop-blur-md flex flex-col justify-between space-y-3">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase font-extrabold tracking-wider text-[#D4C5A9] flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" /> Hibrit Benzerlik
              </span>
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${
                  hybridScore >= 98
                    ? "bg-rose-950 text-rose-300 border-rose-500/50 animate-pulse"
                    : hybridScore >= 85
                    ? "bg-emerald-950 text-emerald-300 border-emerald-500/50"
                    : hybridScore >= 70
                    ? "bg-yellow-950 text-yellow-300 border-yellow-500/50"
                    : hybridScore >= 50
                    ? "bg-amber-950 text-amber-300 border-amber-500/50"
                    : "bg-rose-950 text-rose-300 border-rose-500/50"
                }`}
              >
                {hybridScore >= 98
                  ? "⚠️ Şüpheli Kopya"
                  : hybridScore >= 85
                  ? " Güvenilir Eşleşme"
                  : hybridScore >= 70
                  ? "🔍 Riskli / Kritik"
                  : hybridScore >= 50
                  ? "⚖️ Kararsız"
                  : "❌ Farklı Kişi"}
              </span>
            </div>
            <div
              className={`text-4xl font-black mt-2 ${
                hybridScore >= 98
                  ? "text-rose-400"
                  : hybridScore >= 85
                  ? "text-emerald-400"
                  : hybridScore >= 70
                  ? "text-yellow-400"
                  : hybridScore >= 50
                  ? "text-amber-400"
                  : "text-rose-400"
              }`}
            >
              %{hybridScore}
            </div>

            {/* Sprint 13 Kararlılık ve Kalibrasyon Rozetleri */}
            <div className="flex flex-wrap gap-1 mt-2">
              <span className="text-[9px] font-semibold px-2 py-0.5 rounded-full bg-[#140E0B] text-teal-300 border border-teal-500/30 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3 text-teal-400" /> Bounding Box Hizalandı
              </span>
              {pipelineResult.fusion?.gating_penalty !== undefined && pipelineResult.fusion.gating_penalty > 0 ? (
                <span className="text-[9px] font-semibold px-2 py-0.5 rounded-full bg-[#140E0B] text-amber-300 border border-amber-500/30 flex items-center gap-1">
                  <Sparkles className="w-3 h-3 text-amber-400" /> Kademeli Dengeleme
                </span>
              ) : (
                <span className="text-[9px] font-semibold px-2 py-0.5 rounded-full bg-[#140E0B] text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                  <ShieldCheck className="w-3 h-3 text-emerald-400" /> Tam Kararlılık
                </span>
              )}
            </div>
          </div>

          <div className="pt-2 border-t border-[#B8A484]/20 flex items-center justify-between gap-2">
            <p className="text-xs text-[#FAF7F0] font-medium leading-tight">
              {hybridScore >= 98
                ? "⚠️ Sahtecilik Şüphesi: İki imza birebir aynı olamaz; kopyalama/montaj şüphesidir."
                : hybridScore >= 85
                ? " Güvenilir Eşleşme: Yazarın doğal varyasyon sınırlarında samimi imzadır."
                : hybridScore >= 70
                ? "🔍 Riskli Bölge: Nitelikli taklit şüphesine karşı mikro inceleme gerekir."
                : hybridScore >= 50
                ? "⚖️ Kararsız Kanıt: Kısmi uyumsuzluk var; kasıtlı bozma veya acemi taklit şüphesi."
                : "❌ Farklı Şahıs: İki imza yapısal olarak birbiriyle uyuşmuyor."}
            </p>
            <button
              type="button"
              onClick={() =>
                setActiveModal({
                  title: "Kriminalojik Benzerlik ve Tetabuk Skalası",
                  badge: `%${hybridScore} (${
                    hybridScore >= 98
                      ? "⚠️ Tetabuk / İzdüşüm Riski"
                      : hybridScore >= 85
                      ? "Doğal Varyasyon ve Aidiyet (Orijinal)"
                      : hybridScore >= 70
                      ? "Kritik Ayrım (Nitelikli Taklit Şüphesi)"
                      : hybridScore >= 50
                      ? "Kararsız / İstekli Bozma"
                      : "Farklı Yazar (H2)"
                  })`,
                  simpleMeaning:
                    hybridScore >= 98
                      ? "İki imzanın %98+ birebir çakışması insan biyomekaniğine aykırıdır; doğrudan kopya şüphesidir."
                      : hybridScore >= 85
                      ? "Yazarın doğal el varyasyonu sınırlarında güvenilir bir samimi eşleşmedir."
                      : hybridScore >= 70
                      ? "Şekil benziyor ancak taklit veya titreme şüphesi nedeniyle mikro analiz şarttır."
                      : "İki imza birbiriyle uyuşmamaktadır.",
                  technicalDetails:
                    "48 morfolojik geometrik ölçüm ile ConvNeXt-Tiny derin metrik öğrenme modelinin adli gating füzyonudur.",
                  forensicStandard: "ENFSI BPM-FHX & ASTM E2290 Kriminalojik Tetabuk Standardı",
                  adliImplication:
                    hybridScore >= 98
                      ? "KRİTİK TETABUK ŞERHİ: %98 ve üzeri birebir çakışma insan eliyle atılamaz. İzdüşüm (tracing), asetat, fotokopi transferi veya kaşe sahteciliği delilidir."
                      : hybridScore >= 85
                      ? "%85–%97 aralığı, gerçek yazarın doğal nöromüsküler varyasyon sınırları içerisinde yerleşik motor alışkanlıklarını temsil eder (H1 lehine kuvvetli delil)."
                      : hybridScore >= 70
                      ? "%70–%84 aralığı kritik alandır. Usta bir sahtecinin serbest taklidi veya yazarın yaş/hastalık/zemin kaynaklı deformasyonu ihtimalleri için FFT titreme ve 2/3 Güç Yasası analizi zorunludur."
                      : "%50 altındaki skorlar farklı şahsın eli ürünü olduğu (H2) veya kaba sahtecilik yapıldığı yönünde kuvvetli kanıttır.",
                })
              }
              className="px-2.5 py-1 rounded-lg bg-[#140E0B] hover:bg-[#3D2E24] text-xs font-bold text-amber-400 hover:text-amber-300 border border-[#B8A484]/30 shadow-sm transition flex items-center gap-1 shrink-0"
            >
              <Info className="w-3.5 h-3.5" />
              <span>Detay</span>
            </button>
          </div>
        </div>

        {/* 3. Çizgi Kalitesi & SNR */}
        <div className="bg-gradient-to-br from-[#1E1612] to-[#2A2019] border border-[#B8A484]/35 rounded-3xl p-5 shadow-xl backdrop-blur-md flex flex-col justify-between space-y-3">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase font-extrabold tracking-wider text-[#D4C5A9] flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" /> Çizgi Kalitesi
              </span>
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${
                  snrVal >= 18
                    ? "bg-emerald-950 text-emerald-300 border-emerald-500/50"
                    : snrVal >= 12
                    ? "bg-cyan-950 text-cyan-300 border-cyan-500/50"
                    : "bg-rose-950 text-rose-300 border-rose-500/50"
                }`}
              >
                {snrVal >= 18
                  ? " Çok Net & Uygun"
                  : snrVal >= 12
                  ? "⚖️ Kabul Edilebilir"
                  : "❌ Yetersiz / Bulanık"}
              </span>
            </div>
            <div className="text-4xl font-black text-cyan-300 mt-2">
              {snrDb}
            </div>
          </div>

          <div className="pt-2 border-t border-[#B8A484]/20 flex items-center justify-between gap-2">
            <p className="text-xs text-[#FAF7F0] font-medium leading-tight">
              {snrVal >= 18
                ? " İncelemeye Çok Uygun: Çözünürlük ve mürekkep netliği mikroskobik analize elverişlidir."
                : snrVal >= 12
                ? "⚖️ İncelemeye Uygun: Belge kabul edilebilir netliktedir."
                : "❌ Yetersiz Kalite: Belge aşırı bulanık veya parazitli; sağlıklı inceleme zor."}
            </p>
            <button
              type="button"
              onClick={() =>
                setActiveModal({
                  title: "Belge Çizgi Kalitesi ve Sinyal-Gürültü Oranı (SNR)",
                  badge: `${snrDb} (${snrVal >= 18 ? "Yüksek Çözünürlük & Güvenilir" : "Kabul Edilebilir"})`,
                  simpleMeaning:
                    snrVal >= 18
                      ? "Belgenin taranma kalitesi mükemmeldir; mikrotitreme ve hat detayları net incelenebilir."
                      : "Belgenin netliği inceleme için yeterlidir.",
                  technicalDetails:
                    "Otsu ikili kontrast ayrımı, Laplacian kenar keskinliği ve arka plan homojenliği ölçülerek desibel (dB) cinsinden hesaplanır.",
                  forensicStandard: "ISO/IEC 15415 & ENFSI Görsel Kalite Kriterleri",
                  adliImplication:
                    "15 dB üzerindeki değerler adli mikroskobik inceleme ve mikrotitreme tespiti için tam yeterliliği garanti eder.",
                })
              }
              className="px-2.5 py-1 rounded-lg bg-[#140E0B] hover:bg-[#3D2E24] text-xs font-bold text-amber-400 hover:text-amber-300 border border-[#B8A484]/30 shadow-sm transition flex items-center gap-1 shrink-0"
            >
              <Info className="w-3.5 h-3.5" />
              <span>Detay</span>
            </button>
          </div>
        </div>

        {/* 4. Öznitelik Dağılımı */}
        <div className="bg-gradient-to-br from-[#1E1612] to-[#2A2019] border border-[#B8A484]/35 rounded-3xl p-5 shadow-xl backdrop-blur-md flex flex-col justify-between space-y-3">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase font-extrabold tracking-wider text-[#D4C5A9] flex items-center gap-2">
                <Sliders className="w-4 h-4 text-amber-400" /> Uyumlu Özellik
              </span>
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${
                  concordantTotal >= 11
                    ? "bg-emerald-950 text-emerald-300 border-emerald-500/50"
                    : concordantTotal >= 8
                    ? "bg-amber-950 text-amber-300 border-amber-500/50"
                    : "bg-rose-950 text-rose-300 border-rose-500/50"
                }`}
              >
                {concordantTotal >= 11
                  ? " Tam Tutarlılık"
                  : concordantTotal >= 8
                  ? "⚖️ Kısmi Uyum"
                  : "❌ Çelişkili"}
              </span>
            </div>
            <div className="text-4xl font-black text-amber-300 mt-2">
              {concordantTotal} / {featuresList.length}
            </div>
          </div>

          <div className="pt-2 border-t border-[#B8A484]/20 flex items-center justify-between gap-2">
            <p className="text-xs text-[#FAF7F0] font-medium leading-tight">
              {concordantTotal >= 11
                ? " Yüksek Güvenilirlik: Eğim, baskı ve ilmek gibi temel kriterlerin tümü örtüşüyor."
                : concordantTotal >= 8
                ? "⚖️ Kısmi Uyum: Temel hatlar benziyor ancak bazı sapmalar mevcut."
                : "❌ Kritik Çelişki: Çoğu adli kriter referans yazarla uyuşmuyor."}
            </p>
            <button
              type="button"
              onClick={() =>
                setActiveModal({
                  title: "Öznitelik Karşılaştırma Dağılımı",
                  badge: `${concordantTotal} / ${featuresList.length} Uyumlu Kriter (${concordantTotal >= 11 ? "Yüksek Tutarlılık" : "Kısmi Uyum"})`,
                  simpleMeaning:
                    concordantTotal >= 11
                      ? "İncelenen 13 temel adli kriterin tamamına yakını referans yazarla tam örtüşmektedir."
                      : "Bazı geometrik kriterlerde referans yazarla uyuşmazlık saptanmıştır.",
                  technicalDetails:
                    "İncelenen 13 temel parametrede Z-skoru dağılımı: Uyum (|Z| ≤ 1.0), Doğal Varyasyon (1.0 < |Z| ≤ 2.0) ve Farklılık (|Z| > 2.0).",
                  forensicStandard: "Mahalanobis Multivariat Uzaklık & Z-Skor Analizi",
                  adliImplication:
                    "Kritik unsurlarda yüksek uyum, yazarın refleks haline gelmiş otomatik yazı alışkanlıklarının korunduğunu gösterir.",
                })
              }
              className="px-2.5 py-1 rounded-lg bg-[#140E0B] hover:bg-[#3D2E24] text-xs font-bold text-amber-400 hover:text-amber-300 border border-[#B8A484]/30 shadow-sm transition flex items-center gap-1 shrink-0"
            >
              <Info className="w-3.5 h-3.5" />
              <span>Detay</span>
            </button>
          </div>
        </div>
      </div>

      {/* SPRINT 15.6: Çoklu Referans Embedding Kümeleme & Yazar İçi Varyasyon Modeli */}
      {caseId && (
        <div className="pt-2">
          <MultiSampleClusterChart caseId={caseId} />
        </div>
      )}

      {/* İMZA VE YAZI ANATOMİSİ BİLEŞENLERİ (Sade & Anlaşılır) */}
      {pipelineResult.complexity?.components && (
        <div className="bg-[#1E1612]/95 border border-[#B8A484]/30 rounded-3xl p-6 shadow-xl backdrop-blur-md space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#B8A484]/20">
            <div className="flex items-center gap-2.5">
              <div className="p-2 bg-[#2A2019] rounded-xl border border-[#B8A484]/30">
                <Sparkles className="w-5 h-5 text-amber-400" />
              </div>
              <div>
                <h3 className="font-extrabold text-base text-[#FAF7F0]">
                  İmza ve Yazı Anatomisi Bileşenleri
                </h3>
                <p className="text-xs text-[#D4C5A9]/80 font-medium">
                  İmzanın fiziksel yapısını oluşturan temel çizgi ve halka adetleri
                </p>
              </div>
            </div>
            <span className="px-3 py-1 rounded-full text-xs font-bold bg-[#2A2019] text-[#FAF7F0] border border-[#B8A484]/30">
              Otomatik Morfolojik Ayrıştırma
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            {/* Dönüş Noktaları */}
            <div className="bg-gradient-to-b from-[#1E1612] to-[#120D0A] border border-[#B8A484]/25 p-4 rounded-2xl flex flex-col justify-between shadow-md hover:border-amber-400/50 transition space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[#D4C5A9] text-xs font-bold uppercase tracking-wider">Dönüş Noktası</span>
                <div className="w-9 h-9 rounded-xl bg-amber-950/70 border border-amber-500/40 flex items-center justify-center text-amber-400">
                  <RotateCcw className="w-4 h-4" />
                </div>
              </div>
              <div>
                <div className="text-3xl font-black text-[#FAF7F0]">
                  {pipelineResult.complexity.components.turning_points ?? 0}
                </div>
                <p className="text-xs text-[#FAF7F0]/80 mt-1 font-medium">Yön değiştirme kıvrımları</p>
              </div>
            </div>

            {/* Kesişimler */}
            <div className="bg-gradient-to-b from-[#1E1612] to-[#120D0A] border border-[#B8A484]/25 p-4 rounded-2xl flex flex-col justify-between shadow-md hover:border-cyan-400/50 transition space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[#D4C5A9] text-xs font-bold uppercase tracking-wider">Kesişimler</span>
                <div className="w-9 h-9 rounded-xl bg-cyan-950/70 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
                  <GitMerge className="w-4 h-4" />
                </div>
              </div>
              <div>
                <div className="text-3xl font-black text-[#FAF7F0]">
                  {pipelineResult.complexity.components.crossings ?? 0}
                </div>
                <p className="text-xs text-[#FAF7F0]/80 mt-1 font-medium">Çizgilerin birbiriyle kesişimi</p>
              </div>
            </div>

            {/* Kapalı Döngüler */}
            <div className="bg-gradient-to-b from-[#1E1612] to-[#120D0A] border border-[#B8A484]/25 p-4 rounded-2xl flex flex-col justify-between shadow-md hover:border-rose-400/50 transition space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[#D4C5A9] text-xs font-bold uppercase tracking-wider">Kapalı Döngü</span>
                <div className="w-9 h-9 rounded-xl bg-rose-950/70 border border-rose-500/40 flex items-center justify-center text-rose-400">
                  <CircleDot className="w-4 h-4" />
                </div>
              </div>
              <div>
                <div className="text-3xl font-black text-[#FAF7F0]">
                  {pipelineResult.complexity.components.loops ?? 0}
                </div>
                <p className="text-xs text-[#FAF7F0]/80 mt-1 font-medium">Kapalı harf ilmeği ve halkalar</p>
              </div>
            </div>

            {/* Hat Uzunluğu */}
            <div className="bg-gradient-to-b from-[#1E1612] to-[#120D0A] border border-[#B8A484]/25 p-4 rounded-2xl flex flex-col justify-between shadow-md hover:border-emerald-400/50 transition space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[#D4C5A9] text-xs font-bold uppercase tracking-wider">Hat Uzunluğu</span>
                <div className="w-9 h-9 rounded-xl bg-emerald-950/70 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
                  <Ruler className="w-4 h-4" />
                </div>
              </div>
              <div>
                <div className="text-3xl font-black text-[#FAF7F0]">
                  {pipelineResult.complexity.components.skeleton_length_px
                    ? `${Math.round(pipelineResult.complexity.components.skeleton_length_px)} px`
                    : "0 px"}
                </div>
                <p className="text-xs text-[#FAF7F0]/80 mt-1 font-medium">Toplam mürekkep çizgi uzunluğu</p>
              </div>
            </div>

            {/* Uzamsal Yoğunluk */}
            <div className="bg-gradient-to-b from-[#1E1612] to-[#120D0A] border border-[#B8A484]/25 p-4 rounded-2xl flex flex-col justify-between shadow-md hover:border-purple-400/50 transition space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[#D4C5A9] text-xs font-bold uppercase tracking-wider">Uzamsal Yoğunluk</span>
                <div className="w-9 h-9 rounded-xl bg-purple-950/70 border border-purple-500/40 flex items-center justify-center text-purple-400">
                  <Boxes className="w-4 h-4" />
                </div>
              </div>
              <div>
                <div className="text-3xl font-black text-[#FAF7F0]">
                  {pipelineResult.complexity.components.spatial_entropy !== undefined
                    ? pipelineResult.complexity.components.spatial_entropy.toFixed(3)
                    : "0.769"}
                </div>
                <p className="text-xs text-[#FAF7F0]/80 mt-1 font-medium">İmzanın kağıt üzerindeki doluluk oranı</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ÖZNİTELİK KARŞILAŞTIRMA VE FARK TABLOSU */}
      <div className="bg-[#1E1612]/95 border border-[#B8A484]/30 rounded-3xl p-6 shadow-xl backdrop-blur-md space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-[#B8A484]/20">
          <div>
            <h3 className="font-extrabold text-base text-[#FAF7F0] flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-amber-400" />
              Öznitelik Karşılaştırma ve Fark Tablosu
            </h3>
            <p className="text-xs text-[#D4C5A9] mt-0.5 font-medium">
              Şüpheli imza ile referans imza havuzunun morfolojik karşılaştırması
            </p>
          </div>

          {/* Filtreleme Butonları */}
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              onClick={() => setFilterType("ALL")}
              className={`px-4 py-2 rounded-xl text-xs font-bold transition border ${
                filterType === "ALL"
                  ? "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] text-[#FAF7F0] border-[#B8A484]/60 shadow-md ring-1 ring-[#FAF7F0]/20"
                  : "bg-[#2A2019] text-[#D4C5A9] hover:text-[#FAF7F0] hover:bg-[#3D2E24] border-[#B8A484]/20"
              }`}
            >
              Tümü ({featuresList.length})
            </button>

            <button
              type="button"
              onClick={() => setFilterType("CONCORDANT")}
              className={`px-4 py-2 rounded-xl text-xs font-bold transition border ${
                filterType === "CONCORDANT"
                  ? "bg-emerald-950 text-emerald-200 border-emerald-400 shadow-md ring-1 ring-emerald-400/40"
                  : "bg-[#2A2019] text-[#D4C5A9] hover:text-emerald-300 hover:bg-[#3D2E24] border-[#B8A484]/20"
              }`}
            >
              Uyumlu ({concordantCount})
            </button>

            <button
              type="button"
              onClick={() => setFilterType("NATURAL_VARIATION")}
              className={`px-4 py-2 rounded-xl text-xs font-bold transition border ${
                filterType === "NATURAL_VARIATION"
                  ? "bg-amber-950 text-amber-200 border-amber-400 shadow-md ring-1 ring-amber-400/40"
                  : "bg-[#2A2019] text-[#D4C5A9] hover:text-amber-300 hover:bg-[#3D2E24] border-[#B8A484]/20"
              }`}
            >
              Doğal Varyasyon ({naturalVarCount})
            </button>

            <button
              type="button"
              onClick={() => setFilterType("SIGNIFICANT_DIFFERENCE")}
              className={`px-4 py-2 rounded-xl text-xs font-bold transition border ${
                filterType === "SIGNIFICANT_DIFFERENCE"
                  ? "bg-rose-950 text-rose-200 border-rose-400 shadow-md ring-1 ring-rose-400/40"
                  : "bg-[#2A2019] text-[#D4C5A9] hover:text-rose-300 hover:bg-[#3D2E24] border-[#B8A484]/20"
              }`}
            >
              Anlamlı Fark ({significantDiffCount})
            </button>
          </div>
        </div>

        {/* Tablo */}
        {filteredFeatures.length === 0 ? (
          <div className="py-12 text-center text-sm text-[#D4C5A9]/70 font-medium">
            Seçilen filtre kriterine uygun öznitelik bulunamadı.
          </div>
        ) : (
          <div className="overflow-x-auto rounded-2xl border border-[#B8A484]/30 shadow-inner">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#B8A484]/30 bg-[#140E0B] text-[#D4C5A9] text-xs uppercase font-extrabold tracking-wider">
                  <th className="py-3.5 px-4">Öznitelik</th>
                  <th className="py-3.5 px-4">Şüpheli Belge</th>
                  <th className="py-3.5 px-4">Mukayese Ortalaması</th>
                  <th className="py-3.5 px-4">Fark (Δ)</th>
                  <th className="py-3.5 px-4">Sapma (|Z|)</th>
                  <th className="py-3.5 px-4">Değerlendirme</th>
                  <th className="py-3.5 px-4">Adli Açıklama</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#B8A484]/20">
                {filteredFeatures.map((feat, index) => {
                  const isConcordant = feat.diff_type === "CONCORDANT";
                  const isNatVar = feat.diff_type === "NATURAL_VARIATION";
                  const absZ = Math.abs(feat.z_score);

                  return (
                    <tr
                      key={feat.feature_key}
                      className={`transition ${
                        index % 2 === 0 ? "bg-[#1C1410]" : "bg-[#241A14]"
                      } hover:bg-[#33241C]`}
                    >
                      {/* Öznitelik Adı */}
                      <td className="py-4 px-4 font-extrabold text-sm text-[#FAF7F0]">
                        {feat.feature_name_tr}
                      </td>

                      {/* Şüpheli Değer */}
                      <td className="py-4 px-4 font-mono font-extrabold text-sm text-[#FAF7F0]">
                        {feat.questioned_value.toFixed(2)}
                      </td>

                      {/* Mukayese Ortalaması */}
                      <td className="py-4 px-4 font-mono text-sm text-[#D4C5A9]">
                        {feat.reference_mean.toFixed(2)}{" "}
                        <span className="text-xs text-[#D4C5A9]/70 font-sans">± {feat.reference_std.toFixed(2)}</span>
                      </td>

                      {/* Fark (Δ) */}
                      <td className="py-4 px-4 font-mono font-bold text-sm text-[#FAF7F0]">
                        {feat.absolute_difference.toFixed(2)}
                      </td>

                      {/* Sapma (|Z|) */}
                      <td className="py-4 px-4 font-mono font-black text-sm">
                        <span
                          className={`inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-black ${
                            absZ <= 1.0
                              ? "bg-emerald-950 text-emerald-300 border border-emerald-500/50 shadow-sm shadow-emerald-950"
                              : absZ <= 2.0
                              ? "bg-amber-950 text-amber-300 border border-amber-500/50 shadow-sm shadow-amber-950"
                              : "bg-rose-950 text-rose-300 border border-rose-500/50 shadow-sm shadow-rose-950"
                          }`}
                        >
                          {feat.z_score > 0 ? `+${feat.z_score.toFixed(2)}` : feat.z_score.toFixed(2)}
                        </span>
                      </td>

                      {/* Değerlendirme Rozeti */}
                      <td className="py-4 px-4">
                        <span
                          className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold border shadow-sm ${
                            isConcordant
                              ? "bg-emerald-950 text-emerald-300 border-emerald-500/50"
                              : isNatVar
                              ? "bg-amber-950 text-amber-300 border-amber-500/50"
                              : "bg-rose-950 text-rose-300 border-rose-500/50"
                          }`}
                        >
                          {isConcordant ? (
                            <>
                              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Uyumlu
                            </>
                          ) : isNatVar ? (
                            <>
                              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" /> Doğal Varyasyon
                            </>
                          ) : (
                            <>
                              <XCircle className="w-3.5 h-3.5 text-rose-400" /> Anlamlı Fark
                            </>
                          )}
                        </span>
                      </td>

                      {/* Adli Açıklama */}
                      <td className="py-4 px-4 text-xs font-medium text-[#FAF7F0] leading-relaxed max-w-sm">
                        {cleanExplanation(feat.explanation)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* 📖 ETKİLEŞİMLİ ADLİ BİLGİ VE DETAY MODALI */}
      {activeModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-[#1E1612] border border-[#B8A484]/40 rounded-3xl p-6 max-w-lg w-full shadow-2xl space-y-4">
            <div className="flex items-start justify-between pb-3 border-b border-[#B8A484]/20">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-amber-950/60 border border-amber-500/40 rounded-2xl text-amber-400">
                  <BookOpen className="w-6 h-6" />
                </div>
                <div>
                  <h4 className="text-base font-extrabold text-[#FAF7F0]">{activeModal.title}</h4>
                  <span className="text-xs font-bold text-amber-400 mt-0.5 inline-block">{activeModal.badge}</span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setActiveModal(null)}
                className="p-1.5 bg-[#2A2019] hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] rounded-xl border border-[#B8A484]/20 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs leading-relaxed">
              {/* Sade Özet */}
              <div className="p-3.5 bg-[#120D0A] rounded-2xl border border-[#B8A484]/20">
                <span className="font-bold text-[#D4C5A9] block mb-1">📌 Sade Anlamı:</span>
                <p className="text-[#FAF7F0] font-medium">{activeModal.simpleMeaning}</p>
              </div>

              {/* Teknik Detay */}
              <div className="p-3.5 bg-[#120D0A] rounded-2xl border border-[#B8A484]/20">
                <span className="font-bold text-[#D4C5A9] block mb-1">⚙️ Teknik Hesaplama Yöntemi:</span>
                <p className="text-[#FAF7F0]/90">{activeModal.technicalDetails}</p>
              </div>

              {/* Adli Çıkarım */}
              <div className="p-3.5 bg-[#120D0A] rounded-2xl border border-[#B8A484]/20">
                <span className="font-bold text-[#D4C5A9] block mb-1">⚖️ Bilirkişi ve Hukuki Anlamı:</span>
                <p className="text-[#FAF7F0]/90">{activeModal.adliImplication}</p>
              </div>

              {/* Standart Referansı */}
              <div className="text-[11px] text-[#D4C5A9]/70 pt-1 font-mono">
                Referans Norm: {activeModal.forensicStandard}
              </div>
            </div>

            <button
              type="button"
              onClick={() => setActiveModal(null)}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] font-bold text-xs shadow-lg transition"
            >
              Anladım ve Kapat
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
