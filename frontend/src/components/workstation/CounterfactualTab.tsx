"use client";

import React, { useState, useEffect } from "react";
import {
  Sliders,
  Sparkles,
  RefreshCw,
  TrendingUp,
  ShieldCheck,
  AlertCircle,
  ArrowRight,
  Zap,
  HelpCircle,
  Layers,
  Eye,
} from "lucide-react";
import { api, formatApiError, CounterfactualResponse, CounterfactualScenario } from "@/services/api";

interface CounterfactualTabProps {
  caseId: string;
  hasSamples?: boolean;
  hasPipelineResult?: boolean;
  onNavigateMicroscopy?: () => void;
}

export const CounterfactualTab: React.FC<CounterfactualTabProps> = ({
  caseId,
  hasSamples = false,
  hasPipelineResult = false,
  onNavigateMicroscopy,
}) => {
  const [data, setData] = useState<CounterfactualResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [simulating, setSimulating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Kaydırıcı Durumları (Perturbations)
  const [sliders, setSliders] = useState<{
    slant_angle_deg: number;
    tremor_index_psd: number;
    smoothness_energy: number;
    loop_count: number;
    baseline_deviation_px: number;
  }>({
    slant_angle_deg: 0,
    tremor_index_psd: 0,
    smoothness_energy: 0,
    loop_count: 0,
    baseline_deviation_px: 0,
  });

  const loadInitialData = async () => {
    if (!hasSamples || !hasPipelineResult) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.simulateCounterfactual({
        case_id: caseId,
        custom_perturbations: {},
      });
      setData(res);
    } catch (err: any) {
      setError(formatApiError(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (hasSamples && hasPipelineResult) {
      loadInitialData();
    } else {
      setData(null);
      setLoading(false);
    }
  }, [caseId, hasSamples, hasPipelineResult]);

  const handleSliderChange = async (key: keyof typeof sliders, value: number) => {
    const updatedSliders = { ...sliders, [key]: value };
    setSliders(updatedSliders);

    // REST Simülasyonunu anlık çalıştır
    setSimulating(true);
    try {
      const res = await api.simulateCounterfactual({
        case_id: caseId,
        custom_perturbations: updatedSliders,
      });
      setData(res);
    } catch (err) {
      // Sessiz hata
    } finally {
      setSimulating(false);
    }
  };

  const applyScenario = (scenario: CounterfactualScenario) => {
    const newSliders = { ...sliders };
    scenario.required_perturbations.forEach((p) => {
      const k = p.feature_key as keyof typeof sliders;
      if (k in newSliders) {
        newSliders[k] = p.simulated_value - p.current_value;
      }
    });
    setSliders(newSliders);

    // API simüle et
    api
      .simulateCounterfactual({
        case_id: caseId,
        custom_perturbations: newSliders,
      })
      .then((res) => setData(res))
      .catch(() => {});
  };

  const resetSliders = () => {
    const resetValues = {
      slant_angle_deg: 0,
      tremor_index_psd: 0,
      smoothness_energy: 0,
      loop_count: 0,
      baseline_deviation_px: 0,
    };
    setSliders(resetValues);
    api
      .simulateCounterfactual({
        case_id: caseId,
        custom_perturbations: resetValues,
      })
      .then((res) => setData(res))
      .catch(() => {});
  };

  if (!hasSamples || !hasPipelineResult) {
    return (
      <div className="bg-[#1E1612]/95 border border-[#B8A484]/30 rounded-3xl p-12 shadow-2xl backdrop-blur-md text-center space-y-5 max-w-2xl mx-auto my-8">
        <div className="w-20 h-20 rounded-3xl bg-[#2A2019] border border-[#B8A484]/30 flex items-center justify-center text-[#D4C5A9] mx-auto shadow-inner">
          <Sliders className="w-10 h-10 text-amber-400" />
        </div>
        <div className="space-y-2">
          <h3 className="text-xl font-extrabold text-[#FAF7F0]">Karşıtlı Simülasyon İçin Analiz Sonucu Bekleniyor</h3>
          <p className="text-sm text-[#D4C5A9]/90 max-w-md mx-auto leading-relaxed">
            &quot;Ne Olurdu?&quot; duyarlılık analizi, şüpheli ve mukayese numunelerinin morfolojik öznitelik farkları ve derin metrik füzyon skorları üzerinden hesaplanır. Dosyada görsel numune yüklü değilken veya analiz çalıştırılmamışken varsayımsal çıktı üretilmez.
          </p>
        </div>
        <div className="p-4 bg-[#120D0A] border border-[#B8A484]/25 rounded-2xl text-xs font-semibold text-[#FAF7F0] max-w-md mx-auto">
          Simülasyon yapabilmek için lütfen 1. sekmeden numuneleri yükleyip <strong>&quot;Kapsamlı Adli Analizi Başlat&quot;</strong> adımını tamamlayınız.
        </div>
        {onNavigateMicroscopy && (
          <button
            type="button"
            onClick={onNavigateMicroscopy}
            className="px-5 py-2.5 bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] font-semibold text-xs rounded-xl shadow-lg transition inline-flex items-center gap-2 border border-[#B8A484]/30"
          >
            <Eye className="w-4 h-4" />
            1. Görsel Karşılaştırma Sekmesine Git
          </button>
        )}
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center p-16 space-y-4 text-center">
        <RefreshCw className="w-8 h-8 text-teal-400 animate-spin" />
        <p className="text-sm text-[#D4C5A9]">Karşıtlı karar sınırları ve duyarlılık matrisi hesaplanıyor...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8 bg-amber-950/20 border border-amber-500/30 rounded-2xl text-center space-y-3">
        <AlertCircle className="w-8 h-8 text-amber-400 mx-auto" />
        <p className="text-sm text-amber-200">{error || "Karşıtlı simülasyon verisi bulunamadı."}</p>
        <button
          onClick={loadInitialData}
          className="px-4 py-2 bg-amber-800/40 hover:bg-amber-800/60 rounded-xl text-xs font-semibold text-white transition"
        >
          Yeniden Dene
        </button>
      </div>
    );
  }

  const originalPct = Math.round(data.original_score * 100);
  const simulatedPct = Math.round(data.simulated_hybrid_score * 100);
  const deltaPct = simulatedPct - originalPct;

  return (
    <div className="space-y-6">
      {/* Üst Başlık & Açıklama */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-[#1E1612]/90 border border-[#B8A484]/25 p-5 rounded-2xl shadow-xl backdrop-blur-md">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            <h2 className="text-base font-bold text-[#FAF7F0]">Karşıtlı Açıklamalar & "Ne Olurdu?" Simülatörü</h2>
            <span className="text-[11px] bg-amber-500/20 border border-amber-500/30 text-amber-300 px-2 py-0.5 rounded-full font-mono">
              Counterfactual XAI & Sensitivity Engine
            </span>
          </div>
          <p className="text-xs text-[#D4C5A9]/80">
            Adli öznitelikleri interaktif olarak değiştirerek model karar sınırını ve Adli Gating mekanizmasını test edin.
          </p>
        </div>

        <button
          onClick={resetSliders}
          className="flex items-center gap-2 px-3.5 py-2 bg-[#2A2019] hover:bg-[#382B22] border border-[#B8A484]/30 rounded-xl text-xs text-[#FAF7F0] font-medium transition"
        >
          <RefreshCw className="w-3.5 h-3.5 text-amber-400" />
          Değerleri Sıfırla
        </button>
      </div>

      {/* Simülasyon Skor Değişim Paneli */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Orijinal Skor */}
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 p-5 rounded-2xl shadow-lg flex flex-col justify-between">
          <span className="text-xs text-[#D4C5A9]/80 font-medium">Mevcut Analiz Skoru</span>
          <div className="my-3">
            <span className="text-3xl font-extrabold text-[#FAF7F0] font-mono">%{originalPct}</span>
          </div>
          <span className="text-[11px] text-[#D4C5A9]/60">Gating Durumu: {data.gating_triggered ? "Devrede (Ceza Var)" : "Ceza Yok"}</span>
        </div>

        {/* Simüle Edilmiş Skor */}
        <div className="bg-[#1E1612]/90 border border-amber-500/40 p-5 rounded-2xl shadow-xl flex flex-col justify-between relative overflow-hidden">
          {simulating && (
            <div className="absolute top-2 right-2 flex items-center gap-1 text-[10px] text-amber-400 font-mono">
              <RefreshCw className="w-3 h-3 animate-spin" />
              Hesaplanıyor...
            </div>
          )}
          <span className="text-xs text-amber-300 font-semibold flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5" />
            Simüle Edilen Hibrit Skor
          </span>
          <div className="my-3 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-amber-400 font-mono">%{simulatedPct}</span>
            <span
              className={`text-xs font-bold font-mono px-2 py-0.5 rounded-full ${
                deltaPct >= 0 ? "bg-emerald-500/20 text-emerald-300" : "bg-red-500/20 text-red-300"
              }`}
            >
              {deltaPct >= 0 ? `+${deltaPct}%` : `${deltaPct}%`}
            </span>
          </div>
          <span className="text-[11px] text-[#D4C5A9]/80">
            Hesaplama Süresi: <span className="font-mono text-amber-300">{data.execution_time_ms} ms</span>
          </span>
        </div>

        {/* Uyuşma & Karar Seviyesi */}
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 p-5 rounded-2xl shadow-lg flex flex-col justify-between">
          <span className="text-xs text-[#D4C5A9]/80 font-medium">Öngörülen Karar Sınıfı</span>
          <div className="my-3">
            <span className="text-base font-bold text-[#FAF7F0]">{data.concordance_level}</span>
          </div>
          <span className="text-[11px] text-[#D4C5A9]/70">
            Adli Gating Kesintisi: -%{(data.gating_penalty * 100).toFixed(1)}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* İnteraktif Öznitelik Kaydırıcıları (Sliders) */}
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 p-5 rounded-2xl shadow-xl space-y-5">
          <div className="flex items-center justify-between border-b border-[#B8A484]/15 pb-3">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-amber-400" />
              <h3 className="text-xs font-bold text-[#FAF7F0]">Öznitelik Müdahale Kontrolleri</h3>
            </div>
            <span className="text-[11px] text-[#D4C5A9]/60">Anlık Simülasyon</span>
          </div>

          {/* 1. Yazı Eğim Açısı */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-[#FAF7F0] font-medium">Yazı Eğim Açısı Sapması (Δθ)</span>
              <span className="font-mono text-amber-400">{sliders.slant_angle_deg > 0 ? `+${sliders.slant_angle_deg}°` : `${sliders.slant_angle_deg}°`}</span>
            </div>
            <input
              type="range"
              min="-15"
              max="15"
              step="1"
              value={sliders.slant_angle_deg}
              onChange={(e) => handleSliderChange("slant_angle_deg", parseFloat(e.target.value))}
              className="w-full h-1.5 bg-[#120D0A] rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>

          {/* 2. Titreme İndeksi */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-[#FAF7F0] font-medium">FFT Titreme İndeksi (Δ Tremor)</span>
              <span className="font-mono text-amber-400">{sliders.tremor_index_psd.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="-0.10"
              max="0.10"
              step="0.01"
              value={sliders.tremor_index_psd}
              onChange={(e) => handleSliderChange("tremor_index_psd", parseFloat(e.target.value))}
              className="w-full h-1.5 bg-[#120D0A] rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>

          {/* 3. Kinematik Akıcılık / Enerji */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-[#FAF7F0] font-medium">Kinematik Pürüzsüzlük Enerjisi (Δ Smoothness)</span>
              <span className="font-mono text-amber-400">{sliders.smoothness_energy.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="-0.15"
              max="0.15"
              step="0.01"
              value={sliders.smoothness_energy}
              onChange={(e) => handleSliderChange("smoothness_energy", parseFloat(e.target.value))}
              className="w-full h-1.5 bg-[#120D0A] rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>

          {/* 4. Döngü Sayısı */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-[#FAF7F0] font-medium">Halka / İlik Sayısı Farkı (Δ Loops)</span>
              <span className="font-mono text-amber-400">{sliders.loop_count > 0 ? `+${sliders.loop_count}` : sliders.loop_count}</span>
            </div>
            <input
              type="range"
              min="-5"
              max="5"
              step="1"
              value={sliders.loop_count}
              onChange={(e) => handleSliderChange("loop_count", parseInt(e.target.value))}
              className="w-full h-1.5 bg-[#120D0A] rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>
        </div>

        {/* Önerilen Asgari Karşıtlı Senaryolar (Actionable Scenarios) */}
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 p-5 rounded-2xl shadow-xl space-y-4">
          <div className="flex items-center justify-between border-b border-[#B8A484]/15 pb-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-teal-400" />
              <h3 className="text-xs font-bold text-[#FAF7F0]">Otomatik Üretilen Karşıtlı Senaryolar</h3>
            </div>
            <span className="text-[11px] text-[#D4C5A9]/60">Minimal Pertürbasyonlar</span>
          </div>

          <div className="space-y-3">
            {data.top_scenarios && data.top_scenarios.length > 0 ? (
              data.top_scenarios.map((sc, idx) => (
                <div
                  key={idx}
                  className="p-4 bg-[#2A2019]/60 border border-[#B8A484]/20 hover:border-teal-500/40 rounded-xl space-y-2 transition"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-[#FAF7F0]">{sc.title}</span>
                    <span className="text-[10px] bg-teal-500/20 text-teal-300 px-2 py-0.5 rounded-full font-mono">
                      +{(sc.score_delta * 100).toFixed(0)}% Artış
                    </span>
                  </div>

                  <p className="text-[11px] text-[#D4C5A9]">
                    Hedef: <span className="text-[#FAF7F0] font-medium">{sc.target_goal}</span>
                  </p>

                  <div className="space-y-1 pt-1">
                    {sc.required_perturbations.slice(0, 2).map((p, pIdx) => (
                      <p key={pIdx} className="text-[11px] text-[#EFE7DA]/90 flex items-start gap-1.5">
                        <ArrowRight className="w-3 h-3 text-teal-400 shrink-0 mt-0.5" />
                        <span>{p.explanation_tr}</span>
                      </p>
                    ))}
                  </div>

                  <div className="pt-2 flex justify-end">
                    <button
                      onClick={() => applyScenario(sc)}
                      className="px-3 py-1 bg-teal-600/30 hover:bg-teal-600/50 border border-teal-500/40 rounded-lg text-[11px] text-teal-200 font-semibold transition"
                    >
                      Bu Senaryoyu Simüle Et ➔
                    </button>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-xs text-[#D4C5A9]/60 py-4 text-center italic">
                Mevcut analizde belirgin bir karar çelişkisi bulunmamaktadır.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
