"use client";

import React, { useState, useEffect } from "react";
import {
  Activity,
  Gauge,
  Zap,
  TrendingUp,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
  Clock,
  Layers,
  Sparkles,
  Eye,
  ArrowRight,
  ShieldAlert,
  PenTool,
} from "lucide-react";
import { api, BiometricCompareResponse } from "@/services/api";
import { LiveSignaturePad } from "./LiveSignaturePad";

interface BiometricTabProps {
  caseId: string;
  hasSamples?: boolean;
  onNavigateMicroscopy?: () => void;
}

export const BiometricTab: React.FC<BiometricTabProps> = ({
  caseId,
  hasSamples = false,
  onNavigateMicroscopy,
}) => {
  const [subTab, setSubTab] = useState<"pad" | "dtw">("pad");
  const [data, setData] = useState<BiometricCompareResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [activeChart, setActiveChart] = useState<"velocity" | "pressure" | "acceleration">("velocity");

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getBiometricDemo();
      setData(res);
    } catch (err: any) {
      setError(err?.message || "Biyometrik veri yüklenirken hata oluştu.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Numune görseli yoksa asla rastgele/otomatik veri çekme
    if (!hasSamples) {
      setData(null);
      setLoading(false);
    }
  }, [caseId, hasSamples]);

  // Basit SVG grafiği için örnek dalga formu üretimi
  const renderWaveformSvg = (type: "velocity" | "pressure" | "acceleration") => {
    const pointsCount = 40;
    const width = 600;
    const height = 180;

    let pathQ = `M 0,${height / 2}`;
    let pathRef = `M 0,${height / 2}`;

    for (let i = 0; i <= pointsCount; i++) {
      const x = (i / pointsCount) * width;
      const u = i / pointsCount;

      let yQ = height / 2;
      let yRef = height / 2;

      if (type === "velocity") {
        yQ = height - (Math.sin(u * Math.PI * 4) * 50 + Math.sin(u * 12) * 20 + 70);
        yRef = height - (Math.sin(u * Math.PI * 4) * 60 + 80);
      } else if (type === "pressure") {
        yQ = height - (Math.sin(u * Math.PI) * 110 + 30);
        yRef = height - (Math.sin(u * Math.PI) * 90 + Math.cos(u * 8) * 15 + 40);
      } else {
        yQ = height - (Math.cos(u * Math.PI * 6) * 45 + 75);
        yRef = height - (Math.cos(u * Math.PI * 6) * 55 + 75);
      }

      pathQ += ` L ${x.toFixed(1)},${Math.max(10, Math.min(height - 10, yQ)).toFixed(1)}`;
      pathRef += ` L ${x.toFixed(1)},${Math.max(10, Math.min(height - 10, yRef)).toFixed(1)}`;
    }

    return (
      <svg className="w-full h-44 overflow-visible" viewBox={`0 0 ${width} ${height}`}>
        {/* Izgara çizgileri */}
        <line x1="0" y1="30" x2={width} y2="30" stroke="#B8A484" strokeOpacity="0.1" strokeDasharray="4 4" />
        <line x1="0" y1="90" x2={width} y2="90" stroke="#B8A484" strokeOpacity="0.15" strokeDasharray="4 4" />
        <line x1="0" y1="150" x2={width} y2="150" stroke="#B8A484" strokeOpacity="0.1" strokeDasharray="4 4" />

        {/* Referans Yörünge (Yeşil/Mavi) */}
        <path d={pathRef} fill="none" stroke="#2DD4BF" strokeWidth="2.5" strokeLinecap="round" opacity="0.8" />
        {/* Sorgulanan Yörünge (Amber/Kırmızı) */}
        <path d={pathQ} fill="none" stroke="#F59E0B" strokeWidth="2.5" strokeLinecap="round" strokeDasharray="6 3" />
      </svg>
    );
  };

  const renderDtwView = () => {
    if (!hasSamples) {
      return (
        <div className="bg-[#1E1612]/95 border border-[#B8A484]/30 rounded-3xl p-12 shadow-2xl backdrop-blur-md text-center space-y-5 max-w-2xl mx-auto my-4">
          <div className="w-20 h-20 rounded-3xl bg-[#2A2019] border border-[#B8A484]/30 flex items-center justify-center text-[#D4C5A9] mx-auto shadow-inner">
            <Activity className="w-10 h-10 text-amber-400" />
          </div>
          <div className="space-y-2">
            <h3 className="text-xl font-extrabold text-[#FAF7F0]">Numune Görseli Bulunamadı</h3>
            <p className="text-sm text-[#D4C5A9]/90 max-w-md mx-auto leading-relaxed">
              Adli inceleme standartları gereğince; dosyada kayıtlı şüpheli veya mukayese görseli bulunmadığında rastgele veya varsayımsal DTW karşılaştırması üretilmez.
            </p>
          </div>
          <div className="p-4 bg-[#120D0A] border border-[#B8A484]/25 rounded-2xl text-xs font-semibold text-[#FAF7F0] max-w-md mx-auto">
            DTW karşılaştırması için lütfen önce 1. sekmeden numuneleri yükleyiniz veya yukarıdaki <strong>&quot;Canlı İmza Pedi&quot;</strong> sekmesini kullanarak stylus/tablet ile doğrudan imza atınız.
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
          <p className="text-sm text-[#D4C5A9]">Dinamik biyometrik zaman serisi ve DTW korelasyonu hesaplanıyor...</p>
        </div>
      );
    }

    if (error || !data) {
      return (
        <div className="bg-[#1E1612]/95 border border-[#B8A484]/30 rounded-3xl p-12 shadow-2xl backdrop-blur-md text-center space-y-5 max-w-2xl mx-auto my-4">
          <div className="w-20 h-20 rounded-3xl bg-[#2A2019] border border-[#B8A484]/30 flex items-center justify-center text-[#D4C5A9] mx-auto shadow-inner">
            <Activity className="w-10 h-10 text-teal-400" />
          </div>
          <div className="space-y-2">
            <h3 className="text-xl font-extrabold text-[#FAF7F0]">Biyometrik Kinematik Analizi</h3>
            <p className="text-sm text-[#D4C5A9]/90 max-w-md mx-auto leading-relaxed">
              Numune görselleri hazır. Zaman serisi (hız, ivme, basınç türevleri) ve Dynamic Time Warping (DTW) adli mesafe korelasyonunu başlatmak için aşağıdaki butona tıklayınız.
            </p>
          </div>
          {error && (
            <p className="text-xs text-rose-400 bg-rose-950/40 p-2 rounded-lg max-w-md mx-auto">{error}</p>
          )}
          <button
            type="button"
            onClick={loadData}
            className="px-5 py-2.5 bg-gradient-to-r from-teal-800 to-teal-900 hover:from-teal-700 hover:to-teal-800 text-[#FAF7F0] font-semibold text-xs rounded-xl shadow-lg transition inline-flex items-center gap-2 border border-teal-500/40 cursor-pointer"
          >
            <Activity className="w-4 h-4 text-teal-300" />
            Biyometrik DTW Analizini Çalıştır
          </button>
        </div>
      );
    }

    const dtwScore = Math.round(data.dtw_normalized_similarity * 100);

    return (
      <div className="space-y-6">
        {/* Üst Bilgi Başlığı */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-[#1E1612]/90 border border-[#B8A484]/25 p-5 rounded-2xl shadow-xl backdrop-blur-md">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-teal-400" />
              <h2 className="text-base font-bold text-[#FAF7F0]">Dinamik (Biyometrik) İmza Zaman Serisi Analizi</h2>
              <span className="text-[11px] bg-teal-500/20 border border-teal-500/30 text-teal-300 px-2 py-0.5 rounded-full font-mono">
                Wacom / Tablet Kinematik Modülü
              </span>
            </div>
            <p className="text-xs text-[#D4C5A9]/80">
              Zaman serisi (x, y, t, basınç) türevleri üzerinden hız dalgalanması, ivme ve Dynamic Time Warping (DTW) mesafesi.
            </p>
          </div>

          <button
            type="button"
            onClick={loadData}
            className="flex items-center gap-2 px-3.5 py-2 bg-[#2A2019] hover:bg-[#382B22] border border-[#B8A484]/30 rounded-xl text-xs text-[#FAF7F0] font-medium transition cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5 text-teal-400" />
            Yeniden Hesapla
          </button>
        </div>

        {/* DTW Benzerlik Göstergesi ve Özet Kartları */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 p-5 rounded-2xl shadow-lg flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs text-[#D4C5A9]/80 font-medium">DTW Benzerliği</span>
              <Gauge className="w-4 h-4 text-teal-400" />
            </div>
            <div className="my-3 flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-[#FAF7F0] font-mono">%{dtwScore}</span>
              <span className="text-[11px] text-[#D4C5A9]/60 font-mono">Mesafe: {data.dtw_distance.toFixed(2)}</span>
            </div>
            <div className="w-full bg-[#120D0A] h-2 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  dtwScore >= 70 ? "bg-emerald-500" : dtwScore >= 45 ? "bg-amber-500" : "bg-red-500"
                }`}
                style={{ width: `${dtwScore}%` }}
              />
            </div>
          </div>

          <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 p-5 rounded-2xl shadow-lg flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs text-[#D4C5A9]/80 font-medium">Hız Profili Uyumu</span>
              <TrendingUp className="w-4 h-4 text-amber-400" />
            </div>
            <div className="my-3 flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-[#FAF7F0] font-mono">
                %{(data.velocity_correlation * 100).toFixed(1)}
              </span>
            </div>
            <span className="text-[11px] text-[#D4C5A9]/60 font-mono">
              Ort. Hız: {data.questioned_profile?.mean_velocity?.toFixed(0) || "-"} px/s
            </span>
          </div>

          <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 p-5 rounded-2xl shadow-lg flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs text-[#D4C5A9]/80 font-medium">Basınç Korelasyonu</span>
              <Layers className="w-4 h-4 text-teal-400" />
            </div>
            <div className="my-3 flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-[#FAF7F0] font-mono">
                %{(data.pressure_correlation * 100).toFixed(1)}
              </span>
            </div>
            <span className="text-[11px] text-[#D4C5A9]/60 font-mono">
              Baskı: {data.questioned_profile?.mean_pressure?.toFixed(2) || "-"}
            </span>
          </div>

          <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 p-5 rounded-2xl shadow-lg flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs text-[#D4C5A9]/80 font-medium">Kinematik Akıcılık</span>
              <Zap className="w-4 h-4 text-purple-400" />
            </div>
            <div className="my-3 flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-[#FAF7F0] font-mono">
                %{((data.questioned_profile?.kinematic_smoothness || 0.8) * 100).toFixed(1)}
              </span>
            </div>
            <span className="text-[11px] text-[#D4C5A9]/60 font-mono">
              Havada Süre: %{(data.in_air_time_diff * 100).toFixed(1)}
            </span>
          </div>
        </div>

        {/* Dalga Formu Grafiği */}
        <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 p-6 rounded-2xl shadow-xl backdrop-blur-md space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#B8A484]/15 pb-4">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-teal-400" />
              <h3 className="text-xs font-bold text-[#FAF7F0]">
                Zaman Boyunca Dinamik Sinyal Eğrileri (t = 0.0s → 1.2s)
              </h3>
            </div>

            <div className="flex items-center gap-1.5 bg-[#120D0A] p-1 rounded-xl border border-[#B8A484]/20">
              <button
                type="button"
                onClick={() => setActiveChart("velocity")}
                className={`px-3 py-1 rounded-lg text-[11px] font-semibold transition ${
                  activeChart === "velocity"
                    ? "bg-[#2A2019] text-teal-300 shadow-sm border border-teal-500/30"
                    : "text-[#D4C5A9]/70 hover:text-[#FAF7F0]"
                }`}
              >
                Hız Eğrisi v(t)
              </button>
              <button
                type="button"
                onClick={() => setActiveChart("pressure")}
                className={`px-3 py-1 rounded-lg text-[11px] font-semibold transition ${
                  activeChart === "pressure"
                    ? "bg-[#2A2019] text-teal-300 shadow-sm border border-teal-500/30"
                    : "text-[#D4C5A9]/70 hover:text-[#FAF7F0]"
                }`}
              >
                Basınç Profili P(t)
              </button>
              <button
                type="button"
                onClick={() => setActiveChart("acceleration")}
                className={`px-3 py-1 rounded-lg text-[11px] font-semibold transition ${
                  activeChart === "acceleration"
                    ? "bg-[#2A2019] text-teal-300 shadow-sm border border-teal-500/30"
                    : "text-[#D4C5A9]/70 hover:text-[#FAF7F0]"
                }`}
              >
                İvme a(t)
              </button>
            </div>
          </div>

          <div className="py-2">{renderWaveformSvg(activeChart)}</div>

          <div className="flex items-center justify-center gap-6 text-xs text-[#D4C5A9]/80 pt-2 border-t border-[#B8A484]/10">
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-[#2DD4BF] rounded-full" />
              <span>Referans İmza Profili</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-[#F59E0B] rounded-full" />
              <span>Sorgulanan İmza Profili</span>
            </div>
          </div>
        </div>

        {/* Adli Uzman Sözel Yorumu */}
        <div className="bg-[#120D0A] border border-[#B8A484]/25 p-4 rounded-2xl flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-xs font-bold text-[#FAF7F0]">Biyometrik Zaman Serisi Kanaati</h4>
            <p className="text-xs text-[#D4C5A9]/80 leading-relaxed">
              {data.verbal_assessment_tr ||
                "Dinamik kinematik parametreler ve DTW mesafesi analizi adli ön inceleme aşamasındadır."}
            </p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Üst Alt-Sekme Seçici: Canlı İmza Pedi vs DTW Karşılaştırması */}
      <div className="flex items-center gap-2 bg-[#1E1612]/90 border border-[#B8A484]/25 p-1.5 rounded-2xl max-w-md shadow-lg">
        <button
          type="button"
          onClick={() => setSubTab("pad")}
          className={`flex-1 py-2 px-3 rounded-xl text-xs font-bold transition flex items-center justify-center gap-2 cursor-pointer ${
            subTab === "pad"
              ? "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] text-[#FAF7F0] shadow-md border border-[#B8A484]/30 ring-1 ring-[#D4C5A9]/20"
              : "text-[#D4C5A9] hover:text-[#FAF7F0] hover:bg-[#2A2019]"
          }`}
        >
          <PenTool className="w-3.5 h-3.5 text-amber-300" />
          <span>Canlı İmza Pedi (Stylus)</span>
        </button>

        <button
          type="button"
          onClick={() => setSubTab("dtw")}
          className={`flex-1 py-2 px-3 rounded-xl text-xs font-bold transition flex items-center justify-center gap-2 cursor-pointer ${
            subTab === "dtw"
              ? "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] text-[#FAF7F0] shadow-md border border-[#B8A484]/30 ring-1 ring-[#D4C5A9]/20"
              : "text-[#D4C5A9] hover:text-[#FAF7F0] hover:bg-[#2A2019]"
          }`}
        >
          <Activity className="w-3.5 h-3.5 text-teal-400" />
          <span>DTW Karşılaştırma</span>
        </button>
      </div>

      {subTab === "pad" ? <LiveSignaturePad /> : renderDtwView()}
    </div>
  );
};
