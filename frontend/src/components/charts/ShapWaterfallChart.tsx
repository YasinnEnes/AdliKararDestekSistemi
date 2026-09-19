"use client";

import React, { useState } from "react";
import {
  TrendingUp,
  TrendingDown,
  Sparkles,
  Activity,
  Compass,
  Repeat,
  Layers,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Brain,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { TabularFeatureAttribution } from "@/services/api";

interface ShapWaterfallChartProps {
  positiveFeatures?: TabularFeatureAttribution[];
  negativeFeatures?: TabularFeatureAttribution[];
  summaryText?: string;
}

// Öznitelik anahtarlarına göre ikon ve sade anlam eşleştirmesi
const FEATURE_META: Record<
  string,
  { icon: any; simpleMeaning: string; tip: string }
> = {
  tremor_index_psd: {
    icon: Activity,
    simpleMeaning: "Çizgideki istemsiz el titremesi ve motor akıcılık düzeyi.",
    tip: "Doğal imzada vuruşlar akıcıdır; taklitte duraksama ve titreme artar.",
  },
  smoothness_energy: {
    icon: Zap,
    simpleMeaning: "Vuruşların pürüzsüzlüğü ve eğrilik akış dengesi.",
    tip: "Gerçek yazar alışık olduğu rotayı pürüzsüz çizer.",
  },
  slant_angle_deg: {
    icon: Compass,
    simpleMeaning: "Yazının yatay eksendeki karakteristik eğim açısı.",
    tip: "İmza eğimi bir kişinin en zor değiştirebildiği kaligrafik alışkanlığıdır.",
  },
  loop_count: {
    icon: Repeat,
    simpleMeaning: "İmza içerisindeki kapalı döngü ve ilmek sayısı.",
    tip: "İlmeklerin sayısı ve dönüş yönü yazarın motor hafızasını yansıtır.",
  },
  baseline_deviation_px: {
    icon: Layers,
    simpleMeaning: "İmzanın taban çizgisine oturma ve dalgalanma payı.",
    tip: "İmzanın satır hizasında nasıl hareket ettiği ölçülür.",
  },
  crossing_count: {
    icon: Layers,
    simpleMeaning: "Çizgilerin birbiriyle kesiştiği kavşak noktaları.",
    tip: "İmzanın karmaşıklık ve kesişim topolojisini belirler.",
  },
  terminal_count: {
    icon: CheckCircle2,
    simpleMeaning: "Kalem kalkış ve çizgi bitiş noktaları.",
    tip: "İmzanın nerede başlayıp nerede bittiğini modeller.",
  },
};

export const ShapWaterfallChart: React.FC<ShapWaterfallChartProps> = ({
  positiveFeatures = [],
  negativeFeatures = [],
  summaryText,
}) => {
  const [activeTab, setActiveTab] = useState<"all" | "positive" | "negative">("all");

  const totalPositiveScore = positiveFeatures.reduce((acc, f) => acc + Math.abs(f.impact_score), 0);
  const totalNegativeScore = negativeFeatures.reduce((acc, f) => acc + Math.abs(f.impact_score), 0);

  return (
    <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-3xl p-5 sm:p-6 shadow-2xl backdrop-blur-md space-y-5">
      {/* 🌟 1. ÜST HERO BAŞLIK & YÖNETİCİ ÖZETİ */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-[#B8A484]/20">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-2xl bg-burgundy-950/80 border border-burgundy-700/40 flex items-center justify-center text-amber-300 shadow-inner shrink-0 mt-0.5">
            <Brain className="w-5 h-5 text-amber-400" />
          </div>
          <div className="space-y-0.5">
            <h3 className="text-sm font-bold text-[#FAF7F0]">
              Karara En Çok Etki Eden Özellikler
            </h3>
            <p className="text-xs text-[#D4C5A9]/80">
              Modelin kararını en çok etkileyen öznitelikler aşağıda listelenmiştir.
            </p>
          </div>
        </div>

        {/* Hızlı Özet Göstergeleri */}
        <div className="flex items-center gap-2 text-xs">
          <div className="px-3 py-1.5 rounded-xl bg-[#120D0A] border border-[#B8A484]/20 flex items-center gap-2">
            <span className="text-[11px] text-[#D4C5A9]/70">Orijinallik Desteği:</span>
            <span className="font-mono font-bold text-emerald-400">
              {positiveFeatures.length} Öznitelik (+%{(totalPositiveScore * 100).toFixed(0)})
            </span>
          </div>
          {negativeFeatures.length > 0 && (
            <div className="px-3 py-1.5 rounded-xl bg-[#120D0A] border border-rose-500/20 flex items-center gap-2">
              <span className="text-[11px] text-[#D4C5A9]/70">Sapma:</span>
              <span className="font-mono font-bold text-rose-400">
                {negativeFeatures.length} Öznitelik (-%{(totalNegativeScore * 100).toFixed(0)})
              </span>
            </div>
          )}
        </div>
      </div>

      {/* 📊 2. KARARA EN ÇOK ETKİ EDEN ÖZNİTELİK KARTLARI */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold text-[#FAF7F0] flex items-center gap-2">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            Karara En Çok Katkı Sağlayan 5 Kritik Öznitelik
          </h4>
          <span className="text-[11px] text-[#D4C5A9]/60">Önem sırasına göre</span>
        </div>

        {positiveFeatures.length === 0 && negativeFeatures.length === 0 ? (
          <div className="p-8 text-center bg-[#120D0A] border border-[#B8A484]/15 rounded-2xl text-xs text-[#D4C5A9]/60">
            Henüz öznitelik çıkarımı yapılmadı. Lütfen analizi başlatınız.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Pozitif Destek Kartları (Orijinallik Yönünde) */}
            {positiveFeatures.map((feat) => {
              const meta = FEATURE_META[feat.feature_key] || {
                icon: Activity,
                simpleMeaning: "İmzanın morfolojik geometrik ölçümü.",
                tip: "Referans alışkanlıklarla karşılaştırılmıştır.",
              };
              const IconComp = meta.icon;
              const impactPct = Math.round(feat.impact_score * 100);

              return (
                <div
                  key={feat.feature_key}
                  className="p-3.5 rounded-2xl bg-gradient-to-br from-[#1A130F] to-[#241A14] border border-emerald-500/25 hover:border-emerald-500/50 shadow-md transition space-y-2.5"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-7 h-7 rounded-xl bg-emerald-950/80 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                        <IconComp className="w-3.5 h-3.5" />
                      </div>
                      <div>
                        <span className="text-xs font-bold text-[#FAF7F0] block">
                          {feat.feature_name_tr}
                        </span>
                        <span className="text-[10px] text-emerald-400 font-semibold flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3" /> Tam Doğal Uyum
                        </span>
                      </div>
                    </div>

                    <div className="text-right">
                      <span className="font-mono text-xs font-black text-emerald-400">
                        +{impactPct}% Katkı
                      </span>
                    </div>
                  </div>

                  {/* İnce & Şık İlerleme Çubuğu */}
                  <div className="w-full bg-[#120D0A] rounded-full h-1.5 overflow-hidden border border-[#B8A484]/15">
                    <div
                      className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-full rounded-full transition-all duration-700"
                      style={{ width: `${Math.min(100, Math.max(25, impactPct * 3.5))}%` }}
                    />
                  </div>

                  {/* Sade Anlam Açıklaması */}
                  <p className="text-[11px] text-[#D4C5A9]/85 leading-tight">
                    {meta.simpleMeaning}
                  </p>
                </div>
              );
            })}

            {/* Negatif Kartlar veya Çelişki Yok Durumu */}
            {negativeFeatures.length === 0 ? (
              <div className="p-4 rounded-2xl bg-[#140E0B] border border-emerald-500/20 flex flex-col justify-center items-center text-center space-y-2 shadow-inner">
                <div className="w-9 h-9 rounded-full bg-emerald-950/60 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <div className="space-y-0.5">
                  <h5 className="text-xs font-bold text-emerald-300">
                    Belirgin Negatif Sapma Yok
                  </h5>
                  <p className="text-[11px] text-[#D4C5A9]/75 max-w-xs leading-relaxed">
                    İncelenen tüm hat ve eğim parametreleri yazarın doğal imza varyasyon sınırları içerisindedir.
                  </p>
                </div>
              </div>
            ) : (
              negativeFeatures.map((feat) => {
                const meta = FEATURE_META[feat.feature_key] || {
                  icon: AlertTriangle,
                  simpleMeaning: "Belirlenen parametrede referans yazarla uyuşmazlık.",
                  tip: "Olası taklit şüphesi taşır.",
                };
                const IconComp = meta.icon;
                const impactPct = Math.round(Math.abs(feat.impact_score) * 100);

                return (
                  <div
                    key={feat.feature_key}
                    className="p-3.5 rounded-2xl bg-gradient-to-br from-[#1A130F] to-[#261515] border border-rose-500/30 shadow-md space-y-2.5"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-7 h-7 rounded-xl bg-rose-950/80 border border-rose-500/30 flex items-center justify-center text-rose-400">
                          <IconComp className="w-3.5 h-3.5" />
                        </div>
                        <div>
                          <span className="text-xs font-bold text-[#FAF7F0] block">
                            {feat.feature_name_tr}
                          </span>
                          <span className="text-[10px] text-rose-400 font-semibold flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" /> Şüpheli Sapma
                          </span>
                        </div>
                      </div>

                      <div className="text-right">
                        <span className="font-mono text-xs font-black text-rose-400">
                          -{impactPct}% Etki
                        </span>
                      </div>
                    </div>

                    <div className="w-full bg-[#120D0A] rounded-full h-1.5 overflow-hidden border border-[#B8A484]/15">
                      <div
                        className="bg-rose-500 h-full rounded-full"
                        style={{ width: `${Math.min(100, Math.max(25, impactPct * 3.5))}%` }}
                      />
                    </div>

                    <p className="text-[11px] text-[#D4C5A9]/85 leading-tight">
                      {meta.simpleMeaning}
                    </p>
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>
    </div>
  );
};
