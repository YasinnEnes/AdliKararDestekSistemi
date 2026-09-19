"use client";

import React, { useState, useEffect } from "react";
import { Sparkles, Check, X, ArrowRight } from "lucide-react";

export interface DemoScenario {
  id: string;
  title: string;
  config: {
    slant: number;
    withTremor: boolean;
    style: "standard" | "duplicate" | "tremor" | "angular" | "disguise";
  };
}

export const DEMO_SCENARIOS: DemoScenario[] = [
  {
    id: "scenario_1",
    title: "Örnek 1",
    config: { slant: 14.5, withTremor: false, style: "standard" },
  },
  {
    id: "scenario_2",
    title: "Örnek 2",
    config: { slant: 14.2, withTremor: false, style: "duplicate" },
  },
  {
    id: "scenario_3",
    title: "Örnek 3",
    config: { slant: 16.5, withTremor: true, style: "tremor" },
  },
  {
    id: "scenario_4",
    title: "Örnek 4",
    config: { slant: -8.0, withTremor: false, style: "angular" },
  },
  {
    id: "scenario_5",
    title: "Örnek 5",
    config: { slant: 24.0, withTremor: true, style: "disguise" },
  },
];

// Canvas üzerinde imza çizici
export function renderSignatureToDataUrl(
  slantAngle: number,
  withTremor: boolean,
  style: "standard" | "duplicate" | "tremor" | "angular" | "disguise"
): string {
  if (typeof window === "undefined") return "";
  const canvas = document.createElement("canvas");
  canvas.width = 400;
  canvas.height = 250;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  // Arka plan
  ctx.fillStyle = "#FAF7F0";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Kılavuz çizgisi
  ctx.strokeStyle = "rgba(184, 164, 132, 0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(30, 190);
  ctx.lineTo(370, 190);
  ctx.stroke();

  ctx.save();
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.rotate((slantAngle * Math.PI) / 180);

  ctx.strokeStyle = "#1A1A2E";
  ctx.lineWidth = style === "angular" ? 3.5 : 3.0;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (style === "angular") {
    // Tamamen farklı keskin/köşeli imza
    ctx.beginPath();
    ctx.moveTo(-110, 30);
    ctx.lineTo(-80, -50);
    ctx.lineTo(-50, 15);
    ctx.lineTo(-20, -40);
    ctx.lineTo(10, 20);
    ctx.lineTo(40, -60);
    ctx.lineTo(70, 30);
    ctx.lineTo(120, -15);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(-90, 45);
    ctx.lineTo(130, 45);
    ctx.stroke();
  } else if (style === "disguise") {
    // Kasıtlı bozulmuş dalgalı imza
    ctx.beginPath();
    ctx.moveTo(-120, 40);
    ctx.bezierCurveTo(-100, -80, -40, -30, -20, 20);
    ctx.bezierCurveTo(-5, 60, 15, -40, 40, -15);
    ctx.bezierCurveTo(60, 30, 90, -80, 120, 10);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(-110, 55);
    for (let x = -110; x < 120; x += 12) {
      ctx.lineTo(x, 55 + Math.sin(x * 0.35) * 4.5);
    }
    ctx.stroke();
  } else {
    // Standart, Duplicate veya Tremor'lu imza
    ctx.beginPath();
    ctx.moveTo(-120, 20);
    ctx.bezierCurveTo(-95, -60, -60, -50, -45, 10);
    ctx.bezierCurveTo(-30, 45, -5, 50, 10, 8);
    ctx.bezierCurveTo(30, -75, 75, -60, 90, 15);
    ctx.bezierCurveTo(105, 45, 135, 50, 150, 30);

    if (withTremor) {
      for (let t = -110; t < 140; t += 8) {
        const offY = Math.sin(t * 0.5) * 3.0;
        ctx.lineTo(t, 15 + offY);
      }
    }
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(-105, 48);
    ctx.bezierCurveTo(-30, 64, 70, 34, 135, 48);
    ctx.stroke();
  }

  ctx.restore();
  return canvas.toDataURL("image/png");
}

interface DemoSamplePickerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectScenario: (scenario: DemoScenario, target: "both" | "questioned" | "reference") => void;
  defaultTarget?: "both" | "questioned" | "reference";
}

export const DemoSamplePickerModal: React.FC<DemoSamplePickerModalProps> = ({
  isOpen,
  onClose,
  onSelectScenario,
  defaultTarget = "both",
}) => {
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>(DEMO_SCENARIOS[0].id);
  const [targetType, setTargetType] = useState<"both" | "questioned" | "reference">(defaultTarget);
  const [previewUrls, setPreviewUrls] = useState<Record<string, string>>({});

  // Senaryolar için önizleme görsellerini 1 kez oluştur
  useEffect(() => {
    const urls: Record<string, string> = {};
    DEMO_SCENARIOS.forEach((sc) => {
      urls[sc.id] = renderSignatureToDataUrl(
        sc.config.slant,
        sc.config.withTremor,
        sc.config.style
      );
    });
    setPreviewUrls(urls);
  }, []);

  useEffect(() => {
    setTargetType(defaultTarget);
  }, [defaultTarget]);

  if (!isOpen) return null;

  const activeScenario = DEMO_SCENARIOS.find((s) => s.id === selectedScenarioId) || DEMO_SCENARIOS[0];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-in fade-in duration-200">
      <div className="bg-[#1E1612] border border-[#B8A484]/40 rounded-3xl p-5 sm:p-6 max-w-4xl w-full shadow-2xl space-y-4 flex flex-col max-h-[92vh]">
        {/* MODAL BAŞLIĞI */}
        <div className="flex items-start justify-between pb-3 border-b border-[#B8A484]/20 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-amber-950/70 border border-amber-500/40 flex items-center justify-center text-amber-400 shadow-inner">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm sm:text-base font-extrabold text-[#FAF7F0] flex items-center gap-2">
                Hazır Adli İmza Galerisi
                <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-[#120D0A] text-amber-400 border border-[#B8A484]/20">
                  Görsel Seçim
                </span>
              </h3>
              <p className="text-xs text-[#D4C5A9]/80 mt-0.5">
                Yüklemek istediğiniz adli imza örneğine tıklayarak seçiniz.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 bg-[#2A2019] hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] rounded-xl border border-[#B8A484]/20 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* 🎨 GÖRSEL İMZA GALERİSİ (GRID OF VISUAL THUMBNAILS) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5 overflow-y-auto pr-1 flex-1 py-1">
          {DEMO_SCENARIOS.map((sc) => {
            const isSelected = sc.id === selectedScenarioId;
            const previewUrl = previewUrls[sc.id];

            return (
              <div
                key={sc.id}
                onClick={() => setSelectedScenarioId(sc.id)}
                className={`group relative rounded-2xl border p-3 flex flex-col justify-between transition-all duration-200 cursor-pointer ${
                  isSelected
                    ? "bg-[#2A2019] border-amber-400 ring-2 ring-amber-400/50 shadow-xl shadow-amber-950/40"
                    : "bg-[#140E0B]/90 hover:bg-[#1A130F] border-[#B8A484]/25 hover:border-[#B8A484]/50"
                }`}
              >
                {/* Seçim Rozeti (Sağ Üst) */}
                <div
                  className={`absolute top-2.5 right-2.5 z-10 w-6 h-6 rounded-full flex items-center justify-center border shadow-md transition ${
                    isSelected
                      ? "bg-amber-400 text-black border-amber-300 font-bold"
                      : "bg-[#120D0A]/80 border-[#B8A484]/30 text-transparent group-hover:border-[#B8A484]/60"
                  }`}
                >
                  <Check className="w-3.5 h-3.5" />
                </div>

                <div className="space-y-2.5">
                  {/* Görsel İmza Önizlemesi (Visual Canvas Thumbnail) */}
                  <div className="w-full aspect-[16/10] rounded-xl overflow-hidden border border-[#B8A484]/30 bg-[#FAF7F0] shadow-inner flex items-center justify-center relative">
                    {previewUrl ? (
                      <img
                        src={previewUrl}
                        alt={sc.title}
                        className="w-full h-full object-contain p-1 group-hover:scale-105 transition-transform duration-200"
                      />
                    ) : (
                      <div className="text-xs text-gray-500 font-mono">Çiziliyor...</div>
                    )}
                  </div>

                  {/* Kart Başlığı */}
                  <div className="flex items-center justify-between pt-1">
                    <h4 className="text-xs font-bold text-[#FAF7F0]">
                      {sc.title}
                    </h4>
                    <span className="text-[10px] font-mono text-[#D4C5A9]/50">
                      Adli Numune
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* YÜKLEME TÜRÜ VE AKSİYON BUTONLARI */}
        <div className="pt-3 border-t border-[#B8A484]/20 space-y-3 shrink-0">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className="text-[#D4C5A9] font-semibold text-xs">Yüklenecek Hedef:</span>
            <div className="flex items-center bg-[#120D0A] p-1 rounded-xl border border-[#B8A484]/20 text-[11px]">
              <button
                type="button"
                onClick={() => setTargetType("both")}
                className={`px-3 py-1 rounded-lg font-semibold transition ${
                  targetType === "both"
                    ? "bg-[#7B1E34] text-[#FAF7F0] shadow"
                    : "text-[#D4C5A9]/60 hover:text-[#FAF7F0]"
                }`}
              >
                Hem Şüpheli Hem Mukayese
              </button>
              <button
                type="button"
                onClick={() => setTargetType("questioned")}
                className={`px-3 py-1 rounded-lg font-semibold transition ${
                  targetType === "questioned"
                    ? "bg-[#7B1E34] text-[#FAF7F0] shadow"
                    : "text-[#D4C5A9]/60 hover:text-[#FAF7F0]"
                }`}
              >
                Sadece Şüpheli
              </button>
              <button
                type="button"
                onClick={() => setTargetType("reference")}
                className={`px-3 py-1 rounded-lg font-semibold transition ${
                  targetType === "reference"
                    ? "bg-[#7B1E34] text-[#FAF7F0] shadow"
                    : "text-[#D4C5A9]/60 hover:text-[#FAF7F0]"
                }`}
              >
                Sadece Mukayese
              </button>
            </div>
          </div>

          <div className="flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl bg-[#2A2019] hover:bg-[#3D2E24] text-xs font-semibold text-[#D4C5A9] border border-[#B8A484]/20 transition"
            >
              Kapat
            </button>

            <button
              type="button"
              onClick={() => {
                onSelectScenario(activeScenario, targetType);
                onClose();
              }}
              className="px-6 py-2 rounded-xl bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] font-bold text-xs shadow-lg shadow-[#7B1E34]/30 border border-[#B8A484]/40 hover:scale-105 active:scale-95 transition-all flex items-center gap-2"
            >
              <Sparkles className="w-4 h-4 text-amber-400" />
              <span>Seçilen Görseli Yükle</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
