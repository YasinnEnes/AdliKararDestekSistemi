"use client";

import React from "react";
import {
  Eye,
  Grid,
  Flame,
  GitCommit,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Link2,
  SlidersHorizontal,
  Target,
  Sparkles,
} from "lucide-react";

export interface LayerState {
  showOriginal: boolean;
  showSkeleton: boolean;
  showNodes: boolean;
  showGradCam: boolean;
  showSlantGrid: boolean;
  isSynchronized: boolean;
}

interface LayerControlsProps {
  layers: LayerState;
  onLayerChange: (newLayers: LayerState) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomReset: () => void;
}

export const LayerControls: React.FC<LayerControlsProps> = ({
  layers,
  onLayerChange,
  onZoomIn,
  onZoomOut,
  onZoomReset,
}) => {
  const toggle = (key: keyof LayerState) => {
    onLayerChange({
      ...layers,
      [key]: !layers[key],
    });
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 bg-[#1E1612]/95 border border-[#B8A484]/30 rounded-2xl text-xs shadow-2xl backdrop-blur-md">
      {/* Katman Seçim Butonları (Farklı Renk Kodlamalı ve Net Aktif Göstergeli) */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 px-2 py-1 bg-[#2A2019] rounded-lg border border-[#B8A484]/20 mr-1">
          <SlidersHorizontal className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-[#FAF7F0] font-bold text-[11px] uppercase tracking-wider">Adli Katmanlar:</span>
        </div>

        {/* 1. Orijinal Katman */}
        <button
          type="button"
          onClick={() => toggle("showOriginal")}
          title="Orijinal belge görüntüsünü göster / gizle"
          className={`px-3 py-1.5 rounded-xl flex items-center gap-2 transition-all font-semibold text-xs border ${
            layers.showOriginal
              ? "bg-[#2A2019] text-[#FAF7F0] border-[#B8A484] shadow-md ring-1 ring-[#FAF7F0]/20"
              : "bg-[#18120F] text-[#D4C5A9]/50 border-[#B8A484]/15 hover:text-[#D4C5A9] hover:bg-[#2A2019]"
          }`}
        >
          <div className={`w-2 h-2 rounded-full ${layers.showOriginal ? "bg-[#FAF7F0] shadow-sm shadow-white" : "bg-zinc-600"}`} />
          <Eye className={`w-3.5 h-3.5 ${layers.showOriginal ? "text-[#FAF7F0]" : "text-[#D4C5A9]/40"}`} />
          <span>Orijinal</span>
        </button>

        {/* 2. Çizgi İskeleti Katmanı */}
        <button
          type="button"
          onClick={() => toggle("showSkeleton")}
          title="İncelik ve hat iskeletini (1-pixel thinning) imzanın tam üzerinde göster"
          className={`px-3 py-1.5 rounded-xl flex items-center gap-2 transition-all font-semibold text-xs border ${
            layers.showSkeleton
              ? "bg-cyan-950/80 text-cyan-200 border-cyan-400 shadow-md shadow-cyan-950 ring-1 ring-cyan-400/40"
              : "bg-[#18120F] text-[#D4C5A9]/50 border-[#B8A484]/15 hover:text-cyan-300 hover:bg-[#2A2019]"
          }`}
        >
          <div className={`w-2 h-2 rounded-full ${layers.showSkeleton ? "bg-cyan-400 shadow-sm shadow-cyan-400 animate-pulse" : "bg-zinc-600"}`} />
          <GitCommit className={`w-3.5 h-3.5 ${layers.showSkeleton ? "text-cyan-400" : "text-[#D4C5A9]/40"}`} />
          <span>Çizgi İskeleti</span>
        </button>

        {/* 3. Kavşak ve Uç Noktaları Katmanı */}
        <button
          type="button"
          onClick={() => toggle("showNodes")}
          title="Yazıdaki başlangıç (yeşil), bitiş (kırmızı) ve dönüm (sarı) noktalarını çizgilerin üzerinde göster"
          className={`px-3 py-1.5 rounded-xl flex items-center gap-2 transition-all font-semibold text-xs border ${
            layers.showNodes
              ? "bg-amber-950/80 text-amber-200 border-amber-400 shadow-md shadow-amber-950 ring-1 ring-amber-400/40"
              : "bg-[#18120F] text-[#D4C5A9]/50 border-[#B8A484]/15 hover:text-amber-300 hover:bg-[#2A2019]"
          }`}
        >
          <div className={`w-2 h-2 rounded-full ${layers.showNodes ? "bg-amber-400 shadow-sm shadow-amber-400 animate-pulse" : "bg-zinc-600"}`} />
          <Target className={`w-3.5 h-3.5 ${layers.showNodes ? "text-amber-400" : "text-[#D4C5A9]/40"}`} />
          <span>Kavşak & Uç Noktaları</span>
        </button>

        {/* 4. Yapay Zekâ Odak Haritası Katmanı */}
        <button
          type="button"
          onClick={() => toggle("showGradCam")}
          title="Modelin karar verirken yoğunlaştığı Grad-CAM görsel dikkat ısı haritasını göster"
          className={`px-3 py-1.5 rounded-xl flex items-center gap-2 transition-all font-semibold text-xs border ${
            layers.showGradCam
              ? "bg-rose-950/80 text-rose-200 border-rose-500 shadow-md shadow-rose-950 ring-1 ring-rose-500/40"
              : "bg-[#18120F] text-[#D4C5A9]/50 border-[#B8A484]/15 hover:text-rose-300 hover:bg-[#2A2019]"
          }`}
        >
          <div className={`w-2 h-2 rounded-full ${layers.showGradCam ? "bg-rose-500 shadow-sm shadow-rose-500 animate-pulse" : "bg-zinc-600"}`} />
          <Flame className={`w-3.5 h-3.5 ${layers.showGradCam ? "text-rose-400" : "text-[#D4C5A9]/40"}`} />
          <span>YZ Odak Haritası</span>
        </button>

        {/* 5. Yazı Eğimi Kılavuzu Katmanı */}
        <button
          type="button"
          onClick={() => toggle("showSlantGrid")}
          title="Belge alanında açısal eğim kılavuz çizgilerini ve derece rozetini göster"
          className={`px-3 py-1.5 rounded-xl flex items-center gap-2 transition-all font-semibold text-xs border ${
            layers.showSlantGrid
              ? "bg-yellow-950/80 text-yellow-200 border-yellow-400 shadow-md shadow-yellow-950 ring-1 ring-yellow-400/40"
              : "bg-[#18120F] text-[#D4C5A9]/50 border-[#B8A484]/15 hover:text-yellow-300 hover:bg-[#2A2019]"
          }`}
        >
          <div className={`w-2 h-2 rounded-full ${layers.showSlantGrid ? "bg-yellow-400 shadow-sm shadow-yellow-400" : "bg-zinc-600"}`} />
          <Grid className={`w-3.5 h-3.5 ${layers.showSlantGrid ? "text-yellow-400" : "text-[#D4C5A9]/40"}`} />
          <span>Yazı Eğimi Kılavuzu</span>
        </button>
      </div>

      {/* Büyütme & Senkronizasyon Kontrolleri */}
      <div className="flex items-center gap-2 border-l border-[#B8A484]/20 pl-3">
        <button
          type="button"
          onClick={() => toggle("isSynchronized")}
          title="Çift kanvas eşzamanlı büyütme ve kaydırma"
          className={`px-3 py-1.5 rounded-xl flex items-center gap-1.5 transition font-semibold text-xs border ${
            layers.isSynchronized
              ? "bg-emerald-950/80 text-emerald-200 border-emerald-500 shadow-sm"
              : "bg-[#2A2019] text-[#D4C5A9]/60 hover:text-[#FAF7F0] hover:bg-[#3D2E24] border-[#B8A484]/15"
          }`}
        >
          <Link2 className={`w-3.5 h-3.5 ${layers.isSynchronized ? "text-emerald-400" : "text-[#D4C5A9]/40"}`} />
          <span>{layers.isSynchronized ? "Senkron: Açık" : "Senkron: Kapalı"}</span>
        </button>

        <div className="flex items-center bg-[#2A2019] rounded-xl p-0.5 border border-[#B8A484]/20 shadow-inner">
          <button
            type="button"
            onClick={onZoomOut}
            title="Uzaklaş"
            className="p-1.5 hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] rounded-lg transition"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={onZoomIn}
            title="Yakınlaş"
            className="p-1.5 hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] rounded-lg transition"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={onZoomReset}
            title="Sıfırla (%100)"
            className="p-1.5 hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] rounded-lg transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};

