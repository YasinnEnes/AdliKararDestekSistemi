"use client";

import React from "react";
import {
  FolderOpen,
  PlusCircle,
  ShieldCheck,
  Cpu,
  Fingerprint,
  Sparkles,
  ArrowRight,
  History,
  FileCheck2,
  Lock,
} from "lucide-react";
import { useCaseContext } from "@/context/CaseContext";

interface CaseSelectionDashboardProps {
  onSelectExisting: () => void;
  onSelectNew: () => void;
}

export const CaseSelectionDashboard: React.FC<CaseSelectionDashboardProps> = ({
  onSelectExisting,
  onSelectNew,
}) => {
  const { cases } = useCaseContext();

  return (
    <div className="w-full flex flex-col items-center justify-center flex-1 py-8 px-4 sm:px-6">
      {/* ÜST BAŞLIK & AÇIKLAMA */}
      <div className="text-center mb-10 space-y-3 max-w-2xl">
        <div className="inline-flex items-center gap-2 px-4 py-1 rounded-full bg-[#1C1510]/90 border border-[#B8A484]/25 text-xs font-semibold text-[#FAF7F0] shadow-lg backdrop-blur-md">
          <Sparkles className="w-3.5 h-3.5 text-burgundy-600" />
          TÜBİTAK Destekli Adli Belge & Karar Destek Platformu
        </div>

        <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-[#FAF7F0] tracking-tight drop-shadow-md">
          Vaka Çalışma Alanı
        </h2>

        <p className="text-sm sm:text-base text-[#D4C5A9]/85 max-w-xl mx-auto leading-relaxed">
          Çalışmak istediğiniz vakayı seçin veya yeni bir adli el yazısı ve imza inceleme vakası oluşturun.
        </p>
      </div>

      {/* 2 ANA SEÇENEK KARTI */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl">
        {/* KART 1: MEVCUT VAKAYA DEVAM ET */}
        <div
          onClick={onSelectExisting}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onSelectExisting();
            }
          }}
          className="group relative bg-[#1E1612]/90 hover:bg-[#251B16] border border-[#B8A484]/25 hover:border-[#B8A484]/50 rounded-3xl p-7 sm:p-8 shadow-2xl backdrop-blur-md transition-all duration-300 flex flex-col justify-between cursor-pointer text-left hover:-translate-y-1 hover:shadow-burgundy-950/40"
        >
          {/* Üst Vurgu Rozeti */}
          <div className="flex items-center justify-between mb-6">
            <div className="w-14 h-14 rounded-2xl bg-[#2A2019] border border-[#B8A484]/30 flex items-center justify-center text-teal-400 group-hover:scale-105 transition shadow-inner">
              <FolderOpen className="w-7 h-7 text-cream-200 group-hover:text-amber-400 transition" />
            </div>
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-[#2A2019] text-teal-400 border border-teal-500/30 shadow-sm">
              {cases.length > 0 ? `${cases.length} Kayıtlı Vaka` : "Vaka Arşivi"}
            </span>
          </div>

          {/* Başlık ve Açıklama */}
          <div className="space-y-2.5 mb-6">
            <h3 className="text-xl sm:text-2xl font-bold text-[#FAF7F0] group-hover:text-cream-100 transition flex items-center gap-2">
              Mevcut Vakaya Devam Et
            </h3>
            <p className="text-xs sm:text-sm text-[#D4C5A9]/80 leading-relaxed">
              Kayıtlı adli dosyalardan birini seçerek dondurulmuş özniteliklere, görsel kanıtlara ve kaldığınız aşamaya doğrudan geri dönün.
            </p>
          </div>

          {/* Özellik Maddeleri */}
          <div className="space-y-2 py-3 border-y border-[#B8A484]/15 mb-6 text-xs text-[#D4C5A9]/90">
            <div className="flex items-center gap-2">
              <History className="w-4 h-4 text-burgundy-600 shrink-0" />
              <span>Son çalışılan aşamayı (Öznitelik Analizi, Bilirkişi Raporu vb.) korur</span>
            </div>
            <div className="flex items-center gap-2">
              <Lock className="w-4 h-4 text-teal-400 shrink-0" />
              <span>Dondurulmuş Şüpheli Belge kanıt bütünlüğü mühürlüdür</span>
            </div>
          </div>

          {/* Aksiyon Butonu */}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelectExisting();
            }}
            className="w-full py-3.5 bg-[#2A2019] group-hover:bg-gradient-to-r group-hover:from-[#7B1E34] group-hover:to-[#5C1425] text-[#FAF7F0] font-semibold text-xs sm:text-sm rounded-2xl border border-[#B8A484]/25 group-hover:border-[#B8A484]/40 shadow-lg transition duration-200 flex items-center justify-center gap-2"
          >
            <span>Mevcut Vakalara Git</span>
            <ArrowRight className="w-4 h-4 text-cream-200 group-hover:translate-x-1 transition duration-200" />
          </button>
        </div>

        {/* KART 2: YENİ VAKA OLUŞTUR */}
        <div
          onClick={onSelectNew}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onSelectNew();
            }
          }}
          className="group relative bg-[#1E1612]/90 hover:bg-[#251B16] border border-[#B8A484]/25 hover:border-burgundy-600/60 rounded-3xl p-7 sm:p-8 shadow-2xl backdrop-blur-md transition-all duration-300 flex flex-col justify-between cursor-pointer text-left hover:-translate-y-1 hover:shadow-burgundy-950/40"
        >
          {/* Üst Vurgu Rozeti */}
          <div className="flex items-center justify-between mb-6">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-burgundy-700/60 to-burgundy-900/80 border border-burgundy-600/50 flex items-center justify-center shadow-inner group-hover:scale-105 transition">
              <PlusCircle className="w-7 h-7 text-cream-100 group-hover:text-amber-300 transition" />
            </div>
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-burgundy-900/60 text-cream-200 border border-burgundy-600/50 shadow-sm">
              Yeni İnceleme
            </span>
          </div>

          {/* Başlık ve Açıklama */}
          <div className="space-y-2.5 mb-6">
            <h3 className="text-xl sm:text-2xl font-bold text-[#FAF7F0] group-hover:text-cream-100 transition flex items-center gap-2">
              Yeni Vaka Oluştur
            </h3>
            <p className="text-xs sm:text-sm text-[#D4C5A9]/80 leading-relaxed">
              Yeni bir adli el yazısı, imza veya karma belge incelemesi dosyası açın. Bilişsel izolasyon korumasıyla analizi başlatın.
            </p>
          </div>

          {/* Özellik Maddeleri */}
          <div className="space-y-2 py-3 border-y border-[#B8A484]/15 mb-6 text-xs text-[#D4C5A9]/90">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>Otomatik Vaka No & Bilişsel İzolasyon Durum Makinesi</span>
            </div>
            <div className="flex items-center gap-2">
              <FileCheck2 className="w-4 h-4 text-amber-400 shrink-0" />
              <span>ENFSI BPM-FHX-01 uyumlu resmi adli kayıt</span>
            </div>
          </div>

          {/* Aksiyon Butonu */}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelectNew();
            }}
            className="w-full py-3.5 bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] font-semibold text-xs sm:text-sm rounded-2xl border border-[#B8A484]/30 shadow-lg transition duration-200 flex items-center justify-center gap-2"
          >
            <span>Yeni Vaka Formunu Aç</span>
            <ArrowRight className="w-4 h-4 text-cream-200 group-hover:translate-x-1 transition duration-200" />
          </button>
        </div>
      </div>
    </div>
  );
};
