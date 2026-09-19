"use client";

import React from "react";
import { AlertTriangle, AlertCircle, ArrowLeft, X } from "lucide-react";

interface CaseSwitchConfirmationModalProps {
  isOpen: boolean;
  hasUnsavedChanges?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  currentCaseId?: string;
}

export const CaseSwitchConfirmationModal: React.FC<CaseSwitchConfirmationModalProps> = ({
  isOpen,
  hasUnsavedChanges = false,
  onConfirm,
  onCancel,
  currentCaseId,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in">
      <div className="bg-[#1E1612] border border-[#B8A484]/35 rounded-3xl p-6 max-w-md w-full shadow-2xl space-y-4 text-left relative">
        {/* Kapat İkonu */}
        <button
          type="button"
          onClick={onCancel}
          className="absolute right-4 top-4 p-1.5 rounded-full bg-[#2A2019] hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] transition"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Başlık ve Uyarı İkonu */}
        <div className="flex items-start gap-3.5">
          <div className="w-11 h-11 rounded-2xl bg-amber-950/80 border border-amber-600/50 flex items-center justify-center text-amber-400 shrink-0 shadow-inner">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-[#FAF7F0]">
              Vaka Çalışma Alanına Dön
            </h3>
            <p className="text-xs text-[#D4C5A9]/80 mt-0.5">
              Aktif dosya: <span className="font-mono text-cream-200 font-semibold">{currentCaseId || "Mevcut Vaka"}</span>
            </p>
          </div>
        </div>

        {/* Açıklama */}
        <p className="text-xs text-[#EFE7DA] leading-relaxed">
          Mevcut vakadan çıkıp başka bir vakaya geçmek veya yeni vaka oluşturmak istediğinizden emin misiniz?
        </p>

        {/* Kaydedilmemiş Değişiklik Uyarısı */}
        {hasUnsavedChanges && (
          <div className="p-3 bg-rose-950/80 border border-rose-800/80 rounded-xl text-xs text-rose-200 flex items-start gap-2 shadow-inner">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <strong className="block font-bold text-rose-300">Kaydedilmemiş Değişiklikleriniz Bulunuyor!</strong>
              <span>Dondurulmamış veya kaydedilmemiş analiz verileri kaybolabilir.</span>
            </div>
          </div>
        )}

        {/* Butonlar */}
        <div className="flex items-center justify-end gap-2.5 pt-2 border-t border-[#B8A484]/15">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#2A2019] hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] border border-[#B8A484]/20 transition"
          >
            İncelemeye Devam Et
          </button>

          <button
            type="button"
            onClick={onConfirm}
            className="px-4 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] shadow-lg transition border border-[#B8A484]/30 flex items-center gap-1.5"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Onayla ve Çık</span>
          </button>
        </div>
      </div>
    </div>
  );
};
