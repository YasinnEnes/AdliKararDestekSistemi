"use client";

import React, { useState } from "react";
import {
  PlusCircle,
  ArrowLeft,
  ShieldCheck,
  FolderPlus,
  Fingerprint,
  FileText,
  Building2,
  User,
  Hash,
  AlertCircle,
  CheckCircle2,
  Sparkles,
} from "lucide-react";
import { useCaseContext } from "@/context/CaseContext";
import { ExaminationType, formatApiError } from "@/services/api";

interface CreateCasePanelProps {
  onBack: () => void;
}

export const CreateCasePanel: React.FC<CreateCasePanelProps> = ({ onBack }) => {
  const { createNewCase, selectCaseAndNavigate } = useCaseContext();

  // Form Alanları
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [examinationType, setExaminationType] = useState<ExaminationType>("SIGNATURE");
  const [referenceNumber, setReferenceNumber] = useState("");
  const [authority, setAuthority] = useState("Cumhuriyet Başsavcılığı / Mahkeme");
  const [examinerId, setExaminerId] = useState("BILIRKISI_UZMAN_01");
  const [notes, setNotes] = useState("");

  // Durum Yönetimi
  const [loading, setLoading] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);

    // Zorunlu Alan Doğrulaması
    if (!name.trim()) {
      setValidationError("Lütfen Vaka Adını giriniz (Zorunlu alan).");
      return;
    }

    if (!description.trim()) {
      setValidationError("Lütfen Vaka Açıklamasını giriniz (Zorunlu alan).");
      return;
    }

    setLoading(true);

    try {
      const created = await createNewCase({
        name: name.trim(),
        description: description.trim(),
        examination_type: examinationType,
        reference_number: referenceNumber.trim() || undefined,
        authority: authority.trim() || "Cumhuriyet Başsavcılığı / Mahkeme",
        examiner_id: examinerId.trim() || "BILIRKISI_UZMAN_01",
        notes: notes.trim() || "",
        status: "IN_PROGRESS",
        current_stage: "CASE_CREATED",
      });

      setSuccessMessage(`"${created.name}" (${created.case_id}) başarıyla oluşturuldu. İş istasyonuna aktarılıyorsunuz...`);

      setTimeout(() => {
        selectCaseAndNavigate(created.case_id);
      }, 700);
    } catch (err: any) {
      const formattedErr = formatApiError(err);
      setValidationError(`Vaka oluşturulamadı: ${formattedErr}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto py-4 px-2 flex flex-col flex-1 space-y-4">
      {/* ÜST BAŞLIK & GERİ DÖNÜŞ BARI */}
      <div className="flex items-center justify-between pb-3 border-b border-[#B8A484]/20">
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={onBack}
            className="p-2 rounded-xl bg-[#2A2019] hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] border border-[#B8A484]/20 transition shadow-sm"
            title="Vaka Seçim Alanına Dön"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <h2 className="text-lg font-bold text-[#FAF7F0] flex items-center gap-1.5">
              <FolderPlus className="w-5 h-5 text-burgundy-600" />
              Yeni Vaka Oluştur
            </h2>
            <p className="text-[11px] text-[#D4C5A9]/80">
              Adli el yazısı & imza inceleme dosyası
            </p>
          </div>
        </div>

        <div className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-[#1E1612] border border-[#B8A484]/20 text-[10px] text-[#D4C5A9]">
          <ShieldCheck className="w-3 h-3 text-emerald-400" /> MFHM İzolasyon
        </div>
      </div>

      {/* FORM KARTI */}
      <div className="max-w-md w-full bg-[#1C1510]/85 border border-[#B8A484]/30 rounded-2xl p-6 shadow-2xl backdrop-blur-md space-y-5">
        {/* Bildirim Alanları */}
        {validationError && (
          <div className="p-3 rounded-xl bg-rose-950/80 border border-rose-800 text-rose-200 text-xs flex items-center gap-2 shadow-inner">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{validationError}</span>
          </div>
        )}

        {successMessage && (
          <div className="p-3 rounded-xl bg-emerald-950/80 border border-emerald-800 text-emerald-200 text-xs flex items-center gap-2 shadow-inner">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{successMessage}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4 text-xs">
          {/* Vaka Adı */}
          <div>
            <label className="block text-[#FAF7F0] font-semibold mb-1">
              Vaka Adı <span className="text-rose-400 font-bold">*</span>:
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Örn: Şüpheli Senet İmza İncelemesi"
              required
              className="w-full px-3.5 py-2 bg-[#120D0A]/80 border border-[#B8A484]/20 rounded-xl text-[#FAF7F0] placeholder-[#D4C5A9]/40 focus:outline-none focus:border-[#7B1E34] focus:ring-1 focus:ring-[#7B1E34]/50 transition text-xs shadow-inner"
            />
          </div>

          {/* İnceleme Türü Seçimi */}
          <div>
            <label className="block text-[#FAF7F0] font-semibold mb-1">
              İnceleme Türü <span className="text-rose-400 font-bold">*</span>:
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { id: "SIGNATURE", label: "İmza İncelemesi" },
                { id: "HANDWRITING", label: "El Yazısı İncelemesi" },
                { id: "HANDWRITING_AND_SIGNATURE", label: "El Yazısı + İmza" },
                { id: "OTHER", label: "Diğer İnceleme" },
              ].map((type) => (
                <button
                  key={type.id}
                  type="button"
                  onClick={() => setExaminationType(type.id as ExaminationType)}
                  className={`py-2 px-2.5 rounded-xl font-semibold text-[11px] transition border flex items-center justify-center text-center ${
                    examinationType === type.id
                      ? "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] text-[#FAF7F0] border-[#B8A484]/40 shadow-md"
                      : "bg-[#2A2019] text-[#D4C5A9] hover:bg-[#3D2E24] hover:text-[#FAF7F0] border-[#B8A484]/15"
                  }`}
                >
                  {type.label}
                </button>
              ))}
            </div>
          </div>

          {/* Vaka Açıklaması */}
          <div>
            <label className="block text-[#FAF7F0] font-semibold mb-1">
              Vaka Açıklaması <span className="text-rose-400 font-bold">*</span>:
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="İncelemeye konu belgenin niteliği ve şüpheli unsurlar..."
              rows={2}
              required
              className="w-full px-3.5 py-2 bg-[#120D0A]/80 border border-[#B8A484]/20 rounded-xl text-[#FAF7F0] placeholder-[#D4C5A9]/40 focus:outline-none focus:border-[#7B1E34] focus:ring-1 focus:ring-[#7B1E34]/50 transition text-xs shadow-inner leading-relaxed"
            />
          </div>

          <div className="grid grid-cols-2 gap-2.5">
            {/* Esas No / Referans No */}
            <div>
              <label className="block text-[#D4C5A9] font-medium mb-1">
                Esas No / Ref:
              </label>
              <input
                type="text"
                value={referenceNumber}
                onChange={(e) => setReferenceNumber(e.target.value)}
                placeholder="2026/1492"
                className="w-full px-3 py-1.5 bg-[#120D0A]/80 border border-[#B8A484]/20 rounded-xl text-[#FAF7F0] placeholder-[#D4C5A9]/40 focus:outline-none focus:border-[#7B1E34] focus:ring-1 focus:ring-[#7B1E34]/50 transition text-xs shadow-inner"
              />
            </div>

            {/* İncelemeci Bilirkişi */}
            <div>
              <label className="block text-[#D4C5A9] font-medium mb-1">
                Bilirkişi ID:
              </label>
              <input
                type="text"
                value={examinerId}
                onChange={(e) => setExaminerId(e.target.value)}
                placeholder="UZMAN_01"
                className="w-full px-3 py-1.5 bg-[#120D0A]/80 border border-[#B8A484]/20 rounded-xl text-[#FAF7F0] placeholder-[#D4C5A9]/40 focus:outline-none focus:border-[#7B1E34] focus:ring-1 focus:ring-[#7B1E34]/50 transition text-xs shadow-inner"
              />
            </div>
          </div>

          {/* Adli Makam */}
          <div>
            <label className="block text-[#D4C5A9] font-medium mb-1">
              Talep Eden Makam:
            </label>
            <input
              type="text"
              value={authority}
              onChange={(e) => setAuthority(e.target.value)}
              placeholder="Cumhuriyet Başsavcılığı / Mahkeme"
              className="w-full px-3.5 py-1.5 bg-[#120D0A]/80 border border-[#B8A484]/20 rounded-xl text-[#FAF7F0] placeholder-[#D4C5A9]/40 focus:outline-none focus:border-[#7B1E34] focus:ring-1 focus:ring-[#7B1E34]/50 transition text-xs shadow-inner"
            />
          </div>

          {/* BUTONLAR */}
          <div className="flex flex-col gap-2 pt-2 border-t border-[#B8A484]/20">
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] font-semibold py-3 rounded-xl shadow-lg transition-all border border-[#B8A484]/30 flex items-center justify-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed text-xs sm:text-sm"
            >
              <PlusCircle className="w-4 h-4 text-cream-200" />
              <span>{loading ? "Vaka Oluşturuluyor..." : "İş İstasyonunu Başlat"}</span>
            </button>

            <button
              type="button"
              onClick={onBack}
              disabled={loading}
              className="w-full py-2 rounded-xl text-xs font-medium bg-[#2A2019] hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] border border-[#B8A484]/20 transition text-center"
            >
              Vazgeç / Geri Dön
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
