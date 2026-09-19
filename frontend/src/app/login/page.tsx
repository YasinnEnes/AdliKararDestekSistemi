"use client";

import React, { useState } from "react";
import {
  Shield,
  Lock,
  User,
  Fingerprint,
  AlertCircle,
  ArrowRight,
  ShieldCheck,
  Scale,
  Sparkles,
} from "lucide-react";
import { useAuth } from "@/context/AuthContext";

export default function LoginPage() {
  const { login, loading, error, clearError } = useAuth();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [formValidation, setFormValidation] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    setFormValidation(null);

    if (!username.trim()) {
      setFormValidation("Lütfen kullanıcı adınızı veya sicil numaranızı giriniz.");
      return;
    }
    if (!password) {
      setFormValidation("Lütfen parolanızı giriniz.");
      return;
    }

    try {
      await login(username.trim(), password);
    } catch {
      // Hata AuthContext state'inde işlenir
    }
  };

  const handleQuickFill = (userVal: string, passVal: string) => {
    setUsername(userVal);
    setPassword(passVal);
    clearError();
    setFormValidation(null);
  };

  return (
    <div className="fixed inset-0 w-full h-full min-h-screen bg-[url('/bg-forensic-dashboard.jpg')] bg-cover bg-center bg-no-repeat overflow-y-auto flex flex-col justify-between font-sans selection:bg-[#7B1E34] selection:text-[#FAF7F0]">
      {/* ÜST BİLGİ ÇUBUĞU */}
      <header className="border-b border-[#B8A484]/20 bg-[#1C1510]/90 backdrop-blur-md px-6 py-3.5 flex items-center justify-between shadow-xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-burgundy-700/40 border border-burgundy-600/50 flex items-center justify-center shadow-inner">
            <Fingerprint className="w-5 h-5 text-cream-200" />
          </div>
          <div>
            <h1 className="text-base font-extrabold tracking-wider text-[#FAF7F0] flex items-center gap-2">
              FHDSS WORKSTATION
              <span className="text-[10px] font-normal px-2 py-0.5 rounded-full bg-burgundy-700/60 border border-burgundy-600 text-cream-200">
                v1.0 Adli Güvenlik Sürümü
              </span>
            </h1>
            <p className="text-[11px] text-[#D4C5A9]/75 font-medium">
              Adli Belge ve El Yazısı Karar Destek Platformu
            </p>
          </div>
        </div>

        <div className="hidden sm:flex items-center gap-2 text-xs">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#2A2019] border border-[#B8A484]/20 text-[#FAF7F0] font-medium shadow-sm">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> 256-Bit JWT &amp; RBAC Korumalı
          </span>
        </div>
      </header>

      {/* ANA GİRİŞ FORMU */}
      <main className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-md bg-[#1C1510]/95 border border-[#B8A484]/30 rounded-3xl p-6 md:p-8 shadow-2xl backdrop-blur-xl space-y-6">
          {/* Logo & Başlık */}
          <div className="text-center space-y-2">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-[#7B1E34] to-[#450A17] border border-[#B8A484]/40 flex items-center justify-center text-[#FAF7F0] mx-auto shadow-lg">
              <Scale className="w-7 h-7 text-amber-300" />
            </div>
            <h2 className="text-xl font-extrabold text-[#FAF7F0] tracking-wide">
              Adli Bilirkişi Girişi
            </h2>
            <p className="text-xs text-[#D4C5A9]/80 max-w-xs mx-auto leading-relaxed">
              Vaka dosyalarına, adli analiz modellerine ve mühürlü raporlara erişmek için kimliğinizi doğrulayınız.
            </p>
          </div>

          {/* HATA BİLDİRİMİ */}
          {(formValidation || error) && (
            <div className="bg-rose-950/80 border border-rose-800/80 rounded-2xl p-3.5 text-xs text-rose-200 flex items-start gap-2.5 shadow-inner">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <span>{formValidation || error}</span>
            </div>
          )}

          {/* GİRİŞ FORMU */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#D4C5A9] flex items-center gap-1.5">
                <User className="w-3.5 h-3.5 text-amber-400" />
                Kullanıcı Adı / Sicil No
              </label>
              <div className="relative">
                <input
                  type="text"
                  value={username}
                  onChange={(e) => {
                    setUsername(e.target.value);
                    clearError();
                    setFormValidation(null);
                  }}
                  placeholder="örn: bilirkisi_uzman veya admin"
                  disabled={loading}
                  className="w-full bg-[#120D0A] border border-[#B8A484]/30 focus:border-amber-400 rounded-xl px-3.5 py-2.5 text-sm text-[#FAF7F0] placeholder-[#D4C5A9]/40 focus:outline-none focus:ring-1 focus:ring-amber-400/50 transition font-mono"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#D4C5A9] flex items-center gap-1.5">
                <Lock className="w-3.5 h-3.5 text-amber-400" />
                Güvenli Parola
              </label>
              <div className="relative">
                <input
                  type="password"
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    clearError();
                    setFormValidation(null);
                  }}
                  placeholder="••••••••••••"
                  disabled={loading}
                  className="w-full bg-[#120D0A] border border-[#B8A484]/30 focus:border-amber-400 rounded-xl px-3.5 py-2.5 text-sm text-[#FAF7F0] placeholder-[#D4C5A9]/40 focus:outline-none focus:ring-1 focus:ring-amber-400/50 transition"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] font-bold text-xs uppercase tracking-wider rounded-xl shadow-xl transition duration-200 flex items-center justify-center gap-2 border border-[#B8A484]/30 disabled:opacity-50 cursor-pointer"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
                  <span>Kimlik Doğrulanıyor...</span>
                </>
              ) : (
                <>
                  <Shield className="w-4 h-4 text-amber-400" />
                  <span>Güvenli Oturum Aç</span>
                  <ArrowRight className="w-4 h-4 text-cream-200" />
                </>
              )}
            </button>
          </form>

          {/* HIZLI TEST KİMLİKLERİ (GELİŞTİRME & TEST AMAÇLI) */}
          <div className="pt-2 border-t border-[#B8A484]/15 space-y-2">
            <p className="text-[11px] text-[#D4C5A9]/70 font-semibold text-center uppercase tracking-wider">
              Test &amp; Demonstrasyon Hesapları
            </p>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => handleQuickFill("bilirkisi_uzman", "password123")}
                className="p-2 bg-[#2A2019] hover:bg-[#382B22] border border-[#B8A484]/20 rounded-xl text-left transition group"
              >
                <div className="text-[11px] font-bold text-cream-200 group-hover:text-amber-300">
                  Bilirkişi Uzman
                </div>
                <div className="text-[10px] text-[#D4C5A9]/60 font-mono">
                  bilirkisi_uzman
                </div>
                <span className="inline-block mt-1 text-[9px] px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-800/40">
                  Rol: EXAMINER
                </span>
              </button>

              <button
                type="button"
                onClick={() => handleQuickFill("admin", "password123")}
                className="p-2 bg-[#2A2019] hover:bg-[#382B22] border border-[#B8A484]/20 rounded-xl text-left transition group"
              >
                <div className="text-[11px] font-bold text-cream-200 group-hover:text-amber-300">
                  Baş Bilirkişi / Admin
                </div>
                <div className="text-[10px] text-[#D4C5A9]/60 font-mono">
                  admin
                </div>
                <span className="inline-block mt-1 text-[9px] px-1.5 py-0.5 rounded bg-amber-950/60 text-amber-400 border border-amber-800/40">
                  Rol: ADMIN
                </span>
              </button>
            </div>
          </div>

          {/* ADLİ GÜVENLİK BİLGİLENDİRMESİ */}
          <div className="bg-[#120D0A]/70 border border-[#B8A484]/15 rounded-xl p-3 text-[10px] text-[#D4C5A9]/70 leading-relaxed text-center">
            Bu sisteme yapılan tüm erişimler, 5237 sayılı TCK ve adli bilişim denetim standartları kapsamında kriptografik SHA-256 hash zinciriyle kayıt altına alınmaktadır.
          </div>
        </div>
      </main>

      {/* ALT BİLGİ */}
      <footer className="border-t border-[#B8A484]/15 bg-[#1C1510]/80 backdrop-blur-md px-6 py-3 text-center text-[11px] text-[#D4C5A9]/70">
        FHDSS © 2026 • TÜBİTAK Destekli Adli Belge ve El Yazısı Karar Destek Sistemi
      </footer>
    </div>
  );
}
