"use client";

import React, { useState } from "react";
import {
  Fingerprint,
  ShieldCheck,
  Cpu,
  User,
  LogOut,
  Shield,
  AlertTriangle,
  RotateCw,
} from "lucide-react";
import { CaseSelectionDashboard } from "@/components/cases/CaseSelectionDashboard";
import { ExistingCasesPanel } from "@/components/cases/ExistingCasesPanel";
import { CreateCasePanel } from "@/components/cases/CreateCasePanel";
import { useAuth } from "@/context/AuthContext";

type ViewState = "SELECTION" | "EXISTING_CASES" | "NEW_CASE";

export default function HomePage() {
  const [viewState, setViewState] = useState<ViewState>("SELECTION");
  const { user, logout, loading, sessionState, error, retrySessionVerification } = useAuth();

  if (loading || sessionState === "verifying" || sessionState === "initializing") {
    return (
      <div className="fixed inset-0 w-full h-full min-h-screen bg-[#18120F] flex flex-col items-center justify-center space-y-4">
        <div className="w-10 h-10 rounded-full border-2 border-burgundy-600 border-t-transparent animate-spin" />
        <p className="text-xs text-[#D4C5A9] font-medium">Güvenli adli oturum doğrulanıyor...</p>
      </div>
    );
  }

  if (sessionState === "error" && !user) {
    return (
      <div className="fixed inset-0 w-full h-full min-h-screen bg-[#18120F] flex flex-col items-center justify-center p-4">
        <div className="max-w-md w-full bg-[#1C1510] border border-rose-800/60 rounded-2xl p-6 text-center space-y-4 shadow-2xl">
          <div className="w-12 h-12 rounded-full bg-rose-950/80 border border-rose-700/50 flex items-center justify-center mx-auto text-rose-300">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h2 className="text-sm font-bold text-[#FAF7F0] tracking-wide">
            Adli Oturum Doğrulanamadı
          </h2>
          <p className="text-xs text-[#D4C5A9]/80 leading-relaxed">
            {error || "Sunucuyla bağlantı kurulamadı veya oturum zaman aşımına uğradı."}
          </p>
          <div className="flex items-center justify-center gap-3 pt-2">
            <button
              type="button"
              onClick={() => retrySessionVerification()}
              className="px-4 py-2 bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] text-xs font-semibold rounded-lg transition shadow-md flex items-center gap-1.5"
            >
              <RotateCw className="w-3.5 h-3.5" />
              Tekrar Dene
            </button>
            <button
              type="button"
              onClick={logout}
              className="px-4 py-2 bg-[#2A2019] hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] text-xs font-semibold rounded-lg border border-[#B8A484]/20 transition"
            >
              Giriş Ekranına Dön
            </button>
          </div>
        </div>
      </div>
    );
  }

  const getRoleLabel = (role?: string) => {
    switch (role) {
      case "ADMIN":
        return { label: "Sistem Yöneticisi / Baş Bilirkişi", color: "bg-amber-950/70 text-amber-300 border-amber-600/40" };
      case "SENIOR_REVIEWER":
        return { label: "Kıdemli İncelemeci", color: "bg-purple-950/70 text-purple-300 border-purple-600/40" };
      default:
        return { label: "Bilirkişi Adli Uzman", color: "bg-emerald-950/70 text-emerald-300 border-emerald-600/40" };
    }
  };

  const roleInfo = getRoleLabel(user?.role);

  return (
    <div className='fixed inset-0 w-full h-full min-h-screen bg-[url("/bg-forensic-dashboard.jpg")] bg-cover bg-center bg-no-repeat overflow-y-auto flex flex-col justify-between font-sans selection:bg-[#7B1E34] selection:text-[#FAF7F0]'>
      {/* ÜST BAŞLIK BARI */}
      <header className="border-b border-[#B8A484]/20 bg-[#1C1510]/90 backdrop-blur-md px-4 md:px-8 py-3 flex flex-wrap items-center justify-between gap-3 shadow-xl sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-burgundy-700/40 border border-burgundy-600/50 flex items-center justify-center shadow-inner">
            <Fingerprint className="w-5 h-5 text-cream-200" />
          </div>
          <div>
            <h1 className="text-base font-extrabold tracking-wider text-[#FAF7F0] flex items-center gap-2">
              FHDSS WORKSTATION
              <span className="text-[10px] font-normal px-2 py-0.5 rounded-full bg-burgundy-700/60 border border-burgundy-600 text-cream-200">
                v1.0 Adli Sürüm
              </span>
            </h1>
            <p className="text-[11px] text-[#D4C5A9]/75 font-medium">
              Adli Belge ve El Yazısı Karar Destek Platformu
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Oturum Açan Kullanıcı ve Rol Rozeti */}
          {user && (
            <div className="flex items-center gap-2.5 bg-[#2A2019] border border-[#B8A484]/20 rounded-xl px-3 py-1.5 shadow-sm">
              <div className="w-7 h-7 rounded-lg bg-burgundy-900/60 border border-burgundy-600/40 flex items-center justify-center text-[#FAF7F0]">
                <User className="w-3.5 h-3.5 text-amber-300" />
              </div>
              <div className="text-left hidden sm:block">
                <div className="text-xs font-bold text-[#FAF7F0] leading-tight">
                  {user.full_name || user.username}
                </div>
                <div className="flex items-center gap-1 mt-0.5">
                  <span className={`text-[9px] px-1.5 py-0.2 rounded border font-semibold ${roleInfo.color}`}>
                    {roleInfo.label}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Admin Yönetim Paneli Linki (Yalnızca ADMIN Rolü) */}
          {user?.role === "ADMIN" && (
            <a
              href="/admin"
              className="px-3 py-1.5 bg-amber-950/70 hover:bg-amber-900/80 text-amber-300 border border-amber-600/40 rounded-xl text-xs font-bold transition flex items-center gap-1.5 shadow-sm"
              title="Sistem ve Kullanıcı Yönetim Paneli"
            >
              <Shield className="w-3.5 h-3.5 text-amber-400" />
              <span className="hidden sm:inline">Yönetim Paneli</span>
            </a>
          )}

          {/* Çıkış Yap Butonu */}
          <button
            type="button"
            onClick={logout}
            title="Oturumu Kapat"
            className="px-3 py-1.5 bg-[#2A2019] hover:bg-rose-950/60 text-[#D4C5A9] hover:text-rose-300 border border-[#B8A484]/20 hover:border-rose-700/40 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-sm"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span className="hidden md:inline">Çıkış</span>
          </button>
        </div>
      </header>

      {/* ANA GÖVDE: ORTA ALANDA HAFİF KARARTMA FİLTRESİ VE TEMİZ FORM KONUMLANDIRMA */}
      <main className="relative z-10 flex flex-col items-center justify-center p-4 flex-1">
        {viewState === "SELECTION" && (
          <CaseSelectionDashboard
            onSelectExisting={() => setViewState("EXISTING_CASES")}
            onSelectNew={() => setViewState("NEW_CASE")}
          />
        )}

        {viewState === "EXISTING_CASES" && (
          <ExistingCasesPanel
            onBack={() => setViewState("SELECTION")}
            onSelectNew={() => setViewState("NEW_CASE")}
          />
        )}

        {viewState === "NEW_CASE" && (
          <CreateCasePanel
            onBack={() => setViewState("SELECTION")}
          />
        )}
      </main>

      {/* ALT BİLGİ */}
      <footer className="border-t border-[#B8A484]/15 bg-[#1C1510]/80 backdrop-blur-md px-6 py-3 text-center text-[11px] text-[#D4C5A9]/70">
        FHDSS © 2026 • TÜBİTAK Destekli Adli Belge ve El Yazısı Karar Destek Sistemi
      </footer>
    </div>
  );
}
