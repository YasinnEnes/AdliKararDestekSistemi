import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/context/AuthContext";
import { CaseProvider } from "@/context/CaseContext";

export const metadata: Metadata = {
  title: "FHDSS - Adli Belge ve El Yazısı Karar Destek Sistemi",
  description: "Yapay Zekâ Tabanlı, Modüler, Açıklanabilir ve Denetlenebilir Adli Belge & İmza İnceleme İstasyonu",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="tr" className="dark">
      <body className="bg-[#18120F] text-[#FAF7F0] min-h-screen antialiased selection:bg-[#7B1E34] selection:text-[#FAF7F0] relative">
        {/* Global Koyu Ahşap / Espresso Doku Katmanı */}
        <div className="bg-[url('/bg-forensic-espresso.jpg')] bg-cover bg-center bg-no-repeat fixed inset-0 w-full h-full min-h-screen z-0 pointer-events-none opacity-40" />
        <div className="fixed inset-0 bg-gradient-to-b from-[#18120F]/90 via-[#1C1510]/85 to-[#120D0A]/95 backdrop-blur-[1px] z-0 pointer-events-none" />

        {/* Kimlik Doğrulama ve Vaka Durum Sağlayıcıları */}
        <AuthProvider>
          <CaseProvider>
            <div className="relative z-10 min-h-screen flex flex-col">
              {children}
            </div>
          </CaseProvider>
        </AuthProvider>
      </body>
    </html>
  );
}

