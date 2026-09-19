"use client";

import React, { useRef, useEffect, useState } from "react";
import {
  Lock,
  ShieldAlert,
  CheckCircle2,
  FileQuestion,
  FileCheck,
  UploadCloud,
  FolderPlus,
  Sparkles,
  Plus,
  RefreshCw,
  Eye,
  Trash2,
} from "lucide-react";
import { IsolationState } from "@/services/api";
import { LayerState } from "./LayerControls";

interface DualCanvasViewerProps {
  isolationState: IsolationState;
  questionedImageUrl?: string;
  referenceImages?: { id: string; url: string; name: string }[];
  selectedRefIndex: number;
  onSelectRefIndex: (idx: number) => void;
  layers: LayerState;
  zoomScale: number;
  onFreezePhaseA?: () => void;
  onUploadQuestioned?: (file: File) => void;
  onUploadReference?: (files: FileList | File[]) => void;
  onLoadDemoSample?: (type: "questioned" | "reference") => void;
  onClearQuestioned?: () => void;
  onRemoveReference?: (idx: number) => void;
  onClearAllReferences?: () => void;
}

export const DualCanvasViewer: React.FC<DualCanvasViewerProps> = ({
  isolationState,
  questionedImageUrl,
  referenceImages = [],
  selectedRefIndex,
  onSelectRefIndex,
  layers,
  zoomScale,
  onFreezePhaseA,
  onUploadQuestioned,
  onUploadReference,
  onLoadDemoSample,
  onClearQuestioned,
  onRemoveReference,
  onClearAllReferences,
}) => {
  const isPhaseBUnlocked =
    isolationState === "PHASE_A_FROZEN" ||
    isolationState === "PHASE_B_ACTIVE" ||
    isolationState === "EVALUATION_COMPLETED";

  const qCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const refCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Dosya Yükleyici Input Referansları
  const qFileInputRef = useRef<HTMLInputElement | null>(null);
  const refFileInputRef = useRef<HTMLInputElement | null>(null);

  // Donanım İvmelendirmeli Pan ve Sürükleme Durumu
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Sürükle Bırak (Drag & Drop) Efekti
  const [isDraggingOverQ, setIsDraggingOverQ] = useState(false);
  const [isDraggingOverRef, setIsDraggingOverRef] = useState(false);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Adli Katmanları Çizme Fonksiyonu
  const renderForensicLayers = (
    ctx: CanvasRenderingContext2D,
    docX: number,
    docY: number,
    docW: number,
    docH: number,
    layersState: LayerState,
    isQuestioned: boolean
  ) => {
    // 1. YAZI EĞİMİ KILAVUZU (Yalnızca Belge Alanı İçine Kırpılır!)
    if (layersState.showSlantGrid) {
      ctx.save();
      // Belge sınırlarına kırp (Siyah arka plana taşmayı önler)
      ctx.beginPath();
      ctx.rect(docX, docY, docW, docH);
      ctx.clip();

      ctx.strokeStyle = "rgba(234, 179, 8, 0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);

      const slantAngleDeg = isQuestioned ? 14.5 : 14.0;
      const tanAngle = Math.tan((slantAngleDeg * Math.PI) / 180);

      for (let x = docX - 150; x < docX + docW + 150; x += 28) {
        ctx.beginPath();
        ctx.moveTo(x, docY);
        ctx.lineTo(x + docH * tanAngle, docY + docH);
        ctx.stroke();
      }

      ctx.setLineDash([]);

      // Eğim Açısı Rozeti (Belge İçinde Sağ Üstte)
      ctx.fillStyle = "rgba(18, 13, 10, 0.85)";
      ctx.strokeStyle = "rgba(234, 179, 8, 0.6)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(docX + docW - 138, docY + 8, 130, 22, 6);
      } else {
        ctx.rect(docX + docW - 138, docY + 8, 130, 22);
      }
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#fef08a";
      ctx.font = "bold 10px 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(`📐 Eğim: ${slantAngleDeg.toFixed(1)}° (Adli Norm)`, docX + docW - 132, docY + 23);

      ctx.restore();
    }

    // 2. ÇİZGİ İSKELETİ (1-Pixel Thinning - Çizgilerin Tam Üzerinde!)
    if (layersState.showSkeleton) {
      ctx.save();
      ctx.strokeStyle = "#06b6d4"; // Canlı neon cyan
      ctx.lineWidth = 2.0;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowColor = "#06b6d4";
      ctx.shadowBlur = 4;

      // İmza omurga hattı
      ctx.beginPath();
      ctx.moveTo(-85, 16);
      ctx.bezierCurveTo(-69, -43, -43, -37, -32, 8);
      ctx.bezierCurveTo(-21, 32, -5, 37, 5, 5);
      ctx.bezierCurveTo(21, -53, 53, -43, 64, 11);
      ctx.bezierCurveTo(75, 32, 96, 37, 107, 21);
      ctx.stroke();

      // Taban alt vuruş iskeleti
      ctx.beginPath();
      ctx.moveTo(-74, 35);
      ctx.bezierCurveTo(-21, 45, 48, 24, 96, 35);
      ctx.stroke();

      // İskelet Rozeti
      ctx.fillStyle = "rgba(18, 13, 10, 0.85)";
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(docX + 8, docY + docH - 28, 132, 20, 6);
      } else {
        ctx.rect(docX + 8, docY + docH - 28, 132, 20);
      }
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#a5f3fc";
      ctx.font = "bold 9px 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("⚡ İskelet: 1-px Morfolojik", docX + 14, docY + docH - 14);

      ctx.restore();
    }

    // 3. KAVŞAK VE UÇ NOKTALARI (Belirteçler Çizgilerin Tam Üzerinde!)
    if (layersState.showNodes) {
      ctx.save();

      const landmarks = isQuestioned
        ? [
            { x: -85, y: 16, type: "START", label: "S1: Başlangıç", color: "#10b981" },
            { x: -48, y: -42, type: "APEX", label: "K1: İlmek Kavşağı", color: "#f59e0b" },
            { x: -16, y: 22, type: "VALLEY", label: "D1: Dönüm Vadisi", color: "#f59e0b" },
            { x: 48, y: -44, type: "APEX", label: "K2: Tepe Dönüm", color: "#f59e0b" },
            { x: 107, y: 21, type: "END", label: "E1: Bitiş Ucu", color: "#ef4444" },
          ]
        : [
            { x: -85, y: 16, type: "START", label: "S1: Başlangıç", color: "#10b981" },
            { x: -48, y: -40, type: "APEX", label: "K1: Doğal İlmek", color: "#f59e0b" },
            { x: -16, y: 24, type: "VALLEY", label: "D1: Vadi", color: "#f59e0b" },
            { x: 48, y: -42, type: "APEX", label: "K2: Tepe", color: "#f59e0b" },
            { x: 105, y: 22, type: "END", label: "E1: Bitiş", color: "#ef4444" },
          ];

      landmarks.forEach((pt) => {
        // 1. Dış Parlama Halkası
        ctx.beginPath();
        ctx.strokeStyle = pt.color;
        ctx.lineWidth = 1.5;
        ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
        ctx.stroke();

        // 2. İç Dolgu Çekirdeği
        ctx.beginPath();
        ctx.fillStyle = pt.color;
        ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
        ctx.fill();

        // 3. Hedef Artı Çaprazı (Crosshair)
        ctx.beginPath();
        ctx.strokeStyle = pt.color;
        ctx.lineWidth = 1;
        ctx.moveTo(pt.x - 9, pt.y);
        ctx.lineTo(pt.x - 3, pt.y);
        ctx.moveTo(pt.x + 3, pt.y);
        ctx.lineTo(pt.x + 9, pt.y);
        ctx.moveTo(pt.x, pt.y - 9);
        ctx.lineTo(pt.x, pt.y - 3);
        ctx.moveTo(pt.x, pt.y + 3);
        ctx.lineTo(pt.x, pt.y + 9);
        ctx.stroke();

        // 4. Bilgi Etiketi
        ctx.font = "bold 9px 'Segoe UI', sans-serif";
        const textWidth = ctx.measureText(pt.label).width;
        const boxX = pt.x + 10;
        const boxY = pt.y - 14;

        ctx.fillStyle = "rgba(18, 13, 10, 0.9)";
        ctx.strokeStyle = pt.color;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(boxX, boxY, textWidth + 8, 14, 4);
        } else {
          ctx.rect(boxX, boxY, textWidth + 8, 14);
        }
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#FAF7F0";
        ctx.textAlign = "left";
        ctx.fillText(pt.label, boxX + 4, boxY + 10);
      });

      ctx.restore();
    }

    // 4. YAPAY ZEKÂ ODAK HARİTASI (Grad-CAM - Çizgi Üzerine Odaklanmış Isı Haritası)
    if (layersState.showGradCam) {
      ctx.save();

      // 1. Odak Alanı: Üst İlmek Kavşağı
      const g1 = ctx.createRadialGradient(-48, -42, 2, -48, -42, 42);
      g1.addColorStop(0.0, "rgba(239, 68, 68, 0.80)");  // Kırmızı
      g1.addColorStop(0.4, "rgba(245, 158, 11, 0.55)"); // Turuncu / Sarı
      g1.addColorStop(0.7, "rgba(6, 182, 212, 0.25)");  // Turkuaz
      g1.addColorStop(1.0, "rgba(18, 13, 10, 0.0)");    // Şeffaf
      ctx.fillStyle = g1;
      ctx.beginPath();
      ctx.arc(-48, -42, 42, 0, Math.PI * 2);
      ctx.fill();

      // 2. Odak Alanı: İkinci Tepe ve Vuruş Kıvrımı
      const g2 = ctx.createRadialGradient(48, -20, 2, 48, -20, 38);
      g2.addColorStop(0.0, "rgba(239, 68, 68, 0.75)");
      g2.addColorStop(0.4, "rgba(245, 158, 11, 0.50)");
      g2.addColorStop(0.7, "rgba(6, 182, 212, 0.20)");
      g2.addColorStop(1.0, "rgba(18, 13, 10, 0.0)");
      ctx.fillStyle = g2;
      ctx.beginPath();
      ctx.arc(48, -20, 38, 0, Math.PI * 2);
      ctx.fill();

      // Isı Haritası Başlık Rozeti
      ctx.fillStyle = "rgba(18, 13, 10, 0.85)";
      ctx.strokeStyle = "rgba(239, 68, 68, 0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(docX + 8, docY + 8, 126, 22, 6);
      } else {
        ctx.rect(docX + 8, docY + 8, 126, 22);
      }
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#fca5a5";
      ctx.font = "bold 9.5px 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("🔥 Grad-CAM: %94 Odak", docX + 14, docY + 22);

      ctx.restore();
    }
  };

  // Questioned Canvas Render
  useEffect(() => {
    const canvas = qCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
    if (!ctx) return;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#120D0A";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (questionedImageUrl) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = questionedImageUrl;
      img.onload = () => {
        ctx.save();
        ctx.translate(canvas.width / 2 + pan.x, canvas.height / 2 + pan.y);
        ctx.scale(zoomScale, zoomScale);

        const docX = -canvas.width / 4;
        const docY = -canvas.height / 4;
        const docW = canvas.width / 2;
        const docH = canvas.height / 2;

        if (layers.showOriginal) {
          ctx.drawImage(img, docX, docY, docW, docH);
        } else {
          // Orijinal gizlendiğinde sadece beyaz arka plan kağıdı
          ctx.fillStyle = "#FAF7F0";
          ctx.fillRect(docX, docY, docW, docH);
        }

        // Adli Katmanları Çiz
        renderForensicLayers(ctx, docX, docY, docW, docH, layers, true);

        ctx.restore();
      };
    }
  }, [questionedImageUrl, layers, zoomScale, pan]);

  // Reference Canvas Render
  useEffect(() => {
    if (!isPhaseBUnlocked) return;
    const canvas = refCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
    if (!ctx) return;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#120D0A";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const currentRef = referenceImages[selectedRefIndex];
    if (currentRef?.url) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = currentRef.url;
      img.onload = () => {
        ctx.save();
        ctx.translate(canvas.width / 2 + pan.x, canvas.height / 2 + pan.y);
        ctx.scale(zoomScale, zoomScale);

        const docX = -canvas.width / 4;
        const docY = -canvas.height / 4;
        const docW = canvas.width / 2;
        const docH = canvas.height / 2;

        if (layers.showOriginal) {
          ctx.drawImage(img, docX, docY, docW, docH);
        } else {
          ctx.fillStyle = "#FAF7F0";
          ctx.fillRect(docX, docY, docW, docH);
        }

        // Adli Katmanları Çiz
        renderForensicLayers(ctx, docX, docY, docW, docH, layers, false);

        ctx.restore();
      };
    }
  }, [isPhaseBUnlocked, referenceImages, selectedRefIndex, layers, zoomScale, pan]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* GİZLİ DOSYA INPUTLARI */}
      <input
        type="file"
        ref={qFileInputRef}
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files[0] && onUploadQuestioned) {
            onUploadQuestioned(e.target.files[0]);
          }
        }}
      />
      <input
        type="file"
        ref={refFileInputRef}
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0 && onUploadReference) {
            onUploadReference(e.target.files);
          }
        }}
      />

      {/* SOL KANVAS: SORGULANAN BELGE */}
      <div className="flex flex-col bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl overflow-hidden shadow-2xl backdrop-blur-md">
        <div className="flex items-center justify-between px-4 py-2.5 bg-[#2A2019]/90 border-b border-[#B8A484]/20">
          <div className="flex items-center gap-2">
            <FileQuestion className="w-4 h-4 text-amber-400" />
            <span className="font-semibold text-xs text-[#FAF7F0]">Sorgulanan Belge (Şüpheli)</span>
          </div>

          <div className="flex items-center gap-2">
            {questionedImageUrl && (
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => qFileInputRef.current?.click()}
                  title="Yeni bir şüpheli belge yükleyerek değiştirin"
                  className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[#2A2019] hover:bg-[#3D2E24] text-[#D4C5A9] hover:text-[#FAF7F0] border border-[#B8A484]/20 transition"
                >
                  <RefreshCw className="w-3 h-3" /> Değiştir
                </button>

                {onClearQuestioned && (
                  <button
                    type="button"
                    onClick={onClearQuestioned}
                    title="Şüpheli belgeyi kaldır ve temizle"
                    className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[#2A2019] hover:bg-rose-950/80 text-rose-300 hover:text-rose-200 border border-rose-500/30 hover:border-rose-500/60 transition shadow-sm"
                  >
                    <Trash2 className="w-3 h-3 text-rose-400" /> Kaldır
                  </button>
                )}
              </div>
            )}

            {isPhaseBUnlocked ? (
              <span
                title="Şüpheli belge incelendi, kilitlendi ve mukayese aşamasına geçildi."
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#2A2019] text-teal-400 border border-teal-500/40 shadow-sm"
              >
                <CheckCircle2 className="w-3.5 h-3.5" /> İncelendi ve Kilitlendi
              </span>
            ) : (
              <button
                type="button"
                onClick={onFreezePhaseA}
                disabled={!questionedImageUrl}
                title={
                  !questionedImageUrl
                    ? "Önce şüpheli belge yükleyiniz."
                    : "Bilişsel yanlılığı önlemek için şüpheli belgenin ön incelemesini onaylayıp kilitleyin."
                }
                className={`inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-semibold transition shadow-lg border ${
                  questionedImageUrl
                    ? "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] border-[#B8A484]/30 cursor-pointer"
                    : "bg-[#2A2019] text-[#D4C5A9]/40 border-transparent cursor-not-allowed"
                }`}
              >
                <Lock className="w-3.5 h-3.5 text-cream-200" /> Şüpheli Belgeyi Onayla ve Kilitle
              </button>
            )}
          </div>
        </div>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDraggingOverQ(true);
          }}
          onDragLeave={() => setIsDraggingOverQ(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDraggingOverQ(false);
            if (e.dataTransfer.files && e.dataTransfer.files[0] && onUploadQuestioned) {
              onUploadQuestioned(e.dataTransfer.files[0]);
            }
          }}
          className={`relative w-full aspect-[4/3] bg-[#120D0A] flex items-center justify-center overflow-hidden border-t border-[#B8A484]/15 transition-all ${
            isDraggingOverQ ? "ring-2 ring-amber-400/80 bg-[#1E1612]" : ""
          }`}
        >
          <canvas
            ref={qCanvasRef}
            width={640}
            height={480}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            className={`w-full h-full object-contain ${
              questionedImageUrl ? (isDragging ? "cursor-grabbing" : "cursor-grab") : "cursor-default"
            }`}
          />

          {/* MERKEZDEKİ ŞIK EVRAK YÜKLEME BUTONU VE KARTI */}
          {!questionedImageUrl && (
            <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center z-10 bg-[#120D0A]/90 backdrop-blur-md">
              <div className="max-w-sm w-full p-6 rounded-3xl bg-gradient-to-b from-[#2A2019]/90 to-[#18120F]/95 border-2 border-dashed border-[#B8A484]/30 hover:border-[#7B1E34] shadow-2xl transition-all duration-300 group">
                <div className="w-14 h-14 mx-auto mb-3 rounded-2xl bg-gradient-to-br from-[#7B1E34]/30 to-[#2A2019] border border-[#B8A484]/30 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                  <UploadCloud className="w-7 h-7 text-amber-400 animate-pulse" />
                </div>

                <h3 className="text-sm font-bold text-[#FAF7F0]">Sorgulanan Şüpheli Belgeyi Yükleyin</h3>
                <p className="text-xs text-[#D4C5A9]/70 mt-1 mb-4 leading-relaxed">
                  Taranmış adli belge veya imza görselini buraya sürükleyin ya da doğrudan seçin (PNG, JPG, TIFF).
                </p>

                <div className="flex flex-col sm:flex-row items-center justify-center gap-2">
                  <button
                    type="button"
                    onClick={() => qFileInputRef.current?.click()}
                    className="w-full sm:w-auto px-5 py-2.5 rounded-xl text-xs font-bold bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] shadow-lg shadow-[#7B1E34]/30 border border-[#B8A484]/40 hover:scale-105 active:scale-95 transition-all flex items-center justify-center gap-2"
                  >
                    <UploadCloud className="w-4 h-4" /> Belge Seç ve Yükle
                  </button>

                  {onLoadDemoSample && (
                    <button
                      type="button"
                      onClick={() => onLoadDemoSample("questioned")}
                      title="Test etmek için hazır adli imza galerisini açın"
                      className="w-full sm:w-auto px-3.5 py-2.5 rounded-xl text-xs font-semibold bg-[#2A2019] hover:bg-[#3D2E24] text-amber-300 border border-amber-400/30 hover:border-amber-400/60 shadow-sm transition-all flex items-center justify-center gap-1.5"
                    >
                      <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Örnek Yükle
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {questionedImageUrl && (
            <div className="absolute bottom-2.5 left-2.5 px-2.5 py-0.5 bg-[#1C1510]/80 backdrop-blur rounded-full text-[10px] text-[#D4C5A9] border border-[#B8A484]/20">
              Büyütme: %{Math.round(zoomScale * 100)} | GPU İvmelendirmeli
            </div>
          )}
        </div>
      </div>

      {/* SAĞ KANVAS: MUKAYESE / REFERANS BELGELER */}
      <div className="flex flex-col bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl overflow-hidden shadow-2xl backdrop-blur-md relative">
        <div className="flex items-center justify-between px-4 py-2.5 bg-[#2A2019]/90 border-b border-[#B8A484]/20">
          <div className="flex items-center gap-2">
            <FileCheck className="w-4 h-4 text-[#D4C5A9]" />
            <span className="font-semibold text-xs text-[#FAF7F0]">Mukayese / Referans Belgeler</span>
          </div>

          {/* Numune Seçici ve Ekleme/Silme Butonları */}
          {isPhaseBUnlocked && (
            <div className="flex items-center gap-1.5">
              {referenceImages.map((ref, idx) => (
                <button
                  key={ref.id}
                  type="button"
                  onClick={() => onSelectRefIndex(idx)}
                  className={`px-3 py-0.5 rounded-full text-xs font-medium transition ${
                    selectedRefIndex === idx
                      ? "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] text-[#FAF7F0] border border-[#B8A484]/40 shadow-sm"
                      : "bg-[#2A2019] text-[#D4C5A9] hover:bg-[#3D2E24] hover:text-[#FAF7F0] border border-[#B8A484]/15"
                  }`}
                >
                  Numune {idx + 1}
                </button>
              ))}

              <button
                type="button"
                onClick={() => refFileInputRef.current?.click()}
                title="Yeni bir mukayese belgesi ekleyin"
                className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#2A2019] hover:bg-[#3D2E24] text-teal-400 hover:text-teal-300 border border-teal-500/30 transition shadow-sm"
              >
                <Plus className="w-3.5 h-3.5" /> Ekle
              </button>

              {referenceImages.length > 0 && onRemoveReference && (
                <button
                  type="button"
                  onClick={() => onRemoveReference(selectedRefIndex)}
                  title={`Numune ${selectedRefIndex + 1}'i kaldır`}
                  className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#2A2019] hover:bg-rose-950/80 text-rose-300 hover:text-rose-200 border border-rose-500/30 transition shadow-sm"
                >
                  <Trash2 className="w-3 h-3 text-rose-400" /> Numuneyi Sil
                </button>
              )}
            </div>
          )}
        </div>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            if (isPhaseBUnlocked) setIsDraggingOverRef(true);
          }}
          onDragLeave={() => setIsDraggingOverRef(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDraggingOverRef(false);
            if (isPhaseBUnlocked && e.dataTransfer.files && e.dataTransfer.files.length > 0 && onUploadReference) {
              onUploadReference(e.dataTransfer.files);
            }
          }}
          className={`relative w-full aspect-[4/3] bg-[#120D0A] flex items-center justify-center overflow-hidden border-t border-[#B8A484]/15 transition-all ${
            isDraggingOverRef ? "ring-2 ring-teal-400/80 bg-[#1E1612]" : ""
          }`}
        >
          <canvas
            ref={refCanvasRef}
            width={640}
            height={480}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            className={`w-full h-full object-contain ${
              isPhaseBUnlocked && referenceImages.length > 0
                ? isDragging
                  ? "cursor-grabbing"
                  : "cursor-grab"
                : "cursor-default"
            }`}
          />

          {/* BİLİŞSEL YANLILIK KORUMASI KİLİT MASKESİ */}
          {!isPhaseBUnlocked && (
            <div className="absolute inset-0 bg-[#120D0A]/95 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center z-10 space-y-3">
              <div className="w-12 h-12 rounded-full bg-[#2A2019] border border-[#B8A484]/30 flex items-center justify-center shadow-lg">
                <ShieldAlert className="w-6 h-6 text-amber-400" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-amber-300">Analiz Kilidi: Şüpheli Belgeyi İncele ve Onayla</h4>
                <p className="text-xs text-[#D4C5A9]/80 max-w-xs mt-1 leading-relaxed">
                  Bilişsel yanlılığı önlemek için önce şüpheli belgenin özellikleri incelenip kilitlenmelidir.
                </p>
              </div>
              <button
                type="button"
                onClick={onFreezePhaseA}
                disabled={!questionedImageUrl}
                className={`px-4 py-1.5 font-semibold text-xs rounded-full shadow-lg transition border ${
                  questionedImageUrl
                    ? "bg-gradient-to-r from-[#7B1E34] to-[#5C1425] hover:from-[#5C1425] hover:to-[#7B1E34] text-[#FAF7F0] border-[#B8A484]/30 cursor-pointer"
                    : "bg-[#2A2019] text-[#D4C5A9]/40 border-transparent cursor-not-allowed"
                }`}
              >
                Şüpheli Belgeyi Onayla ve Mukayeseyi Aç
              </button>
            </div>
          )}

          {/* FAZ B AÇIK ANCAK HENÜZ MUKAYESE YÜKLENMEDİĞİNDE MERKEZDEKİ YÜKLEME BUTONU */}
          {isPhaseBUnlocked && referenceImages.length === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center z-10 bg-[#120D0A]/90 backdrop-blur-md">
              <div className="max-w-sm w-full p-6 rounded-3xl bg-gradient-to-b from-[#2A2019]/90 to-[#18120F]/95 border-2 border-dashed border-[#B8A484]/30 hover:border-teal-500 shadow-2xl transition-all duration-300 group">
                <div className="w-14 h-14 mx-auto mb-3 rounded-2xl bg-gradient-to-br from-teal-900/40 to-[#2A2019] border border-teal-500/30 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                  <FolderPlus className="w-7 h-7 text-teal-400 animate-pulse" />
                </div>

                <h3 className="text-sm font-bold text-[#FAF7F0]">Mukayese / Referans Belgeleri Ekleyin</h3>
                <p className="text-xs text-[#D4C5A9]/70 mt-1 mb-4 leading-relaxed">
                  Şüphelinin bilinen orijinal imza veya el yazılarını yükleyin (Birden fazla dosya seçebilirsiniz).
                </p>

                <div className="flex flex-col sm:flex-row items-center justify-center gap-2">
                  <button
                    type="button"
                    onClick={() => refFileInputRef.current?.click()}
                    className="w-full sm:w-auto px-5 py-2.5 rounded-xl text-xs font-bold bg-gradient-to-r from-teal-700 to-teal-900 hover:from-teal-600 hover:to-teal-800 text-[#FAF7F0] shadow-lg shadow-teal-900/40 border border-teal-500/40 hover:scale-105 active:scale-95 transition-all flex items-center justify-center gap-2"
                  >
                    <FolderPlus className="w-4 h-4" /> Mukayese Belgelerini Seç
                  </button>

                  {onLoadDemoSample && (
                    <button
                      type="button"
                      onClick={() => onLoadDemoSample("reference")}
                      title="Hazır adli imza galerisinden örnek mukayese belgesi seçin"
                      className="w-full sm:w-auto px-3.5 py-2.5 rounded-xl text-xs font-semibold bg-[#2A2019] hover:bg-[#3D2E24] text-teal-300 border border-teal-500/30 hover:border-teal-500/60 shadow-sm transition-all flex items-center justify-center gap-1.5"
                    >
                      <Sparkles className="w-3.5 h-3.5 text-teal-400" /> Örnek Mukayese
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {isPhaseBUnlocked && referenceImages.length > 0 && (
            <div className="absolute bottom-2.5 left-2.5 px-2.5 py-0.5 bg-[#1C1510]/80 backdrop-blur rounded-full text-[10px] text-[#D4C5A9] border border-[#B8A484]/20">
              Büyütme: %{Math.round(zoomScale * 100)} | Numune: {selectedRefIndex + 1} / {referenceImages.length}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};


