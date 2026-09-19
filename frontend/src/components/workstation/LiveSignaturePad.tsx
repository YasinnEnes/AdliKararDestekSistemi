"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import {
  RotateCcw,
  Trash2,
  Download,
  Play,
  Activity,
  PenTool,
  CheckCircle2,
  AlertTriangle,
  Info,
  ShieldCheck,
} from "lucide-react";
import {
  BrowserPointerProvider,
  MockSignatureProvider,
  RawBiometricPoint,
  BiometricDynamicFeatures,
} from "@/services/biometricInput";

interface LiveSignaturePadProps {
  onAnalyze?: (points: RawBiometricPoint[], features: BiometricDynamicFeatures) => void;
  onSaveImage?: (dataUrl: string, points: RawBiometricPoint[]) => void;
}

export const LiveSignaturePad: React.FC<LiveSignaturePadProps> = ({
  onAnalyze,
  onSaveImage,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const providerRef = useRef<BrowserPointerProvider>(new BrowserPointerProvider());

  const [isDrawing, setIsDrawing] = useState(false);
  const [strokeHistory, setStrokeHistory] = useState<RawBiometricPoint[][]>([]);
  const [currentStroke, setCurrentStroke] = useState<RawBiometricPoint[]>([]);
  const [activePointerType, setActivePointerType] = useState<string>("mouse");
  const [livePressure, setLivePressure] = useState<number>(0);
  const [features, setFeatures] = useState<BiometricDynamicFeatures | null>(null);
  const [isMockActive, setIsMockActive] = useState(false);

  // Canvas Boyutlandırma ve High-DPI Ayarı
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    ctx.scale(dpr, dpr);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#1A1A1A";

    // Arka planı temiz adli kağıt tonu yap
    ctx.fillStyle = "#FAF7F0";
    ctx.fillRect(0, 0, rect.width, rect.height);
  }, []);

  // Çizgiyi Yeniden Çiz (Undo ve Redraw için)
  const redrawCanvas = useCallback((strokes: RawBiometricPoint[][]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#FAF7F0";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();

    strokes.forEach((stroke) => {
      if (stroke.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);

      for (let i = 1; i < stroke.length; i++) {
        const p0 = stroke[i - 1];
        const p1 = stroke[i];
        const midX = (p0.x + p1.x) / 2;
        const midY = (p0.y + p1.y) / 2;

        const pressure = p1.pressure > 0 ? p1.pressure : 0.5;
        ctx.lineWidth = Math.max(pressure * 4.5, 1.5);
        ctx.quadraticCurveTo(p0.x, p0.y, midX, midY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(midX, midY);
      }
      ctx.stroke();
    });
  }, []);

  // Pointer Down (Kalem / Dokunmatik / Fare Başlangıcı)
  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.setPointerCapture(e.pointerId);

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const pressure = e.pressure > 0 ? e.pressure : 0.5;
    const t = performance.now();

    setIsDrawing(true);
    setActivePointerType(e.pointerType || "mouse");
    setLivePressure(pressure);

    const firstPoint: RawBiometricPoint = {
      x,
      y,
      t,
      pressure,
      is_pen_down: true,
      pointerType: e.pointerType,
    };

    providerRef.current.recordPoint(firstPoint);
    setCurrentStroke([firstPoint]);
  };

  // Pointer Move (Dinamik Yörünge ve Basınç Kaydı)
  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const pressure = e.pressure > 0 ? e.pressure : 0.5;
    const t = performance.now();

    setLivePressure(pressure);

    const newPoint: RawBiometricPoint = {
      x,
      y,
      t,
      pressure,
      is_pen_down: true,
      pointerType: e.pointerType,
    };

    providerRef.current.recordPoint(newPoint);
    const updated = [...currentStroke, newPoint];
    setCurrentStroke(updated);

    // Canlı Çizim
    if (updated.length > 1) {
      const p0 = updated[updated.length - 2];
      const p1 = updated[updated.length - 1];

      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineWidth = Math.max(pressure * 4.5, 1.5);
      ctx.strokeStyle = "#1A1A1A";
      ctx.lineTo(p1.x, p1.y);
      ctx.stroke();
    }
  };

  // Pointer Up (Vuruş Tamamlandı)
  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    setIsDrawing(false);
    setLivePressure(0);

    const canvas = canvasRef.current;
    if (canvas) {
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
    }

    if (currentStroke.length > 0) {
      const newHistory = [...strokeHistory, currentStroke];
      setStrokeHistory(newHistory);
      setCurrentStroke([]);

      // Dinamik öznitelikleri anlık hesapla
      const f = providerRef.current.computeFeatures();
      setFeatures(f);
    }
  };

  // Temizle
  const handleClear = () => {
    providerRef.current.clear();
    setStrokeHistory([]);
    setCurrentStroke([]);
    setFeatures(null);
    setLivePressure(0);
    setIsMockActive(false);

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#FAF7F0";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  // Geri Al (Undo Last Stroke)
  const handleUndo = () => {
    if (strokeHistory.length === 0) return;
    const newHistory = strokeHistory.slice(0, -1);
    setStrokeHistory(newHistory);

    providerRef.current.clear();
    newHistory.forEach((stroke) => {
      stroke.forEach((p) => providerRef.current.recordPoint(p));
    });

    redrawCanvas(newHistory);
    setFeatures(providerRef.current.computeFeatures());
  };

  // Mock İmzayı Yükle (Tablet Donanımı Olmayan Geliştiriciler İçin)
  const handleLoadMock = () => {
    handleClear();
    setIsMockActive(true);
    const mockProvider = new MockSignatureProvider();
    const mockPoints = mockProvider.getPoints();

    providerRef.current.clear();
    mockPoints.forEach((p) => providerRef.current.recordPoint(p));

    // Tek vuruş halinde geçmişe ekle
    setStrokeHistory([mockPoints]);
    redrawCanvas([mockPoints]);

    const f = providerRef.current.computeFeatures();
    setFeatures(f);
    setActivePointerType("mock_stylus");
  };

  // Analizi Başlat
  const handleTriggerAnalysis = () => {
    const pts = providerRef.current.getPoints();
    const f = features || providerRef.current.computeFeatures();
    if (onAnalyze && pts.length > 0) {
      onAnalyze(pts, f);
    }
  };

  // Görsel Olarak Kaydet / Numune Yap
  const handleSaveImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dataUrl = canvas.toDataURL("image/png");
    const pts = providerRef.current.getPoints();
    if (onSaveImage) {
      onSaveImage(dataUrl, pts);
    } else {
      // Doğrudan indir
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = `fhdss_canli_imza_${Date.now()}.png`;
      a.click();
    }
  };

  return (
    <div className="bg-[#1C1510] border border-[#7B1E34]/40 rounded-xl p-5 shadow-2xl flex flex-col gap-4">
      {/* Üst Başlık & Donanım Durumu */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#FAF7F0]/10">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-lg bg-[#7B1E34]/20 border border-[#7B1E34]/50 flex items-center justify-center text-[#E5A93C]">
            <PenTool className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-[#FAF7F0] flex items-center gap-2">
              Canlı Biyometrik İmza Padi (Hardware Abstraction Layer)
            </h3>
            <p className="text-xs text-[#FAF7F0]/60">
              Wacom, Stylus Kalem, iPad/Tablet ve Dokunmatik Basınç Sensör Entegrasyonu
            </p>
          </div>
        </div>

        {/* Aktif Giriş Cihazı Göstergesi */}
        <div className="flex items-center gap-2 bg-[#120D0A] px-3 py-1.5 rounded-lg border border-[#FAF7F0]/10 text-xs">
          <span className="text-[#FAF7F0]/60">Giriş Türü:</span>
          <span className="font-mono text-[#E5A93C] uppercase font-medium flex items-center gap-1">
            <Activity className="w-3.5 h-3.5 text-[#2E7D32]" />
            {activePointerType === "pen"
              ? "STYLUS / KALEM (DONANIM)"
              : isMockActive
              ? "MOCK BİYOMETRİ (TEST)"
              : activePointerType}
          </span>
          <div className="h-3 w-px bg-[#FAF7F0]/20 mx-1" />
          <span className="text-[#FAF7F0]/60">Basınç:</span>
          <span className="font-mono text-[#FAF7F0]">
            {(livePressure * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Canvas Çizim Alanı */}
      <div className="relative rounded-lg overflow-hidden border-2 border-dashed border-[#7B1E34]/30 bg-[#FAF7F0] shadow-inner">
        <canvas
          ref={canvasRef}
          className="w-full h-72 touch-none cursor-crosshair block"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
        />

        {/* İmza Kılavuz Çizgisi */}
        <div className="absolute left-8 right-8 bottom-12 border-b border-[#1A1A1A]/20 pointer-events-none flex justify-between text-[10px] text-[#1A1A1A]/40 uppercase tracking-widest font-mono pt-1">
          <span>Keşideci / Yetkili İmza Hattı</span>
          <span>Basınç & Kinematik Kayıt Aktif</span>
        </div>
      </div>

      {/* Dinamik Kinematik Öznitelik Göstergeleri */}
      {features && features.totalPoints > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2.5 bg-[#120D0A]/70 p-3 rounded-lg border border-[#FAF7F0]/5 text-xs">
          <div>
            <div className="text-[#FAF7F0]/50 text-[10px]">Toplam Süre</div>
            <div className="font-mono font-semibold text-[#FAF7F0]">
              {features.durationSec.toFixed(2)} sn
            </div>
          </div>
          <div>
            <div className="text-[#FAF7F0]/50 text-[10px]">Vuruş Sayısı</div>
            <div className="font-mono font-semibold text-[#E5A93C]">
              {features.strokeCount}
            </div>
          </div>
          <div>
            <div className="text-[#FAF7F0]/50 text-[10px]">Ortalama Hız</div>
            <div className="font-mono font-semibold text-[#FAF7F0]">
              {features.meanVelocity.toFixed(0)} px/s
            </div>
          </div>
          <div>
            <div className="text-[#FAF7F0]/50 text-[10px]">Maksimum Hız</div>
            <div className="font-mono font-semibold text-[#FAF7F0]">
              {features.maxVelocity.toFixed(0)} px/s
            </div>
          </div>
          <div>
            <div className="text-[#FAF7F0]/50 text-[10px]">Ortalama Basınç</div>
            <div className="font-mono font-semibold text-[#2E7D32]">
              {(features.meanPressure * 100).toFixed(0)}%
            </div>
          </div>
          <div>
            <div className="text-[#FAF7F0]/50 text-[10px]">Basınç Tepe (Peaks)</div>
            <div className="font-mono font-semibold text-[#FAF7F0]">
              {features.pressurePeaksCount}
            </div>
          </div>
        </div>
      )}

      {/* Aksiyon Butonları */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleClear}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-[#FAF7F0]/5 hover:bg-[#FAF7F0]/10 text-[#FAF7F0]/80 border border-[#FAF7F0]/10 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Temizle
          </button>
          <button
            type="button"
            onClick={handleUndo}
            disabled={strokeHistory.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-[#FAF7F0]/5 hover:bg-[#FAF7F0]/10 text-[#FAF7F0]/80 border border-[#FAF7F0]/10 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Geri Al
          </button>
          <button
            type="button"
            onClick={handleLoadMock}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-[#E5A93C]/10 hover:bg-[#E5A93C]/20 text-[#E5A93C] border border-[#E5A93C]/30 transition-colors"
            title="Stylus tableti olmayan cihazlar için kalibre edilmiş sentetik adli imza yükler"
          >
            <Activity className="w-3.5 h-3.5" />
            Mock İmza Yükle (Test)
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleSaveImage}
            disabled={strokeHistory.length === 0}
            className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium rounded-lg bg-[#FAF7F0]/10 hover:bg-[#FAF7F0]/20 text-[#FAF7F0] border border-[#FAF7F0]/20 transition-colors disabled:opacity-30"
          >
            <Download className="w-3.5 h-3.5" />
            PNG Olarak Kaydet
          </button>
          <button
            type="button"
            onClick={handleTriggerAnalysis}
            disabled={strokeHistory.length === 0}
            className="flex items-center gap-1.5 px-4 py-1.5 text-xs font-semibold rounded-lg bg-[#7B1E34] hover:bg-[#8F233D] text-[#FAF7F0] shadow-md transition-colors disabled:opacity-30"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            Dinamik Biyometrik Analizi Başlat
          </button>
        </div>
      </div>

      {/* Yasal Uyarı ve 5070 E-İmza Hazır Mimari Bildirimi */}
      <div className="bg-[#120D0A] p-3 rounded-lg border border-[#E5A93C]/20 text-[11px] text-[#FAF7F0]/70 flex items-start gap-2.5">
        <ShieldCheck className="w-4 h-4 text-[#E5A93C] shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          <span className="font-semibold text-[#FAF7F0]">
            5070 Sayılı Elektronik İmza Kanunu Entegrasyonuna Hazır Adli Mimari:
          </span>{" "}
          Bu arayüz imzanın yalnızca iki boyutlu görüntüsünü değil; zaman, hız, ivme ve basınç serilerini (DTW) adli standartlarda toplar.
          Sistem araştırma ve adli karar destek amacıyla tasarlanmış olup doğrudan Kamu SM / Nitelikli Elektronik Sertifika (NES) yerine geçmez.
        </div>
      </div>
    </div>
  );
};
