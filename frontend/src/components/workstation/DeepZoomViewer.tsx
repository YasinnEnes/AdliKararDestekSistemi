"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
  Eye,
  Crosshair,
  Sliders,
  Move,
  Info,
} from "lucide-react";

interface DeepZoomViewerProps {
  imageUrl: string;
  title?: string;
  sampleType?: "questioned" | "reference";
}

export function DeepZoomViewer({
  imageUrl,
  title = "Adli Yüksek Çözünürlüklü İnceleme",
  sampleType = "questioned",
}: DeepZoomViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Zoom ve Pan Durumu
  const [scale, setScale] = useState<number>(1.0);
  const [offset, setOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Piksel İnceleme (Pixel Inspector)
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [pixelRgb, setPixelRgb] = useState<{ r: number; g: number; b: number } | null>(null);
  const [isInspectorActive, setIsInspectorActive] = useState<boolean>(true);

  // Görüntü Özellikleri
  const [imageObj, setImageObj] = useState<HTMLImageElement | null>(null);
  const [imgDimensions, setImgDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

  // Görüntüyü yükle
  useEffect(() => {
    if (!imageUrl) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;
    img.onload = () => {
      setImageObj(img);
      setImgDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      // Merkeze hizala
      setScale(1.0);
      setOffset({ x: 0, y: 0 });
    };
  }, [imageUrl]);

  // Canvas Render Döngüsü
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageObj) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Temizle
    ctx.clearRect(0, 0, width, height);

    // Kılavuz / Milimetrik ızgara arka planı
    ctx.save();
    ctx.strokeStyle = "rgba(184, 164, 132, 0.08)";
    ctx.lineWidth = 1;
    const gridSize = 20 * scale;
    if (gridSize >= 10) {
      for (let x = (offset.x % gridSize); x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = (offset.y % gridSize); y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }
    ctx.restore();

    // Görüntüyü ölçeklendir ve çiz
    ctx.save();
    ctx.translate(width / 2 + offset.x, height / 2 + offset.y);
    ctx.scale(scale, scale);

    // Yüksek büyütmede pikselleri bulanıklaştırmak yerine keskin (nearest-neighbor) göster
    ctx.imageSmoothingEnabled = scale < 3.0;

    ctx.drawImage(
      imageObj,
      -imageObj.naturalWidth / 2,
      -imageObj.naturalHeight / 2,
      imageObj.naturalWidth,
      imageObj.naturalHeight
    );
    ctx.restore();
  }, [imageObj, scale, offset]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Boyutlandırma
  useEffect(() => {
    function resizeCanvas() {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      if (container && canvas) {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        renderCanvas();
      }
    }
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    return () => window.removeEventListener("resize", resizeCanvas);
  }, [renderCanvas]);

  // Fare Tekerleği ile Zoom (1.0x - 50.0x)
  function handleWheel(e: React.WheelEvent<HTMLCanvasElement>) {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.2 : 0.83;
    setScale((prev) => Math.min(50.0, Math.max(0.2, prev * zoomFactor)));
  }

  // Sürükleme (Pan)
  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }

    // Piksel İnceleme
    if (isInspectorActive && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      setCursorPos({ x: Math.round(mouseX), y: Math.round(mouseY) });

      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        try {
          const pixel = ctx.getImageData(mouseX, mouseY, 1, 1).data;
          setPixelRgb({ r: pixel[0], g: pixel[1], b: pixel[2] });
        } catch {
          // Cross-origin güvenlik kısıtlaması durumunda sessiz kal
        }
      }
    }
  }

  function handleMouseUp() {
    setIsDragging(false);
  }

  function handleReset() {
    setScale(1.0);
    setOffset({ x: 0, y: 0 });
  }

  return (
    <div className="flex flex-col h-full bg-[#1C1510] border border-[#B8A484]/20 rounded-2xl overflow-hidden shadow-2xl">
      {/* KONTROL ARAÇ ÇUBUĞU */}
      <div className="bg-[#241A14] border-b border-[#B8A484]/20 px-4 py-2.5 flex items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-amber-300" />
          <span className="font-bold text-[#FAF7F0]">{title}</span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#18120F] border border-[#B8A484]/30 text-[#D4C5A9]">
            {imgDimensions.width} × {imgDimensions.height} px
          </span>
        </div>

        {/* Büyütme Butonları */}
        <div className="flex items-center gap-1.5 bg-[#18120F] border border-[#B8A484]/30 rounded-xl px-2 py-1">
          <button
            type="button"
            onClick={() => setScale((s) => Math.max(0.2, s * 0.8))}
            className="p-1 hover:bg-[#2A2019] rounded text-[#D4C5A9] hover:text-[#FAF7F0]"
            title="Uzaklaştır"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="font-mono font-bold text-amber-300 w-12 text-center text-[11px]">
            {scale.toFixed(1)}x
          </span>
          <button
            type="button"
            onClick={() => setScale((s) => Math.min(50.0, s * 1.25))}
            className="p-1 hover:bg-[#2A2019] rounded text-[#D4C5A9] hover:text-[#FAF7F0]"
            title="Yakınlaştır (Maks 50x)"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <div className="w-[1px] h-3 bg-[#B8A484]/20 mx-1" />
          <button
            type="button"
            onClick={handleReset}
            className="p-1 hover:bg-[#2A2019] rounded text-[#D4C5A9] hover:text-[#FAF7F0]"
            title="Görünümü Sıfırla"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Piksel Denetleyici Toggle */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setIsInspectorActive(!isInspectorActive)}
            className={`px-2.5 py-1 rounded-lg border text-[11px] font-semibold flex items-center gap-1 transition ${
              isInspectorActive
                ? "bg-burgundy-950/80 border-burgundy-600/50 text-cream-200"
                : "bg-[#18120F] border-[#B8A484]/20 text-[#D4C5A9]/70"
            }`}
          >
            <Crosshair className="w-3 h-3" />
            <span>Piksel İnceleyici</span>
          </button>
        </div>
      </div>

      {/* CANVAS GÖRÜNTÜLEME ALANI */}
      <div
        ref={containerRef}
        className="relative flex-1 w-full h-[450px] bg-[#120D0A] cursor-grab active:cursor-grabbing overflow-hidden"
      >
        <canvas
          ref={canvasRef}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="w-full h-full block"
        />

        {/* PİKSEL BİLGİ KUTUSU (HUD) */}
        {isInspectorActive && cursorPos && (
          <div className="absolute bottom-3 left-3 bg-[#1C1510]/90 backdrop-blur-md border border-[#B8A484]/30 rounded-xl px-3 py-1.5 text-[10px] font-mono text-[#FAF7F0] shadow-xl flex items-center gap-3 pointer-events-none">
            <div>
              X: <span className="text-amber-300">{cursorPos.x}</span> Y:{" "}
              <span className="text-amber-300">{cursorPos.y}</span>
            </div>
            {pixelRgb && (
              <div className="flex items-center gap-1.5">
                <div
                  className="w-3 h-3 rounded-full border border-white/40"
                  style={{ backgroundColor: `rgb(${pixelRgb.r}, ${pixelRgb.g}, ${pixelRgb.b})` }}
                />
                <span>
                  RGB({pixelRgb.r}, {pixelRgb.g}, {pixelRgb.b})
                </span>
              </div>
            )}
            <div className="text-[#D4C5A9]/60">Büyütme: {scale.toFixed(1)}x</div>
          </div>
        )}

        {/* NAVIGATOR KÜÇÜK HARİTA (MİNİMAP) */}
        {imageObj && scale > 1.5 && (
          <div className="absolute top-3 right-3 w-28 h-20 bg-[#18120F]/90 border border-[#B8A484]/30 rounded-lg overflow-hidden shadow-2xl pointer-events-none p-1">
            <div
              className="w-full h-full bg-contain bg-center bg-no-repeat relative opacity-60"
              style={{ backgroundImage: `url(${imageUrl})` }}
            >
              <div className="absolute inset-0 border border-amber-500/80 rounded" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
