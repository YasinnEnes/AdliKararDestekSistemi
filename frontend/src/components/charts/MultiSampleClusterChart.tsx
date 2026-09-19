"use client";

import React, { useState, useEffect } from "react";
import {
  ScatterChart,
  Layers,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Info,
  ShieldAlert,
} from "lucide-react";
import { apiClient, formatApiError } from "@/services/api";

interface ClusterData {
  case_id: string;
  questioned: {
    sample_id: string;
    pca_x: number;
    pca_y: number;
    distance_to_centroid: number;
    normalized_ellipse_distance: number;
    is_inside_tolerance_ellipse: boolean;
    forensic_status: string;
  };
  references: Array<{
    sample_id: string;
    pca_x: number;
    pca_y: number;
    distance_to_centroid: number;
  }>;
  variability_ellipse: {
    center_x: number;
    center_y: number;
    radius_x: number;
    radius_y: number;
    angle_deg: number;
    confidence_level?: number;
  };
  tolerance_ellipse?: {
    center_x: number;
    center_y: number;
    radius_x: number;
    radius_y: number;
    angle_deg: number;
  };
  intra_writer_variability: {
    mean_distance: number;
    max_distance: number;
    std_dev: number;
    sample_count?: number;
    consistency_rating?: string;
    reference_count?: number;
  };
  concordance_conclusion: string;
  pca_explained_variance_ratio?: number[];
}

interface Props {
  caseId: string;
}

export const MultiSampleClusterChart: React.FC<Props> = ({ caseId }) => {
  const [data, setData] = useState<ClusterData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchClustering = async () => {
    if (!caseId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.get<ClusterData>(`/analysis/multi-sample-cluster/${caseId}`);
      setData(res.data);
    } catch (err: any) {
      setError(formatApiError(err) || "Kümeleme verisi alınamadı.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchClustering();
  }, [caseId]);

  if (loading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 flex flex-col items-center justify-center min-h-[350px]">
        <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin mb-3" />
        <p className="text-slate-300 font-mono text-sm">
          ONNX Derin Metrik Embedding Uzayı & PCA 2D Kümeleme Hesaplanıyor...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-slate-900/80 border border-amber-500/30 rounded-xl p-6 flex flex-col items-center justify-center text-center">
        <AlertTriangle className="w-8 h-8 text-amber-400 mb-2" />
        <p className="text-slate-200 font-semibold mb-1">Çoklu Numune Kümeleme Uyarısı</p>
        <p className="text-slate-400 text-sm mb-4 max-w-md">{error}</p>
        <button
          onClick={fetchClustering}
          className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-cyan-400 rounded-lg text-xs font-mono transition flex items-center gap-2"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Tekrar Dene
        </button>
      </div>
    );
  }

  if (!data) return null;

  // SVG Projeksiyon Koordinat Haritası (Genişlik: 500, Yükseklik: 360)
  // Tüm noktaları (Q + Refs) toplayıp min/max sınırları bul
  const allX = [
    data.questioned.pca_x,
    ...data.references.map((r) => r.pca_x),
    data.variability_ellipse.center_x - data.variability_ellipse.radius_x,
    data.variability_ellipse.center_x + data.variability_ellipse.radius_x,
  ];
  const allY = [
    data.questioned.pca_y,
    ...data.references.map((r) => r.pca_y),
    data.variability_ellipse.center_y - data.variability_ellipse.radius_y,
    data.variability_ellipse.center_y + data.variability_ellipse.radius_y,
  ];

  const minX = Math.min(...allX) - 0.2;
  const maxX = Math.max(...allX) + 0.2;
  const minY = Math.min(...allY) - 0.2;
  const maxY = Math.max(...allY) + 0.2;

  const width = 500;
  const height = 340;
  const padding = 45;

  const scaleX = (val: number) =>
    padding + ((val - minX) / (maxX - minX || 1)) * (width - 2 * padding);
  const scaleY = (val: number) =>
    height - padding - ((val - minY) / (maxY - minY || 1)) * (height - 2 * padding);

  const qSvgX = scaleX(data.questioned.pca_x);
  const qSvgY = scaleY(data.questioned.pca_y);

  const elCenterX = scaleX(data.variability_ellipse.center_x);
  const elCenterY = scaleY(data.variability_ellipse.center_y);

  // Ölçek faktörlerine göre elips yarıçapları
  const rx = (data.variability_ellipse.radius_x / (maxX - minX || 1)) * (width - 2 * padding);
  const ry = (data.variability_ellipse.radius_y / (maxY - minY || 1)) * (height - 2 * padding);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-2xl">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-white tracking-wide">
              Çoklu Referans Kümeleme & Yazar İçi Varyasyon (Intra-Writer) Modeli
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            ConvNeXt 128D Embedding PCA 2D İndirgemesi & %95 Güven Aralığı Tolerans Elipsi
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-xs font-bold font-mono flex items-center gap-1.5 border ${
              data.questioned.is_inside_tolerance_ellipse
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                : "bg-rose-500/10 text-rose-400 border-rose-500/30"
            }`}
          >
            {data.questioned.is_inside_tolerance_ellipse ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5" /> Doğal Varyasyon Sınırları İçinde
              </>
            ) : (
              <>
                <AlertTriangle className="w-3.5 h-3.5" /> Tolerans Elipsi Dışında (Anomali)
              </>
            )}
          </span>

          <button
            onClick={fetchClustering}
            className="p-1.5 text-slate-400 hover:text-cyan-400 hover:bg-slate-800 rounded-lg transition"
            title="Yeniden Hesapla"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Grid: SVG Scatter Plot & Metrics Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-6 items-center">
        {/* SVG Chart */}
        <div className="lg:col-span-7 bg-slate-950 border border-slate-800/80 rounded-xl p-3 flex flex-col items-center justify-center relative overflow-hidden">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-auto max-h-[340px] select-none"
          >
            <defs>
              <linearGradient id="gridLineGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#334155" stopOpacity="0.2" />
                <stop offset="100%" stopColor="#334155" stopOpacity="0.6" />
              </linearGradient>
            </defs>

            {/* Grid & Axis Lines */}
            <line
              x1={padding}
              y1={height - padding}
              x2={width - padding}
              y2={height - padding}
              stroke="#475569"
              strokeWidth="1.5"
            />
            <line
              x1={padding}
              y1={padding}
              x2={padding}
              y2={height - padding}
              stroke="#475569"
              strokeWidth="1.5"
            />

            {/* Labels */}
            <text
              x={width / 2}
              y={height - 12}
              textAnchor="middle"
              className="text-[11px] fill-slate-400 font-mono font-semibold"
            >
              Temel Bileşen 1 (PC1 - %{Math.round(((data.pca_explained_variance_ratio && data.pca_explained_variance_ratio[0]) || 0) * 100)} Varyans)
            </text>
            <text
              x={14}
              y={height / 2}
              textAnchor="middle"
              transform={`rotate(-90 14 ${height / 2})`}
              className="text-[11px] fill-slate-400 font-mono font-semibold"
            >
              PC2 (%{Math.round(((data.pca_explained_variance_ratio && data.pca_explained_variance_ratio[1]) || 0) * 100)})
            </text>

            {/* 95% Confidence Tolerance Ellipse */}
            <g transform={`rotate(${data.variability_ellipse.angle_deg} ${elCenterX} ${elCenterY})`}>
              <ellipse
                cx={elCenterX}
                cy={elCenterY}
                rx={Math.max(rx, 15)}
                ry={Math.max(ry, 12)}
                fill="rgba(6, 182, 212, 0.08)"
                stroke="#06b6d4"
                strokeWidth="2"
                strokeDasharray="4 3"
              />
            </g>

            {/* Centroid Crosshair */}
            <line
              x1={elCenterX - 8}
              y1={elCenterY}
              x2={elCenterX + 8}
              y2={elCenterY}
              stroke="#38bdf8"
              strokeWidth="2"
            />
            <line
              x1={elCenterX}
              y1={elCenterY - 8}
              x2={elCenterX}
              y2={elCenterY + 8}
              stroke="#38bdf8"
              strokeWidth="2"
            />

            {/* Reference Sample Dots */}
            {data.references.map((r, i) => {
              const sx = scaleX(r.pca_x);
              const sy = scaleY(r.pca_y);
              return (
                <g key={r.sample_id} className="cursor-pointer group">
                  <circle
                    cx={sx}
                    cy={sy}
                    r="6"
                    className="fill-cyan-400 stroke-slate-900 stroke-2 hover:r-8 transition-all"
                  />
                  <text
                    x={sx + 8}
                    y={sy + 3}
                    className="text-[9px] fill-slate-300 font-mono pointer-events-none opacity-80"
                  >
                    Ref #{i + 1}
                  </text>
                </g>
              );
            })}

            {/* Questioned Sample Marker (Diamond) */}
            <g className="cursor-pointer">
              <polygon
                points={`${qSvgX},${qSvgY - 9} ${qSvgX + 9},${qSvgY} ${qSvgX},${qSvgY + 9} ${qSvgX - 9},${qSvgY}`}
                className={`${
                  data.questioned.is_inside_tolerance_ellipse ? "fill-emerald-400" : "fill-rose-500"
                } stroke-white stroke-2 animate-pulse`}
              />
              <text
                x={qSvgX + 12}
                y={qSvgY - 4}
                className="text-[10px] font-bold fill-white font-mono drop-shadow"
              >
                Sorgulanan (Q)
              </text>
            </g>
          </svg>

          {/* Legend */}
          <div className="flex items-center gap-4 text-xs font-mono text-slate-400 mt-2">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block" />
              <span>Mukayese ({data.references.length} adet)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rotate-45 bg-rose-500 inline-block" />
              <span>Sorgulanan (Q)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3.5 h-2 border border-dashed border-cyan-400 inline-block" />
              <span>%95 Yazar İçi Tolerans Elipsi</span>
            </div>
          </div>
        </div>

        {/* Metrics & Interpretation */}
        <div className="lg:col-span-5 space-y-3">
          <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-4">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
              Yazar İçi Tutarlılık Derecesi
            </span>
            <div className="text-sm font-bold text-cyan-300 mt-1">
              {data.intra_writer_variability.consistency_rating}
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Referans imzalar arası ortalama varyasyon mesafesi:{" "}
              <strong className="text-white font-mono">
                {data.intra_writer_variability.mean_distance.toFixed(3)}
              </strong>
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="bg-slate-950/40 border border-slate-800/80 rounded-lg p-3">
              <span className="text-[10px] font-mono text-slate-400">Maksimum Varyasyon</span>
              <div className="text-sm font-bold text-slate-200 font-mono mt-0.5">
                {data.intra_writer_variability.max_distance.toFixed(3)}
              </div>
            </div>
            <div className="bg-slate-950/40 border border-slate-800/80 rounded-lg p-3">
              <span className="text-[10px] font-mono text-slate-400">Standart Sapma</span>
              <div className="text-sm font-bold text-slate-200 font-mono mt-0.5">
                {data.intra_writer_variability.std_dev.toFixed(3)}
              </div>
            </div>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-4">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
              Sorgulanan İmzanın Küme Durumu
            </span>
            <div className="flex items-center justify-between mt-1">
              <span className="text-xs text-slate-300">Merkeze (Centroid) Uzaklık:</span>
              <span className="text-xs font-mono font-bold text-white">
                {data.questioned.distance_to_centroid.toFixed(3)}
              </span>
            </div>
            <div className="flex items-center justify-between mt-1">
              <span className="text-xs text-slate-300">Normalize Elips İndeksi:</span>
              <span
                className={`text-xs font-mono font-bold ${
                  data.questioned.is_inside_tolerance_ellipse ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {data.questioned.normalized_ellipse_distance.toFixed(2)} / 1.00
              </span>
            </div>
          </div>

          <div className="p-3 bg-cyan-950/20 border border-cyan-900/30 rounded-lg flex items-start gap-2.5 text-xs text-cyan-200/90">
            <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
            <p>
              Adli kriterlere göre, çoklu referans imzaların doğal varyasyonu tekil imzaya kıyasla
              yanlış pozitif oranını düşürür. Sorgulanan imza elipsin içinde kaldığında yazar içi
              doğal değişim kabul edilir.
            </p>
          </div>

          <div className="p-3 bg-amber-950/25 border border-amber-800/40 rounded-lg flex items-start gap-2.5 text-xs text-amber-200/90">
            <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <p>
              <strong>Adli Karar Destek Uyarısı (ISO 17025):</strong> Bu kümeleme görselleştirmesi, bilirkişiye varyasyon analizi sağlamak için hesaplanmıştır. Tek başına nihai aidiyet veya kesin hukuki hüküm oluşturmaz.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
