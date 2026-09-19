"use client";

import React, { useState } from "react";
import { History, ShieldCheck, Copy, Check } from "lucide-react";

export interface AuditItem {
  timestamp: string;
  eventType: string;
  action: string;
  examinerId: string;
  sha256Hash: string;
  detailsSummary?: string;
}

interface AuditTrailTimelineProps {
  items?: AuditItem[];
  merkleRoot?: string;
}

export const AuditTrailTimeline: React.FC<AuditTrailTimelineProps> = ({ items = [], merkleRoot }) => {
  const [copiedHash, setCopiedHash] = useState<string | null>(null);

  const copyToClipboard = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopiedHash(hash);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  const defaultRoot = merkleRoot || (items.length > 0 ? items[items.length - 1].sha256Hash : "0".repeat(64));

  return (
    <div className="bg-[#1E1612]/90 border border-[#B8A484]/25 rounded-2xl p-5 shadow-2xl backdrop-blur-md text-xs space-y-4">
      <div className="flex items-center justify-between pb-2 border-b border-[#B8A484]/20">
        <h3 className="font-bold text-xs text-[#FAF7F0] flex items-center gap-2">
          <History className="w-4 h-4 text-teal-400" />
          Kriptografik Denetim İzi ve Merkle Ağacı Mührü
        </h3>
        <span className="text-[11px] text-[#D4C5A9]/70">Değiştirilemez Bütünlük Zinciri</span>
      </div>

      {/* Merkle Root Hash Mührü */}
      <div className="p-3 bg-[#120D0A] border border-teal-500/30 rounded-xl flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-full bg-teal-500/20 text-teal-400 flex items-center justify-center shrink-0">
            <ShieldCheck className="w-3.5 h-3.5" />
          </div>
          <div>
            <div className="text-[10px] text-teal-300 font-semibold uppercase tracking-wider">Merkle Root Hash (Kök Mühür)</div>
            <div className="font-mono text-[11px] text-[#FAF7F0] truncate max-w-[280px] sm:max-w-md">
              {defaultRoot}
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={() => copyToClipboard(defaultRoot)}
          className="px-2 py-1 bg-[#2A2019] hover:bg-[#382B22] border border-[#B8A484]/20 rounded-lg text-[10px] text-teal-200 font-mono flex items-center gap-1 shrink-0 transition"
        >
          {copiedHash === defaultRoot ? (
            <>
              <Check className="w-3 h-3 text-emerald-400" />
              Kopyalandı
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              Kökü Kopyala
            </>
          )}
        </button>
      </div>

      {items.length === 0 ? (
        <p className="text-[#D4C5A9]/60 italic py-2 text-center">Henüz kayıtlı bir adli işlem olayı bulunmamaktadır.</p>
      ) : (
        <div className="space-y-2.5">
          {items.map((item, idx) => (
            <div key={idx} className="flex items-start gap-2.5 pb-2.5 border-b border-[#B8A484]/10 last:border-0 last:pb-0">
              <div className="mt-0.5 w-5 h-5 rounded-full bg-[#2A2019] border border-teal-500/40 text-teal-400 flex items-center justify-center shrink-0 shadow-sm">
                <ShieldCheck className="w-3 h-3" />
              </div>

              <div className="flex-1 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-[#FAF7F0]">{item.eventType}</span>
                  <span className="text-[10px] text-[#D4C5A9] font-mono">{item.timestamp}</span>
                </div>

                <div className="text-[11px] text-[#EFE7DA]">
                  <span>İşlem: {item.action}</span> | <span className="text-[#D4C5A9]">Bilirkişi: {item.examinerId}</span>
                </div>

                <div className="flex items-center gap-1.5 pt-0.5">
                  <span className="font-mono text-[10px] text-[#D4C5A9] bg-[#120D0A] px-2.5 py-0.5 rounded-full border border-[#B8A484]/15 truncate max-w-[280px]">
                    SHA-256: {item.sha256Hash}
                  </span>
                  <button
                    type="button"
                    onClick={() => copyToClipboard(item.sha256Hash)}
                    title="Özeti Kopyala"
                    className="p-1 hover:bg-[#2A2019] rounded text-[#D4C5A9] hover:text-[#FAF7F0] transition"
                  >
                    {copiedHash === item.sha256Hash ? (
                      <Check className="w-3 h-3 text-emerald-400" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
