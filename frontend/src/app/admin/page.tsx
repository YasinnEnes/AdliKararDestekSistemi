"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  ShieldAlert,
  ShieldCheck,
  Users,
  KeyRound,
  History,
  UserPlus,
  ArrowLeft,
  CheckCircle2,
  XCircle,
  AlertCircle,
  RefreshCw,
  Search,
  Filter,
} from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { api, UserResponse, UserRole, formatApiError } from "@/services/api";

type AdminTab = "users" | "roles" | "audit";

interface AuditLogItem {
  id: string;
  case_id?: string;
  event_type: string;
  action: string;
  examiner_id?: string;
  sha256_hash: string;
  details_json?: any;
  created_at_utc: string;
}

export default function AdminDashboardPage() {
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();

  const [activeTab, setActiveTab] = useState<AdminTab>("users");
  const [users, setUsers] = useState<UserResponse[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLogItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Yeni Kullanıcı Modalı
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newFullName, setNewFullName] = useState("");
  const [newRole, setNewRole] = useState<UserRole>("EXAMINER");

  // Şifre Sıfırlama Modalı
  const [resetModalUserId, setResetModalUserId] = useState<string | null>(null);
  const [resetModalUsername, setResetModalUsername] = useState<string>("");
  const [newPasswordInput, setNewPasswordInput] = useState("");

  // Audit Arama Filtreleri
  const [auditSearchCaseId, setAuditSearchCaseId] = useState("");
  const [auditSearchExaminer, setAuditSearchExaminer] = useState("");

  useEffect(() => {
    if (!authLoading && user && user.role === "ADMIN") {
      loadData();
    }
  }, [user, authLoading, activeTab]);

  async function loadData() {
    setLoading(true);
    setErrorMsg(null);
    try {
      if (activeTab === "users") {
        const uList = await api.getAdminUsers();
        setUsers(uList);
      } else if (activeTab === "audit") {
        const logs = await api.getAdminAuditLogs({
          case_id: auditSearchCaseId.trim() || undefined,
          examiner_id: auditSearchExaminer.trim() || undefined,
          limit: 100,
        });
        setAuditLogs(logs as AuditLogItem[]);
      }
    } catch (err) {
      setErrorMsg(formatApiError(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateUser(e: React.FormEvent) {
    e.preventDefault();
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      await api.createAdminUser({
        username: newUsername,
        password: newPassword,
        full_name: newFullName,
        role: newRole,
      });
      setSuccessMsg(`'${newUsername}' kullanıcısı başarıyla oluşturuldu.`);
      setIsCreateModalOpen(false);
      setNewUsername("");
      setNewPassword("");
      setNewFullName("");
      await loadData();
    } catch (err) {
      setErrorMsg(formatApiError(err));
    }
  }

  async function handleRoleChange(userId: string, targetRole: string) {
    setErrorMsg(null);
    try {
      await api.updateAdminUserRole(userId, targetRole);
      setSuccessMsg("Kullanıcı rolü başarıyla güncellendi.");
      await loadData();
    } catch (err) {
      setErrorMsg(formatApiError(err));
    }
  }

  async function handleToggleStatus(userId: string, currentStatus: boolean) {
    setErrorMsg(null);
    try {
      await api.updateAdminUserStatus(userId, !currentStatus);
      setSuccessMsg(`Kullanıcı ${!currentStatus ? "aktifleştirildi" : "pasifleştirildi"}.`);
      await loadData();
    } catch (err) {
      setErrorMsg(formatApiError(err));
    }
  }

  async function handleResetPassword(e: React.FormEvent) {
    e.preventDefault();
    if (!resetModalUserId) return;
    setErrorMsg(null);
    try {
      await api.resetAdminUserPassword(resetModalUserId, newPasswordInput);
      setSuccessMsg(`'${resetModalUsername}' kullanıcısının şifresi başarıyla sıfırlandı.`);
      setResetModalUserId(null);
      setNewPasswordInput("");
    } catch (err) {
      setErrorMsg(formatApiError(err));
    }
  }

  if (authLoading) {
    return (
      <div className="min-h-screen bg-[#18120F] flex items-center justify-center">
        <div className="w-8 h-8 rounded-full border-2 border-burgundy-600 border-t-transparent animate-spin" />
      </div>
    );
  }

  if (!user || user.role !== "ADMIN") {
    return (
      <div className="min-h-screen bg-[#18120F] flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 rounded-2xl bg-rose-950/60 border border-rose-700/50 flex items-center justify-center mb-4 text-rose-300">
          <ShieldAlert className="w-8 h-8" />
        </div>
        <h1 className="text-xl font-bold text-[#FAF7F0] mb-2">Yetkisiz Erişim (403 Forbidden)</h1>
        <p className="text-sm text-[#D4C5A9]/80 max-w-md mb-6">
          Sistem Yönetimi (Admin) paneline yalnızca <strong>ADMIN</strong> yetkisine sahip kullanıcılar erişebilir.
        </p>
        <button
          type="button"
          onClick={() => router.push("/")}
          className="px-4 py-2 bg-burgundy-700 hover:bg-burgundy-800 text-[#FAF7F0] rounded-xl text-sm font-semibold transition flex items-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Ana Sayfaya Dön
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#18120F] text-[#FAF7F0] font-sans flex flex-col">
      {/* ÜST BAŞLIK */}
      <header className="border-b border-[#B8A484]/20 bg-[#1C1510]/90 backdrop-blur-md px-6 py-4 flex items-center justify-between shadow-xl sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => router.push("/")}
            className="p-2 hover:bg-[#2A2019] rounded-xl text-[#D4C5A9] transition"
            title="Ana Sayfaya Dön"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-lg font-black tracking-wide flex items-center gap-2">
              FHDSS SİSTEM YÖNETİMİ
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-950/80 border border-amber-600/50 text-amber-300">
                ADMIN PANEL
              </span>
            </h1>
            <p className="text-xs text-[#D4C5A9]/70">Rol tabanlı erişim denetimi ve kriptografik log izleme</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => loadData()}
            className="px-3 py-1.5 bg-[#2A2019] hover:bg-[#382B22] border border-[#B8A484]/20 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Yenile
          </button>
        </div>
      </header>

      {/* BİLDİRİM BANTLARI */}
      {errorMsg && (
        <div className="mx-6 mt-4 p-3 bg-rose-950/60 border border-rose-700/50 rounded-xl text-xs text-rose-200 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{errorMsg}</span>
        </div>
      )}
      {successMsg && (
        <div className="mx-6 mt-4 p-3 bg-emerald-950/60 border border-emerald-700/50 rounded-xl text-xs text-emerald-200 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
          <span>{successMsg}</span>
        </div>
      )}

      {/* SEKME GEÇİŞLERİ */}
      <div className="px-6 mt-6">
        <div className="flex border-b border-[#B8A484]/20 gap-4">
          <button
            type="button"
            onClick={() => setActiveTab("users")}
            className={`pb-3 text-sm font-semibold flex items-center gap-2 border-b-2 transition ${
              activeTab === "users"
                ? "border-amber-500 text-amber-300"
                : "border-transparent text-[#D4C5A9]/70 hover:text-[#FAF7F0]"
            }`}
          >
            <Users className="w-4 h-4" />
            Kullanıcı Yönetimi ({users.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("roles")}
            className={`pb-3 text-sm font-semibold flex items-center gap-2 border-b-2 transition ${
              activeTab === "roles"
                ? "border-amber-500 text-amber-300"
                : "border-transparent text-[#D4C5A9]/70 hover:text-[#FAF7F0]"
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            Roller ve Yetkiler
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("audit")}
            className={`pb-3 text-sm font-semibold flex items-center gap-2 border-b-2 transition ${
              activeTab === "audit"
                ? "border-amber-500 text-amber-300"
                : "border-transparent text-[#D4C5A9]/70 hover:text-[#FAF7F0]"
            }`}
          >
            <History className="w-4 h-4" />
            Denetim Kütükleri (Audit Logs)
          </button>
        </div>
      </div>

      {/* ANA İÇERİK ALANI */}
      <main className="p-6 flex-1">
        {/* SEKME 1: KULLANICILAR */}
        {activeTab === "users" && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <p className="text-xs text-[#D4C5A9]/80">
                Sistemde kayıtlı adli bilirkişileri, kıdemli incelemecileri ve yöneticileri yönetin.
              </p>
              <button
                type="button"
                onClick={() => setIsCreateModalOpen(true)}
                className="px-3.5 py-1.5 bg-burgundy-700 hover:bg-burgundy-800 text-[#FAF7F0] rounded-xl text-xs font-bold transition flex items-center gap-1.5 shadow-md"
              >
                <UserPlus className="w-4 h-4" />
                Yeni Bilirkişi / Kullanıcı Ekle
              </button>
            </div>

            <div className="bg-[#241A14] border border-[#B8A484]/20 rounded-2xl overflow-hidden shadow-xl">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-[#1C1510] border-b border-[#B8A484]/20 text-[#D4C5A9]/80 font-bold uppercase tracking-wider">
                    <th className="p-3.5">Kullanıcı</th>
                    <th className="p-3.5">Ad Soyad</th>
                    <th className="p-3.5">Rol (Yetki)</th>
                    <th className="p-3.5">Durum</th>
                    <th className="p-3.5">Kayıt Tarihi</th>
                    <th className="p-3.5 text-right">İşlemler</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#B8A484]/10">
                  {users.map((u) => (
                    <tr key={u.id} className="hover:bg-[#2A2019]/60 transition">
                      <td className="p-3.5 font-bold text-[#FAF7F0]">{u.username}</td>
                      <td className="p-3.5 text-[#D4C5A9]">{u.full_name || "-"}</td>
                      <td className="p-3.5">
                        <select
                          value={u.role}
                          onChange={(e) => handleRoleChange(u.id, e.target.value)}
                          disabled={u.id === user.id}
                          className="bg-[#18120F] border border-[#B8A484]/30 rounded-lg px-2 py-1 text-xs text-[#FAF7F0] focus:outline-none focus:border-amber-500"
                        >
                          <option value="EXAMINER">EXAMINER (Bilirkişi)</option>
                          <option value="SENIOR_REVIEWER">SENIOR_REVIEWER (Kıdemli)</option>
                          <option value="ADMIN">ADMIN (Yönetici)</option>
                        </select>
                      </td>
                      <td className="p-3.5">
                        <span
                          className={`px-2 py-0.5 rounded-full font-bold text-[10px] border ${
                            u.is_active
                              ? "bg-emerald-950/70 border-emerald-600/40 text-emerald-300"
                              : "bg-rose-950/70 border-rose-600/40 text-rose-300"
                          }`}
                        >
                          {u.is_active ? "Aktif" : "Pasif"}
                        </span>
                      </td>
                      <td className="p-3.5 text-[#D4C5A9]/70 text-[11px]">
                        {new Date(u.created_at_utc).toLocaleString("tr-TR")}
                      </td>
                      <td className="p-3.5 text-right space-x-2">
                        <button
                          type="button"
                          onClick={() => {
                            setResetModalUserId(u.id);
                            setResetModalUsername(u.username);
                          }}
                          className="px-2.5 py-1 bg-[#18120F] hover:bg-amber-950/60 text-amber-300 border border-amber-600/30 rounded-lg text-[11px] font-semibold transition"
                          title="Şifre Sıfırla"
                        >
                          Şifre Sıfırla
                        </button>
                        <button
                          type="button"
                          onClick={() => handleToggleStatus(u.id, u.is_active)}
                          disabled={u.id === user.id}
                          className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold border transition ${
                            u.is_active
                              ? "bg-rose-950/40 hover:bg-rose-950/80 text-rose-300 border-rose-700/40"
                              : "bg-emerald-950/40 hover:bg-emerald-950/80 text-emerald-300 border-emerald-700/40"
                          } ${u.id === user.id ? "opacity-50 cursor-not-allowed" : ""}`}
                        >
                          {u.is_active ? "Pasifleştir" : "Aktifleştir"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* SEKME 2: ROLLER VE İZİNLER */}
        {activeTab === "roles" && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-[#241A14] border border-emerald-600/30 rounded-2xl p-5 shadow-xl space-y-3">
              <div className="flex items-center gap-2 text-emerald-300">
                <Users className="w-5 h-5" />
                <h2 className="font-bold text-base">EXAMINER (Adli Bilirkişi)</h2>
              </div>
              <p className="text-xs text-[#D4C5A9]/80 leading-relaxed">
                Temel adli inceleme yetkisi. Vaka oluşturabilir, soru konusu ve mukayese numuneleri yükleyebilir,
                Bilişsel İzolasyon adımlarını yürütür, XAI haritalarını ve ENFSI Bayesyen analizlerini inceler.
              </p>
              <div className="text-[11px] text-[#FAF7F0] space-y-1 font-mono">
                <div>✓ Vaka Açma / Düzenleme</div>
                <div>✓ Numune Yükleme</div>
                <div>✓ Pipeline & XAI İnceleme</div>
                <div>✓ Rapor Önizleme & İndirme</div>
                <div>✗ Kullanıcı ve Rol Yönetimi</div>
              </div>
            </div>

            <div className="bg-[#241A14] border border-purple-600/30 rounded-2xl p-5 shadow-xl space-y-3">
              <div className="flex items-center gap-2 text-purple-300">
                <ShieldCheck className="w-5 h-5" />
                <h2 className="font-bold text-base">SENIOR_REVIEWER (Kıdemli)</h2>
              </div>
              <p className="text-xs text-[#D4C5A9]/80 leading-relaxed">
                İkinci göz denetimi (Blind Peer-Review). Bilirkişi tarafından tamamlanan analizleri doğrular,
                karşıtlı (counterfactual) simülasyonları onaylar ve nihai adli kanaat raporunu onaylar.
              </p>
              <div className="text-[11px] text-[#FAF7F0] space-y-1 font-mono">
                <div>✓ Tüm Examiner İzinleri</div>
                <div>✓ Çift Kör İnceleme Onayı</div>
                <div>✓ Hassasiyet & Gating Denetimi</div>
                <div>✓ Kurumsal Arşiv İhracı (.ZIP)</div>
                <div>✗ Sistem Konfigürasyonu</div>
              </div>
            </div>

            <div className="bg-[#241A14] border border-amber-600/30 rounded-2xl p-5 shadow-xl space-y-3">
              <div className="flex items-center gap-2 text-amber-300">
                <KeyRound className="w-5 h-5" />
                <h2 className="font-bold text-base">ADMIN (Sistem Yöneticisi)</h2>
              </div>
              <p className="text-xs text-[#D4C5A9]/80 leading-relaxed">
                Tam yetkili sistem yöneticisi. Yeni bilirkişi hesapları tanımlar, rolleri düzenler,
                kullanıcıları dondurur ve kriptografik Merkle denetim kütüklerini (Audit Logs) filtreler.
              </p>
              <div className="text-[11px] text-[#FAF7F0] space-y-1 font-mono">
                <div>✓ Kullanıcı Oluşturma & Pasifleştirme</div>
                <div>✓ Rol Değiştirme (RBAC)</div>
                <div>✓ Şifre Sıfırlama</div>
                <div>✓ Sistem Audit Log Arama</div>
                <div>✓ Donanım ve Model Durum Denetimi</div>
              </div>
            </div>
          </div>
        )}

        {/* SEKME 3: DENETİM İZİ (AUDIT LOGS) */}
        {activeTab === "audit" && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3 bg-[#241A14] p-3.5 border border-[#B8A484]/20 rounded-2xl">
              <div className="flex items-center gap-2 flex-1 min-w-[200px]">
                <Search className="w-4 h-4 text-[#D4C5A9]/60" />
                <input
                  type="text"
                  placeholder="Vaka ID ara..."
                  value={auditSearchCaseId}
                  onChange={(e) => setAuditSearchCaseId(e.target.value)}
                  className="bg-[#18120F] border border-[#B8A484]/30 rounded-lg px-3 py-1.5 text-xs text-[#FAF7F0] w-full focus:outline-none focus:border-amber-500"
                />
              </div>
              <div className="flex items-center gap-2 flex-1 min-w-[200px]">
                <Filter className="w-4 h-4 text-[#D4C5A9]/60" />
                <input
                  type="text"
                  placeholder="Bilirkişi kullanıcı adı..."
                  value={auditSearchExaminer}
                  onChange={(e) => setAuditSearchExaminer(e.target.value)}
                  className="bg-[#18120F] border border-[#B8A484]/30 rounded-lg px-3 py-1.5 text-xs text-[#FAF7F0] w-full focus:outline-none focus:border-amber-500"
                />
              </div>
              <button
                type="button"
                onClick={() => loadData()}
                className="px-4 py-1.5 bg-burgundy-700 hover:bg-burgundy-800 text-[#FAF7F0] rounded-lg text-xs font-bold transition"
              >
                Filtrele
              </button>
            </div>

            <div className="bg-[#241A14] border border-[#B8A484]/20 rounded-2xl overflow-hidden shadow-xl">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-[#1C1510] border-b border-[#B8A484]/20 text-[#D4C5A9]/80 font-bold uppercase tracking-wider">
                    <th className="p-3">Tarih (UTC)</th>
                    <th className="p-3">Vaka ID</th>
                    <th className="p-3">Bilirkişi</th>
                    <th className="p-3">Olay Türü</th>
                    <th className="p-3">Eylem (Action)</th>
                    <th className="p-3">SHA-256 Hash</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#B8A484]/10">
                  {auditLogs.map((log) => (
                    <tr key={log.id} className="hover:bg-[#2A2019]/60 transition font-mono">
                      <td className="p-3 text-[#D4C5A9]/70 text-[11px]">
                        {new Date(log.created_at_utc).toLocaleString("tr-TR")}
                      </td>
                      <td className="p-3 font-bold text-amber-300">{log.case_id || "-"}</td>
                      <td className="p-3 text-[#FAF7F0]">{log.examiner_id || "SYSTEM"}</td>
                      <td className="p-3 text-purple-300 text-[11px]">{log.event_type}</td>
                      <td className="p-3 text-emerald-300 text-[11px]">{log.action}</td>
                      <td className="p-3 text-[#D4C5A9]/60 text-[10px] truncate max-w-[140px]" title={log.sha256_hash}>
                        {log.sha256_hash.slice(0, 16)}...
                      </td>
                    </tr>
                  ))}
                  {auditLogs.length === 0 && (
                    <tr>
                      <td colSpan={6} className="p-6 text-center text-[#D4C5A9]/60">
                        Kayıt bulunamadı.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>

      {/* YENİ KULLANICI EKLEME MODALI */}
      {isCreateModalOpen && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-[#241A14] border border-[#B8A484]/30 rounded-2xl p-6 max-w-md w-full shadow-2xl space-y-4">
            <h3 className="text-base font-bold text-[#FAF7F0] flex items-center gap-2">
              <UserPlus className="w-5 h-5 text-amber-300" />
              Yeni Adli Kullanıcı Tanımla
            </h3>
            <form onSubmit={handleCreateUser} className="space-y-3 text-xs">
              <div>
                <label className="block text-[#D4C5A9] font-medium mb-1">Kullanıcı Adı</label>
                <input
                  type="text"
                  required
                  value={newUsername}
                  onChange={(e) => setNewUsername(e.target.value)}
                  placeholder="uzman_ahmet"
                  className="w-full bg-[#18120F] border border-[#B8A484]/30 rounded-xl px-3 py-2 text-[#FAF7F0] focus:outline-none focus:border-amber-500"
                />
              </div>
              <div>
                <label className="block text-[#D4C5A9] font-medium mb-1">Ad Soyad ve Ünvan</label>
                <input
                  type="text"
                  required
                  value={newFullName}
                  onChange={(e) => setNewFullName(e.target.value)}
                  placeholder="Ahmet Yılmaz (Adli Belge İnceleme Uzmanı)"
                  className="w-full bg-[#18120F] border border-[#B8A484]/30 rounded-xl px-3 py-2 text-[#FAF7F0] focus:outline-none focus:border-amber-500"
                />
              </div>
              <div>
                <label className="block text-[#D4C5A9] font-medium mb-1">Geçici Şifre</label>
                <input
                  type="password"
                  required
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-[#18120F] border border-[#B8A484]/30 rounded-xl px-3 py-2 text-[#FAF7F0] focus:outline-none focus:border-amber-500"
                />
              </div>
              <div>
                <label className="block text-[#D4C5A9] font-medium mb-1">Rol (Yetki Seviyesi)</label>
                <select
                  value={newRole}
                  onChange={(e) => setNewRole(e.target.value as UserRole)}
                  className="w-full bg-[#18120F] border border-[#B8A484]/30 rounded-xl px-3 py-2 text-[#FAF7F0] focus:outline-none focus:border-amber-500"
                >
                  <option value="EXAMINER">EXAMINER (Bilirkişi Adli Uzman)</option>
                  <option value="SENIOR_REVIEWER">SENIOR_REVIEWER (Kıdemli İncelemeci)</option>
                  <option value="ADMIN">ADMIN (Sistem Yöneticisi)</option>
                </select>
              </div>

              <div className="flex justify-end gap-2 pt-3">
                <button
                  type="button"
                  onClick={() => setIsCreateModalOpen(false)}
                  className="px-4 py-2 bg-[#18120F] hover:bg-[#382B22] text-[#D4C5A9] rounded-xl font-semibold transition"
                >
                  İptal
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-burgundy-700 hover:bg-burgundy-800 text-[#FAF7F0] rounded-xl font-bold transition shadow-md"
                >
                  Kullanıcıyı Kaydet
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ŞİFRE SIFIRLAMA MODALI */}
      {resetModalUserId && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-[#241A14] border border-[#B8A484]/30 rounded-2xl p-6 max-w-sm w-full shadow-2xl space-y-4">
            <h3 className="text-base font-bold text-[#FAF7F0] flex items-center gap-2">
              <KeyRound className="w-5 h-5 text-amber-300" />
              Şifre Sıfırlama
            </h3>
            <p className="text-xs text-[#D4C5A9]/80">
              <strong>{resetModalUsername}</strong> kullanıcısı için yeni bir şifre belirleyin.
            </p>
            <form onSubmit={handleResetPassword} className="space-y-3 text-xs">
              <div>
                <label className="block text-[#D4C5A9] font-medium mb-1">Yeni Şifre</label>
                <input
                  type="password"
                  required
                  minLength={6}
                  value={newPasswordInput}
                  onChange={(e) => setNewPasswordInput(e.target.value)}
                  placeholder="En az 6 karakter"
                  className="w-full bg-[#18120F] border border-[#B8A484]/30 rounded-xl px-3 py-2 text-[#FAF7F0] focus:outline-none focus:border-amber-500"
                />
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setResetModalUserId(null)}
                  className="px-3 py-1.5 bg-[#18120F] hover:bg-[#382B22] text-[#D4C5A9] rounded-xl font-semibold transition"
                >
                  İptal
                </button>
                <button
                  type="submit"
                  className="px-3 py-1.5 bg-amber-600 hover:bg-amber-700 text-amber-950 font-bold rounded-xl transition shadow-md"
                >
                  Şifreyi Güncelle
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
