"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  api,
  formatApiError,
  CaseResponse,
  CaseCreatePayload,
  CaseUpdatePayload,
  CaseStatus,
  CaseStage,
  ExaminationType,
} from "@/services/api";

interface CaseContextType {
  activeCaseId: string | null;
  activeCase: CaseResponse | null;
  cases: CaseResponse[];
  loadingCases: boolean;
  error: string | null;
  hasUnsavedChanges: boolean;
  setActiveCaseId: (id: string | null) => void;
  setActiveCase: (caseData: CaseResponse | null) => void;
  setHasUnsavedChanges: (hasChanges: boolean) => void;
  fetchCases: () => Promise<CaseResponse[]>;
  selectCaseAndNavigate: (caseId: string) => Promise<void>;
  createNewCase: (payload: CaseCreatePayload) => Promise<CaseResponse>;
  updateCurrentCase: (payload: CaseUpdatePayload) => Promise<CaseResponse | null>;
  deleteCase: (caseId: string) => Promise<boolean>;
}

const CaseContext = createContext<CaseContextType | undefined>(undefined);

const ACTIVE_CASE_STORAGE_KEY = "fhdss_active_case_id";

export const CaseProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const router = useRouter();
  const [activeCaseId, setActiveCaseIdState] = useState<string | null>(null);
  const [activeCase, setActiveCase] = useState<CaseResponse | null>(null);
  const [cases, setCases] = useState<CaseResponse[]>([]);
  const [loadingCases, setLoadingCases] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState<boolean>(false);

  // Aktif vaka ID'sini localStorage ile senkronize et
  const setActiveCaseId = useCallback((id: string | null) => {
    setActiveCaseIdState(id);
    if (typeof window !== "undefined") {
      if (id) {
        localStorage.setItem(ACTIVE_CASE_STORAGE_KEY, id);
      } else {
        localStorage.removeItem(ACTIVE_CASE_STORAGE_KEY);
      }
    }
  }, []);

  // Tüm vakaları API'den getir
  const fetchCases = useCallback(async (): Promise<CaseResponse[]> => {
    setLoadingCases(true);
    setError(null);
    try {
      const data = await api.getCases();
      setCases(data || []);
      return data || [];
    } catch (err: any) {
      console.warn("Vaka listesi API'den alınamadı:", err?.message || err);
      const errorMessage = formatApiError(err);
      setError(errorMessage);
      setCases([]);
      return [];
    } finally {
      setLoadingCases(false);
    }
  }, []);

  // İlk yüklemede vakaları ve aktif vaka ID'sini geri yükle
  useEffect(() => {
    fetchCases();
    if (typeof window !== "undefined") {
      const savedId = localStorage.getItem(ACTIVE_CASE_STORAGE_KEY);
      if (savedId) {
        setActiveCaseIdState(savedId);
      }
    }
  }, [fetchCases]);

  // Aktif vaka ID değiştiğinde activeCase nesnesini eşle
  useEffect(() => {
    if (!activeCaseId) {
      setActiveCase(null);
      return;
    }
    const found = cases.find((c) => c.case_id === activeCaseId);
    if (found) {
      setActiveCase(found);
    }
  }, [activeCaseId, cases]);

  // Vaka seç ve İş İstasyonuna Yönlendir
  const selectCaseAndNavigate = async (caseId: string) => {
    setActiveCaseId(caseId);
    setHasUnsavedChanges(false);
    router.push(`/cases/${encodeURIComponent(caseId)}`);
  };

  // Yeni Vaka Oluştur ve İş İstasyonuna Yönlendir
  const createNewCase = async (payload: CaseCreatePayload): Promise<CaseResponse> => {
    setError(null);
    try {
      const created = await api.createCase(payload);
      setCases((prev) => [created, ...prev.filter((c) => c.case_id !== created.case_id)]);
      setActiveCaseId(created.case_id);
      setActiveCase(created);
      setHasUnsavedChanges(false);
      return created;
    } catch (err: any) {
      const errMsg = formatApiError(err);
      console.error("Vaka oluşturulamadı:", errMsg);
      setError(errMsg);
      throw err;
    }
  };

  // Mevcut Vakayı Güncelle
  const updateCurrentCase = async (payload: CaseUpdatePayload): Promise<CaseResponse | null> => {
    if (!activeCaseId) return null;
    try {
      const updated = await api.updateCase(activeCaseId, payload);
      setCases((prev) => prev.map((c) => (c.case_id === activeCaseId ? updated : c)));
      setActiveCase(updated);
      setHasUnsavedChanges(false);
      return updated;
    } catch (err: any) {
      console.warn("Vaka güncellenemedi:", err?.message || err);
      if (activeCase) {
        const localUpdated: CaseResponse = {
          ...activeCase,
          ...payload,
          updated_at_utc: new Date().toISOString(),
        };
        setActiveCase(localUpdated);
        setCases((prev) => prev.map((c) => (c.case_id === activeCaseId ? localUpdated : c)));
        return localUpdated;
      }
      return null;
    }
  };

  // Vakayı Kalıcı Olarak Sil
  const deleteCase = async (caseId: string): Promise<boolean> => {
    setError(null);
    try {
      await api.deleteCase(caseId);
      // Listeden çıkar
      setCases((prev) => prev.filter((c) => c.case_id !== caseId));

      // Eğer silinen vaka şu anda aktifse, aktif vaka durumunu ve depolamayı temizle
      if (activeCaseId === caseId) {
        setActiveCaseId(null);
        setActiveCase(null);
        setHasUnsavedChanges(false);
      }

      return true;
    } catch (err: any) {
      const errMsg = formatApiError(err);
      console.error("Vaka silinemedi:", errMsg);
      setError(errMsg);
      throw err;
    }
  };

  return (
    <CaseContext.Provider
      value={{
        activeCaseId,
        activeCase,
        cases,
        loadingCases,
        error,
        hasUnsavedChanges,
        setActiveCaseId,
        setActiveCase,
        setHasUnsavedChanges,
        fetchCases,
        selectCaseAndNavigate,
        createNewCase,
        updateCurrentCase,
        deleteCase,
      }}
    >
      {children}
    </CaseContext.Provider>
  );
};

export const useCaseContext = (): CaseContextType => {
  const context = useContext(CaseContext);
  if (!context) {
    throw new Error("useCaseContext must be used within a CaseProvider");
  }
  return context;
};
