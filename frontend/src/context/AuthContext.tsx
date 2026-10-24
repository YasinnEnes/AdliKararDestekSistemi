"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from "react";
import { useRouter, usePathname } from "next/navigation";
import {
  api,
  formatApiError,
  UserResponse,
  TokenResponse,
  TOKEN_STORAGE_KEY,
  REFRESH_TOKEN_STORAGE_KEY,
  USER_STORAGE_KEY,
} from "@/services/api";

export type SessionState =
  | "idle"
  | "initializing"
  | "authenticated"
  | "verifying"
  | "verified"
  | "expired"
  | "unauthorized"
  | "error";

interface AuthContextType {
  user: UserResponse | null;
  token: string | null;
  sessionState: SessionState;
  loading: boolean;
  error: string | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  hasRole: (allowedRoles: string[]) => boolean;
  clearError: () => void;
  retrySessionVerification: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const router = useRouter();
  const pathname = usePathname();

  const [user, setUser] = useState<UserResponse | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [sessionState, setSessionState] = useState<SessionState>("initializing");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const isVerifyingRef = useRef<boolean>(false);

  const clearError = () => setError(null);

  // Oturumu Güvenli Şekilde Temizle (Logout)
  const logout = useCallback(() => {
    isVerifyingRef.current = false;
    if (typeof window !== "undefined") {
      const storedRefreshToken = localStorage.getItem(REFRESH_TOKEN_STORAGE_KEY);
      if (storedRefreshToken) {
        api.logout(storedRefreshToken).catch(() => {});
      }
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY);
      localStorage.removeItem(USER_STORAGE_KEY);
    }
    setUser(null);
    setToken(null);
    setError(null);
    setSessionState("unauthorized");
    setLoading(false);
    router.replace("/login");
  }, [router]);

  // Giriş Yap (Login)
  const login = async (username: string, password: string) => {
    setLoading(true);
    setSessionState("initializing");
    setError(null);
    try {
      const res: TokenResponse = await api.login(username, password);
      if (typeof window !== "undefined") {
        localStorage.setItem(TOKEN_STORAGE_KEY, res.access_token);
        if (res.refresh_token) {
          localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, res.refresh_token);
        }
        localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(res.user));
      }
      setToken(res.access_token);
      setUser(res.user);
      setSessionState("verified");
      setLoading(false);
      router.replace("/");
    } catch (err: any) {
      const msg = formatApiError(err);
      setError(msg);
      setSessionState("unauthorized");
      setLoading(false);
      throw err;
    }
  };

  // Rol Denetimi
  const hasRole = useCallback(
    (allowedRoles: string[]): boolean => {
      if (!user) return false;
      return allowedRoles.includes(user.role);
    },
    [user]
  );

  // Oturum Doğrulama (Akıllı Arka Plan Revalidasyonu & Kesin Timeout Koruması)
  const verifySession = useCallback(async (isSilentBackground: boolean = false) => {
    if (typeof window === "undefined") {
      setSessionState("unauthorized");
      setLoading(false);
      return;
    }

    if (isVerifyingRef.current) {
      return;
    }
    isVerifyingRef.current = true;

    const storedToken = localStorage.getItem(TOKEN_STORAGE_KEY);
    const storedUserStr = localStorage.getItem(USER_STORAGE_KEY);
    const storedRefreshToken = localStorage.getItem(REFRESH_TOKEN_STORAGE_KEY);

    if (!storedToken) {
      setUser(null);
      setToken(null);
      setSessionState("unauthorized");
      setLoading(false);
      isVerifyingRef.current = false;
      return;
    }

    // Sessiz modda değilse (kullanıcı 'Tekrar Dene' yaptıysa) yükleme durumunu aktifleştir
    if (!isSilentBackground) {
      setLoading(true);
      setSessionState("verifying");
    }

    // Maksimum 4 saniye timeout emniyeti
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("SESSION_TIMEOUT")), 4000)
    );

    try {
      const currentUser = (await Promise.race([
        api.getMe(),
        timeoutPromise,
      ])) as UserResponse;

      setUser(currentUser);
      localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(currentUser));
      setSessionState("verified");
      setError(null);
    } catch (err: any) {
      console.warn("Oturum doğrulama bildirimi:", err?.message || err);

      const status = err?.response?.status;
      if (status === 401 || status === 403) {
        // Token geçersiz veya süresi dolmuş
        if (storedRefreshToken) {
          try {
            const refreshRes = await api.refreshToken(storedRefreshToken);
            localStorage.setItem(TOKEN_STORAGE_KEY, refreshRes.access_token);
            if (refreshRes.refresh_token) {
              localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, refreshRes.refresh_token);
            }
            setToken(refreshRes.access_token);
            const verifiedUser = await api.getMe();
            setUser(verifiedUser);
            localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(verifiedUser));
            setSessionState("verified");
            setError(null);
            return;
          } catch (rfErr) {
            console.warn("Refresh token ile yenileme başarısız:", rfErr);
          }
        }

        // Kurtarılamaz 401: Oturumu temizle ve yetkisiz yap
        localStorage.removeItem(TOKEN_STORAGE_KEY);
        localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY);
        localStorage.removeItem(USER_STORAGE_KEY);
        setUser(null);
        setToken(null);
        setSessionState("unauthorized");
        setError("Oturum süreniz doldu. Lütfen tekrar giriş yapınız.");
      } else {
        // Ağ hatası veya timeout durumunda: Eğer yerel kullanıcı bilgisi varsa oturumu bozma!
        if (storedUserStr) {
          try {
            const cachedUser = JSON.parse(storedUserStr);
            setUser(cachedUser);
            setSessionState("verified");
            // Kullanıcıyı engelleme, arka planda cached oturum ile devam et
          } catch {
            setSessionState("error");
            setError(formatApiError(err));
          }
        } else {
          setSessionState("error");
          setError(formatApiError(err));
        }
      }
    } finally {
      setLoading(false);
      isVerifyingRef.current = false;
    }
  }, []);

  // 1. ANLIK EŞZAMANLI YEREL HYDRATION (Sayfa açılır açılmaz spinner olmadan anında açılış)
  useEffect(() => {
    if (typeof window === "undefined") return;

    const storedToken = localStorage.getItem(TOKEN_STORAGE_KEY);
    const storedUserStr = localStorage.getItem(USER_STORAGE_KEY);

    if (storedToken && storedUserStr) {
      try {
        const cachedUser = JSON.parse(storedUserStr);
        setUser(cachedUser);
        setToken(storedToken);
        setSessionState("verified");
        setLoading(false);
      } catch {
        localStorage.removeItem(USER_STORAGE_KEY);
        setLoading(false);
      }
    } else if (!storedToken) {
      setUser(null);
      setToken(null);
      setSessionState("unauthorized");
      setLoading(false);
    } else {
      setLoading(false);
    }

    // 2. Arka planda sessiz revalidasyon
    verifySession(true);

    // 3. Global acil durum güvenlik zamanlayıcısı (loading asla 2 saniyeden uzun kalamaz)
    const safetyTimer = setTimeout(() => {
      setLoading(false);
      isVerifyingRef.current = false;
    }, 2000);

    return () => {
      clearTimeout(safetyTimer);
    };
  }, [verifySession]);

  // Korumalı Rota Yönlendirmesi
  useEffect(() => {
    if (loading) return;

    const isLoginPage = pathname === "/login";

    if (!user && !isLoginPage && (sessionState === "unauthorized" || sessionState === "idle" || sessionState === "expired")) {
      router.replace("/login");
    } else if (user && isLoginPage && sessionState === "verified") {
      router.replace("/");
    }
  }, [user, loading, pathname, router, sessionState]);

  const isAuthenticated = Boolean(user && token && sessionState === "verified");

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        sessionState,
        loading,
        error,
        isAuthenticated,
        login,
        logout,
        hasRole,
        clearError,
        retrySessionVerification: () => verifySession(false),
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
