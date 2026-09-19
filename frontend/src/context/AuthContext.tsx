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

  const verificationAbortControllerRef = useRef<AbortController | null>(null);
  const isVerifyingRef = useRef<boolean>(false);

  const clearError = () => setError(null);

  // Oturumu Güvenli Şekilde Temizle (Logout)
  const logout = useCallback(() => {
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
    setSessionState("idle");
    setLoading(false);
    router.push("/login");
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
      router.push("/");
    } catch (err: any) {
      const msg = formatApiError(err);
      setError(msg);
      setSessionState("unauthorized");
      throw err;
    } finally {
      setLoading(false);
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

  // Oturum Doğrulama Fonksiyonu (Deterministik State Machine + Timeout Koruması)
  const verifySession = useCallback(async () => {
    if (typeof window === "undefined") {
      setSessionState("idle");
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
      setSessionState("idle");
      setLoading(false);
      isVerifyingRef.current = false;
      return;
    }

    setToken(storedToken);
    setSessionState("verifying");
    setLoading(true);

    if (storedUserStr) {
      try {
        setUser(JSON.parse(storedUserStr));
      } catch {
        // Hatalı JSON durumunda yut
      }
    }

    // Zaman Aşımı Emniyeti (Timeout): 8 saniye içinde yanıt alınamazsa kilitlenmeyi engelle
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("SESSION_TIMEOUT")), 8000)
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
      console.warn("Oturum doğrulama hatası:", err?.message || err);

      if (err?.message === "SESSION_TIMEOUT") {
        setSessionState("error");
        setError("Adli oturum sunucudan zamanında doğrulanamadı. Lütfen ağ bağlantınızı veya sunucu durumunu kontrol edin.");
      } else if (err?.response?.status === 401 || err?.response?.status === 403) {
        // Refresh token ile kurtarma dene
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
            isVerifyingRef.current = false;
            setLoading(false);
            return;
          } catch (rfErr) {
            console.warn("Refresh token yenileme de başarısız oldu:", rfErr);
          }
        }

        // Kurtarılamaz 401/403
        localStorage.removeItem(TOKEN_STORAGE_KEY);
        localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY);
        localStorage.removeItem(USER_STORAGE_KEY);
        setUser(null);
        setToken(null);
        setSessionState("unauthorized");
        setError("Oturum süresi doldu veya yetkisiz erişim. Lütfen tekrar giriş yapın.");
      } else {
        // Ağ veya sunucu bağlantı hatası
        setSessionState("error");
        setError(formatApiError(err));
      }
    } finally {
      setLoading(false);
      isVerifyingRef.current = false;
    }
  }, []);

  // Sayfa İlk Yüklendiğinde Oturum Durumunu Doğrula
  useEffect(() => {
    verifySession();

    return () => {
      if (verificationAbortControllerRef.current) {
        verificationAbortControllerRef.current.abort();
      }
    };
  }, [verifySession]);

  // Protected Routes Yönlendirmesi
  useEffect(() => {
    if (loading) return;

    const isLoginPage = pathname === "/login";
    const isPublic = isLoginPage;

    if (!user && !isPublic && (sessionState === "idle" || sessionState === "unauthorized" || sessionState === "expired")) {
      router.push("/login");
    } else if (user && isLoginPage && sessionState === "verified") {
      router.push("/");
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
        retrySessionVerification: verifySession,
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
