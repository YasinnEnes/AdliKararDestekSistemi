"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import {
  api,
  formatApiError,
  UserResponse,
  TokenResponse,
  TOKEN_STORAGE_KEY,
  USER_STORAGE_KEY,
} from "@/services/api";

interface AuthContextType {
  user: UserResponse | null;
  token: string | null;
  loading: boolean;
  error: string | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  hasRole: (allowedRoles: string[]) => boolean;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const router = useRouter();
  const pathname = usePathname();

  const [user, setUser] = useState<UserResponse | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const clearError = () => setError(null);

  // Oturumu Güvenli Şekilde Temizle (Logout)
  const logout = useCallback(() => {
    if (typeof window !== "undefined") {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      localStorage.removeItem(USER_STORAGE_KEY);
    }
    setUser(null);
    setToken(null);
    setError(null);
    router.push("/login");
  }, [router]);

  // Giriş Yap (Login)
  const login = async (username: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const res: TokenResponse = await api.login(username, password);
      if (typeof window !== "undefined") {
        localStorage.setItem(TOKEN_STORAGE_KEY, res.access_token);
        localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(res.user));
      }
      setToken(res.access_token);
      setUser(res.user);
      router.push("/");
    } catch (err: any) {
      const msg = formatApiError(err);
      setError(msg);
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

  // Sayfa İlk Yüklendiğinde Oturum Durumunu Doğrula
  useEffect(() => {
    const initializeAuth = async () => {
      if (typeof window === "undefined") {
        setLoading(false);
        return;
      }

      const storedToken = localStorage.getItem(TOKEN_STORAGE_KEY);
      const storedUserStr = localStorage.getItem(USER_STORAGE_KEY);

      if (!storedToken) {
        setUser(null);
        setToken(null);
        setLoading(false);
        return;
      }

      setToken(storedToken);

      if (storedUserStr) {
        try {
          setUser(JSON.parse(storedUserStr));
        } catch {
          // ignore
        }
      }

      // Backend'den profil geçerliliğini teyit et
      try {
        const currentUser = await api.getMe();
        setUser(currentUser);
        localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(currentUser));
      } catch (err: any) {
        console.warn("Oturum doğrulanamadı, yeniden giriş bekleniyor:", err?.message);
        localStorage.removeItem(TOKEN_STORAGE_KEY);
        localStorage.removeItem(USER_STORAGE_KEY);
        setUser(null);
        setToken(null);
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();
  }, []);

  // Protected Routes Yönlendirmesi
  useEffect(() => {
    if (loading) return;

    const isLoginPage = pathname === "/login";
    const isPublic = isLoginPage;

    if (!user && !isPublic) {
      router.push("/login");
    } else if (user && isLoginPage) {
      router.push("/");
    }
  }, [user, loading, pathname, router]);

  const isAuthenticated = Boolean(user && token);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        error,
        isAuthenticated,
        login,
        logout,
        hasRole,
        clearError,
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
