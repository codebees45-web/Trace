import { createContext, useContext, useState, useEffect, useCallback } from "react";
import api from "../api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  const loadMe = useCallback(async () => {
    const token = localStorage.getItem("trace_token");
    if (!token) { setLoading(false); return; }
    try {
      const res = await api.get("/auth/me");
      setUser(res.data);
    } catch {
      localStorage.removeItem("trace_token");
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadMe(); }, [loadMe]);

  const login = async (email, password) => {
    setAuthError(null);
    const form = new URLSearchParams();
    form.append("username", email);
    form.append("password", password);
    try {
      const res = await api.post("/auth/login", form, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });
      localStorage.setItem("trace_token", res.data.access_token);
      await loadMe();
      return true;
    } catch (err) {
      setAuthError(err.response?.data?.detail || "Login failed");
      return false;
    }
  };

  const register = async (email, password) => {
    setAuthError(null);
    try {
      await api.post("/auth/register", { email, password });
      return login(email, password);
    } catch (err) {
      setAuthError(err.response?.data?.detail || "Registration failed");
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem("trace_token");
    setUser(null);
  };

  const changePassword = async (currentPassword, newPassword) => {
    try {
      await api.post("/auth/change-password", {
        current_password: currentPassword,
        new_password: newPassword,
      });
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err.response?.data?.detail || "Couldn't change password" };
    }
  };

  const forgotPassword = async (email) => {
    try {
      await api.post("/auth/forgot-password", { email });
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err.response?.data?.detail || "Something went wrong" };
    }
  };

  const resetPassword = async (token, newPassword) => {
    try {
      await api.post("/auth/reset-password", { token, new_password: newPassword });
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err.response?.data?.detail || "That reset link isn't valid" };
    }
  };

  const deleteAccount = async () => {
    try {
      await api.delete("/auth/me");
      logout();
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err.response?.data?.detail || "Couldn't delete your account" };
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        authError,
        login,
        register,
        logout,
        changePassword,
        forgotPassword,
        resetPassword,
        deleteAccount,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}