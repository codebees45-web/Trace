import { useState } from "react";
import { useAuth } from "../context/AuthContext";

export default function AuthModal({ onClose }) {
  const { login, register, authError, forgotPassword } = useAuth();
  const [mode, setMode] = useState("login"); // "login" | "register" | "forgot"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [forgotError, setForgotError] = useState(null);
  const [forgotSent, setForgotSent] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    if (mode === "forgot") {
      setForgotError(null);
      const result = await forgotPassword(email);
      setSubmitting(false);
      if (result.ok) setForgotSent(true);
      else setForgotError(result.error);
      return;
    }
    const ok = mode === "login" ? await login(email, password) : await register(email, password);
    setSubmitting(false);
    if (ok) onClose();
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        {mode !== "forgot" && (
          <div className="modal-tabs">
            <button className={mode === "login" ? "active" : ""} onClick={() => setMode("login")} type="button">Log in</button>
            <button className={mode === "register" ? "active" : ""} onClick={() => setMode("register")} type="button">Sign up</button>
          </div>
        )}
        {mode === "forgot" && <h3 className="modal-title">Reset your password</h3>}

        {mode === "forgot" && forgotSent ? (
          <p className="modal-note">
            If <strong>{email}</strong> has an account, a reset link is on its way. It expires in 30 minutes.
          </p>
        ) : (
          <form onSubmit={handleSubmit} className="modal-form">
            <label>Email
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
            </label>
            {mode !== "forgot" && (
              <label>Password
                <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8}
                  autoComplete={mode === "login" ? "current-password" : "new-password"} />
              </label>
            )}
            {mode === "register" && <p className="modal-hint">At least 8 characters.</p>}
            {mode === "login" && (
              <button type="button" className="link-btn modal-forgot-link" onClick={() => setMode("forgot")}>
                Forgot password?
              </button>
            )}
            {mode === "forgot" && <p className="modal-hint">We'll send a reset link if that email has an account.</p>}
            {(mode === "forgot" ? forgotError : authError) && (
              <p className="modal-error">{mode === "forgot" ? forgotError : authError}</p>
            )}
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting
                ? "Please wait..."
                : mode === "login" ? "Log in" : mode === "register" ? "Create account" : "Send reset link"}
            </button>
            {mode === "forgot" && (
              <button type="button" className="link-btn" onClick={() => setMode("login")}>
                Back to log in
              </button>
            )}
          </form>
        )}

        {mode !== "forgot" && (
          <p className="modal-note">
            Signed-in scans are saved to your history. Guests can still run the demo — nothing is saved without an account.
          </p>
        )}
      </div>
    </div>
  );
}