import { useState } from "react";
import { useAuth } from "../context/AuthContext";

export default function ResetPasswordModal({ token, onClose, onDone }) {
  const { resetPassword } = useAuth();
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [done, setDone] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    const result = await resetPassword(token, password);
    setSubmitting(false);
    if (result.ok) {
      setDone(true);
    } else {
      setError(result.error);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        <h3 className="modal-title">Set a new password</h3>

        {done ? (
          <>
            <p className="modal-note">Your password has been reset. You can log in with it now.</p>
            <button className="btn btn-primary" onClick={onDone}>Log in</button>
          </>
        ) : (
          <form onSubmit={handleSubmit} className="modal-form">
            <label>New password
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={8}
                autoComplete="new-password"
              />
            </label>
            <p className="modal-hint">At least 8 characters. This reset link expires 30 minutes after it was sent.</p>
            {error && <p className="modal-error">{error}</p>}
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting ? "Please wait..." : "Reset password"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}