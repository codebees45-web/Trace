import { useState } from "react";
import { useAuth } from "../context/AuthContext";

export default function SettingsPanel({ onClose }) {
  const { user, changePassword, deleteAccount } = useAuth();

  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [pwSubmitting, setPwSubmitting] = useState(false);
  const [pwError, setPwError] = useState(null);
  const [pwSuccess, setPwSuccess] = useState(false);

  const [confirmText, setConfirmText] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const handleChangePassword = async (e) => {
    e.preventDefault();
    setPwSubmitting(true);
    setPwError(null);
    setPwSuccess(false);
    const result = await changePassword(currentPw, newPw);
    setPwSubmitting(false);
    if (result.ok) {
      setPwSuccess(true);
      setCurrentPw("");
      setNewPw("");
    } else {
      setPwError(result.error);
    }
  };

  const handleDelete = async () => {
    setDeleting(true);
    setDeleteError(null);
    const result = await deleteAccount();
    setDeleting(false);
    if (result.ok) {
      onClose();
    } else {
      setDeleteError(result.error);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card history-card" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        <div className="history-head">
          <h3>Account settings</h3>
          <p className="modal-note">Signed in as {user?.email}</p>
        </div>

        <form onSubmit={handleChangePassword} className="modal-form settings-section">
          <h4 className="settings-subhead">Change password</h4>
          <label>Current password
            <input type="password" value={currentPw} onChange={(e) => setCurrentPw(e.target.value)}
              required autoComplete="current-password" />
          </label>
          <label>New password
            <input type="password" value={newPw} onChange={(e) => setNewPw(e.target.value)}
              required minLength={8} autoComplete="new-password" />
          </label>
          {pwError && <p className="modal-error">{pwError}</p>}
          {pwSuccess && <p className="modal-success">Password updated.</p>}
          <button type="submit" className="btn btn-primary" disabled={pwSubmitting}>
            {pwSubmitting ? "Saving..." : "Update password"}
          </button>
        </form>

        <div className="settings-section settings-danger">
          <h4 className="settings-subhead">Delete account</h4>
          <p className="modal-note">
            Permanently deletes your account and every scan in your history. This can't be undone.
          </p>
          {!showDeleteConfirm ? (
            <button className="btn btn-danger-ghost" onClick={() => setShowDeleteConfirm(true)}>
              Delete my account
            </button>
          ) : (
            <div className="settings-danger-confirm">
              <label>Type <strong>DELETE</strong> to confirm
                <input value={confirmText} onChange={(e) => setConfirmText(e.target.value)} />
              </label>
              {deleteError && <p className="modal-error">{deleteError}</p>}
              <div className="settings-danger-actions">
                <button
                  className="btn btn-danger"
                  disabled={confirmText !== "DELETE" || deleting}
                  onClick={handleDelete}
                >
                  {deleting ? "Deleting..." : "Permanently delete"}
                </button>
                <button className="link-btn" onClick={() => { setShowDeleteConfirm(false); setConfirmText(""); }}>
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}