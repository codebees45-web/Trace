import { useEffect, useState } from "react";
import api from "../api";
import { useAuth } from "../context/AuthContext";

function timeAgo(iso) {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

export default function HistoryPanel({ onClose }) {
  const { user } = useAuth();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await api.get("/history");
        if (!cancelled) setItems(res.data);
      } catch (err) {
        if (!cancelled) {
          setError(err.response?.data?.detail || "Couldn't load your history.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleDelete = async (id) => {
    setDeletingId(id);
    try {
      await api.delete(`/history/${id}`);
      setItems((prev) => prev.filter((it) => it.id !== id));
    } catch (err) {
      console.error(err);
      setError("Couldn't delete that scan — try again.");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card history-card" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        <div className="history-head">
          <h3>Your scan history</h3>
          <p className="modal-note">Signed in as {user?.email}. Only you can see these.</p>
        </div>

        {loading && <p className="stage-text">Loading history...</p>}
        {error && <p className="stage-text stage-error">{error}</p>}

        {!loading && !error && items.length === 0 && (
          <p className="history-empty">
            No scans saved yet — run a prediction above while signed in and it'll show up here.
          </p>
        )}

        <div className="history-list">
          {items.map((item) => {
            const isUnknown = item.identity === "Unknown";
            return (
              <div className="history-row" key={item.id}>
                <div className="history-thumbs">
                  {item.input_thumb && (
                    <img
                      src={`data:image/jpeg;base64,${item.input_thumb}`}
                      alt="Masked input thumbnail"
                      className="history-thumb"
                    />
                  )}
                  {item.generated_thumb && (
                    <img
                      src={`data:image/jpeg;base64,${item.generated_thumb}`}
                      alt="Reconstructed face thumbnail"
                      className="history-thumb"
                    />
                  )}
                </div>
                <div className="history-meta">
                  <span className={"history-identity" + (isUnknown ? " identity-name-unknown" : "")}>
                    {isUnknown ? "No confident match" : item.identity}
                  </span>
                  <span className="history-sub">
                    {(item.confidence * 100).toFixed(0)}% confidence · {timeAgo(item.created_at)}
                  </span>
                </div>
                <button
                  className="history-delete"
                  onClick={() => handleDelete(item.id)}
                  disabled={deletingId === item.id}
                  aria-label="Delete scan"
                  title="Delete this scan"
                >
                  {deletingId === item.id ? "…" : "🗑"}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}