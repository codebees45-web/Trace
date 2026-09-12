import React, { useState, useEffect, useCallback, useRef } from "react";
import api from "../api";

// Enroll 2+ people from a couple of reference photos each, then run a
// masked-face video against that gallery. Mirrors the /enroll and
// /identify/video endpoints in backend/main.py — enrollment is additive
// (call /enroll again to add more photos to an existing identity) and the
// gallery persists across page reloads since it's stored server-side.
//
// CHANGE: /enroll now auto-generates a synthetic-masked embedding per
// photo (see backend/mask_synth.py). This component now calls the new
// /enroll/preview endpoint as soon as photos are selected, so the person
// can see exactly what the synthetic mask looks like on each photo
// *before* committing to Enroll — a badly-positioned mask (extreme angle,
// glasses throwing off landmarks, etc.) is obvious at a glance but not
// worth guessing about from the embeddings alone.
const VideoIdentify = () => {
  const [identities, setIdentities] = useState([]);
  const [identitiesLoading, setIdentitiesLoading] = useState(true);

  const [newIdentityName, setNewIdentityName] = useState("");
  const [newIdentityFiles, setNewIdentityFiles] = useState([]);
  const [enrolling, setEnrolling] = useState(false);
  const [enrollError, setEnrollError] = useState(null);
  const [enrollNotice, setEnrollNotice] = useState(null);

  // One entry per selected file: { name, sourceUrl, previewUrl, previewError, previewLoading }
  const [maskPreviews, setMaskPreviews] = useState([]);
  const objectUrlsRef = useRef([]); // tracks every blob: URL we create, so we can revoke them on cleanup

  const [videoFile, setVideoFile] = useState(null);
  const [identifying, setIdentifying] = useState(false);
  const [identifyError, setIdentifyError] = useState(null);
  const [videoResult, setVideoResult] = useState(null);

  const refreshIdentities = useCallback(async () => {
    setIdentitiesLoading(true);
    try {
      const res = await api.get("/enroll/identities");
      setIdentities(res.data.identities || []);
    } catch (err) {
      console.error("Failed to load enrolled identities", err);
    } finally {
      setIdentitiesLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshIdentities();
  }, [refreshIdentities]);

  // Revoke any blob: URLs we created whenever the component unmounts, so
  // we don't leak memory across repeated file selections.
  useEffect(() => {
    return () => {
      objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  const clearPreviews = () => {
    objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlsRef.current = [];
    setMaskPreviews([]);
  };

  const handleFilesSelected = async (files) => {
    setNewIdentityFiles(files);
    clearPreviews();
    if (files.length === 0) return;

    // Show the original photo immediately, then fill in the masked
    // preview per-file as each /enroll/preview call resolves — the user
    // shouldn't wait for every photo to finish before seeing anything.
    const initial = files.map((f) => {
      const sourceUrl = URL.createObjectURL(f);
      objectUrlsRef.current.push(sourceUrl);
      return { name: f.name, sourceUrl, previewUrl: null, previewError: null, previewLoading: true };
    });
    setMaskPreviews(initial);

    files.forEach(async (f, idx) => {
      try {
        const formData = new FormData();
        formData.append("file", f);
        const res = await api.post("/enroll/preview", formData, {
          headers: { "Content-Type": "multipart/form-data" },
          responseType: "blob",
        });
        const previewUrl = URL.createObjectURL(res.data);
        objectUrlsRef.current.push(previewUrl);
        setMaskPreviews((prev) =>
          prev.map((p, i) => (i === idx ? { ...p, previewUrl, previewLoading: false } : p))
        );
      } catch (err) {
        const detail = err.response?.data
          ? await err.response.data.text?.().catch(() => null)
          : null;
        let message = "Couldn't generate a mask preview for this photo.";
        try {
          if (detail) message = JSON.parse(detail).detail || message;
        } catch {
          /* non-JSON error body — keep the generic message */
        }
        setMaskPreviews((prev) =>
          prev.map((p, i) => (i === idx ? { ...p, previewError: message, previewLoading: false } : p))
        );
      }
    });
  };

  const handleEnroll = async (e) => {
    e.preventDefault();
    if (!newIdentityName.trim() || newIdentityFiles.length === 0 || enrolling) return;

    setEnrolling(true);
    setEnrollError(null);
    setEnrollNotice(null);

    try {
      const formData = new FormData();
      newIdentityFiles.forEach((f) => formData.append("files", f));

      const res = await api.post(
        `/enroll?identity=${encodeURIComponent(newIdentityName.trim())}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setEnrollNotice(
        `Enrolled "${res.data.identity}" — ${res.data.embeddings_added} embeddings added ` +
          `(includes auto-generated masked variants; ${res.data.total_identities} identities total).`
      );
      setNewIdentityName("");
      setNewIdentityFiles([]);
      clearPreviews();
      await refreshIdentities();
    } catch (err) {
      setEnrollError(err.response?.data?.detail || "Enrollment failed — try again.");
    } finally {
      setEnrolling(false);
    }
  };

  const handleDeleteIdentity = async (identity) => {
    try {
      await api.delete(`/enroll/${encodeURIComponent(identity)}`);
      await refreshIdentities();
    } catch (err) {
      console.error("Failed to delete identity", identity, err);
    }
  };

  const handleIdentifyVideo = async () => {
    if (!videoFile || identifying) return;

    setIdentifying(true);
    setIdentifyError(null);
    setVideoResult(null);

    try {
      const formData = new FormData();
      formData.append("file", videoFile, videoFile.name);

      const res = await api.post("/identify/video", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        // Video processing with InsightFace (SCRFD + ArcFace) on CPU can
        // take 30-120+ seconds depending on video length. The default
        // axios/browser timeout is too short — give it up to 5 minutes.
        timeout: 5 * 60 * 1000,
      });
      setVideoResult(res.data);
    } catch (err) {
      if (err.code === "ECONNABORTED" || err.message?.includes("timeout")) {
        setIdentifyError(
          "The video took too long to process and the request timed out. " +
          "Try a shorter video or a lower resolution."
        );
      } else if (err.response?.status === 415) {
        setIdentifyError(
          "Unsupported video format. Please upload an MP4, MOV, AVI, or WebM video."
        );
      } else if (err.response?.status === 422) {
        setIdentifyError(
          err.response?.data?.detail ||
          "Could not process the video — make sure identities are enrolled and the video contains faces."
        );
      } else if (err.response?.status === 503) {
        setIdentifyError(
          "The video pipeline is not available on the server. " +
          "Make sure insightface and onnxruntime are installed."
        );
      } else {
        setIdentifyError(
          err.response?.data?.detail ||
          "Couldn't process that video — check that the backend is running and try again."
        );
      }
    } finally {
      setIdentifying(false);
    }
  };

  return (
    <div className="video-identify-container">
      {/* ---------- Enrollment ---------- */}
      <div className="enroll-panel">
        <h3>1. Enroll identities</h3>
        <p className="section-sub">
          Add each candidate with a couple of reference photos. Each photo is embedded as-is,
          plus a synthetic masked variant is generated automatically — so 2 photos become up to
          4 embeddings, spanning both the unmasked and masked domains.
        </p>

        <form onSubmit={handleEnroll} className="enroll-form">
          <input
            type="text"
            placeholder="Identity name"
            value={newIdentityName}
            onChange={(e) => setNewIdentityName(e.target.value)}
            disabled={enrolling}
          />
          <label className="upload-zone upload-zone-compact">
            <input
              type="file"
              accept="image/*"
              multiple
              hidden
              disabled={enrolling}
              onChange={(e) =>
                handleFilesSelected(e.target.files ? Array.from(e.target.files) : [])
              }
            />
            <span className="upload-icon" aria-hidden="true">↑</span>
            <span>
              {newIdentityFiles.length > 0
                ? `${newIdentityFiles.length} photo(s) selected`
                : "Select reference photos"}
            </span>
          </label>

          {maskPreviews.length > 0 && (
            <div className="mask-preview-grid">
              {maskPreviews.map((p, i) => (
                <div key={`${p.name}-${i}`} className="mask-preview-cell">
                  <div className="mask-preview-pair">
                    <div className="mask-preview-item">
                      <img src={p.sourceUrl} alt={`${p.name} original`} />
                      <span className="mask-preview-label">original</span>
                    </div>
                    <div className="mask-preview-item">
                      {p.previewLoading && <div className="mask-preview-placeholder">generating…</div>}
                      {p.previewError && (
                        <div className="mask-preview-placeholder mask-preview-error" title={p.previewError}>
                          preview failed
                        </div>
                      )}
                      {p.previewUrl && <img src={p.previewUrl} alt={`${p.name} with synthetic mask`} />}
                      <span className="mask-preview-label">masked (auto)</span>
                    </div>
                  </div>
                  <span className="mask-preview-filename muted">{p.name}</span>
                </div>
              ))}
            </div>
          )}
          {maskPreviews.some((p) => p.previewError) && (
            <p className="form-notice">
              Some photos couldn't get a synthetic mask preview (usually an extreme angle or
              obscured landmarks). Those will still enroll with their original, unmasked
              embedding only — consider swapping in a clearer photo, or a real masked photo,
              for those cases.
            </p>
          )}

          <button
            type="submit"
            className="btn btn-primary"
            disabled={enrolling || !newIdentityName.trim() || newIdentityFiles.length === 0}
          >
            {enrolling ? "Enrolling..." : "Enroll"}
          </button>
        </form>

        {enrollError && <p className="form-error">{enrollError}</p>}
        {enrollNotice && <p className="form-notice">{enrollNotice}</p>}

        <div className="enrolled-list">
          <h4>Enrolled gallery</h4>
          {identitiesLoading ? (
            <p className="muted">Loading...</p>
          ) : identities.length === 0 ? (
            <p className="muted">No identities enrolled yet.</p>
          ) : (
            <ul>
              {identities.map((identity) => (
                <li key={identity} className="enrolled-row">
                  <span>{identity}</span>
                  <button
                    className="remove-btn"
                    aria-label={`Remove ${identity}`}
                    onClick={() => handleDeleteIdentity(identity)}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* ---------- Video identification ---------- */}
      <div className="video-panel">
        <h3>2. Identify from video</h3>
        <p className="section-sub">
          Upload the masked-face video. Frames are sampled evenly across it and
          matched against the gallery above, with a quality-weighted vote deciding
          the final identity.
        </p>

        <label className="upload-zone">
          <input
            type="file"
            accept="video/*"
            hidden
            onChange={(e) => {
              setVideoFile(e.target.files?.[0] || null);
              setVideoResult(null);
              setIdentifyError(null);
            }}
          />
          <span className="upload-icon" aria-hidden="true">🎬</span>
          <span>{videoFile ? videoFile.name : "Drop a video or click to upload"}</span>
        </label>

        <button
          className="btn btn-primary"
          onClick={handleIdentifyVideo}
          disabled={!videoFile || identifying || identities.length === 0}
        >
          {identifying ? "Analyzing video..." : "Identify"}
        </button>
        {identities.length === 0 && (
          <p className="muted">Enroll at least one identity before identifying a video.</p>
        )}

        {identifyError && <p className="form-error">{identifyError}</p>}

        {videoResult && (
          <div className="video-result-card">
            <div className="video-result-headline">
              <span className="video-result-identity">{videoResult.identity}</span>
              {typeof videoResult.confidence === "number" && (
                <span className="video-result-confidence">
                  {(videoResult.confidence * 100).toFixed(1)}% confidence
                </span>
              )}
            </div>
            {typeof videoResult.raw_confidence === "number" && (
              <p className="muted">
                raw similarity: {videoResult.raw_confidence.toFixed(3)}{" "}
                <span title="Uncalibrated cosine similarity, before the logistic calibration curve. Useful for re-tuning calibrate.py.">
                  (?)
                </span>
              </p>
            )}
            <p className="muted">
              {videoResult.frames_used} usable frame(s) analyzed
              {videoResult.detail ? ` — ${videoResult.detail}` : ""}
            </p>

            {videoResult.candidates && videoResult.candidates.length > 0 && (
              <ul className="candidate-list">
                {videoResult.candidates.map((c) => (
                  <li key={c.identity}>
                    <span>{c.identity}</span>
                    <span className="muted">score {c.score}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoIdentify;