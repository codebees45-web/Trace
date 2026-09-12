/**
 * LiveCamera.jsx
 *
 * Real-time face recognition from the user's webcam.
 *
 * Architecture:
 *   webcam → <video> → canvas.toBlob(jpeg) every ~100ms
 *         → WebSocket binary message [4-byte frame_id + JPEG]
 *         → backend /ws/live
 *         → JSON { faces, latency_ms, frame_id, fps }
 *         → overlay canvas draws bounding boxes + identity labels
 *
 * Falls back to HTTP POST /predict/frame if WebSocket is unavailable.
 */
import React, { useRef, useState, useEffect, useCallback } from "react";
import api from "../api";

// Base URL for WebSocket (swap http(s) → ws(s)).
function getWsUrl(path) {
  const base = (api.defaults.baseURL || "http://localhost:8000").replace(
    /^http/,
    "ws"
  );
  return `${base}${path}`;
}

// ─── colour palette for identity labels ──────────────────────────────────────
const IDENTITY_COLORS = [
  "#6EE7B7", // emerald
  "#93C5FD", // sky
  "#FCA5A5", // rose
  "#FCD34D", // amber
  "#A78BFA", // violet
  "#34D399", // green
  "#F472B6", // pink
  "#60A5FA", // blue
];
const colorForIdentity = (() => {
  const cache = {};
  let idx = 0;
  return (id) => {
    if (!cache[id]) cache[id] = IDENTITY_COLORS[idx++ % IDENTITY_COLORS.length];
    return cache[id];
  };
})();

const CAPTURE_QUALITY = 0.80;   // JPEG quality — high enough for reliable face detection
const TARGET_W = 640;           // must match backend _process_live_frame target_w=640
const FRAME_INTERVAL_MS = 100;  // ~10 fps capture rate

export default function LiveCamera() {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null); // off-screen canvas for frame capture
  const overlayCanvasRef = useRef(null); // visible overlay for face boxes
  const wsRef = useRef(null);
  const frameIdRef = useRef(0);
  const animFrameRef = useRef(null);
  const lastSendRef = useRef(0);
  const pendingRef = useRef(false);   // true while a frame is in flight (HTTP fallback)
  const streamRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [usingFallback, setUsingFallback] = useState(false);
  const [stats, setStats] = useState({ latency: null, fps: null, detected: 0 });
  const [lastFaces, setLastFaces] = useState([]);

  // ─── draw overlay ───────────────────────────────────────────────────────────
  const drawOverlay = useCallback((faces, videoEl, canvas) => {
    if (!canvas || !videoEl) return;
    const vw = videoEl.videoWidth  || videoEl.clientWidth;
    const vh = videoEl.videoHeight || videoEl.clientHeight;
    const dw = canvas.offsetWidth;
    const dh = canvas.offsetHeight;

    canvas.width  = dw;
    canvas.height = dh;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, dw, dh);

    if (!faces || faces.length === 0) return;

    // The frame sent to the backend was resized to TARGET_W; we need to map
    // coordinates back to the displayed video size.
    const sentW  = Math.min(vw, TARGET_W);
    const sentH  = Math.round(vh * (sentW / vw));
    const scaleX = dw / sentW;
    const scaleY = dh / sentH;

    // Apply a horizontal mirror on the canvas so the webcam looks like a
    // selfie camera (matching natural user expectation). Text is drawn with
    // a local counter-transform so it stays readable.
    ctx.save();
    ctx.translate(dw, 0);
    ctx.scale(-1, 1);

    faces.forEach((face) => {
      const x = face.x * scaleX;
      const y = face.y * scaleY;
      const w = face.w * scaleX;
      const h = face.h * scaleY;

      const isUnknown  = face.identity === "Unknown";
      const color      = isUnknown ? "#94A3B8" : colorForIdentity(face.identity);
      const label      = isUnknown
        ? "Unknown"
        : `${face.identity} ${(face.confidence * 100).toFixed(0)}%`;

      // Bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2.5;
      ctx.shadowColor = color;
      ctx.shadowBlur  = 8;
      ctx.strokeRect(x, y, w, h);
      ctx.shadowBlur  = 0;

      // Corner accents
      const cs = Math.min(w, h) * 0.18;
      ctx.lineWidth = 3;
      [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].forEach(([cx, cy], i) => {
        ctx.beginPath();
        if (i === 0) { ctx.moveTo(cx + cs, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + cs); }
        if (i === 1) { ctx.moveTo(cx - cs, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + cs); }
        if (i === 2) { ctx.moveTo(cx + cs, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy - cs); }
        if (i === 3) { ctx.moveTo(cx - cs, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy - cs); }
        ctx.strokeStyle = color;
        ctx.stroke();
      });

      // Label pill — drawn with a counter-mirror so text is readable.
      ctx.font = "bold 13px 'Inter', sans-serif";
      const tw = ctx.measureText(label).width;
      const pad = 6;
      const lh  = 22;
      const lx  = x;
      const ly  = y > lh + 4 ? y - lh - 4 : y + h + 4;

      // Draw pill background (still in mirrored space — looks fine).
      ctx.fillStyle = color + "DD";
      ctx.beginPath();
      ctx.roundRect(lx, ly, tw + pad * 2, lh, 5);
      ctx.fill();

      // Counter-mirror just for the text so it reads left-to-right.
      ctx.save();
      ctx.scale(-1, 1);
      ctx.fillStyle = "#0F172A";
      // The mirrored x of the pill's left edge is -(lx + pad); the pill right
      // edge in mirrored space is -(lx + tw + pad). We want text to start at
      // the left edge of the pill, which in counter-mirrored coords is:
      ctx.fillText(label, -(lx + tw + pad), ly + lh - 5);
      ctx.restore();
    });

    ctx.restore(); // undo the global scaleX(-1) mirror
  }, []);


  // ─── send one frame ─────────────────────────────────────────────────────────
  const sendFrame = useCallback(() => {
    const video  = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;

    const now = performance.now();
    if (now - lastSendRef.current < FRAME_INTERVAL_MS) return;

    // Resize for capture
    const cw = Math.min(vw, TARGET_W);
    const ch = Math.round(vh * (cw / vw));
    canvas.width  = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, cw, ch);

    lastSendRef.current = now;

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const frameId = frameIdRef.current++;

        // ── WebSocket path ──────────────────────────────────────────────
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then((ab) => {
            const header = new Uint8Array(4);
            new DataView(header.buffer).setInt32(0, frameId, true);
            const msg = new Uint8Array(4 + ab.byteLength);
            msg.set(header, 0);
            msg.set(new Uint8Array(ab), 4);
            try { wsRef.current.send(msg.buffer); } catch (_) {}
          });
          return;
        }

        // ── HTTP fallback ───────────────────────────────────────────────
        if (pendingRef.current) return; // skip if previous request still in flight
        pendingRef.current = true;
        const form = new FormData();
        form.append("file", blob, "frame.jpg");
        api
          .post("/predict/frame", form, {
            headers: { "Content-Type": "multipart/form-data", "X-Frame-ID": String(frameId) },
            timeout: 3000,
          })
          .then((res) => handleResult(res.data))
          .catch(() => {})
          .finally(() => { pendingRef.current = false; });
      },
      "image/jpeg",
      CAPTURE_QUALITY
    );
  }, []);

  const handleResult = useCallback(
    (data) => {
      setLastFaces(data.faces || []);
      setStats({
        latency : data.latency_ms,
        fps     : data.fps ?? null,
        detected: (data.faces || []).length,
      });
      drawOverlay(data.faces, videoRef.current, overlayCanvasRef.current);
    },
    [drawOverlay]
  );

  // ─── WebSocket lifecycle ─────────────────────────────────────────────────
  const connectWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
    }

    const url = getWsUrl("/ws/live");
    let ws;
    try { ws = new WebSocket(url); }
    catch (_) { setUsingFallback(true); return; }
    wsRef.current = ws;
    ws.binaryType = "arraybuffer";

    ws.onopen  = ()  => { setWsConnected(true); setUsingFallback(false); };
    ws.onclose = ()  => { setWsConnected(false); setUsingFallback(true); };
    ws.onerror = ()  => { setWsConnected(false); setUsingFallback(true); };
    ws.onmessage = (ev) => {
      try { handleResult(JSON.parse(ev.data)); } catch (_) {}
    };
  }, [handleResult]);

  // ─── animation loop ──────────────────────────────────────────────────────
  const loop = useCallback(() => {
    sendFrame();
    animFrameRef.current = requestAnimationFrame(loop);
  }, [sendFrame]);

  // ─── start / stop camera ─────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      connectWs();
      setCameraActive(true);
      animFrameRef.current = requestAnimationFrame(loop);
    } catch (err) {
      setCameraError(
        err.name === "NotAllowedError"
          ? "Camera permission denied. Please allow camera access and try again."
          : `Could not open camera: ${err.message}`
      );
    }
  }, [connectWs, loop]);

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close(); wsRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
    setWsConnected(false);
    setCameraActive(false);
    setLastFaces([]);
    setStats({ latency: null, fps: null, detected: 0 });
    // Clear overlay
    const oc = overlayCanvasRef.current;
    if (oc) oc.getContext("2d").clearRect(0, 0, oc.width, oc.height);
  }, []);

  // cleanup on unmount
  useEffect(() => () => stopCamera(), [stopCamera]);

  // ─── render ───────────────────────────────────────────────────────────────
  return (
    <div className="live-camera-container">
      <div className="live-camera-header">
        <h3 className="live-camera-title">
          <span className={`live-dot${cameraActive ? " live-dot--active" : ""}`} />
          Live Camera Recognition
        </h3>
        <p className="live-camera-sub">
          Millisecond-class face detection and identity prediction directly from your
          webcam — no upload required.
        </p>
      </div>

      {/* ── Video viewport ── */}
      <div className="live-viewport">
        <video
          ref={videoRef}
          className="live-video"
          playsInline
          muted
          autoPlay
        />
        <canvas ref={overlayCanvasRef} className="live-overlay" />

        {/* off-screen capture canvas (never displayed) */}
        <canvas ref={captureCanvasRef} style={{ display: "none" }} />

        {/* No-camera placeholder */}
        {!cameraActive && (
          <div className="live-placeholder">
            <div className="live-placeholder-icon">📷</div>
            <p>Camera not active</p>
          </div>
        )}

        {/* Stats HUD */}
        {cameraActive && (
          <div className="live-hud">
            <span className="live-hud-chip">
              {wsConnected
                ? <span className="hud-dot hud-dot--green" />
                : usingFallback
                ? <span className="hud-dot hud-dot--yellow" />
                : <span className="hud-dot hud-dot--red" />}
              {wsConnected ? "WebSocket" : usingFallback ? "HTTP fallback" : "Connecting…"}
            </span>
            {stats.latency !== null && (
              <span className="live-hud-chip">⚡ {stats.latency.toFixed(0)} ms</span>
            )}
            {stats.fps !== null && stats.fps > 0 && (
              <span className="live-hud-chip">🎞 {stats.fps.toFixed(1)} fps</span>
            )}
            <span className="live-hud-chip">
              👤 {stats.detected} face{stats.detected !== 1 ? "s" : ""}
            </span>
          </div>
        )}
      </div>

      {/* ── Controls ── */}
      <div className="live-controls">
        {!cameraActive ? (
          <button className="btn btn-primary live-btn" onClick={startCamera} id="live-start-btn">
            <span>▶ Start Camera</span>
          </button>
        ) : (
          <button className="btn live-btn live-btn--stop" onClick={stopCamera} id="live-stop-btn">
            <span>■ Stop Camera</span>
          </button>
        )}
      </div>

      {cameraError && <p className="form-error live-error">{cameraError}</p>}

      {/* ── Live results list ── */}
      {lastFaces.length > 0 && (
        <div className="live-results">
          <h4 className="live-results-title">Detected this frame</h4>
          <div className="live-results-grid">
            {lastFaces.map((f, i) => (
              <div key={i} className="live-result-chip"
                style={{ "--chip-color": f.identity === "Unknown" ? "#94A3B8" : colorForIdentity(f.identity) }}>
                <span className="live-chip-dot" />
                <span className="live-chip-identity">{f.identity}</span>
                {f.identity !== "Unknown" && (
                  <span className="live-chip-conf">{(f.confidence * 100).toFixed(0)}%</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Tips ── */}
      <div className="live-tips">
        <p className="live-tips-text">
          💡 <strong>Tips:</strong> Ensure good lighting and face the camera directly.
          Enroll identities via the <em>Video Identification</em> tab first for named recognition.
          The pipeline runs entirely on CPU — expect ~30–150 ms latency depending on hardware.
        </p>
      </div>
    </div>
  );
}
