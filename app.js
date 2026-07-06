/* ============================================================
   CHATTERX — live audio spectrum + peak analysis
   Pipeline: mic → AnalyserNode (FFT) → peak detection → draw
   Extension point: feed `peaks` + spindle params into your
   stability / chatter-adjustment algorithm (see onPeaks()).
   ============================================================ */

(() => {
  "use strict";

  // ---------- config ----------
  const FFT_SIZE = 4096;          // at 48 kHz → ~11.7 Hz per bin
  const SMOOTHING = 0.55;         // analyser time-smoothing for a steadier trace
  const MIN_DB = -100;            // analyser floor
  const MAX_DB = -10;             // analyser ceiling
  const MAX_PEAKS = 5;            // peaks reported
  const PEAK_MIN_SEP_HZ = 60;     // minimum spacing between reported peaks
  const PEAK_ABOVE_FLOOR_DB = 12; // peak must clear the noise floor by this much
  const TABLE_UPDATE_MS = 250;    // throttle DOM updates

  // ---------- state ----------
  let audioCtx = null;
  let analyser = null;
  let mediaStream = null;
  let running = false;
  let rafId = null;
  let freqData = null;            // Float32Array of dB magnitudes
  let binHz = 0;                  // Hz per FFT bin
  let displayMaxHz = 5000;
  let lastTableUpdate = 0;

  // ---------- elements ----------
  const $ = (id) => document.getElementById(id);
  const toggleBtn = $("toggleBtn");
  const statusEl = $("status");
  const statusText = $("statusText");
  const peakFreqEl = $("peakFreq");
  const peakDbEl = $("peakDb");
  const peakBody = $("peakBody");
  const errorMsg = $("errorMsg");
  const rangeSelect = $("rangeSelect");
  const specCanvas = $("spectrum");
  const gramCanvas = $("spectrogram");
  const specCtx = specCanvas.getContext("2d");
  const gramCtx = gramCanvas.getContext("2d");

  // ---------- canvas sizing (crisp on high-DPI phones) ----------
  function sizeCanvas(canvas, ctx) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssW = canvas.clientWidth;
    const cssH = parseInt(canvas.getAttribute("height"), 10);
    canvas.style.height = cssH + "px";
    // Preserve spectrogram history through resizes by snapshotting first.
    let snapshot = null;
    if (canvas === gramCanvas && canvas.width > 0) {
      snapshot = document.createElement("canvas");
      snapshot.width = canvas.width;
      snapshot.height = canvas.height;
      snapshot.getContext("2d").drawImage(canvas, 0, 0);
    }
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (snapshot) {
      ctx.drawImage(snapshot, 0, 0, cssW, cssH);
    }
  }

  function sizeAllCanvases() {
    sizeCanvas(specCanvas, specCtx);
    sizeCanvas(gramCanvas, gramCtx);
    if (!running) drawIdleSpectrum();
  }

  window.addEventListener("resize", sizeAllCanvases);

  // ---------- start / stop ----------
  async function start() {
    hideError();
    try {
      // Measurement settings: turn OFF the phone's voice processing,
      // which would otherwise filter exactly the content we care about.
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
    } catch (err) {
      showError(micErrorMessage(err));
      return;
    }

    const AC = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AC();
    // iOS starts contexts suspended; resume inside this user-gesture handler.
    if (audioCtx.state === "suspended") {
      await audioCtx.resume();
    }

    analyser = audioCtx.createAnalyser();
    analyser.fftSize = FFT_SIZE;
    analyser.smoothingTimeConstant = SMOOTHING;
    analyser.minDecibels = MIN_DB;
    analyser.maxDecibels = MAX_DB;

    const source = audioCtx.createMediaStreamSource(mediaStream);
    source.connect(analyser);
    // Note: analyser is NOT connected to destination — we analyze, not play back.

    freqData = new Float32Array(analyser.frequencyBinCount);
    binHz = audioCtx.sampleRate / analyser.fftSize;

    running = true;
    setUiRunning(true);
    loop();
  }

  function stop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
    analyser = null;
    setUiRunning(false);
  }

  toggleBtn.addEventListener("click", () => (running ? stop() : start()));

  rangeSelect.addEventListener("change", () => {
    displayMaxHz = Number(rangeSelect.value);
    if (!running) drawIdleSpectrum();
  });

  function setUiRunning(on) {
    toggleBtn.textContent = on ? "STOP" : "START LISTENING";
    toggleBtn.className = on ? "btn-stop" : "btn-start";
    statusEl.classList.toggle("live", on);
    statusText.textContent = on ? "listening" : "idle";
    if (!on) {
      peakFreqEl.textContent = "----";
      peakDbEl.textContent = "--";
    }
  }

  // ---------- main loop ----------
  function loop() {
    if (!running) return;
    analyser.getFloatFrequencyData(freqData);

    const maxBin = Math.min(
      Math.floor(displayMaxHz / binHz),
      freqData.length - 1
    );

    const peaks = findPeaks(freqData, binHz, maxBin);

    drawSpectrum(freqData, maxBin, peaks);
    drawSpectrogramColumn(freqData, maxBin);

    const now = performance.now();
    if (now - lastTableUpdate > TABLE_UPDATE_MS) {
      lastTableUpdate = now;
      updateReadout(peaks);
      updateTable(peaks);
      onPeaks(peaks); // ← your stability algorithm hooks in here
    }

    rafId = requestAnimationFrame(loop);
  }

  // ---------- peak detection ----------
  // Local-maxima search above an adaptive noise floor, with parabolic
  // interpolation for sub-bin frequency accuracy and a minimum-separation
  // rule so one wide lobe doesn't fill all five slots.
  function findPeaks(mags, hzPerBin, maxBin) {
    // Adaptive floor: median magnitude of the displayed band.
    const band = mags.slice(1, maxBin).filter(Number.isFinite);
    if (band.length === 0) return [];
    const sorted = Float32Array.from(band).sort();
    const floor = sorted[sorted.length >> 1];
    const threshold = floor + PEAK_ABOVE_FLOOR_DB;

    const candidates = [];
    for (let i = 2; i < maxBin - 2; i++) {
      const v = mags[i];
      if (v < threshold) continue;
      if (
        v > mags[i - 1] &&
        v >= mags[i + 1] &&
        v > mags[i - 2] &&
        v >= mags[i + 2]
      ) {
        // Parabolic interpolation around the bin maximum.
        const a = mags[i - 1];
        const b = v;
        const c = mags[i + 1];
        const denom = a - 2 * b + c;
        const delta = denom !== 0 ? (0.5 * (a - c)) / denom : 0;
        candidates.push({
          freq: (i + delta) * hzPerBin,
          db: b - 0.25 * (a - c) * delta,
          bin: i,
        });
      }
    }

    candidates.sort((p, q) => q.db - p.db);

    const selected = [];
    for (const p of candidates) {
      if (selected.every((s) => Math.abs(s.freq - p.freq) > PEAK_MIN_SEP_HZ)) {
        selected.push(p);
      }
      if (selected.length >= MAX_PEAKS) break;
    }
    return selected;
  }

  // ---------- extension point ----------
  // Called ~4×/second with the current top peaks:
  //   peaks = [{ freq: Hz, db: level, bin: fftBin }, ...] sorted by level.
  // This is where the stability check and RPM-adjustment algorithm will go,
  // e.g. compare peak frequencies against tooth-passing harmonics computed
  // from spindle RPM × flute count entered by the user.
  function onPeaks(peaks) {
    void peaks;
  }

  // ---------- drawing: spectrum ----------
  function drawSpectrum(mags, maxBin, peaks) {
    const w = specCanvas.clientWidth;
    const h = parseInt(specCanvas.getAttribute("height"), 10);
    const ctx = specCtx;

    ctx.clearRect(0, 0, w, h);
    drawGrid(ctx, w, h);

    // Amber trace
    ctx.beginPath();
    for (let i = 1; i <= maxBin; i++) {
      const x = (i / maxBin) * w;
      const y = dbToY(mags[i], h);
      if (i === 1) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#FFB24D";
    ctx.lineWidth = 1.6;
    ctx.stroke();

    // Soft fill under the trace
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = "rgba(255, 178, 77, 0.10)";
    ctx.fill();

    // Peak markers
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    peaks.forEach((p, idx) => {
      const x = (p.freq / (maxBin * binHz)) * w;
      const y = dbToY(p.db, h);
      ctx.fillStyle = idx === 0 ? "#7FD8E8" : "rgba(127, 216, 232, 0.55)";
      ctx.beginPath();
      ctx.moveTo(x, y - 4);
      ctx.lineTo(x - 4, y - 11);
      ctx.lineTo(x + 4, y - 11);
      ctx.closePath();
      ctx.fill();
      if (idx === 0) {
        ctx.fillText(formatHz(p.freq), clamp(x, 24, w - 24), Math.max(y - 15, 10));
      }
    });
  }

  function drawIdleSpectrum() {
    const w = specCanvas.clientWidth;
    const h = parseInt(specCanvas.getAttribute("height"), 10);
    specCtx.clearRect(0, 0, w, h);
    drawGrid(specCtx, w, h);
  }

  function drawGrid(ctx, w, h) {
    ctx.strokeStyle = "rgba(44, 51, 62, 0.8)";
    ctx.fillStyle = "#8A919C";
    ctx.lineWidth = 1;
    ctx.font = "9px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left";

    // Vertical gridlines every 1 kHz
    const step = 1000;
    for (let f = step; f < displayMaxHz; f += step) {
      const x = (f / displayMaxHz) * w;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.fillText(f / 1000 + "k", x + 3, h - 5);
    }
    // Horizontal gridlines every 20 dB
    for (let db = MAX_DB - 20; db > MIN_DB; db -= 20) {
      const y = dbToY(db, h);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  }

  function dbToY(db, h) {
    const t = (db - MIN_DB) / (MAX_DB - MIN_DB);
    return h - clamp(t, 0, 1) * h;
  }

  // ---------- drawing: spectrogram ----------
  // Scrolls left one column per frame; newest audio at the right edge.
  // Drawn in raw device pixels (identity transform) so the self-copy
  // shift is exact regardless of devicePixelRatio.
  function drawSpectrogramColumn(mags, maxBin) {
    const ctx = gramCtx;
    const pw = gramCanvas.width;   // device pixels
    const ph = gramCanvas.height;
    const col = 2;                 // scroll speed in device pixels per frame

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Shift existing image left
    ctx.drawImage(gramCanvas, -col, 0);
    ctx.clearRect(pw - col, 0, col, ph);

    // Draw the new column: low freq at bottom, high at top
    for (let y = 0; y < ph; y++) {
      const frac = 1 - y / ph;
      const bin = Math.min(Math.round(frac * maxBin), maxBin);
      const db = mags[Math.max(bin, 1)];
      ctx.fillStyle = dbToColor(db);
      ctx.fillRect(pw - col, y, col, 1);
    }

    ctx.restore();
  }

  // Dark → ember → amber → near-white heat map
  function dbToColor(db) {
    const t = clamp((db - MIN_DB) / (MAX_DB - MIN_DB), 0, 1);
    if (t < 0.35) {
      const k = t / 0.35;
      return rgb(16 + 30 * k, 20 + 14 * k, 26 + 6 * k); // deep gunmetal → warm dark
    } else if (t < 0.7) {
      const k = (t - 0.35) / 0.35;
      return rgb(46 + 180 * k, 34 + 110 * k, 32 + 15 * k); // → ember orange
    } else {
      const k = (t - 0.7) / 0.3;
      return rgb(226 + 29 * k, 144 + 90 * k, 47 + 160 * k); // → hot amber-white
    }
  }

  function rgb(r, g, b) {
    return `rgb(${r | 0},${g | 0},${b | 0})`;
  }

  // ---------- readout + table ----------
  function updateReadout(peaks) {
    if (peaks.length === 0) {
      peakFreqEl.textContent = "----";
      peakDbEl.textContent = "--";
      return;
    }
    peakFreqEl.textContent = Math.round(peaks[0].freq).toString();
    peakDbEl.textContent = peaks[0].db.toFixed(1);
  }

  function updateTable(peaks) {
    if (peaks.length === 0) {
      peakBody.innerHTML =
        '<tr class="empty-row"><td colspan="3">No peaks above the noise floor.</td></tr>';
      return;
    }
    peakBody.innerHTML = peaks
      .map(
        (p, i) =>
          `<tr class="${i === 0 ? "top" : ""}">
             <td class="rank">${i + 1}</td>
             <td class="freq">${formatHz(p.freq)}</td>
             <td>${p.db.toFixed(1)} dB</td>
           </tr>`
      )
      .join("");
  }

  // ---------- helpers ----------
  function formatHz(f) {
    return f >= 1000 ? (f / 1000).toFixed(2) + " kHz" : Math.round(f) + " Hz";
  }

  function clamp(v, lo, hi) {
    return Math.min(Math.max(v, lo), hi);
  }

  function micErrorMessage(err) {
    if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) {
      return "Microphone access was denied. Allow the microphone in your browser settings, then try again.";
    }
    if (err && err.name === "NotFoundError") {
      return "No microphone was found on this device.";
    }
    return "Could not start the microphone. Close other apps using it and try again.";
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.hidden = false;
  }

  function hideError() {
    errorMsg.hidden = true;
  }

  // ---------- lifecycle: pause cleanly when backgrounded ----------
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && running) stop();
  });

  // ---------- iOS install hint ----------
  (function iosHint() {
    const isIOS = /iphone|ipad|ipod/i.test(navigator.userAgent);
    const standalone =
      window.matchMedia("(display-mode: standalone)").matches ||
      window.navigator.standalone === true;
    if (isIOS && !standalone && !localStorage.getItem("iosHintDismissed")) {
      const hint = $("iosHint");
      hint.hidden = false;
      $("iosHintClose").addEventListener("click", () => {
        hint.hidden = true;
        localStorage.setItem("iosHintDismissed", "1");
      });
    }
  })();

  // ---------- service worker ----------
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("./sw.js").catch(() => {
        /* offline support unavailable; app still works online */
      });
    });
  }

  // ---------- init ----------
  sizeAllCanvases();
})();
