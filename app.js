/* ============================================================
   ChatteRx — live audio spectrum, peak analysis, session replay
   Modes:
     idle   — nothing running
     live   — mic → AnalyserNode (display) + raw capture (recording)
     review — scrubbable session spectrogram, playback, WAV export
   Extension point: onPeaks() receives detected peaks in both
   live and review modes — the stability algorithm plugs in there.
   ============================================================ */

(() => {
  "use strict";

  // ---------- config ----------
  const FFT_SIZE_LIVE = 4096;     // live analyser; ~11.7 Hz/bin @48 kHz
  const SMOOTHING = 0.55;
  const MIN_DB = -100;
  const MAX_DB = -10;
  const MAX_PEAKS = 5;
  const PEAK_MIN_SEP_HZ = 60;
  const PEAK_ABOVE_FLOOR_DB = 12;
  const UI_UPDATE_MS = 250;

  const MAX_REC_SECONDS = 300;    // 5 min cap ≈ 55 MB of Float32 @48 kHz
  const FFT_SIZE_REC = 2048;      // offline spectrogram resolution
  const HOP_REC = 1024;           // 50% overlap
  const STORE_MAX_HZ = 10000;     // spectrogram bins kept up to 10 kHz

  // ---------- state ----------
  let mode = "idle";              // idle | live | review
  let audioCtx = null;
  let analyser = null;
  let mediaStream = null;
  let captureNode = null;
  let rafId = null;
  let freqData = null;
  let binHzLive = 0;
  let displayMaxHz = 5000;
  let lastUiUpdate = 0;
  let liveStartTime = 0;

  // recording
  let recChunks = [];
  let recSamples = 0;
  let recRate = 48000;

  // review
  let recBuffer = null;           // Float32Array, full session
  let gram = null;                // { cols, bins, data(Uint8), hzPerBin, hopSec }
  let gramImage = null;           // offscreen canvas of full spectrogram
  let playCtx = null;
  let playBuffer = null;          // AudioBuffer built once per session
  let playSource = null;
  let playing = false;
  let playhead = 0;               // seconds
  let playStartCtxTime = 0;
  let playStartOffset = 0;

  // ---------- elements ----------
  const $ = (id) => document.getElementById(id);
  const appEl = $("app");
  const toggleBtn = $("toggleBtn");
  const statusEl = $("status");
  const statusText = $("statusText");
  const peakFreqEl = $("peakFreq");
  const peakDbEl = $("peakDb");
  const peakChips = $("peakChips");
  const errorMsg = $("errorMsg");
  const rangeSelect = $("rangeSelect");
  const gramTitle = $("gramTitle");
  const gramNote = $("gramNote");
  const transport = $("transport");
  const playBtn = $("playBtn");
  const timeLabel = $("timeLabel");
  const exportBtn = $("exportBtn");
  const processingEl = $("processing");
  const specCanvas = $("spectrum");
  const gramCanvas = $("spectrogram");
  const specCtx = specCanvas.getContext("2d");
  const gramCtx = gramCanvas.getContext("2d");

  // ---------- canvas sizing via ResizeObserver ----------
  const DPR = () => Math.min(window.devicePixelRatio || 1, 2);

  function fitCanvas(canvas, ctx, preserve) {
    const w = Math.round(canvas.clientWidth * DPR());
    const h = Math.round(canvas.clientHeight * DPR());
    if (w === canvas.width && h === canvas.height) return;
    let snap = null;
    if (preserve && canvas.width > 0 && canvas.height > 0) {
      snap = document.createElement("canvas");
      snap.width = canvas.width;
      snap.height = canvas.height;
      snap.getContext("2d").drawImage(canvas, 0, 0);
    }
    canvas.width = w;
    canvas.height = h;
    ctx.setTransform(DPR(), 0, 0, DPR(), 0, 0);
    if (snap) {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(snap, 0, 0, w, h);
      ctx.restore();
    }
  }

  const ro = new ResizeObserver(() => {
    fitCanvas(specCanvas, specCtx, false);
    fitCanvas(gramCanvas, gramCtx, mode === "live");
    if (mode !== "live") redrawStatic();
  });
  ro.observe(specCanvas.parentElement);
  ro.observe(gramCanvas.parentElement);

  function redrawStatic() {
    if (mode === "review" && gramImage) {
      drawTimeline();
      drawReviewSpectrum();
    } else if (mode === "idle") {
      drawGrid(specCtx, specCanvas.clientWidth, specCanvas.clientHeight);
    }
  }

  // ============================================================
  // LIVE MODE
  // ============================================================

  async function startLive() {
    hideError();
    try {
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
    if (audioCtx.state === "suspended") await audioCtx.resume();
    recRate = audioCtx.sampleRate;

    analyser = audioCtx.createAnalyser();
    analyser.fftSize = FFT_SIZE_LIVE;
    analyser.smoothingTimeConstant = SMOOTHING;
    analyser.minDecibels = MIN_DB;
    analyser.maxDecibels = MAX_DB;

    const source = audioCtx.createMediaStreamSource(mediaStream);
    source.connect(analyser);

    freqData = new Float32Array(analyser.frequencyBinCount);
    binHzLive = recRate / analyser.fftSize;

    recChunks = [];
    recSamples = 0;
    await attachCapture(source);

    setMode("live");
    liveStartTime = performance.now();
    clearGram();
    loop();
  }

  // Raw sample capture: AudioWorklet preferred, ScriptProcessor fallback.
  async function attachCapture(source) {
    const onChunk = (chunk) => {
      recChunks.push(chunk);
      recSamples += chunk.length;
      if (recSamples >= MAX_REC_SECONDS * recRate) {
        stopLive(); // auto-stop at the cap; session goes to review
      }
    };

    if (audioCtx.audioWorklet) {
      const workletCode = `
        class Cap extends AudioWorkletProcessor {
          constructor() { super(); this.buf = new Float32Array(4096); this.n = 0; }
          process(inputs) {
            const ch = inputs[0] && inputs[0][0];
            if (ch) {
              let i = 0;
              while (i < ch.length) {
                const c = Math.min(ch.length - i, 4096 - this.n);
                this.buf.set(ch.subarray(i, i + c), this.n);
                this.n += c; i += c;
                if (this.n === 4096) {
                  this.port.postMessage(this.buf, [this.buf.buffer]);
                  this.buf = new Float32Array(4096); this.n = 0;
                }
              }
            }
            return true;
          }
        }
        registerProcessor("chatterx-capture", Cap);
      `;
      const url = URL.createObjectURL(
        new Blob([workletCode], { type: "application/javascript" })
      );
      try {
        await audioCtx.audioWorklet.addModule(url);
        captureNode = new AudioWorkletNode(audioCtx, "chatterx-capture");
        captureNode.port.onmessage = (e) => onChunk(e.data);
        const mute = audioCtx.createGain();
        mute.gain.value = 0;
        source.connect(captureNode);
        captureNode.connect(mute);
        mute.connect(audioCtx.destination);
        return;
      } catch (e) {
        /* fall through to ScriptProcessor */
      } finally {
        URL.revokeObjectURL(url);
      }
    }

    // Fallback for older browsers
    captureNode = audioCtx.createScriptProcessor(4096, 1, 1);
    captureNode.onaudioprocess = (e) =>
      onChunk(new Float32Array(e.inputBuffer.getChannelData(0)));
    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    source.connect(captureNode);
    captureNode.connect(mute);
    mute.connect(audioCtx.destination);
  }

  function stopLive() {
    if (rafId) cancelAnimationFrame(rafId);
    if (captureNode) {
      try { captureNode.disconnect(); } catch (e) {}
      if (captureNode.port) captureNode.port.onmessage = null;
      captureNode = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
    analyser = null;

    // Enough audio to review? (> 0.5 s)
    if (recSamples > recRate * 0.5) {
      enterReview();
    } else {
      setMode("idle");
    }
  }

  function loop() {
    if (mode !== "live") return;
    analyser.getFloatFrequencyData(freqData);

    const maxBin = Math.min(
      Math.floor(displayMaxHz / binHzLive),
      freqData.length - 1
    );

    const peaks = findPeaks(freqData, binHzLive, maxBin, 1);

    drawSpectrumTrace(freqData, maxBin, binHzLive, peaks, 1);
    drawLiveGramColumn(freqData, maxBin);

    const now = performance.now();
    if (now - lastUiUpdate > UI_UPDATE_MS) {
      lastUiUpdate = now;
      updateReadout(peaks);
      updateChips(peaks);
      onPeaks(peaks);
      const secs = (now - liveStartTime) / 1000;
      statusText.textContent = "listening · " + fmtTime(secs);
    }

    rafId = requestAnimationFrame(loop);
  }

  // ============================================================
  // PEAK DETECTION (shared by live + review)
  // startBin lets review skip the DC bin.
  // ============================================================

  function findPeaks(mags, hzPerBin, maxBin, startBin) {
    const band = [];
    for (let i = startBin; i < maxBin; i++) {
      if (Number.isFinite(mags[i])) band.push(mags[i]);
    }
    if (band.length === 0) return [];
    band.sort((a, b) => a - b);
    const floor = band[band.length >> 1];
    const threshold = floor + PEAK_ABOVE_FLOOR_DB;

    const candidates = [];
    for (let i = Math.max(startBin, 2); i < maxBin - 2; i++) {
      const v = mags[i];
      if (v < threshold) continue;
      if (v > mags[i-1] && v >= mags[i+1] && v > mags[i-2] && v >= mags[i+2]) {
        const a = mags[i-1], b = v, c = mags[i+1];
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
  // Called ~4×/s (live) and on every scrub/playback frame (review) with
  // peaks = [{ freq, db, bin }] sorted by level. Feed these plus spindle
  // RPM / flute count into the stability + adjustment algorithm here.
  function onPeaks(peaks) {
    void peaks;
  }

  // ============================================================
  // REVIEW MODE — timeline, scrubbing, playback, export
  // ============================================================

  async function enterReview() {
    setMode("review");
    processingEl.hidden = false;

    // Assemble the session into one Float32Array
    recBuffer = new Float32Array(recSamples);
    let off = 0;
    for (const c of recChunks) {
      recBuffer.set(c, off);
      off += c.length;
    }
    recChunks = [];

    await computeGram();      // offline spectrogram (chunked, keeps UI alive)
    renderGramImage();
    processingEl.hidden = true;

    playhead = 0;
    drawTimeline();
    drawReviewSpectrum();
    updateTimeLabel();
  }

  // Offline STFT with a Hann window, magnitudes quantized to Uint8.
  async function computeGram() {
    const fft = new FFT(FFT_SIZE_REC);
    const hzPerBin = recRate / FFT_SIZE_REC;
    const bins = Math.min(
      Math.floor(STORE_MAX_HZ / hzPerBin),
      FFT_SIZE_REC / 2
    );
    const cols = Math.max(
      1,
      Math.floor((recBuffer.length - FFT_SIZE_REC) / HOP_REC) + 1
    );
    const data = new Uint8Array(cols * bins);

    const window = new Float32Array(FFT_SIZE_REC);
    for (let i = 0; i < FFT_SIZE_REC; i++) {
      window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE_REC - 1)));
    }
    // Amplitude → dBFS correction for the windowed, one-sided spectrum
    const norm = 2 / (FFT_SIZE_REC * 0.5); // 0.5 = Hann coherent gain

    const re = new Float32Array(FFT_SIZE_REC);
    const im = new Float32Array(FFT_SIZE_REC);

    for (let c = 0; c < cols; c++) {
      const start = c * HOP_REC;
      for (let i = 0; i < FFT_SIZE_REC; i++) {
        re[i] = recBuffer[start + i] * window[i];
        im[i] = 0;
      }
      fft.transform(re, im);
      for (let b = 0; b < bins; b++) {
        const mag = Math.hypot(re[b], im[b]) * norm;
        const db = 20 * Math.log10(mag + 1e-12);
        data[c * bins + b] = dbToU8(db);
      }
      if ((c & 63) === 0) await nextFrame(); // yield to keep UI responsive
    }

    gram = { cols, bins, data, hzPerBin, hopSec: HOP_REC / recRate };
  }

  function dbToU8(db) {
    const t = (db - MIN_DB) / (MAX_DB - MIN_DB);
    return Math.max(0, Math.min(255, Math.round(t * 255)));
  }

  function u8ToDb(u) {
    return (u / 255) * (MAX_DB - MIN_DB) + MIN_DB;
  }

  // Render the whole session once into an offscreen canvas
  // (x = time column, y = frequency, row 0 = highest stored freq).
  function renderGramImage() {
    const { cols, bins, data } = gram;
    gramImage = document.createElement("canvas");
    gramImage.width = cols;
    gramImage.height = bins;
    const ictx = gramImage.getContext("2d");
    const img = ictx.createImageData(cols, bins);
    const px = img.data;
    for (let c = 0; c < cols; c++) {
      for (let b = 0; b < bins; b++) {
        const y = bins - 1 - b; // low freq at bottom
        const [r, g, bl] = heat(data[c * bins + b] / 255);
        const o = (y * cols + c) * 4;
        px[o] = r; px[o + 1] = g; px[o + 2] = bl; px[o + 3] = 255;
      }
    }
    ictx.putImageData(img, 0, 0);
  }

  function duration() {
    return recBuffer ? recBuffer.length / recRate : 0;
  }

  // Draw timeline: spectrogram image (cropped to selected range) + playhead
  function drawTimeline() {
    if (!gramImage || !gram) return;
    const w = gramCanvas.clientWidth;
    const h = gramCanvas.clientHeight;
    const ctx = gramCtx;
    ctx.clearRect(0, 0, w, h);

    const shownBins = Math.min(
      Math.floor(displayMaxHz / gram.hzPerBin),
      gram.bins
    );
    const srcTop = gram.bins - shownBins; // crop rows above the range
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(
      gramImage,
      0, srcTop, gram.cols, shownBins,
      0, 0, w, h
    );

    // playhead
    const x = (playhead / duration()) * w;
    ctx.strokeStyle = "#7FD8E8";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillStyle = "#7FD8E8";
    ctx.beginPath();
    ctx.moveTo(x - 5, 0);
    ctx.lineTo(x + 5, 0);
    ctx.lineTo(x, 7);
    ctx.closePath();
    ctx.fill();
  }

  // Spectrum panel shows the FFT column under the playhead
  function drawReviewSpectrum() {
    if (!gram) return;
    const col = Math.min(
      Math.floor(playhead / gram.hopSec),
      gram.cols - 1
    );
    const maxBin = Math.min(
      Math.floor(displayMaxHz / gram.hzPerBin),
      gram.bins - 1
    );
    const mags = new Float32Array(maxBin + 1);
    for (let b = 0; b <= maxBin; b++) {
      mags[b] = u8ToDb(gram.data[col * gram.bins + b]);
    }
    const peaks = findPeaks(mags, gram.hzPerBin, maxBin, 2);
    drawSpectrumTrace(mags, maxBin, gram.hzPerBin, peaks, 0);
    updateReadout(peaks);
    updateChips(peaks);
    onPeaks(peaks);
  }

  // ---------- scrubbing ----------
  let scrubbing = false;

  function scrubTo(clientX) {
    const rect = gramCanvas.getBoundingClientRect();
    const frac = clamp((clientX - rect.left) / rect.width, 0, 1);
    playhead = frac * duration();
    if (playing) restartPlaybackAt(playhead);
    drawTimeline();
    drawReviewSpectrum();
    updateTimeLabel();
  }

  gramCanvas.addEventListener("pointerdown", (e) => {
    if (mode !== "review") return;
    scrubbing = true;
    gramCanvas.setPointerCapture(e.pointerId);
    scrubTo(e.clientX);
  });
  gramCanvas.addEventListener("pointermove", (e) => {
    if (scrubbing && mode === "review") scrubTo(e.clientX);
  });
  gramCanvas.addEventListener("pointerup", () => (scrubbing = false));
  gramCanvas.addEventListener("pointercancel", () => (scrubbing = false));

  // ---------- playback ----------
  async function ensurePlayCtx() {
    if (!playCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      playCtx = new AC();
    }
    if (playCtx.state === "suspended") await playCtx.resume();
  }

  async function play() {
    await ensurePlayCtx();
    if (playhead >= duration() - 0.02) playhead = 0;
    startSourceAt(playhead);
    playing = true;
    playBtn.textContent = "❚❚";
    playBtn.setAttribute("aria-label", "Pause");
    reviewRaf();
  }

  function startSourceAt(offset) {
    stopSource();
    if (!playBuffer) {
      playBuffer = playCtx.createBuffer(1, recBuffer.length, recRate);
      playBuffer.copyToChannel(recBuffer, 0);
    }
    playSource = playCtx.createBufferSource();
    playSource.buffer = playBuffer;
    playSource.connect(playCtx.destination);
    playSource.onended = () => {
      if (playing && playCtx && playCtx.currentTime - playStartCtxTime >= duration() - playStartOffset - 0.05) {
        playing = false;
        playhead = duration();
        playBtn.textContent = "▶";
        playBtn.setAttribute("aria-label", "Play");
        drawTimeline();
        updateTimeLabel();
      }
    };
    playStartCtxTime = playCtx.currentTime;
    playStartOffset = offset;
    playSource.start(0, offset);
  }

  function restartPlaybackAt(offset) {
    startSourceAt(offset);
  }

  function stopSource() {
    if (playSource) {
      playSource.onended = null;
      try { playSource.stop(); } catch (e) {}
      playSource = null;
    }
  }

  function pause() {
    playing = false;
    playhead = playStartOffset + (playCtx.currentTime - playStartCtxTime);
    stopSource();
    playBtn.textContent = "▶";
    playBtn.setAttribute("aria-label", "Play");
    drawTimeline();
    updateTimeLabel();
  }

  playBtn.addEventListener("click", () => (playing ? pause() : play()));

  function reviewRaf() {
    if (!playing || mode !== "review") return;
    playhead = Math.min(
      playStartOffset + (playCtx.currentTime - playStartCtxTime),
      duration()
    );
    drawTimeline();
    drawReviewSpectrum();
    updateTimeLabel();
    if (playhead >= duration()) {
      playing = false;
      playBtn.textContent = "▶";
      playBtn.setAttribute("aria-label", "Play");
      return;
    }
    requestAnimationFrame(reviewRaf);
  }

  function teardownReview() {
    playing = false;
    stopSource();
    if (playCtx) {
      playCtx.close();
      playCtx = null;
    }
    recBuffer = null;
    playBuffer = null;
    gram = null;
    gramImage = null;
    playhead = 0;
  }

  // ---------- WAV export (16-bit PCM mono) ----------
  exportBtn.addEventListener("click", () => {
    if (!recBuffer) return;
    const blob = encodeWav(recBuffer, recRate);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
    a.href = url;
    a.download = `chatterx-${stamp}.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  });

  function encodeWav(samples, rate) {
    const n = samples.length;
    const buffer = new ArrayBuffer(44 + n * 2);
    const v = new DataView(buffer);
    const writeStr = (off, s) => {
      for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    v.setUint32(4, 36 + n * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    v.setUint32(16, 16, true);        // PCM chunk size
    v.setUint16(20, 1, true);         // PCM format
    v.setUint16(22, 1, true);         // mono
    v.setUint32(24, rate, true);
    v.setUint32(28, rate * 2, true);  // byte rate
    v.setUint16(32, 2, true);         // block align
    v.setUint16(34, 16, true);        // bits per sample
    writeStr(36, "data");
    v.setUint32(40, n * 2, true);
    let off = 44;
    for (let i = 0; i < n; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
    return new Blob([buffer], { type: "audio/wav" });
  }

  // ============================================================
  // DRAWING (shared)
  // ============================================================

  function drawSpectrumTrace(mags, maxBin, hzPerBin, peaks, startBin) {
    const w = specCanvas.clientWidth;
    const h = specCanvas.clientHeight;
    const ctx = specCtx;

    ctx.clearRect(0, 0, w, h);
    drawGrid(ctx, w, h);

    ctx.beginPath();
    for (let i = startBin; i <= maxBin; i++) {
      const x = (i / maxBin) * w;
      const y = dbToY(mags[i], h);
      if (i === startBin) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#FFB24D";
    ctx.lineWidth = 1.6;
    ctx.stroke();

    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = "rgba(255, 178, 77, 0.10)";
    ctx.fill();

    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    const topHz = maxBin * hzPerBin;
    peaks.forEach((p, idx) => {
      const x = (p.freq / topHz) * w;
      const y = dbToY(p.db, h);
      ctx.fillStyle = idx === 0 ? "#7FD8E8" : "rgba(127, 216, 232, 0.55)";
      ctx.beginPath();
      ctx.moveTo(x, y - 4);
      ctx.lineTo(x - 4, y - 11);
      ctx.lineTo(x + 4, y - 11);
      ctx.closePath();
      ctx.fill();
      if (idx === 0) {
        ctx.fillText(fmtHz(p.freq), clamp(x, 26, w - 26), Math.max(y - 15, 10));
      }
    });
  }

  function drawGrid(ctx, w, h) {
    ctx.strokeStyle = "rgba(44, 51, 62, 0.8)";
    ctx.fillStyle = "#8A919C";
    ctx.lineWidth = 1;
    ctx.font = "9px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left";
    for (let f = 1000; f < displayMaxHz; f += 1000) {
      const x = (f / displayMaxHz) * w;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.fillText(f / 1000 + "k", x + 3, h - 5);
    }
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

  // live spectrogram: scroll left, newest column at right (device pixels)
  function drawLiveGramColumn(mags, maxBin) {
    const ctx = gramCtx;
    const pw = gramCanvas.width;
    const ph = gramCanvas.height;
    const col = 2;

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(gramCanvas, -col, 0);
    ctx.clearRect(pw - col, 0, col, ph);
    for (let y = 0; y < ph; y++) {
      const frac = 1 - y / ph;
      const bin = Math.min(Math.round(frac * maxBin), maxBin);
      const db = mags[Math.max(bin, 1)];
      const t = clamp((db - MIN_DB) / (MAX_DB - MIN_DB), 0, 1);
      const [r, g, b] = heat(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(pw - col, y, col, 1);
    }
    ctx.restore();
  }

  function clearGram() {
    gramCtx.save();
    gramCtx.setTransform(1, 0, 0, 1, 0, 0);
    gramCtx.clearRect(0, 0, gramCanvas.width, gramCanvas.height);
    gramCtx.restore();
  }

  // Dark → ember → amber → near-white heat map. t in [0,1].
  function heat(t) {
    if (t < 0.35) {
      const k = t / 0.35;
      return [16 + 30 * k, 20 + 14 * k, 26 + 6 * k];
    } else if (t < 0.7) {
      const k = (t - 0.35) / 0.35;
      return [46 + 180 * k, 34 + 110 * k, 32 + 15 * k];
    }
    const k = (t - 0.7) / 0.3;
    return [226 + 29 * k, 144 + 90 * k, 47 + 160 * k];
  }

  // ============================================================
  // UI
  // ============================================================

  function setMode(next) {
    // leaving review: free memory
    if (mode === "review" && next !== "review") teardownReview();

    mode = next;
    appEl.dataset.mode = mode;

    statusEl.className = "status" + (mode === "live" ? " live" : mode === "review" ? " review" : "");
    statusText.textContent =
      mode === "live" ? "listening" : mode === "review" ? "review" : "idle";

    transport.hidden = mode !== "review";
    gramTitle.textContent = mode === "review" ? "SESSION TIMELINE" : "SPECTROGRAM";
    gramNote.textContent = mode === "review" ? "tap or drag to scrub" : "time →";

    if (mode === "idle") {
      toggleBtn.textContent = "START LISTENING";
      toggleBtn.className = "btn-main btn-start";
      peakFreqEl.textContent = "----";
      peakDbEl.textContent = "--";
      peakChips.innerHTML =
        '<span class="chip chip-empty">start listening to detect peaks</span>';
      clearGram();
      drawGrid(specCtx, specCanvas.clientWidth, specCanvas.clientHeight);
    } else if (mode === "live") {
      toggleBtn.textContent = "STOP \u0026 REVIEW";
      toggleBtn.className = "btn-main btn-stop";
    } else {
      toggleBtn.textContent = "NEW SESSION";
      toggleBtn.className = "btn-main btn-new";
    }
  }

  toggleBtn.addEventListener("click", () => {
    if (mode === "idle") startLive();
    else if (mode === "live") stopLive();
    else setMode("idle");
  });

  rangeSelect.addEventListener("change", () => {
    displayMaxHz = Number(rangeSelect.value);
    redrawStatic();
  });

  function updateReadout(peaks) {
    if (peaks.length === 0) {
      peakFreqEl.textContent = "----";
      peakDbEl.textContent = "--";
      return;
    }
    peakFreqEl.textContent = Math.round(peaks[0].freq).toString();
    peakDbEl.textContent = peaks[0].db.toFixed(1);
  }

  function updateChips(peaks) {
    if (peaks.length === 0) {
      peakChips.innerHTML =
        '<span class="chip chip-empty">no peaks above the noise floor</span>';
      return;
    }
    peakChips.innerHTML = peaks
      .map(
        (p, i) =>
          `<span class="chip${i === 0 ? " chip-top" : ""}"><b>${fmtHz(p.freq)}</b>${p.db.toFixed(0)} dB</span>`
      )
      .join("");
  }

  function updateTimeLabel() {
    timeLabel.textContent = fmtTime(playhead) + " / " + fmtTime(duration());
  }

  // ============================================================
  // FFT — iterative radix-2, in-place
  // ============================================================

  function FFT(size) {
    this.size = size;
    this.levels = Math.log2(size) | 0;
    this.cos = new Float32Array(size / 2);
    this.sin = new Float32Array(size / 2);
    for (let i = 0; i < size / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / size);
      this.sin[i] = Math.sin((2 * Math.PI * i) / size);
    }
  }

  FFT.prototype.transform = function (re, im) {
    const n = this.size;
    // bit-reversal permutation
    for (let i = 0; i < n; i++) {
      let j = 0;
      for (let k = 0; k < this.levels; k++) j = (j << 1) | ((i >>> k) & 1);
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size2 = 2; size2 <= n; size2 *= 2) {
      const half = size2 / 2;
      const step = n / size2;
      for (let i = 0; i < n; i += size2) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half;
          const tre = re[l] * this.cos[k] + im[l] * this.sin[k];
          const tim = -re[l] * this.sin[k] + im[l] * this.cos[k];
          re[l] = re[j] - tre;
          im[l] = im[j] - tim;
          re[j] += tre;
          im[j] += tim;
        }
      }
    }
  };

  // ============================================================
  // helpers / lifecycle
  // ============================================================

  function fmtHz(f) {
    return f >= 1000 ? (f / 1000).toFixed(2) + " kHz" : Math.round(f) + " Hz";
  }

  function fmtTime(s) {
    const m = Math.floor(s / 60);
    const ss = Math.floor(s % 60);
    return m + ":" + String(ss).padStart(2, "0");
  }

  function clamp(v, lo, hi) {
    return Math.min(Math.max(v, lo), hi);
  }

  function nextFrame() {
    // rAF stalls when the tab is hidden (e.g. auto-stop on backgrounding),
    // so fall back to a timeout to keep the timeline build moving.
    return document.hidden
      ? new Promise((r) => setTimeout(r, 0))
      : new Promise((r) => requestAnimationFrame(r));
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

  // Live capture can't continue in the background — stop cleanly into
  // review so the recording so far is kept. Review mode survives.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && mode === "live") stopLive();
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
      navigator.serviceWorker.register("./sw.js").catch(() => {});
    });
  }

  // ---------- init ----------
  requestAnimationFrame(() => {
    fitCanvas(specCanvas, specCtx, false);
    fitCanvas(gramCanvas, gramCtx, false);
    drawGrid(specCtx, specCanvas.clientWidth, specCanvas.clientHeight);
  });
})();
