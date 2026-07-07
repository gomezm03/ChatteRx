/* ============================================================
   ChatteRx — live audio spectrum, peak analysis, session replay,
   and time-domain milling simulation.
   Tabs:
     ANALYZE — idle | live | review (mic, loaded file, or sim output)
     SIMULATE — Tony's fixed_slices simulation with a CFC database
   Extension point: onPeaks() receives detected peaks in both live
   and review modes — the stability algorithm plugs in there.
   ============================================================ */

(() => {
  "use strict";

  // ---------- config ----------
  const FFT_SIZE_LIVE = 4096;
  const SMOOTHING = 0.55;
  const MIN_DB = -100;
  const MAX_DB = -10;
  const MAX_PEAKS = 5;
  const PEAK_MIN_SEP_HZ = 60;
  const PEAK_ABOVE_FLOOR_DB = 12;
  const UI_UPDATE_MS = 250;

  const MAX_REC_SECONDS = 300;
  const FFT_SIZE_REC = 2048;
  const HOP_REC = 1024;
  const STORE_MAX_HZ = 10000;

  const AUDIO_RATE = 48000;       // rate for simulation-rendered audio
  const SIM_STEPS_REV = 7200;     // fixed simulation internals
  const SIM_AXIAL = 50;

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
  let recBuffer = null;
  let gram = null;
  let gramImage = null;
  let playCtx = null;
  let playBuffer = null;
  let playSource = null;
  let playing = false;
  let playhead = 0;
  let playStartCtxTime = 0;
  let playStartOffset = 0;

  // ---------- elements ----------
  const $ = (id) => document.getElementById(id);
  const toggleBtn = $("toggleBtn");
  const loadLabel = $("loadLabel");
  const loadInput = $("loadInput");
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

  // ---------- canvas sizing ----------
  const DPR = () => Math.min(window.devicePixelRatio || 1, 2);

  function fitCanvas(canvas, ctx, preserve) {
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (cssW === 0 || cssH === 0) return; // hidden tab
    const w = Math.round(cssW * DPR());
    const h = Math.round(cssH * DPR());
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
  // TABS
  // ============================================================

  const tabAnalyze = $("tab-analyze");
  const tabSim = $("tab-sim");
  const tabBtns = document.querySelectorAll(".tab");

  function switchTab(name) {
    tabBtns.forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    tabAnalyze.hidden = name !== "analyze";
    tabSim.hidden = name !== "sim";
    if (name === "analyze") {
      requestAnimationFrame(() => {
        fitCanvas(specCanvas, specCtx, false);
        fitCanvas(gramCanvas, gramCtx, false);
        redrawStatic();
      });
    }
  }

  tabBtns.forEach((b) =>
    b.addEventListener("click", () => switchTab(b.dataset.tab))
  );

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

  async function attachCapture(source) {
    const onChunk = (chunk) => {
      recChunks.push(chunk);
      recSamples += chunk.length;
      if (recSamples >= MAX_REC_SECONDS * recRate) stopLive();
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
        /* fall through */
      } finally {
        URL.revokeObjectURL(url);
      }
    }

    captureNode = audioCtx.createScriptProcessor(4096, 1, 1);
    captureNode.onaudioprocess = (e) =>
      onChunk(new Float32Array(e.inputBuffer.getChannelData(0)));
    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    source.connect(captureNode);
    captureNode.connect(mute);
    mute.connect(audioCtx.destination);
  }

  function stopLive(discard) {
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

    if (!discard && recSamples > recRate * 0.5) {
      enterReview();
    } else {
      recChunks = [];
      recSamples = 0;
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
      statusText.textContent =
        "listening · " + fmtTime((now - liveStartTime) / 1000);
    }

    rafId = requestAnimationFrame(loop);
  }

  // ============================================================
  // PEAK DETECTION (shared)
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
  function onPeaks(peaks) {
    void peaks;
  }

  // ============================================================
  // REVIEW MODE
  // ============================================================

  async function enterReview() {
    const buf = new Float32Array(recSamples);
    let off = 0;
    for (const c of recChunks) {
      buf.set(c, off);
      off += c.length;
    }
    recChunks = [];
    await openInReview(buf, recRate);
  }

  // Open ANY signal in review mode: mic recording, decoded file,
  // or simulation output.
  async function openInReview(samples, rate) {
    if (mode === "review") teardownReview();
    recBuffer = samples;
    recRate = rate;
    recSamples = samples.length;

    setMode("review");
    processingEl.hidden = false;

    await computeGram();
    renderGramImage();
    processingEl.hidden = true;

    playhead = 0;
    drawTimeline();
    drawReviewSpectrum();
    updateTimeLabel();
  }

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

    const win = new Float32Array(FFT_SIZE_REC);
    for (let i = 0; i < FFT_SIZE_REC; i++) {
      win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE_REC - 1)));
    }
    const norm = 2 / (FFT_SIZE_REC * 0.5);

    const re = new Float32Array(FFT_SIZE_REC);
    const im = new Float32Array(FFT_SIZE_REC);

    for (let c = 0; c < cols; c++) {
      const start = c * HOP_REC;
      for (let i = 0; i < FFT_SIZE_REC; i++) {
        re[i] = (recBuffer[start + i] || 0) * win[i];
        im[i] = 0;
      }
      fft.transform(re, im);
      for (let b = 0; b < bins; b++) {
        const mag = Math.hypot(re[b], im[b]) * norm;
        const db = 20 * Math.log10(mag + 1e-12);
        data[c * bins + b] = dbToU8(db);
      }
      if ((c & 63) === 0) await nextFrame();
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
        const y = bins - 1 - b;
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

  function drawTimeline() {
    if (!gramImage || !gram) return;
    const w = gramCanvas.clientWidth;
    const h = gramCanvas.clientHeight;
    if (w === 0 || h === 0) return;
    const ctx = gramCtx;
    ctx.clearRect(0, 0, w, h);

    const shownBins = Math.min(
      Math.floor(displayMaxHz / gram.hzPerBin),
      gram.bins
    );
    const srcTop = gram.bins - shownBins;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(gramImage, 0, srcTop, gram.cols, shownBins, 0, 0, w, h);

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

  function drawReviewSpectrum() {
    if (!gram) return;
    const col = Math.min(Math.floor(playhead / gram.hopSec), gram.cols - 1);
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
    if (playing) startSourceAt(playhead);
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
    playStartCtxTime = playCtx.currentTime;
    playStartOffset = offset;
    playSource.start(0, offset);
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
      stopSource();
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
    playBtn.textContent = "▶";
    recBuffer = null;
    playBuffer = null;
    gram = null;
    gramImage = null;
    playhead = 0;
  }

  // ---------- WAV export ----------
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
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);
    v.setUint16(22, 1, true);
    v.setUint32(24, rate, true);
    v.setUint32(28, rate * 2, true);
    v.setUint16(32, 2, true);
    v.setUint16(34, 16, true);
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
  // FILE LOADING (.wav, .mp3, .m4a, .mp4 — anything the browser decodes)
  // ============================================================

  loadInput.addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    if (!file) return;
    hideError();
    let ctx = null;
    try {
      const ab = await file.arrayBuffer();
      const AC = window.AudioContext || window.webkitAudioContext;
      ctx = new AC();
      const audio = await ctx.decodeAudioData(ab);
      let ch = audio.getChannelData(0);
      const maxLen = Math.floor(audio.sampleRate * MAX_REC_SECONDS);
      const samples = new Float32Array(
        ch.length > maxLen ? ch.subarray(0, maxLen) : ch
      );
      const rate = audio.sampleRate;
      ctx.close();
      ctx = null;
      await openInReview(samples, rate);
    } catch (err) {
      if (ctx) ctx.close();
      showError(
        "Could not decode that file. Try a .wav, .mp3, .m4a, or .mp4 with an audio track."
      );
    }
  });

  // Make the label act like a button for keyboard users
  loadLabel.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      loadInput.click();
    }
  });

  // ============================================================
  // SIMULATE TAB
  // ============================================================

  const CFC_STORE_KEY = "chatterx-cfc-presets";
  const SIM_STORE_KEY = "chatterx-sim-inputs";

  // Built-in coefficient sets. Values are N/mm² (cutting) and N/mm (edge).
  // "Typical" sets are approximate literature-style starting points — edit
  // to match measured data for real predictions.
  const BUILTIN_CFC = [
    {
      name: "Measured set (default)",
      Ktc: 1720.33, Krc: 628.99, Kac: -11.23,
      Kte: 76.12, Kre: 118.32, Kae: 30.84,
    },
    {
      name: "Aluminum alloy (typical)",
      Ktc: 750, Krc: 250, Kac: 0,
      Kte: 0, Kre: 0, Kae: 0,
    },
    {
      name: "Low-carbon steel (typical)",
      Ktc: 2100, Krc: 700, Kac: 0,
      Kte: 0, Kre: 0, Kae: 0,
    },
  ];

  const cfcPreset = $("cfcPreset");
  const cfcSummary = $("cfcSummary");
  const cfcFields = ["Ktc", "Krc", "Kac", "Kte", "Kre", "Kae"];
  const savePresetBtn = $("savePresetBtn");
  const delPresetBtn = $("delPresetBtn");
  const modeRowsX = $("modeRowsX");
  const modeRowsY = $("modeRowsY");
  const runSimBtn = $("runSimBtn");
  const simProgress = $("simProgress");
  const simProgressFill = $("simProgressFill");
  const simResults = $("simResults");
  const simError = $("simError");
  const playSimBtn = $("playSimBtn");
  const sendBtn = $("sendBtn");
  const forceChan = $("forceChan");
  const forcePlot = $("forcePlot");
  const soundPlot = $("soundPlot");
  const forceStats = $("forceStats");
  const soundStats = $("soundStats");

  let customPresets = loadJson(CFC_STORE_KEY, []);
  let simWorker = null;
  let simAudio = null;   // { samples: Float32Array, rate, metric, fs }
  let simRaw = null;     // { fx, fy, xt, fs, res (lazy) }
  let simPlayCtx = null;
  let simPlaySource = null;
  let simPlayT0 = 0;     // playCtx time at start, for the playhead

  function loadJson(key, fallback) {
    try {
      const s = localStorage.getItem(key);
      return s ? JSON.parse(s) : fallback;
    } catch (e) {
      return fallback;
    }
  }

  function saveJson(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) { /* storage full or unavailable */ }
  }

  // ---------- CFC preset UI ----------
  function refreshPresetSelect(selectName) {
    cfcPreset.innerHTML = "";
    const all = BUILTIN_CFC.concat(customPresets);
    all.forEach((p, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = p.name;
      cfcPreset.appendChild(opt);
    });
    const idx = selectName
      ? all.findIndex((p) => p.name === selectName)
      : 0;
    cfcPreset.value = String(Math.max(idx, 0));
    applyPreset();
  }

  function currentPreset() {
    const all = BUILTIN_CFC.concat(customPresets);
    return all[Number(cfcPreset.value)] || all[0];
  }

  function applyPreset() {
    const p = currentPreset();
    cfcFields.forEach((f) => ($("c" + f).value = p[f]));
    cfcSummary.textContent = p.name;
    delPresetBtn.hidden = Number(cfcPreset.value) < BUILTIN_CFC.length;
  }

  cfcPreset.addEventListener("change", applyPreset);

  savePresetBtn.addEventListener("click", () => {
    const name = prompt("Preset name (e.g. \"6061-T6 / 12.7 mm carbide\"):");
    if (!name) return;
    const p = { name: name.trim() };
    cfcFields.forEach((f) => (p[f] = Number($("c" + f).value) || 0));
    const existing = customPresets.findIndex((x) => x.name === p.name);
    if (existing >= 0) customPresets[existing] = p;
    else customPresets.push(p);
    saveJson(CFC_STORE_KEY, customPresets);
    refreshPresetSelect(p.name);
  });

  delPresetBtn.addEventListener("click", () => {
    const idx = Number(cfcPreset.value) - BUILTIN_CFC.length;
    if (idx < 0) return;
    customPresets.splice(idx, 1);
    saveJson(CFC_STORE_KEY, customPresets);
    refreshPresetSelect();
  });

  // ---------- dynamics modes editor ----------
  const DEFAULT_MODES = [
    { fn: 500, k: 10, zeta: 0.05 },
    { fn: 1000, k: 4, zeta: 0.02 },
    { fn: 2300, k: 5, zeta: 0.03 },
  ];

  function addModeRow(container, m) {
    const row = document.createElement("div");
    row.className = "modes-row";
    row.innerHTML = `
      <input type="number" step="any" inputmode="decimal" class="m-fn" value="${m.fn}">
      <input type="number" step="any" inputmode="decimal" class="m-k" value="${m.k}">
      <input type="number" step="any" inputmode="decimal" class="m-z" value="${m.zeta}">
      <button class="mode-del" aria-label="Remove mode">✕</button>`;
    row.querySelector(".mode-del").addEventListener("click", () => row.remove());
    container.appendChild(row);
  }

  function readModes(container) {
    const rows = container.querySelectorAll(".modes-row");
    const out = [];
    rows.forEach((r) => {
      const fn = Number(r.querySelector(".m-fn").value);
      const k = Number(r.querySelector(".m-k").value);
      const zeta = Number(r.querySelector(".m-z").value);
      if (fn > 0 && k > 0 && zeta > 0) out.push({ fn, k, zeta });
    });
    return out;
  }

  function setModes(container, list) {
    container.innerHTML = "";
    list.forEach((m) => addModeRow(container, m));
  }

  $("addModeXBtn").addEventListener("click", () =>
    addModeRow(modeRowsX, { fn: 1000, k: 5, zeta: 0.03 })
  );
  $("addModeYBtn").addEventListener("click", () =>
    addModeRow(modeRowsY, { fn: 1000, k: 5, zeta: 0.03 })
  );
  $("copyXYBtn").addEventListener("click", () =>
    setModes(modeRowsY, readModes(modeRowsX))
  );

  // ---------- persistence of sim inputs ----------
  const SIM_FIELDS = ["sOmega", "sB", "sNt", "sD", "sBeta", "sFt", "sRd", "sUd", "sTime"];

  function persistSimInputs() {
    const data = {
      fields: {},
      modesX: readModes(modeRowsX),
      modesY: readModes(modeRowsY),
      preset: currentPreset().name,
    };
    SIM_FIELDS.forEach((id) => (data.fields[id] = $(id).value));
    cfcFields.forEach((f) => (data.fields["c" + f] = $("c" + f).value));
    saveJson(SIM_STORE_KEY, data);
  }

  function restoreSimInputs() {
    const data = loadJson(SIM_STORE_KEY, null);
    refreshPresetSelect(data && data.preset);
    // back-compat: older saves had a single "modes" list for both directions
    const legacy = data && data.modes && data.modes.length ? data.modes : null;
    const mx = (data && data.modesX && data.modesX.length && data.modesX) || legacy || DEFAULT_MODES;
    const my = (data && data.modesY && data.modesY.length && data.modesY) || legacy || DEFAULT_MODES;
    setModes(modeRowsX, mx);
    setModes(modeRowsY, my);
    if (data && data.fields) {
      Object.keys(data.fields).forEach((id) => {
        const el = $(id);
        if (el) el.value = data.fields[id];
      });
    }
  }

  // ---------- run simulation ----------
  let simRunMeta = null; // { omega, Nt } of the last run, for the results card

  runSimBtn.addEventListener("click", () => {
    if (simWorker) return; // already running
    simError.hidden = true;

    const num = (id) => Number($(id).value);
    const omega = num("sOmega");
    const b = num("sB") * 1e-3;
    const Nt = Math.round(num("sNt"));
    const d = num("sD") * 1e-3;
    const beta = num("sBeta");
    const ft = num("sFt") * 1e-3;
    let rd = num("sRd") * 1e-3;
    const ud = Number($("sUd").value);
    const simTime = num("sTime");

    if (!(omega > 0 && b > 0 && Nt >= 1 && d > 0 && ft > 0 && rd > 0 && simTime > 0)) {
      showSimError("Check the inputs — every value must be a positive number.");
      return;
    }
    if (rd > d) rd = d; // radial depth cannot exceed diameter

    // Simulation time → whole revolutions (at least 20 so the
    // steady-state half still contains enough revs for the metric).
    const numRevs = Math.max(20, Math.round((simTime * omega) / 60));
    if (numRevs * SIM_STEPS_REV > 40e6) {
      const maxT = ((40e6 / SIM_STEPS_REV) * 60) / omega;
      showSimError(
        "Simulation time too long for this spindle speed — keep it under " +
          maxT.toFixed(1) + " s."
      );
      return;
    }

    // SI conversion: N/mm² → N/m², N/mm → N/m; k N/µm → N/m; fn Hz → rad/s
    const K = {};
    cfcFields.forEach((f) => (K[f] = Number($("c" + f).value) || 0));
    const toSI = (m) => ({ wn: m.fn * 2 * Math.PI, k: m.k * 1e6, zeta: m.zeta });
    const modesX = readModes(modeRowsX).map(toSI);
    const modesY = readModes(modeRowsY).map(toSI);

    persistSimInputs();
    simRunMeta = { omega, Nt };

    simResults.hidden = true;
    stopSimPlayback();
    simAudio = null;
    simRaw = null;
    simProgress.hidden = false;
    simProgressFill.style.width = "0%";
    runSimBtn.textContent = "RUNNING…";
    runSimBtn.disabled = true;

    simWorker = new Worker("./sim-worker.js");
    simWorker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "progress") {
        simProgressFill.style.width = (msg.value * 100).toFixed(1) + "%";
      } else if (msg.type === "done") {
        finishSim(msg, modesX.length > 0);
      } else if (msg.type === "error") {
        showSimError("Simulation failed: " + msg.message);
        resetSimUi();
      }
    };
    simWorker.onerror = (err) => {
      showSimError("Simulation failed: " + (err.message || "worker error"));
      resetSimUi();
    };

    simWorker.postMessage({
      Ktc: K.Ktc * 1e6, Krc: K.Krc * 1e6,
      Kte: K.Kte * 1e3, Kre: K.Kre * 1e3,
      omega, b, d, Nt, beta, ft, rd, ud,
      modesX, modesY,
      numRevs, stepsRev: SIM_STEPS_REV, stepsAxial: SIM_AXIAL,
    });
  });

  function finishSim(msg, hasModes) {
    resetSimUi();

    // Sonify tool displacement when the tool is flexible (that's the
    // vibration you hear); fall back to force for a rigid tool.
    const src = hasModes ? msg.xt : msg.fx;
    const samples = resampleToRate(src, msg.fs, AUDIO_RATE);
    normalizePeak(samples, 0.7);

    simAudio = {
      samples,
      rate: AUDIO_RATE,
      metric: msg.metric,
      fs: msg.fs,
    };
    simRaw = { fx: msg.fx, fy: msg.fy, xt: msg.xt, fs: msg.fs, res: null };

    $("rMetric").textContent = Number.isFinite(msg.metric)
      ? msg.metric.toPrecision(3) + " µm"
      : "–";
    $("rDur").textContent = (samples.length / AUDIO_RATE).toFixed(2) + " s";
    const ftooth = simRunMeta ? (simRunMeta.Nt * simRunMeta.omega) / 60 : 0;
    $("rFtooth").textContent = ftooth > 0 ? fmtHz(ftooth) : "–";
    simResults.hidden = false;

    // canvases need layout before they have a width
    requestAnimationFrame(() => {
      drawForcePlot();
      drawSoundPlot(null);
      simResults.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }

  // ---------- result plots ----------

  function forceData() {
    const ch = forceChan.value;
    if (ch === "fx") return simRaw.fx;
    if (ch === "fy") return simRaw.fy;
    if (!simRaw.res) {
      const n = simRaw.fx.length;
      const r = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        r[i] = Math.hypot(simRaw.fx[i], simRaw.fy[i]);
      }
      simRaw.res = r;
    }
    return simRaw.res;
  }

  // Per-pixel min/max envelope of a long signal.
  function envelope(data, w) {
    const mins = new Float32Array(w);
    const maxs = new Float32Array(w);
    const n = data.length;
    for (let px = 0; px < w; px++) {
      const a = Math.floor((px * n) / w);
      const b = Math.max(a + 1, Math.floor(((px + 1) * n) / w));
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = a; i < b && i < n; i++) {
        const v = data[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      mins[px] = lo;
      maxs[px] = hi;
    }
    return { mins, maxs };
  }

  function prepPlotCanvas(canvas) {
    const dpr = DPR();
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (cssW === 0 || cssH === 0) return null;
    const w = Math.round(cssW * dpr);
    const h = Math.round(cssH * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    return { ctx, w: cssW, h: cssH };
  }

  function drawWave(canvas, data, color) {
    const p = prepPlotCanvas(canvas);
    if (!p) return null;
    const { ctx, w, h } = p;
    const pad = 4;

    const { mins, maxs } = envelope(data, w);
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < w; i++) {
      if (mins[i] < lo) lo = mins[i];
      if (maxs[i] > hi) hi = maxs[i];
    }
    if (!(hi > lo)) {
      hi = lo + 1;
    }
    const span = hi - lo;
    const yOf = (v) => pad + (1 - (v - lo) / span) * (h - 2 * pad);

    // zero line if zero is in range
    if (lo <= 0 && hi >= 0) {
      ctx.strokeStyle = "rgba(44, 51, 62, 0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, yOf(0));
      ctx.lineTo(w, yOf(0));
      ctx.stroke();
    }

    // filled min/max envelope
    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const y = yOf(maxs[x]);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let x = w - 1; x >= 0; x--) {
      ctx.lineTo(x, yOf(mins[x]));
    }
    ctx.closePath();
    ctx.fillStyle = color.fill;
    ctx.fill();
    ctx.strokeStyle = color.line;
    ctx.lineWidth = 1;
    ctx.stroke();

    return { hi, lo, w, h };
  }

  function drawForcePlot() {
    if (!simRaw) return;
    const info = drawWave(forcePlot, forceData(), {
      line: "#7FD8E8",
      fill: "rgba(127, 216, 232, 0.14)",
    });
    if (info) {
      const peak = Math.max(Math.abs(info.hi), Math.abs(info.lo));
      forceStats.textContent = "peak " + fmtForce(peak);
    }
  }

  function drawSoundPlot(playT) {
    if (!simAudio) return;
    const info = drawWave(soundPlot, simAudio.samples, {
      line: "#FFB24D",
      fill: "rgba(255, 178, 77, 0.14)",
    });
    if (!info) return;
    const dur = simAudio.samples.length / simAudio.rate;
    soundStats.textContent = dur.toFixed(2) + " s";
    if (playT != null && dur > 0) {
      const ctx = soundPlot.getContext("2d");
      const x = clamp(playT / dur, 0, 1) * info.w;
      ctx.strokeStyle = "#7FD8E8";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, info.h);
      ctx.stroke();
    }
  }

  function fmtForce(n) {
    return n >= 1000 ? (n / 1000).toFixed(2) + " kN" : n.toFixed(1) + " N";
  }

  forceChan.addEventListener("change", drawForcePlot);

  window.addEventListener("resize", () => {
    if (simRaw && !simResults.hidden) {
      drawForcePlot();
      drawSoundPlot(simPlaySource ? currentSimPlayT() : null);
    }
  });

  function resetSimUi() {
    if (simWorker) {
      simWorker.terminate();
      simWorker = null;
    }
    simProgress.hidden = true;
    runSimBtn.textContent = "RUN SIMULATION";
    runSimBtn.disabled = false;
  }

  function showSimError(msg) {
    simError.textContent = msg;
    simError.hidden = false;
  }

  // Box-filtered resampling (prefix sums) for decimation; linear for upsampling.
  function resampleToRate(x, fsIn, fsOut) {
    const outLen = Math.max(1, Math.floor((x.length * fsOut) / fsIn));
    const out = new Float32Array(outLen);
    const ratio = fsIn / fsOut;

    if (ratio <= 1) {
      for (let n = 0; n < outLen; n++) {
        const t = n * ratio;
        const i = Math.floor(t);
        const fr = t - i;
        const a = x[Math.min(i, x.length - 1)];
        const b = x[Math.min(i + 1, x.length - 1)];
        out[n] = a + (b - a) * fr;
      }
      return out;
    }

    // moving-average anti-alias then decimate
    const pre = new Float64Array(x.length + 1);
    for (let i = 0; i < x.length; i++) pre[i + 1] = pre[i] + x[i];
    const half = ratio / 2;
    for (let n = 0; n < outLen; n++) {
      const c = n * ratio;
      let a = Math.max(0, Math.round(c - half));
      let b = Math.min(x.length, Math.round(c + half));
      if (b <= a) b = a + 1;
      out[n] = (pre[b] - pre[a]) / (b - a);
    }
    return out;
  }

  function normalizePeak(x, target) {
    let peak = 0;
    for (let i = 0; i < x.length; i++) {
      const a = Math.abs(x[i]);
      if (a > peak) peak = a;
    }
    if (peak < 1e-12) return;
    const s = target / peak;
    for (let i = 0; i < x.length; i++) x[i] *= s;
  }

  // ---------- quick play in sim tab ----------
  function currentSimPlayT() {
    return simPlayCtx ? simPlayCtx.currentTime - simPlayT0 : 0;
  }

  function simPlayheadRaf() {
    if (!simPlaySource) return;
    drawSoundPlot(currentSimPlayT());
    requestAnimationFrame(simPlayheadRaf);
  }

  playSimBtn.addEventListener("click", async () => {
    if (!simAudio) return;
    if (simPlaySource) {
      stopSimPlayback();
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!simPlayCtx) simPlayCtx = new AC();
    if (simPlayCtx.state === "suspended") await simPlayCtx.resume();
    const buf = simPlayCtx.createBuffer(1, simAudio.samples.length, simAudio.rate);
    buf.copyToChannel(simAudio.samples, 0);
    simPlaySource = simPlayCtx.createBufferSource();
    simPlaySource.buffer = buf;
    simPlaySource.connect(simPlayCtx.destination);
    simPlaySource.onended = () => stopSimPlayback();
    simPlayT0 = simPlayCtx.currentTime;
    simPlaySource.start();
    playSimBtn.textContent = "■ STOP";
    simPlayheadRaf();
  });

  function stopSimPlayback() {
    if (simPlaySource) {
      simPlaySource.onended = null;
      try { simPlaySource.stop(); } catch (e) {}
      simPlaySource = null;
      drawSoundPlot(null); // clear the playhead
    }
    playSimBtn.textContent = "▶ PLAY";
  }

  // ---------- send to analyzer ----------
  sendBtn.addEventListener("click", async () => {
    if (!simAudio) return;
    stopSimPlayback();
    if (mode === "live") stopLive(true); // discard any live capture
    switchTab("analyze");
    // copy so the sim result can be sent again later
    await openInReview(new Float32Array(simAudio.samples), simAudio.rate);
  });

  // ============================================================
  // DRAWING (shared)
  // ============================================================

  function drawSpectrumTrace(mags, maxBin, hzPerBin, peaks, startBin) {
    const w = specCanvas.clientWidth;
    const h = specCanvas.clientHeight;
    if (w === 0 || h === 0) return;
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
    if (w === 0 || h === 0) return;
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

  function drawLiveGramColumn(mags, maxBin) {
    const ctx = gramCtx;
    const pw = gramCanvas.width;
    const ph = gramCanvas.height;
    if (pw === 0 || ph === 0) return;
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
  // UI STATE
  // ============================================================

  function setMode(next) {
    if (mode === "review" && next !== "review") teardownReview();

    mode = next;

    statusEl.className =
      "status" + (mode === "live" ? " live" : mode === "review" ? " review" : "");
    statusText.textContent =
      mode === "live" ? "listening" : mode === "review" ? "review" : "idle";

    transport.hidden = mode !== "review";
    loadLabel.hidden = mode !== "idle";
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
    else if (mode === "live") stopLive(false);
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

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && mode === "live") stopLive(false);
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
  restoreSimInputs();
  requestAnimationFrame(() => {
    fitCanvas(specCanvas, specCtx, false);
    fitCanvas(gramCanvas, gramCtx, false);
    drawGrid(specCtx, specCanvas.clientWidth, specCanvas.clientHeight);
  });
})();
