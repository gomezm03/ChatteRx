/* ============================================================
   ChatteRx simulation worker
   Port of Tony's "fixed_slices" digital-endmill time-domain
   milling simulation (Python/numba original). Runs off the main
   thread; posts progress and the steady-state signals when done.

   Input message (all SI units):
     { Ktc, Krc, Kte, Kre,            cutting/edge coefficients
       omega, b, d, Nt, beta, ft, rd, ud,
       modes: [{ wn (rad/s), k (N/m), zeta }],   tool-point, X and Y
       numRevs, stepsRev, stepsAxial }

   Output messages:
     { type: "progress", value: 0..1 }
     { type: "done", fs, metric, xt: Float32Array, fx: Float32Array }
     { type: "error", message }

   Notes vs. the Python original:
   - Uniform pitch, zero runout, no process damping (Ct = Cr = 0),
     rigid workpiece; identical mode set applied to X and Y.
   - Kac/Kae omitted: they only produce Fz, which we don't use.
   - Steady state = second half of the record (fraction = 0.5).
   - Stability metric = per-rev sampling of xt (SAMPLE_MODE="per_rev"),
     mean absolute successive difference in micrometers.
   ============================================================ */

"use strict";

self.onmessage = (e) => {
  try {
    run(e.data);
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};

function run(p) {
  const {
    Ktc, Krc, Kte, Kre,
    omega, b, d, Nt, beta, ft, rd, ud,
    modes, numRevs, stepsRev, stepsAxial,
  } = p;

  // --- engagement angles (deg) ---
  const acosArg = Math.min(1, Math.max(-1, (d / 2 - rd) / (d / 2)));
  let phistart, phiexit;
  if (ud === 1) {
    phistart = 0;
    phiexit = (Math.acos(acosArg) * 180) / Math.PI;
  } else {
    phistart = 180 - (Math.acos(acosArg) * 180) / Math.PI;
    phiexit = 180;
  }

  // --- time stepping ---
  const dt = 60 / (stepsRev * omega);
  const dphi = 360 / stepsRev;
  const fs = 1 / dt;
  const steps = numRevs * stepsRev;

  // --- Tony's 2D axial grid: per-disk helix lag in angular indices ---
  const db = b / stepsAxial;
  const dphiRad = (dphi * Math.PI) / 180;
  const tanB = Math.tan((beta * Math.PI) / 180);
  const lag = new Int32Array(stepsAxial);
  for (let disk = 0; disk < stepsAxial; disk++) {
    lag[disk] = Math.round((2 * (disk * db) * tanB) / d / dphiRad);
  }

  // --- precomputed angular tables ---
  const sinT = new Float64Array(stepsRev);
  const cosT = new Float64Array(stepsRev);
  const eng = new Uint8Array(stepsRev);
  for (let i = 0; i < stepsRev; i++) {
    const pv = i * dphi;
    const pr = (pv * Math.PI) / 180;
    sinT[i] = Math.sin(pr);
    cosT[i] = Math.cos(pr);
    eng[i] = pv >= phistart && pv <= phiexit ? 1 : 0;
  }

  // --- teeth pointers (uniform pitch) ---
  const teeth = new Int32Array(Nt);
  for (let i = 0; i < Nt; i++) {
    teeth[i] = Math.round(((i * 360) / Nt) / dphi) % stepsRev;
  }

  // --- coefficient premultiplication ---
  const Ktc_db = Ktc * db;
  const Kte_db = Kte * db;
  const Krc_db = Krc * db;
  const Kre_db = Kre * db;

  // --- surface regeneration store: (disk, angular index) ---
  const surface = new Float64Array(stepsAxial * stepsRev);

  // --- modal state (same modes in X and Y) ---
  const nm = modes.length;
  const mk = new Float64Array(nm);
  const mc = new Float64Array(nm);
  const mmass = new Float64Array(nm);
  for (let i = 0; i < nm; i++) {
    const k = modes[i].k;
    const wn = modes[i].wn;
    const zeta = modes[i].zeta;
    const mass = k / (wn * wn);
    mk[i] = k;
    mmass[i] = mass;
    mc[i] = 2 * zeta * Math.sqrt(mass * k);
  }
  const pt = new Float64Array(nm);
  const dpt = new Float64Array(nm);
  const qt = new Float64Array(nm);
  const dqt = new Float64Array(nm);

  // --- outputs ---
  const xt = new Float32Array(steps);
  const fx = new Float32Array(steps);
  const fy = new Float32Array(steps);

  const progEvery = Math.max(1, Math.floor(steps / 50));

  // displacements from the PREVIOUS step feed the force calculation
  let sxt = 0;
  let syt = 0;

  for (let cnt = 0; cnt < steps; cnt++) {
    // rotate cutter by one dphi
    for (let i = 0; i < Nt; i++) teeth[i] = (teeth[i] + 1) % stepsRev;

    let Fx = 0;
    let Fy = 0;

    for (let tooth = 0; tooth < Nt; tooth++) {
      const ctp = teeth[tooth];
      for (let disk = 0; disk < stepsAxial; disk++) {
        let idx = (ctp - lag[disk]) % stepsRev;
        if (idx < 0) idx += stepsRev;
        if (!eng[idx]) continue;

        const s = sinT[idx];
        const c = cosT[idx];
        const n = -(sxt * s + syt * c); // rigid workpiece: xw = yw = 0
        const off = disk * stepsRev + idx;
        const h = ft * s + surface[off] - n;

        if (h < 0) {
          // tooth jumped out of cut
          surface[off] += ft * s;
          continue;
        }
        const ftan = Ktc_db * h + Kte_db;
        const frad = Krc_db * h + Kre_db;
        surface[off] = n;

        Fx += -frad * s - ftan * c;
        Fy += -frad * c + ftan * s;
      }
    }

    fx[cnt] = Fx;
    fy[cnt] = Fy;

    // Euler integration — tool modes, X and Y
    sxt = 0;
    syt = 0;
    for (let m = 0; m < nm; m++) {
      let dd = (Fx - mc[m] * dpt[m] - mk[m] * pt[m]) / mmass[m];
      dpt[m] += dd * dt;
      pt[m] += dpt[m] * dt;
      sxt += pt[m];

      dd = (Fy - mc[m] * dqt[m] - mk[m] * qt[m]) / mmass[m];
      dqt[m] += dd * dt;
      qt[m] += dqt[m] * dt;
      syt += qt[m];
    }
    xt[cnt] = sxt;

    if (cnt % progEvery === 0) {
      self.postMessage({ type: "progress", value: cnt / steps });
    }
  }

  // --- steady state (second half) + per-rev stability metric ---
  const si = Math.floor(steps * 0.5);
  let metric = NaN;
  {
    let prev = NaN;
    let sum = 0;
    let count = 0;
    for (let i = si; i < steps; i += stepsRev) {
      const v = xt[i] * 1e6; // um
      if (!Number.isNaN(prev)) sum += Math.abs(v - prev);
      prev = v;
      count++;
    }
    if (count > 1) metric = sum / count;
  }

  const xtss = xt.slice(si);
  const fxss = fx.slice(si);
  const fyss = fy.slice(si);
  self.postMessage(
    { type: "done", fs, metric, xt: xtss, fx: fxss, fy: fyss },
    [xtss.buffer, fxss.buffer, fyss.buffer]
  );
}
