// Kesan chord daripada audio: kromagram (puncak spektrum) + padanan templat + Viterbi.
// Tiada model AI: pantas, berjalan dalam pelayar. Ketepatan sederhana (lebih baik pada audio harmonik tanpa dram/vokal).

const NOTES_S = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const NOTES_F = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];

export const TYPES = [
  { id: 'maj', suffix: '', w: [1, 0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0], bonus: 0.04 },
  { id: 'min', suffix: 'm', w: [1, 0, 0, 0.8, 0, 0, 0, 0.8, 0, 0, 0, 0], bonus: 0.04 },
  { id: '7', suffix: '7', w: [1, 0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0.6, 0], bonus: 0 },
  { id: 'm7', suffix: 'm7', w: [1, 0, 0, 0.8, 0, 0, 0, 0.8, 0, 0, 0.6, 0], bonus: 0 },
  { id: 'maj7', suffix: 'maj7', w: [1, 0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0.6], bonus: 0 },
];
const NT = TYPES.length;
const NS = 12 * NT + 1;           // + keadaan "tiada chord"
const NOCHORD = NS - 1;

// templat ternormal L2 untuk setiap (root, jenis)
const TEMPLATES = [];
for (let r = 0; r < 12; r++) for (let t = 0; t < NT; t++) {
  const v = new Float32Array(12);
  for (let i = 0; i < 12; i++) v[(i + r) % 12] = TYPES[t].w[i];
  let n = 0; for (let i = 0; i < 12; i++) n += v[i] * v[i];
  n = Math.sqrt(n); for (let i = 0; i < 12; i++) v[i] /= n;
  TEMPLATES.push(v);
}

// profil kunci Krumhansl-Kessler
const KMAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KMIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

// ---------- FFT ----------
function makeFFT(n) {
  const levels = Math.log2(n);
  const cos = new Float32Array(n / 2), sin = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) { let x = i, r = 0; for (let j = 0; j < levels; j++) { r = (r << 1) | (x & 1); x >>= 1; } rev[i] = r; }
  return function fft(re, im) {
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half;
          const tr = re[l] * cos[k] + im[l] * sin[k];
          const ti = -re[l] * sin[k] + im[l] * cos[k];
          re[l] = re[j] - tr; im[l] = im[j] - ti;
          re[j] += tr; im[j] += ti;
        }
      }
    }
  };
}

const tick = () => new Promise(r => setTimeout(r, 0));

/** Susun nama chord. */
export function noteName(pc, flats) { return (flats ? NOTES_F : NOTES_S)[((pc % 12) + 12) % 12]; }
export function chordLabel(seg, transpose = 0, flats = false) {
  if (seg.type < 0) return 'N.C.';
  return noteName(seg.root + transpose, flats) + TYPES[seg.type].suffix;
}
/** Adakah kunci ini lazim ditulis dengan flat? */
export function prefersFlats(tonic, mode) {
  const maj = [5, 10, 3, 8, 1, 6], min = [2, 7, 0, 5, 10, 3];
  return (mode === 'maj' ? maj : min).includes(((tonic % 12) + 12) % 12);
}

/** Anggar kunci daripada profil kromatik keseluruhan. */
export function estimateKey(profile) {
  let best = { score: -Infinity, tonic: 0, mode: 'maj' };
  const mean = profile.reduce((a, b) => a + b, 0) / 12;
  for (let t = 0; t < 12; t++) for (const [mode, prof] of [['maj', KMAJ], ['min', KMIN]]) {
    const pm = prof.reduce((a, b) => a + b, 0) / 12;
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < 12; i++) {
      const a = profile[(i + t) % 12] - mean, b = prof[i] - pm;
      num += a * b; da += a * a; db += b * b;
    }
    const s = num / Math.sqrt(da * db + 1e-12);
    if (s > best.score) best = { score: s, tonic: t, mode };
  }
  return best;
}

/**
 * detectChords(left, right, sr, opts) -> { segments:[{start,end,root,type}], key:{tonic,mode}, duration }
 * opts.onProgress(0..1), opts.switchCost (lalai 1.6)
 */
export async function detectChords(left, right, sr, opts = {}) {
  const onProgress = opts.onProgress || (() => {});
  const FS = 11025, N = 8192, HOP = 2048;
  const factor = Math.max(1, Math.round(sr / FS));
  const fs = sr / factor;
  const n = Math.floor(left.length / factor);

  // mono + turun-sampel (purata blok = tapis lulus rendah kasar)
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0; const o = i * factor;
    for (let j = 0; j < factor; j++) s += left[o + j] + (right ? right[o + j] : left[o + j]);
    x[i] = s / (2 * factor);
  }
  const frames = Math.max(0, Math.floor((n - N) / HOP) + 1);
  if (frames < 3) return { segments: [], key: { tonic: 0, mode: 'maj' }, duration: left.length / sr };

  const fft = makeFFT(N);
  const win = new Float32Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1));
  const re = new Float32Array(N), im = new Float32Array(N), mag = new Float32Array(N / 2);
  const kmin = Math.ceil(55 * N / fs), kmax = Math.min(N / 2 - 2, Math.floor(2600 * N / fs));

  // pas 1: puncak spektrum bagi setiap bingkai
  const PK = 48;
  const pf = new Float32Array(frames * PK), pm = new Float32Array(frames * PK);
  const energy = new Float32Array(frames);
  let devC = 0, devS = 0;
  for (let t = 0; t < frames; t++) {
    const o = t * HOP;
    for (let i = 0; i < N; i++) { re[i] = x[o + i] * win[i]; im[i] = 0; }
    fft(re, im);
    let maxm = 0;
    for (let k = kmin - 1; k <= kmax + 1; k++) { mag[k] = Math.hypot(re[k], im[k]); if (mag[k] > maxm) maxm = mag[k]; }
    energy[t] = maxm;
    // pilih puncak tempatan, simpan PK terkuat
    const cand = [];
    const thr = maxm * 0.04;
    for (let k = kmin; k <= kmax; k++) {
      const m = mag[k];
      if (m > thr && m > mag[k - 1] && m >= mag[k + 1]) {
        const a = Math.log(mag[k - 1] + 1e-9), b = Math.log(m + 1e-9), c = Math.log(mag[k + 1] + 1e-9);
        const d = 0.5 * (a - c) / (a - 2 * b + c);
        cand.push([(k + (isFinite(d) ? d : 0)) * fs / N, m]);
      }
    }
    cand.sort((p, q) => q[1] - p[1]);
    for (let i = 0; i < Math.min(PK, cand.length); i++) {
      pf[t * PK + i] = cand[i][0]; pm[t * PK + i] = cand[i][1];
      const p = 12 * Math.log2(cand[i][0] / 440);
      const dev = p - Math.round(p), w = Math.sqrt(cand[i][1]);
      devC += w * Math.cos(2 * Math.PI * dev); devS += w * Math.sin(2 * Math.PI * dev);
    }
    if (t % 80 === 0) { onProgress(0.6 * t / frames); await tick(); }
  }
  const tuning = Math.atan2(devS, devC) / (2 * Math.PI);   // -0.5..0.5 semiton

  // pas 2: kromagram (+ bass)
  const chroma = new Float32Array(frames * 12), bass = new Float32Array(frames * 12);
  for (let t = 0; t < frames; t++) {
    for (let i = 0; i < PK; i++) {
      const f = pf[t * PK + i]; if (f === 0) break;
      const p = 12 * Math.log2(f / 440) - tuning;
      const r = Math.round(p), dev = Math.abs(p - r);
      const wdev = Math.max(0, 1 - 2 * dev);
      if (wdev === 0) continue;
      const pc = (((r + 9) % 12) + 12) % 12;              // 440Hz = A = pc 9
      let w = Math.sqrt(pm[t * PK + i]) * wdev;
      if (f > 1500) w *= 0.5;
      chroma[t * 12 + pc] += w;
      if (f < 260) bass[t * 12 + pc] += w;
    }
    for (const arr of [chroma, bass]) {
      let s = 0; for (let i = 0; i < 12; i++) s += arr[t * 12 + i] * arr[t * 12 + i];
      s = Math.sqrt(s) + 1e-9; for (let i = 0; i < 12; i++) arr[t * 12 + i] /= s;
    }
  }
  // licinkan masa: [.25 .5 .25]
  const cs = new Float32Array(frames * 12);
  for (let t = 0; t < frames; t++) for (let i = 0; i < 12; i++) {
    const a = chroma[Math.max(0, t - 1) * 12 + i], b = chroma[t * 12 + i], c = chroma[Math.min(frames - 1, t + 1) * 12 + i];
    cs[t * 12 + i] = 0.25 * a + 0.5 * b + 0.25 * c;
  }

  // pas 3: Viterbi
  const medE = Array.from(energy).sort((a, b) => a - b)[Math.floor(frames / 2)];
  const BETA = 8, LAMBDA = opts.switchCost ?? 1.6, GAMMA = 0.12;
  const emis = new Float32Array(NS);
  let score = new Float32Array(NS), next = new Float32Array(NS);
  const back = new Uint8Array(frames * NS);
  const prof = new Float32Array(12);
  for (let t = 0; t < frames; t++) {
    let cn = 0; for (let i = 0; i < 12; i++) cn += cs[t * 12 + i] * cs[t * 12 + i];
    cn = Math.sqrt(cn) + 1e-9;
    for (let i = 0; i < 12; i++) prof[i] += cs[t * 12 + i];
    for (let r = 0; r < 12; r++) for (let ty = 0; ty < NT; ty++) {
      const tpl = TEMPLATES[r * NT + ty];
      let d = 0; for (let i = 0; i < 12; i++) d += cs[t * 12 + i] * tpl[i];
      emis[r * NT + ty] = d / cn + TYPES[ty].bonus + GAMMA * bass[t * 12 + r];
    }
    emis[NOCHORD] = energy[t] < 0.08 * medE ? 1.2 : 0.25;
    if (t === 0) { for (let s = 0; s < NS; s++) score[s] = BETA * emis[s]; continue; }
    let bi = 0; for (let s = 1; s < NS; s++) if (score[s] > score[bi]) bi = s;
    const sw = score[bi] - LAMBDA;
    for (let s = 0; s < NS; s++) {
      if (score[s] >= sw) { next[s] = score[s]; back[t * NS + s] = s; }
      else { next[s] = sw; back[t * NS + s] = bi; }
      next[s] += BETA * emis[s];
    }
    [score, next] = [next, score];
    if (t % 200 === 0) { onProgress(0.6 + 0.4 * t / frames); await tick(); }
  }
  let s = 0; for (let k = 1; k < NS; k++) if (score[k] > score[s]) s = k;
  const path = new Int16Array(frames);
  for (let t = frames - 1; t >= 0; t--) { path[t] = s; if (t > 0) s = back[t * NS + s]; }

  // segmen
  const segments = [];
  const tOf = t => Math.max(0, ((t * HOP + N / 2) - HOP / 2) / fs);
  const total = left.length / sr;
  for (let t = 0; t < frames; t++) {
    const st = path[t];
    const root = st === NOCHORD ? -1 : Math.floor(st / NT), type = st === NOCHORD ? -1 : st % NT;
    const last = segments[segments.length - 1];
    if (last && last.root === root && last.type === type) continue;
    if (last) last.end = t === 0 ? 0 : tOf(t);
    segments.push({ start: t === 0 ? 0 : tOf(t), end: total, root, type });
  }
  if (segments.length) segments[segments.length - 1].end = total;
  onProgress(1);
  return { segments, key: estimateKey(Array.from(prof)), duration: total, tuning };
}

/** Chord pada masa t (saat) → indeks segmen. */
export function segmentAt(segments, t) {
  let lo = 0, hi = segments.length - 1, ans = 0;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (segments[mid].start <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1; }
  return ans;
}
