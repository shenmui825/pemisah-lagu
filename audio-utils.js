// Fungsi audio tulen (tiada DOM) — digunakan oleh app dan ujian.

/** Kira gandaan akhir setiap stem: bisu / solo / kelantangan. */
export function effectiveGains(states) {
  const anySolo = states.some(s => s.solo);
  return states.map(s => (s.mute || (anySolo && !s.solo) ? 0 : s.vol));
}

/** Campur beberapa stem {left,right} dengan gandaan → {left,right} (dihadkan -1..1). */
export function mixDown(stems, gains) {
  const n = stems[0].left.length;
  const left = new Float32Array(n);
  const right = new Float32Array(n);
  for (let s = 0; s < stems.length; s++) {
    const g = gains[s];
    if (g === 0) continue;
    const L = stems[s].left, R = stems[s].right;
    for (let i = 0; i < n; i++) {
      left[i] += L[i] * g;
      right[i] += R[i] * g;
    }
  }
  for (let i = 0; i < n; i++) {
    left[i] = Math.max(-1, Math.min(1, left[i]));
    right[i] = Math.max(-1, Math.min(1, right[i]));
  }
  return { left, right };
}

export function floatToInt16(f) {
  const o = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) {
    const v = Math.max(-1, Math.min(1, f[i]));
    o[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return o;
}

/** WAV 16-bit stereo. */
export function encodeWav(left, right, sampleRate) {
  const n = left.length;
  const buf = new ArrayBuffer(44 + n * 4);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + n * 4, true); w(8, 'WAVE');
  w(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 4, true);
  v.setUint16(32, 4, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, n * 4, true);
  const L = floatToInt16(left), R = floatToInt16(right);
  let o = 44;
  for (let i = 0; i < n; i++) { v.setInt16(o, L[i], true); v.setInt16(o + 2, R[i], true); o += 4; }
  return new Blob([buf], { type: 'audio/wav' });
}

/** MP3 melalui lamejs (global `lamejs`). Beri ruang kepada UI bernafas setiap beberapa blok. */
export async function encodeMp3(left, right, sampleRate, kbps, onProgress, lame = globalThis.lamejs) {
  if (!lame) throw new Error('lamejs tidak dimuatkan');
  const enc = new lame.Mp3Encoder(2, sampleRate, kbps);
  const L = floatToInt16(left), R = floatToInt16(right);
  const block = 1152, parts = [];
  for (let i = 0; i < L.length; i += block) {
    const out = enc.encodeBuffer(L.subarray(i, i + block), R.subarray(i, i + block));
    if (out.length) parts.push(new Uint8Array(out));
    if ((i / block) % 400 === 0) {
      onProgress && onProgress(i / L.length);
      await new Promise(r => setTimeout(r, 0));
    }
  }
  const end = enc.flush();
  if (end.length) parts.push(new Uint8Array(end));
  return new Blob(parts, { type: 'audio/mpeg' });
}

/** Puncak bentuk gelombang untuk lukisan ringkas. */
export function peaks(left, right, buckets) {
  const n = left.length, out = new Float32Array(buckets);
  const step = Math.max(1, Math.floor(n / buckets));
  for (let b = 0; b < buckets; b++) {
    let m = 0;
    const s = b * step, e = Math.min(n, s + step);
    for (let i = s; i < e; i += 8) {
      const a = Math.abs(left[i] + right[i]) * 0.5;
      if (a > m) m = a;
    }
    out[b] = m;
  }
  return out;
}

export function fmtTime(t) {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return m + ':' + String(s).padStart(2, '0');
}

export function safeName(name) {
  return (name || 'lagu').replace(/\.[^.]+$/, '').replace(/[^\p{L}\p{N}\-_ ]+/gu, '').trim().replace(/\s+/g, '_') || 'lagu';
}
