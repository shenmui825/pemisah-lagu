// Worker: muat model (ONNX) dan jalankan pemisahan di luar thread UI.
// Dua jenis model:
//  m4 = htdemucs 4 stem (timcsy, STFT dikira di luar graf oleh DemucsProcessor)
//  m6 = htdemucs 6 stem (adowu, input gelombang `mix`, output `stems` [1,6,2,T])
import * as ort from './lib/ort.webgpu.min.mjs';
import { DemucsProcessor } from './lib/demucs/processor.js';

const CACHE = 'pemisah-model-v1';
const T = 343980;                       // panjang segmen model (7.8s @ 44.1k)
const M6_KEYS = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'];

ort.env.wasm.numThreads = 1; // GitHub Pages tiada cross-origin isolation
ort.env.wasm.wasmPaths = new URL('./lib/', import.meta.url).href;

let kind = '', processor = null, session = null, provider = '';
let modelBuf = null, modelUrl = '', fromCache = false;

const post = (m, t) => self.postMessage(m, t || []);
const errText = e => String((e && e.message) ? e.message : e).slice(0, 300);

async function readModel(url, file) {
  const cache = await caches.open(CACHE);
  fromCache = false;
  if (file) {
    const b = await file.arrayBuffer();
    try { await cache.put(url, new Response(b.slice(0))); fromCache = true; } catch (e) {}
    return b;
  }
  const hit = await cache.match(url);
  if (hit) { fromCache = true; post({ type: 'status', text: 'Membaca model dari simpanan peranti…' }); return await hit.arrayBuffer(); }

  post({ type: 'status', text: 'Memuat turun model (kali pertama sahaja)…' });
  const r = await fetch(url);
  if (!r.ok) throw new Error('Muat turun model gagal (HTTP ' + r.status + ')');
  const total = +r.headers.get('Content-Length') || 0;
  const reader = r.body.getReader();
  const chunks = []; let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.length;
    post({ type: 'dl', loaded, total });
  }
  const u8 = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { u8.set(c, o); o += c.length; }
  chunks.length = 0;
  if (total && loaded < total) throw new Error('Muat turun tidak lengkap, cuba lagi');
  try { await cache.put(url, new Response(u8.buffer.slice(0))); fromCache = true; } catch (e) { /* storan penuh */ }
  return u8.buffer;
}

async function createSession(eps) {
  session = null; processor = null;
  if (kind === 'm4') {
    processor = new DemucsProcessor({
      ort,
      sessionOptions: { executionProviders: eps },
      onProgress: p => post({ type: 'progress', p: p.progress, seg: p.currentSegment, total: p.totalSegments }),
    });
    await processor.loadModel(modelBuf.slice(0));
  } else {
    session = await ort.InferenceSession.create(modelBuf.slice(0), {
      executionProviders: eps,
      graphOptimizationLevel: 'basic',
    });
  }
  provider = eps[0];
}

const ready = () => (kind === 'm4' ? !!processor : !!session);

const OLD_URLS = ['https://huggingface.co/adowu/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx', 'https://huggingface.co/adowu/htdemucs-6s-onnx/resolve/main/htdemucs_6s.onnx'];
async function load(k, url, file, forceWasm) {
  try { const c = await caches.open(CACHE); for (const u of OLD_URLS) await c.delete(u); } catch (e) {}
  if (k !== kind || url !== modelUrl || file) {
    processor = null; session = null; modelBuf = null;
    kind = k;
    modelBuf = await readModel(url, file);
    modelUrl = url;
  }
  post({ type: 'status', text: 'Menyediakan enjin AI…' });
  const order = forceWasm ? [['wasm']] : [['webgpu'], ['wasm']];
  const errs = [];
  for (const eps of order) {
    try {
      await createSession(eps);
      post({ type: 'ready', provider });
      return true;
    } catch (e) {
      errs.push(eps[0] + ': ' + errText(e)); processor = null; session = null;
      if (eps[0] === 'webgpu') {
        // enjin WASM yang sama mungkin rosak selepas WebGPU gagal: main akan mula semula worker
        post({ type: 'needcpu', reason: 'sedia WebGPU: ' + errText(e) });
        return false;
      }
    }
  }
  const msg = errs.join(' | ');
  // model rosak: buang supaya muat turun semula pada percubaan seterusnya
  if (/protobuf|pars|invalid model|Failed to load model/i.test(msg)) {
    try { await (await caches.open(CACHE)).delete(url); } catch (e) {}
    modelBuf = null; modelUrl = ''; fromCache = false;
  }
  throw new Error('Gagal menyediakan enjin AI (' + msg + ')');
}

// ---- pemisahan 6 stem: overlap-add dengan tetingkap ----
async function separate6(left, right) {
  const N = left.length;
  const stride = Math.floor(T * 0.75);
  const F = T - stride;                       // panjang pertindihan
  const win = new Float32Array(T);
  for (let i = 0; i < T; i++) win[i] = Math.min(1, (i + 1) / F, (T - i) / F);

  const K = M6_KEYS.length;
  const outs = M6_KEYS.map(() => ({ left: new Float32Array(N), right: new Float32Array(N) }));
  const wsum = new Float32Array(N);
  const nSeg = Math.max(1, Math.ceil(Math.max(0, N - T) / stride) + 1);
  const inName = session.inputNames[0];
  const outName = session.outputNames[0];

  let seg = 0;
  for (let start = 0; ; start += stride) {
    const len = Math.min(T, N - start);
    const buf = new Float32Array(2 * T);       // [L(T), R(T)], sisa = sifar
    buf.set(left.subarray(start, start + len), 0);
    buf.set(right.subarray(start, start + len), T);
    const res = await session.run({ [inName]: new ort.Tensor('float32', buf, [1, 2, T]) });
    const d = res[outName].data;               // [1, K, 2, T]
    for (let k = 0; k < K; k++) {
      const oL = (k * 2) * T, oR = (k * 2 + 1) * T;
      const L = outs[k].left, R = outs[k].right;
      for (let i = 0; i < len; i++) {
        const w = win[i];
        L[start + i] += d[oL + i] * w;
        R[start + i] += d[oR + i] * w;
      }
    }
    for (let i = 0; i < len; i++) wsum[start + i] += win[i];
    seg++;
    post({ type: 'progress', p: Math.min(1, seg / nSeg), seg, total: nSeg });
    if (start + T >= N) break;
  }
  for (let i = 0; i < N; i++) {
    const w = wsum[i];
    if (w > 0) for (let k = 0; k < K; k++) { outs[k].left[i] /= w; outs[k].right[i] /= w; }
  }
  const res = {};
  M6_KEYS.forEach((key, k) => { res[key] = outs[k]; });
  return res;
}

async function runSeparate(left, right) {
  try {
    return await (kind === 'm4' ? processor.separate(left, right) : separate6(left, right));
  } catch (e) {
    if (provider === 'webgpu') { post({ type: 'needcpu', reason: 'larian WebGPU: ' + errText(e) }); return null; }
    throw new Error('Larian (' + provider + '): ' + errText(e));
  }
}

self.onmessage = async ev => {
  const m = ev.data;
  try {
    if (m.type === 'check') {
      const cache = await caches.open(CACHE);
      post({ type: 'cached', value: !!(await cache.match(m.url)) });
    } else if (m.type === 'load') {
      await load(m.kind, m.url, m.file, m.forceWasm);
    } else if (m.type === 'separate') {
      if (!ready() || kind !== m.kind || (m.forceWasm && provider !== 'wasm')) {
        if (!(await load(m.kind, m.url, null, !!m.forceWasm))) return;
      }
      const res = await runSeparate(m.left, m.right);
      if (!res) return;
      const out = {}, tr = [];
      for (const k of Object.keys(res)) {
        out[k] = res[k];
        tr.push(res[k].left.buffer, res[k].right.buffer);
      }
      post({ type: 'done', stems: out }, tr);
    }
  } catch (e) {
    post({ type: 'error', message: (e && e.message) ? e.message : String(e) });
  }
};
