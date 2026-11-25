// Worker: muat model Demucs (ONNX) dan jalankan pemisahan di luar thread UI.
import * as ort from './lib/ort.webgpu.min.mjs';
import { DemucsProcessor } from './lib/demucs/processor.js';

const CACHE = 'pemisah-model-v1';
ort.env.wasm.numThreads = 1; // GitHub Pages tiada cross-origin isolation
ort.env.wasm.wasmPaths = new URL('./lib/', import.meta.url).href;

let processor = null;
let provider = '';
let modelBuf = null;
let modelUrl = '';
let fromCache = false;

const post = (m, t) => self.postMessage(m, t || []);

async function readModel(url, file) {
  const cache = await caches.open(CACHE);
  fromCache = false;
  if (file) return await file.arrayBuffer();
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
  return u8.buffer;
}

async function createSession(eps) {
  processor = new DemucsProcessor({
    ort,
    sessionOptions: { executionProviders: eps },
    onProgress: p => post({ type: 'progress', p: p.progress, seg: p.currentSegment, total: p.totalSegments }),
  });
  await processor.loadModel(modelBuf.slice(0));
  provider = eps[0];
}

async function load(url, file, forceWasm) {
  if (!modelBuf || modelUrl !== url || file) {
    modelBuf = await readModel(url, file);
    modelUrl = url;
  }
  post({ type: 'status', text: 'Menyediakan enjin AI…' });
  const order = forceWasm ? [['wasm']] : [['webgpu'], ['wasm']];
  let last;
  for (const eps of order) {
    try {
      await createSession(eps);
      // simpan model dalam peranti hanya selepas terbukti sah
      if (!fromCache) {
        try { await (await caches.open(CACHE)).put(url, new Response(modelBuf.slice(0))); fromCache = true; } catch (e) { /* storan penuh */ }
      }
      post({ type: 'ready', provider });
      return;
    } catch (e) { last = e; processor = null; }
  }
  // model rosak: buang supaya muat turun semula pada percubaan seterusnya
  try { await (await caches.open(CACHE)).delete(url); } catch (e) {}
  modelBuf = null; fromCache = false;
  throw last || new Error('Gagal menyediakan enjin AI');
}

async function runSeparate(left, right) {
  try {
    return await processor.separate(left, right);
  } catch (e) {
    if (provider !== 'webgpu') throw e;
    // WebGPU gagal semasa larian → ulang dengan CPU (WASM)
    post({ type: 'status', text: 'WebGPU bermasalah, tukar ke mod CPU (lebih perlahan)…' });
    await createSession(['wasm']);
    post({ type: 'ready', provider });
    return await processor.separate(left, right);
  }
}

self.onmessage = async ev => {
  const m = ev.data;
  try {
    if (m.type === 'check') {
      const cache = await caches.open(CACHE);
      post({ type: 'cached', value: !!(await cache.match(m.url)) });
    } else if (m.type === 'load') {
      await load(m.url, m.file, m.forceWasm);
    } else if (m.type === 'separate') {
      if (!processor) await load(m.url, null, false);
      const res = await runSeparate(m.left, m.right);
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
