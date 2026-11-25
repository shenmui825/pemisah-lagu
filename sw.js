// Cache app shell + pustaka supaya boleh guna tanpa internet. Model disimpan berasingan (Cache API dalam worker).
const V = 'pemisah-app-v4';
const SHELL = ['./', 'index.html', 'app.js', 'audio-utils.js', 'worker.js', 'manifest.webmanifest', 'icons/icon.svg', 'icons/icon-192.png', 'icons/icon-512.png',
  'lib/lame.min.js', 'lib/ort.webgpu.min.mjs', 'lib/demucs/processor.js', 'lib/demucs/fft.js', 'lib/demucs/constants.js'];
self.addEventListener('install', e => { e.waitUntil(caches.open(V).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k.startsWith('pemisah-app-') && k !== V).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  e.respondWith(caches.open(V).then(async c => {
    const hit = await c.match(e.request);
    if (hit) { fetch(e.request).then(r => { if (r.ok) c.put(e.request, r.clone()); }).catch(() => {}); return hit; }
    const r = await fetch(e.request); if (r.ok) c.put(e.request, r.clone()); return r;
  }));
});
