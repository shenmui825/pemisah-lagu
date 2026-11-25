import { effectiveGains, mixDown, encodeWav, encodeMp3, peaks, fmtTime, safeName } from './audio-utils.js';

const MODEL_URL = 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx';
const SR = 44100;
const STEMS = [
  { key: 'vocals', name: 'Vokal' },
  { key: 'drums', name: 'Dram' },
  { key: 'bass', name: 'Bass' },
  { key: 'other', name: 'Lain-lain (gitar, piano…)' },
];

const $ = s => document.querySelector(s);
const el = {
  file: $('#file'), pick: $('#btnPick'), drop: $('#drop'), info: $('#fileInfo'), sep: $('#btnSep'),
  barWrap: $('#barWrap'), bar: $('#bar'), status: $('#status'), err: $('#err'),
  modelInfo: $('#modelInfo'), modelFile: $('#modelFile'), btnModelFile: $('#btnModelFile'), forceCpu: $('#forceCpu'),
  player: $('#cardPlayer'), exp: $('#cardExport'), wave: $('#wave'), tNow: $('#tNow'), tTot: $('#tTot'), tLoop: $('#tLoop'),
  play: $('#btnPlay'), stop: $('#btnStop'), A: $('#btnA'), B: $('#btnB'), loop: $('#btnLoop'), clr: $('#btnClr'),
  stems: $('#stems'), stemBtns: $('#stemBtns'), mp3: $('#btnMp3'), wav: $('#btnWav'), expStatus: $('#expStatus'),
};

let worker, workerBusy = false;
let song = null;            // {left,right,name}
let stems = null;           // [{left,right}] ikut urutan STEMS
let states = STEMS.map(() => ({ vol: 1, mute: false, solo: false }));
let ac = null, gains = [], buffers = [], sources = [];
let playing = false, startAt = 0, offsetAt = 0, pos = 0, dur = 0;
let loopA = null, loopB = null, loopOn = false;
let peakData = null, raf = 0, wake = null;

// ---------- util UI ----------
const showErr = m => { el.err.textContent = m || ''; el.err.classList.toggle('hidden', !m); };
const setStatus = t => { el.status.textContent = t || ''; };
const setBar = p => {
  el.barWrap.classList.toggle('hidden', p == null);
  if (p != null) el.bar.style.width = Math.round(Math.max(0, Math.min(1, p)) * 100) + '%';
};
const mb = n => (n / 1048576).toFixed(0) + 'MB';

// ---------- worker ----------
function startWorker() {
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = onWorker;
  worker.onerror = e => { workerBusy = false; showErr('Ralat worker: ' + (e.message || 'tidak diketahui')); el.sep.disabled = !song; };
  worker.postMessage({ type: 'check', url: MODEL_URL });
}
let pendingResolve = null;
function onWorker(ev) {
  const m = ev.data;
  switch (m.type) {
    case 'cached':
      el.modelInfo.textContent = m.value
        ? 'Model sudah ada dalam peranti. Boleh guna tanpa internet.'
        : 'Model belum dimuat turun. Kali pertama akan memuat turun ~172MB (guna Wi-Fi), selepas itu disimpan dalam peranti.';
      break;
    case 'status': setStatus(m.text); break;
    case 'dl':
      setBar(m.total ? m.loaded / m.total : null);
      setStatus('Memuat turun model… ' + mb(m.loaded) + (m.total ? ' / ' + mb(m.total) : ''));
      break;
    case 'ready':
      el.modelInfo.textContent = 'Model sedia. Mod: ' + (m.provider === 'webgpu' ? 'WebGPU (laju)' : 'CPU (perlahan)');
      break;
    case 'progress':
      setBar(m.p);
      setStatus('Mengasingkan… ' + Math.round(m.p * 100) + '% (bahagian ' + m.seg + '/' + m.total + ')');
      break;
    case 'done': finishSeparate(m.stems); break;
    case 'error':
      workerBusy = false; setBar(null); setStatus('');
      showErr('Ralat: ' + m.message);
      el.sep.disabled = !song;
      releaseWake();
      break;
  }
}

// ---------- muat lagu ----------
async function loadFile(f) {
  if (!f) return;
  showErr('');
  setStatus('Membaca fail…');
  try {
    const arr = await f.arrayBuffer();
    const tmp = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
    const buf = await tmp.decodeAudioData(arr);
    tmp.close();
    const left = buf.getChannelData(0).slice();
    const right = (buf.numberOfChannels > 1 ? buf.getChannelData(1) : buf.getChannelData(0)).slice();
    song = { left, right, name: f.name, seconds: buf.length / buf.sampleRate };
    el.info.textContent = f.name + ' · ' + fmtTime(song.seconds) + (song.seconds > 600 ? ' (lagu panjang, memori mungkin tak cukup di telefon)' : '');
    el.sep.disabled = false;
    setStatus('');
  } catch (e) {
    song = null; el.sep.disabled = true; setStatus('');
    showErr('Tak dapat baca fail audio ini. Cuba MP3/WAV/M4A lain.');
  }
}
el.pick.onclick = () => el.file.click();
el.file.onchange = () => loadFile(el.file.files[0]);
['dragenter', 'dragover'].forEach(t => el.drop.addEventListener(t, e => { e.preventDefault(); el.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => el.drop.addEventListener(t, e => { e.preventDefault(); el.drop.classList.remove('over'); }));
el.drop.addEventListener('drop', e => loadFile(e.dataTransfer.files[0]));

el.btnModelFile.onclick = () => el.modelFile.click();
el.modelFile.onchange = () => {
  const f = el.modelFile.files[0];
  if (!f) return;
  setStatus('Memuatkan model dari fail…'); showErr('');
  worker.postMessage({ type: 'load', url: MODEL_URL, file: f, forceWasm: el.forceCpu.checked });
};

// ---------- pisah ----------
async function takeWake() { try { wake = await navigator.wakeLock?.request('screen'); } catch (e) {} }
function releaseWake() { try { wake?.release(); } catch (e) {} wake = null; }

el.sep.onclick = async () => {
  if (!song || workerBusy) return;
  stopPlayback(); showErr('');
  workerBusy = true; el.sep.disabled = true;
  setBar(0); setStatus('Bermula…');
  await takeWake();
  // salin supaya lagu asal kekal untuk ulangan
  const left = song.left.slice(), right = song.right.slice();
  worker.postMessage({ type: 'separate', url: MODEL_URL, left, right, forceWasm: el.forceCpu.checked }, [left.buffer, right.buffer]);
};

function finishSeparate(res) {
  workerBusy = false; releaseWake();
  stems = STEMS.map(s => res[s.key]);
  setBar(null); setStatus('Siap. Mixer di bawah.');
  el.sep.disabled = false;
  setupPlayer();
}

// ---------- pemain ----------
function setupPlayer() {
  if (!ac) ac = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  gains.forEach(g => g.disconnect());
  gains = []; buffers = [];
  stems.forEach(() => { const g = ac.createGain(); g.connect(ac.destination); gains.push(g); });
  buffers = stems.map(s => {
    const b = ac.createBuffer(2, s.left.length, SR);
    b.copyToChannel(s.left, 0); b.copyToChannel(s.right, 1);
    return b;
  });
  dur = stems[0].left.length / SR; pos = 0; loopA = loopB = null; loopOn = false;
  states = STEMS.map(() => ({ vol: 1, mute: false, solo: false }));
  const all = mixDown(stems, states.map(() => 1));
  peakData = peaks(all.left, all.right, 800);
  el.tTot.textContent = fmtTime(dur);
  buildMixerUI(); applyGains(); updateLoopUI();
  el.player.classList.remove('hidden'); el.exp.classList.remove('hidden');
  el.player.scrollIntoView({ behavior: 'smooth', block: 'start' });
  cancelAnimationFrame(raf); raf = requestAnimationFrame(tick);
}

function buildMixerUI() {
  el.stems.innerHTML = '';
  el.stemBtns.innerHTML = '';
  STEMS.forEach((s, i) => {
    const d = document.createElement('div');
    d.className = 'stem';
    d.innerHTML = `<span class="nm">${s.name}</span><button data-m>Bisu</button><button data-s>Solo</button>
      <input type="range" min="0" max="1.5" step="0.01" value="1" aria-label="Kelantangan ${s.name}">`;
    d.querySelector('[data-m]').onclick = e => { states[i].mute = !states[i].mute; applyGains(); };
    d.querySelector('[data-s]').onclick = e => { states[i].solo = !states[i].solo; applyGains(); };
    d.querySelector('input').oninput = e => { states[i].vol = +e.target.value; applyGains(); };
    el.stems.appendChild(d);
    const b = document.createElement('button');
    b.textContent = s.name.split(' ')[0];
    b.onclick = () => exportStem(i);
    el.stemBtns.appendChild(b);
  });
}

function applyGains() {
  const g = effectiveGains(states);
  g.forEach((v, i) => { if (gains[i]) gains[i].gain.value = v; });
  [...el.stems.children].forEach((d, i) => {
    d.classList.toggle('muted', g[i] === 0);
    d.querySelector('[data-m]').classList.toggle('on', states[i].mute);
    d.querySelector('[data-s]').classList.toggle('on', states[i].solo);
  });
}

document.querySelectorAll('[data-p]').forEach(b => b.onclick = () => {
  const p = b.dataset.p;
  states.forEach(s => { s.mute = false; s.solo = false; });
  const idx = k => STEMS.findIndex(s => s.key === k);
  if (p === 'nogtr') states[idx('other')].mute = true;
  if (p === 'novox') states[idx('vocals')].mute = true;
  if (p === 'nodrum') states[idx('drums')].mute = true;
  if (p === 'vox') states[idx('vocals')].solo = true;
  applyGains();
});

function curPos() {
  if (!playing) return pos;
  let t = offsetAt + (ac.currentTime - startAt);
  if (loopOn && loopA != null && loopB != null && loopB > loopA && t >= loopA) {
    if (t >= loopB) t = loopA + ((t - loopA) % (loopB - loopA));
  }
  return Math.min(t, dur);
}

function startSources(from) {
  stopSources();
  const useLoop = loopOn && loopA != null && loopB != null && loopB > loopA;
  if (useLoop && (from < loopA || from >= loopB)) from = loopA;
  sources = buffers.map((b, i) => {
    const s = ac.createBufferSource();
    s.buffer = b; s.connect(gains[i]);
    if (useLoop) { s.loop = true; s.loopStart = loopA; s.loopEnd = loopB; }
    s.start(0, from);
    return s;
  });
  offsetAt = from; startAt = ac.currentTime;
}
function stopSources() {
  sources.forEach(s => { try { s.stop(); } catch (e) {} try { s.disconnect(); } catch (e) {} });
  sources = [];
}

async function play() {
  if (!stems) return;
  await ac.resume();
  if (pos >= dur - 0.05) pos = 0;
  startSources(pos);
  playing = true; el.play.textContent = '⏸ Jeda';
}
function pause() {
  if (!playing) return;
  pos = curPos(); playing = false; stopSources(); el.play.textContent = '▶ Main';
}
function stopPlayback() {
  if (playing) { playing = false; stopSources(); }
  pos = 0; el.play.textContent = '▶ Main';
}
function seek(t) {
  pos = Math.max(0, Math.min(dur, t));
  if (playing) startSources(pos);
}

el.play.onclick = () => (playing ? pause() : play());
el.stop.onclick = () => stopPlayback();
el.A.onclick = () => { loopA = curPos(); if (loopB != null && loopB <= loopA) loopB = null; updateLoopUI(); };
el.B.onclick = () => { loopB = curPos(); if (loopA == null || loopB <= loopA) { loopB = null; } updateLoopUI(); };
el.loop.onclick = () => {
  if (loopA == null || loopB == null) return;
  loopOn = !loopOn; updateLoopUI();
  if (playing) { pos = curPos(); startSources(pos); }
};
el.clr.onclick = () => { loopA = loopB = null; loopOn = false; updateLoopUI(); if (playing) { pos = curPos(); startSources(pos); } };
function updateLoopUI() {
  el.loop.classList.toggle('on', loopOn);
  el.tLoop.textContent = loopA != null ? 'A ' + fmtTime(loopA) + (loopB != null ? '  →  B ' + fmtTime(loopB) : '') : '';
}

// ---------- gelombang ----------
const ctx2d = el.wave.getContext('2d');
function seekFromEvent(e) {
  const r = el.wave.getBoundingClientRect();
  seek(((e.clientX - r.left) / r.width) * dur);
}
el.wave.addEventListener('pointerdown', e => { if (stems) seekFromEvent(e); });

function draw() {
  const W = el.wave.width, H = el.wave.height;
  ctx2d.clearRect(0, 0, W, H);
  if (!peakData) return;
  if (loopA != null) {
    const x1 = (loopA / dur) * W, x2 = loopB != null ? (loopB / dur) * W : x1 + 2;
    ctx2d.fillStyle = loopOn ? 'rgba(163,230,53,.25)' : 'rgba(45,212,191,.2)';
    ctx2d.fillRect(x1, 0, Math.max(2, x2 - x1), H);
  }
  const n = peakData.length, t = curPos(), px = (t / dur) * W;
  for (let i = 0; i < n; i++) {
    const x = (i / n) * W, h = Math.max(2, peakData[i] * H * 0.95);
    ctx2d.fillStyle = x < px ? '#2dd4bf' : '#3b6068';
    ctx2d.fillRect(x, (H - h) / 2, Math.max(1, W / n - 1), h);
  }
  ctx2d.fillStyle = '#fff'; ctx2d.fillRect(px - 1, 0, 2, H);
}
function tick() {
  if (stems) {
    const t = curPos();
    if (playing && t >= dur - 0.02 && !(loopOn)) { pos = 0; playing = false; stopSources(); el.play.textContent = '▶ Main'; }
    el.tNow.textContent = fmtTime(playing ? t : pos);
    draw();
  }
  raf = requestAnimationFrame(tick);
}

// ---------- eksport ----------
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30000);
}
const baseName = () => safeName(song?.name);
const mixSuffix = () => {
  const off = STEMS.filter((s, i) => effectiveGains(states)[i] === 0).map(s => s.key);
  return off.length ? '_tanpa-' + off.join('-') : '_campuran';
};
async function exportMix(kind) {
  if (!stems) return;
  el.expStatus.textContent = 'Menyediakan…';
  try {
    const m = mixDown(stems, effectiveGains(states));
    const name = baseName() + mixSuffix();
    if (kind === 'wav') download(encodeWav(m.left, m.right, SR), name + '.wav');
    else download(await encodeMp3(m.left, m.right, SR, 192, p => { el.expStatus.textContent = 'Menukar ke MP3… ' + Math.round(p * 100) + '%'; }), name + '.mp3');
    el.expStatus.textContent = 'Siap: ' + name + '.' + kind;
  } catch (e) { el.expStatus.textContent = ''; showErr('Gagal simpan: ' + e.message); }
}
function exportStem(i) {
  const s = stems[i];
  download(encodeWav(s.left, s.right, SR), baseName() + '_' + STEMS[i].key + '.wav');
  el.expStatus.textContent = 'Siap: ' + STEMS[i].name;
}
el.mp3.onclick = () => exportMix('mp3');
el.wav.onclick = () => exportMix('wav');

// ---------- mula ----------
startWorker();
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
