import { detectChords, chordLabel, prefersFlats, segmentAt } from './chords.js';
import { effectiveGains, mixDown, encodeWav, encodeMp3, peaks, fmtTime, safeName } from './audio-utils.js';

const VERSION = 'v17';
const SR = 44100;
const NAMES = { vocals: 'Vokal', drums: 'Dram', bass: 'Bass', guitar: 'Gitar', piano: 'Piano', other: 'Lain-lain' };
const MODELS = {
  m6: { label: '6 stem', size: '285MB', url: 'https://huggingface.co/kramp/htdemucs-6s-webgpu-onnx/resolve/main/htdemucs_6s.onnx',
        keys: ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'] },
  m6f: { label: '6 stem laju', size: '115MB', url: 'https://huggingface.co/leejsool/dsmuse-models/resolve/main/htdemucs_6s.onnx',
        keys: ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'] },
  m4: { label: '4 stem', size: '172MB', url: 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx',
        keys: ['vocals', 'drums', 'bass', 'other'] },
};
let kind = 'm6';
let STEMS = MODELS[kind].keys.map(k => ({ key: k, name: NAMES[k] }));
const MODEL_URL = () => MODELS[kind].url;

const $ = s => document.querySelector(s);
const el = {
  file: $('#file'), pick: $('#btnPick'), sep: $('#btnSep'), sepHint: $('#sepHint'),
  barWrap: $('#barWrap'), bar: $('#bar'), status: $('#status'), err: $('#err'),
  modelInfo: $('#modelInfo'), modelFile: $('#modelFile'), btnModelFile: $('#btnModelFile'), forceCpu: $('#forceCpu'),
  wave: $('#wave'), tNow: $('#tNow'), tTot: $('#tTot'), tLoop: $('#tLoop'),
  play: $('#btnPlay'), back: $('#btnBack'), fwd: $('#btnFwd'), A: $('#btnA'), B: $('#btnB'), loop: $('#btnLoop'), clr: $('#btnClr'),
  stems: $('#stems'), stemBtns: $('#stemBtns'), mp3: $('#btnMp3'), wav: $('#btnWav'), expStatus: $('#expStatus'),
  songName: $('#songName'), songSub: $('#songSub'), mixCard: $('#mixCard'),
  strip: $('#strip'), keyInfo: $('#keyInfo'), trVal: $('#trVal'),
  btnChord: $('#btnChord'), cBarWrap: $('#cBarWrap'), cBar: $('#cBar'), chordStatus: $('#chordStatus'), chordList: $('#chordList'), chordSrc: $('#chordSrc'),
};

let worker, workerBusy = false;
let song = null;            // {left,right,name}
let stems = null;           // [{left,right}] trek yang dimainkan (lagu asal atau stem)
let raw = null, layout = '6'; // hasil mentah model + susunan stem
let states = [];
let ac = null, gains = [], buffers = [], sources = [];
let playing = false, startAt = 0, offsetAt = 0, pos = 0, dur = 0;
let loopA = null, loopB = null, loopOn = false;
let peakData = null, raf = 0, wake = null;

// ---------- util UI ----------
const diag = () => [VERSION, kind, 'WebGPU ' + (navigator.gpu ? 'ada' : 'tiada'), 'RAM ' + (navigator.deviceMemory || '?') + 'GB', 'CPU ' + (navigator.hardwareConcurrency || '?')].join(' · ');
const showErr = m => { el.err.textContent = m ? m + '\n[' + diag() + ']' : ''; el.err.classList.toggle('hidden', !m); };
const setStatus = t => { el.status.textContent = t || ''; };
const setBar = p => {
  el.barWrap.classList.toggle('hidden', p == null);
  if (p != null) el.bar.style.width = Math.round(Math.max(0, Math.min(1, p)) * 100) + '%';
};
const mb = n => (n / 1048576).toFixed(0) + 'MB';

// ---------- worker ----------
let tStart = 0;
let lastJob = null;       // untuk ulang dalam mod CPU
let gpuReason = '';
function sendSeparate(forceWasm) {
  const left = song.left.slice(), right = song.right.slice();
  worker.postMessage({ type: 'separate', kind, url: MODEL_URL(), left, right, forceWasm, fast: $('#fastMode').checked }, [left.buffer, right.buffer]);
}
function retryCpu(reason) {
  gpuReason = reason;
  worker.terminate();
  el.forceCpu.checked = true;
  setBar(null);
  setStatus('WebGPU gagal, mula semula dalam mod CPU (lebih perlahan)… [' + reason.slice(0, 160) + ']');
  startWorker();
  if (!lastJob) { workerBusy = false; return; }
  if (lastJob.type === 'separate') { setBar(0); sendSeparate(true); }
  else worker.postMessage({ type: 'load', kind, url: MODEL_URL(), file: lastJob.file, forceWasm: true });
}
const friendly = m => {
  let t = m;
  if (/\(wasm: \d+\)|^\d+$/.test(m)) t += '\n\nEnjin CPU berhenti tanpa mesej (kemungkinan model tak serasi atau memori tak cukup).';
  else if (/Aborted|memory|alloc/i.test(m)) t += '\n\nKemungkinan memori tak cukup atau model tak serasi dengan enjin.';
  if (gpuReason) t += '\n\nPunca WebGPU gagal: ' + gpuReason;
  return t + '\n\nCuba model 4 stem (pilihan di bahagian Model AI) untuk bandingkan.';
};
function startWorker() {
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = onWorker;
  worker.onerror = e => { workerBusy = false; showErr('Ralat worker: ' + (e.message || 'tidak diketahui')); setSepState(); };
  checkCached();
}
function checkCached() { worker.postMessage({ type: 'check', url: MODEL_URL() }); }
document.querySelectorAll('input[name=mk]').forEach(r => r.onchange = () => {
  if (workerBusy) { document.querySelector(`input[name=mk][value=${kind}]`).checked = true; return; }
  kind = r.value; el.modelInfo.textContent = 'Menyemak…'; checkCached();
});
function onWorker(ev) {
  const m = ev.data;
  switch (m.type) {
    case 'cached':
      el.modelInfo.textContent = m.value
        ? 'Model sudah ada dalam peranti. Boleh guna tanpa internet.'
        : 'Model belum dimuat turun. Kali pertama akan memuat turun ~' + MODELS[kind].size + ' (guna Wi-Fi), selepas itu disimpan dalam peranti.';
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
      {
        const el_s = (Date.now() - tStart) / 1000, left = m.p > 0 ? el_s / m.p * (1 - m.p) : 0;
        const fmt = s => (s >= 60 ? Math.floor(s / 60) + ' min ' : '') + Math.round(s % 60) + ' s';
        setStatus('Mengasingkan… ' + Math.round(m.p * 100) + '% (bahagian ' + m.seg + '/' + m.total + ') · ' + (el_s / m.seg).toFixed(1) + ' s/bahagian · baki ~' + fmt(left));
      }
      break;
    case 'needcpu': retryCpu(m.reason); break;
    case 'done': finishSeparate(m.stems); break;
    case 'error':
      workerBusy = false; setBar(null); setStatus('');
      showErr('Ralat: ' + friendly(m.message));
      setSepState();
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
    raw = null; chords = null; tr = 0; layout = '6';
    el.songName.textContent = f.name.replace(/\.[^.]+$/, '');
    el.songSub.textContent = fmtTime(song.seconds) + (song.seconds > 600 ? ' · lagu panjang, memori telefon mungkin tak cukup' : '');
    STEMS = [{ key: 'mix', name: 'Lagu asal' }];
    stems = [{ left, right }];
    setupPlayer(false);
    renderChords(); renderChordList();
    el.btnChord.disabled = false; el.chordStatus.textContent = 'Tekan “Kesan chord”.'; el.chordSrc.textContent = '';
    setSepState();
    setStatus('');
  } catch (e) {
    setStatus('');
    showErr('Tak dapat baca fail audio ini. Cuba MP3/WAV/M4A lain.');
  }
}
el.pick.onclick = () => el.file.click();
el.file.onchange = () => { stopPlayback(); loadFile(el.file.files[0]); };
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => { e.preventDefault(); if (e.dataTransfer.files[0]) { stopPlayback(); loadFile(e.dataTransfer.files[0]); } });

el.btnModelFile.onclick = () => el.modelFile.click();
el.modelFile.onchange = () => {
  const f = el.modelFile.files[0];
  if (!f) return;
  setStatus('Memuatkan model dari fail…'); showErr('');
  lastJob = { type: 'load', file: f };
  worker.postMessage({ type: 'load', kind, url: MODEL_URL(), file: f, forceWasm: el.forceCpu.checked });
};

// ---------- pisah ----------
async function takeWake() { try { wake = await navigator.wakeLock?.request('screen'); } catch (e) {} }
function releaseWake() { try { wake?.release(); } catch (e) {} wake = null; }

function setSepState() {
  el.sep.disabled = !song || workerBusy;
  el.sepHint.textContent = !song ? 'Pilih lagu dahulu.' : raw ? 'Sudah diasingkan. Tekan lagi untuk ulang.' : 'Lagu sedia. ' + MODELS[kind].label + ' akan digunakan.';
  el.mp3.disabled = el.wav.disabled = !stems;
  $('#saveHint').classList.toggle('hidden', !!raw || !stems);
}

el.sep.onclick = async () => {
  if (!song || workerBusy) return;
  const keepPos = curPos();
  stopPlayback(); showErr('');
  workerBusy = true; setSepState();
  setBar(0); setStatus('Bermula…');
  await takeWake();
  lastJob = { type: 'separate' }; gpuReason = ''; tStart = Date.now();
  sendSeparate(el.forceCpu.checked);
  pos = keepPos;
};

function finishSeparate(res) {
  workerBusy = false; releaseWake();
  raw = res; layout = '6';
  const has6 = MODELS[kind].keys.includes('guitar');
  $('#layoutRow').classList.toggle('hidden', !has6);
  applyLayout();
  setBar(null); setStatus('Siap. Gunakan mixer di bawah.');
  // semakan kualiti: jumlah semua stem sepatutnya hampir sama dengan lagu asal
  try {
    const n = song.left.length; let sig = 0, err = 0;
    for (let i = 0; i < n; i += 7) {
      let sl = 0, sr = 0;
      for (const k of MODELS[kind].keys) { sl += raw[k].left[i]; sr += raw[k].right[i]; }
      sig += song.left[i] ** 2 + song.right[i] ** 2;
      err += (song.left[i] - sl) ** 2 + (song.right[i] - sr) ** 2;
    }
    const snr = 10 * Math.log10((sig + 1e-12) / (err + 1e-12));
    el.sepHint.dataset.snr = snr.toFixed(1);
    if (snr < 10) {
      showErr('Amaran: hasil pengasingan tak sepadan dengan lagu asal (padanan ' + snr.toFixed(1) + ' dB; biasanya >15 dB). Bunyi mungkin rosak/berdesir dengan model ini. Cuba model "6 stem (stabil)" di tab Tetapan, atau tick Paksa mod CPU.');
    } else setStatus('Siap. Padanan stem ' + snr.toFixed(1) + ' dB (baik). Gunakan mixer di bawah.');
  } catch (e) {}
  setupPlayer(false, pos);
  setSepState();
  if (chords) runChords(true);
}

// Susunan 5 stem: Gitar 1 = saluran kiri, Gitar 2 = saluran kanan (hanya bermakna jika gitar di-pan). Jumlah semua stem = asal.
function applyLayout() {
  const has6 = MODELS[kind].keys.includes('guitar');
  if (has6 && layout === '5') {
    const n = raw.vocals.left.length, zero = new Float32Array(n);
    const mixL = new Float32Array(n), mixR = new Float32Array(n);
    for (const k of ['bass', 'piano', 'other']) {
      const s = raw[k];
      for (let i = 0; i < n; i++) { mixL[i] += s.left[i]; mixR[i] += s.right[i]; }
    }
    STEMS = [
      { key: 'vocals', name: 'Vokal' }, { key: 'drums', name: 'Dram' },
      { key: 'gitar1', name: 'Gitar 1 (kiri)' }, { key: 'gitar2', name: 'Gitar 2 (kanan)' },
      { key: 'other', name: 'Lain-lain (bass, piano…)' },
    ];
    stems = [raw.vocals, raw.drums, { left: raw.guitar.left, right: zero }, { left: zero, right: raw.guitar.right }, { left: mixL, right: mixR }];
  } else {
    STEMS = MODELS[kind].keys.map(k => ({ key: k, name: NAMES[k] }));
    stems = STEMS.map(s => raw[s.key]);
  }
  document.querySelectorAll('[data-lay]').forEach(b => b.classList.toggle('on', b.dataset.lay === layout));
  $('#stemNote').classList.toggle('hidden', has6);
  $('#layNote').classList.toggle('hidden', !(has6 && layout === '5'));
}
document.querySelectorAll('[data-lay]').forEach(b => b.onclick = () => {
  if (!raw || layout === b.dataset.lay) return;
  const p = curPos(), was = playing; stopPlayback(); layout = b.dataset.lay; applyLayout(); setupPlayer(false, p);
  if (was) play();
});

// ---------- pemain ----------
const ICONS = { vocals: '🎤', drums: '🥁', bass: '🎸', guitar: '🎸', gitar1: '🎸', gitar2: '🎸', piano: '🎹', other: '🎶', mix: '🎵' };
function setupPlayer(scroll, keepPos = 0) {
  if (!ac) ac = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  gains.forEach(g => g.disconnect());
  gains = []; buffers = [];
  stems.forEach(() => { const g = ac.createGain(); g.connect(ac.destination); gains.push(g); });
  buffers = stems.map(s => {
    const b = ac.createBuffer(2, s.left.length, SR);
    b.copyToChannel(s.left, 0); b.copyToChannel(s.right, 1);
    return b;
  });
  const first = !dur || Math.abs(dur - stems[0].left.length / SR) > 0.5;
  dur = stems[0].left.length / SR; pos = Math.min(keepPos, dur);
  if (first) { loopA = loopB = null; loopOn = false; }
  states = STEMS.map(() => ({ vol: 1, mute: false, solo: false }));
  const all = mixDown(stems, states.map(() => 1));
  peakData = peaks(all.left, all.right, 800);
  el.tTot.textContent = fmtTime(dur);
  buildMixerUI(); applyGains(); updateLoopUI();
  el.mixCard.classList.remove('hidden');
  cancelAnimationFrame(raf); raf = requestAnimationFrame(tick);
  setSepState();
}

function buildMixerUI() {
  el.stems.innerHTML = '';
  el.stemBtns.innerHTML = '';
  STEMS.forEach((s, i) => {
    const d = document.createElement('div');
    d.className = 'stem';
    d.innerHTML = `<button class="ic" data-m aria-label="Bisu ${s.name}">${ICONS[s.key] || '🎶'}</button>
      <div class="mid"><div class="nm">${s.name}</div><input type="range" min="0" max="1.5" step="0.01" value="1" aria-label="Kelantangan ${s.name}"></div>
      <button class="s" data-s>S</button>
      <button class="dl" data-d aria-label="Simpan ${s.name}">⬇</button>`;
    d.querySelector('[data-m]').onclick = () => { states[i].mute = !states[i].mute; applyGains(); };
    d.querySelector('[data-s]').onclick = () => { states[i].solo = !states[i].solo; applyGains(); };
    d.querySelector('[data-d]').onclick = () => exportStem(i);
    d.querySelector('input').oninput = e => { states[i].vol = +e.target.value; applyGains(); };
    el.stems.appendChild(d);
    if (raw) {
      const b = document.createElement('button');
      b.textContent = s.name.replace(/\s*\(.*$/, '');
      b.onclick = () => exportStem(i);
      el.stemBtns.appendChild(b);
    }
  });
}

function applyGains() {
  const g = effectiveGains(states);
  g.forEach((v, i) => { if (gains[i]) gains[i].gain.value = v; });
  [...el.stems.children].forEach((d, i) => {
    d.querySelector('[data-m]').classList.toggle('off', g[i] === 0);
    d.querySelector('[data-s]').classList.toggle('on', states[i].solo);
  });
}

document.querySelectorAll('[data-p]').forEach(b => b.onclick = () => {
  if (!raw) return;
  const p = b.dataset.p;
  states.forEach(s => { s.mute = false; s.solo = false; });
  const idx = k => STEMS.findIndex(s => s.key === k);
  if (p === 'nogtr') {
    if (idx('gitar1') >= 0) { states[idx('gitar1')].mute = true; states[idx('gitar2')].mute = true; }
    else states[idx(idx('guitar') >= 0 ? 'guitar' : 'other')].mute = true;
  }
  if (p === 'novox') states[idx('vocals')].mute = true;
  if (p === 'nodrum') states[idx('drums')].mute = true;
  if (p === 'vox') states[idx('vocals')].solo = true;
  applyGains();
  document.querySelectorAll('.stem input').forEach((r, i) => r.value = states[i].vol);
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
  playing = true; el.play.textContent = '⏸';
}
function pause() {
  if (!playing) return;
  pos = curPos(); playing = false; stopSources(); el.play.textContent = '▶';
}
function stopPlayback() {
  if (playing) { pos = curPos(); playing = false; stopSources(); }
  el.play.textContent = '▶';
}
function seek(t) {
  pos = Math.max(0, Math.min(dur, t));
  if (playing) startSources(pos);
}

el.play.onclick = () => (playing ? pause() : play());
el.back.onclick = () => stems && seek(curPos() - 10);
el.fwd.onclick = () => stems && seek(curPos() + 10);
el.A.onclick = () => { if (!stems) return; loopA = curPos(); if (loopB != null && loopB <= loopA) loopB = null; updateLoopUI(); };
el.B.onclick = () => { if (!stems) return; loopB = curPos(); if (loopA == null || loopB <= loopA) { loopB = null; } updateLoopUI(); };
el.loop.onclick = () => {
  if (loopA == null || loopB == null) return;
  loopOn = !loopOn; updateLoopUI();
  if (playing) { pos = curPos(); startSources(pos); }
};
el.clr.onclick = () => { loopA = loopB = null; loopOn = false; updateLoopUI(); if (playing) { pos = curPos(); startSources(pos); } };
function updateLoopUI() {
  el.loop.classList.toggle('on', loopOn);
  el.A.classList.toggle('on', loopA != null);
  el.B.classList.toggle('on', loopB != null);
  el.tLoop.textContent = loopA != null ? 'A ' + fmtTime(loopA) + (loopB != null ? '  →  B ' + fmtTime(loopB) : '') : '';
}

// ---------- gelombang ----------
const ctx2d = el.wave.getContext('2d');
el.wave.addEventListener('pointerdown', e => {
  if (!stems) return;
  const r = el.wave.getBoundingClientRect();
  seek(((e.clientX - r.left) / r.width) * dur);
});
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
let lastSeg = -2;
function tick() {
  if (stems) {
    const t = curPos();
    if (playing && t >= dur - 0.02 && !loopOn) { pos = 0; playing = false; stopSources(); el.play.textContent = '▶'; }
    el.tNow.textContent = fmtTime(playing ? t : pos);
    draw();
    if (chords?.segments.length) {
      const si = segmentAt(chords.segments, playing ? t : pos);
      if (si !== lastSeg) { lastSeg = si; highlightChord(si); }
    }
  }
  raf = requestAnimationFrame(tick);
}

// ---------- chord ----------
let chords = null, tr = 0;
const flatsNow = () => chords ? prefersFlats(((chords.key.tonic + tr) % 12 + 12) % 12, chords.key.mode) : false;
const segLabel = s => chordLabel(s, tr, flatsNow());
function renderChords() {
  lastSeg = -2;
  el.trVal.textContent = tr === 0 ? '±0' : (tr > 0 ? '+' : '') + tr;
  if (!chords || !chords.segments.length) {
    el.strip.innerHTML = '<div class="empty">' + (chords ? 'Tiada chord jelas dikesan.' : 'Pilih lagu, kemudian tab <b>Chord</b> → “Kesan chord”.') + '</div>';
    el.keyInfo.textContent = 'Chord · belum dikesan';
    return;
  }
  const kn = chordLabel({ root: (chords.key.tonic + tr) % 12, type: chords.key.mode === 'min' ? 1 : 0 }, 0, flatsNow());
  el.keyInfo.innerHTML = 'Kunci dianggar: <b>' + kn.replace(/m$/, ' minor').replace(/^([A-G][b#]?)$/, '$1 major') + '</b>';
  el.strip.innerHTML = '';
  chords.segments.forEach((s, i) => {
    const d = document.createElement('div');
    d.className = 'cc' + (s.type < 0 ? ' nc' : '');
    d.innerHTML = segLabel(s) + '<small>' + fmtTime(s.start) + '</small>';
    d.onclick = () => seek(s.start);
    el.strip.appendChild(d);
  });
}
function renderChordList() {
  el.chordList.innerHTML = '';
  if (!chords) return;
  chords.segments.forEach((s, i) => {
    const b = document.createElement('button');
    b.innerHTML = segLabel(s) + '<small>' + fmtTime(s.start) + '</small>';
    b.onclick = () => seek(s.start);
    el.chordList.appendChild(b);
  });
}
function highlightChord(si) {
  const cells = el.strip.children;
  for (let i = 0; i < cells.length; i++) cells[i].classList.toggle('cur', i === si);
  const c = cells[si];
  if (c && c.classList) el.strip.scrollTo({ left: c.offsetLeft - el.strip.clientWidth / 2 + c.offsetWidth / 2 - el.strip.offsetLeft, behavior: 'smooth' });
  [...el.chordList.children].forEach((b, i) => b.classList.toggle('cur', i === si));
}
$('#trUp').onclick = () => { tr = (tr + 1 + 12) % 12; if (tr > 6) tr -= 12; renderChords(); renderChordList(); };
$('#trDn').onclick = () => { tr = tr - 1; if (tr < -6) tr += 12; renderChords(); renderChordList(); };

let chordBusy = false;
async function runChords(auto) {
  if (!song || chordBusy) return;
  chordBusy = true; el.btnChord.disabled = true;
  el.btnChord.textContent = 'Mengesan… 0%';
  let L, R, src;
  if (raw) {
    const ks = MODELS[kind].keys.filter(k => k !== 'vocals' && k !== 'drums');
    L = new Float32Array(raw.vocals.left.length); R = new Float32Array(L.length);
    for (const k of ks) { const s = raw[k]; for (let i = 0; i < L.length; i++) { L[i] += s.left[i]; R[i] += s.right[i]; } }
    src = 'sumber: stem harmonik (tanpa dram & vokal)';
  } else { L = song.left; R = song.right; src = 'sumber: lagu asal'; }
  el.chordStatus.textContent = 'Mengesan chord…'; el.chordSrc.textContent = src;
  try {
    const r = await detectChords(L, R, SR, { onProgress: p => { el.btnChord.textContent = 'Mengesan… ' + Math.round(p * 100) + '%'; } });
    chords = r; tr = 0;
    renderChords(); renderChordList();
    el.chordStatus.textContent = r.segments.length + ' perubahan chord dikesan. Ketepatan sederhana: semak dengan telinga, terutama chord 7th. Gunakan − / + untuk transpose.';
  } catch (e) { el.chordStatus.textContent = 'Gagal kesan chord: ' + e.message; }
  chordBusy = false; el.btnChord.disabled = !song;
  el.btnChord.textContent = 'Kesan chord';
}
el.btnChord.onclick = () => runChords(false);

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

// ---------- tab ----------
document.querySelectorAll('.tabs [data-tab]').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs [data-tab]').forEach(x => x.classList.toggle('on', x === b));
  ['mix', 'chord', 'save', 'set'].forEach(t => $('#p-' + t).classList.toggle('hidden', t !== b.dataset.tab));
});

// ---------- mula ----------
$('#ver').textContent = VERSION;
startWorker();
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
