# Pemisah Lagu

Asingkan vokal, dram, bass dan lain-lain (gitar/piano) daripada lagu sendiri. Semua proses dalam pelayar/peranti, tiada server.

- Enjin: Demucs v4 (htdemucs, MIT) melalui ONNX Runtime Web (WebGPU, jatuh balik ke CPU/WASM)
- Model (~172MB) dimuat turun sekali dari Hugging Face, kemudian disimpan dalam peranti
- Ciri: mixer (bisu/solo/kelantangan), ulang A–B, simpan campuran MP3/WAV, simpan stem WAV
- Boleh dipasang di skrin utama telefon (PWA)

Pasang di GitHub Pages: Settings → Pages → Deploy from branch → main / (root).
