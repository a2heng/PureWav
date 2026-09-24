/* PureWav Web — 纯前端、离线、单文件降噪
 * WebAudio 解码 → JS STFT(Bluestein, 960/480/hann) → onnxruntime-web 推理 → ISTFT → WAV 导出
 */
'use strict';
(function () {
  const $ = (s) => document.querySelector(s);
  const FS = 48000, NFFT = 960, HOP = 480, BINS = NFFT / 2 + 1; // 481
  const BAND_LO = 20000, BAND_HI = 24000, BAND_W = 1000;
  const VMIN = -80, VMAX = 0;
  const ROWS_LIN = BAND_LO / (FS / NFFT);       // 400
  const ROWS_BAND = (BAND_HI - BAND_LO) / BAND_W; // 4
  const ROWS = ROWS_LIN + ROWS_BAND;            // 404
  const CHUNK_SEC = 15;                          // 每段推理时长
  const MAX_COLS = 2000;                         // 频谱显示最大列数

  let session = null, modelBytes = null, cmap = null;
  let srcBuf = null, denBuf = null, srcName = 'audio';
  let actx = null, playSrc = null, playWhich = null, playStart = 0, playOff = 0;

  // ── 工具 ──────────────────────────────────────────────
  function setStatus(t) { $('#status').textContent = t; }
  function setProgress(p) { $('#bar').style.width = Math.max(0, Math.min(1, p)) * 100 + '%'; }
  function b64ToBytes(id) {
    const s = document.getElementById(id).textContent.trim();
    const bin = atob(s), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  function fmtTime(t) {
    t = Math.max(0, t || 0);
    const m = Math.floor(t / 60), s = t - m * 60;
    return m + ':' + s.toFixed(1).padStart(4, '0');
  }
  function fmtSize(b) {
    return b >= 1048576 ? (b / 1048576).toFixed(1) + 'MB'
      : Math.max(1, Math.round(b / 1024)) + 'KB';
  }

  // ── 基2 FFT (原位, 长度 2 的幂) ────────────────────────
  function fftPow2(re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (inverse ? 2 : -2) * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cwr = 1, cwi = 0;
        for (let j = 0; j < len / 2; j++) {
          const ur = re[i + j], ui = im[i + j];
          const vr = re[i + j + len / 2] * cwr - im[i + j + len / 2] * cwi;
          const vi = re[i + j + len / 2] * cwi + im[i + j + len / 2] * cwr;
          re[i + j] = ur + vr; im[i + j] = ui + vi;
          re[i + j + len / 2] = ur - vr; im[i + j + len / 2] = ui - vi;
          const t = cwr; cwr = cwr * wr - cwi * wi; cwi = t * wi + cwi * wr;
        }
      }
    }
    if (inverse) { for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; } }
  }

  // ── Bluestein: 任意长度 FFT (960 = 2^6·3·5, 非 2 的幂) ──
  function Bluestein(n) {
    this.n = n;
    let m = 1; while (m < 2 * n - 1) m <<= 1;
    this.m = m;
    const wre = new Float64Array(n), wim = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const k2 = (k * k) % (2 * n), ang = -Math.PI * k2 / n;
      wre[k] = Math.cos(ang); wim[k] = Math.sin(ang);
    }
    this.wre = wre; this.wim = wim;
    const bre = new Float64Array(m), bim = new Float64Array(m);
    for (let j = 0; j < n; j++) { bre[j] = wre[j]; bim[j] = -wim[j]; }
    for (let j = 1; j < n; j++) { bre[m - j] = wre[j]; bim[m - j] = -wim[j]; }
    fftPow2(bre, bim, false);
    this.bre = bre; this.bim = bim;
    this.are = new Float64Array(m); this.aim = new Float64Array(m);
  }
  Bluestein.prototype.transform = function (reIn, imIn, reOut, imOut) {
    const n = this.n, m = this.m, are = this.are, aim = this.aim;
    are.fill(0); aim.fill(0);
    for (let j = 0; j < n; j++) {
      const xr = reIn[j], xi = imIn ? imIn[j] : 0;
      are[j] = xr * this.wre[j] - xi * this.wim[j];
      aim[j] = xr * this.wim[j] + xi * this.wre[j];
    }
    fftPow2(are, aim, false);
    const bre = this.bre, bim = this.bim;
    for (let i = 0; i < m; i++) {
      const ar = are[i], ai = aim[i], br = bre[i], bi = bim[i];
      are[i] = ar * br - ai * bi; aim[i] = ar * bi + ai * br;
    }
    fftPow2(are, aim, true);
    for (let k = 0; k < n; k++) {
      const cr = are[k], ci = aim[k];
      reOut[k] = cr * this.wre[k] - ci * this.wim[k];
      imOut[k] = cr * this.wim[k] + ci * this.wre[k];
    }
  };

  function hann(n) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
    return w;
  }

  // ── 解码 + 重采样为 48k 单声道 ─────────────────────────
  async function decodeToMono48k(file) {
    const arr = await file.arrayBuffer();
    const AC = window.AudioContext || window.webkitAudioContext;
    const tmp = new AC();
    let decoded;
    try { decoded = await tmp.decodeAudioData(arr.slice(0)); }
    finally { tmp.close && tmp.close(); }
    const dur = decoded.duration;
    const outLen = Math.max(1, Math.ceil(dur * FS));
    const off = new OfflineAudioContext(1, outLen, FS);
    const s = off.createBufferSource(); s.buffer = decoded; s.connect(off.destination);
    s.start(0);
    const rendered = await off.startRendering();
    return rendered.getChannelData(0).slice(); // 拷贝为独立 Float32Array
  }

  // ── 单段: STFT → 推理 → ISTFT + 显示谱 ─────────────────
  async function denoise(x) {
    const win = hann(NFFT);
    const bs = new Bluestein(NFFT);
    const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
    const fRe = new Float64Array(BINS), fIm = new Float64Array(BINS);

    const totalFrames = Math.max(1, 1 + Math.floor((x.length - NFFT) / HOP));
    const dispCols = Math.min(MAX_COLS, totalFrames);
    const pool = Math.max(1, Math.ceil(totalFrames / dispCols));
    const dispO = new Float32Array(ROWS * dispCols);
    const dispD = new Float32Array(ROWS * dispCols);
    const cnt = new Float32Array(dispCols);

    const out = new Float32Array(x.length);
    const winSum = new Float32Array(x.length);
    const chunkSamples = CHUNK_SEC * FS;
    const invN = 1 / NFFT;

    // 混合轴行映射: 400 线性 bin + 4 个 1kHz 带(带内均值)
    const bandGroups = [];
    {
      const lo = ROWS_LIN;
      const nBins = BINS - lo; // 81
      for (let b = 0; b < ROWS_BAND; b++) {
        const a = lo + Math.floor(b * nBins / ROWS_BAND);
        const e = lo + Math.floor((b + 1) * nBins / ROWS_BAND);
        bandGroups.push([a, Math.max(a + 1, e)]);
      }
    }
    const rowBuf = new Float32Array(ROWS);
    function hybridRow(dbmag) {
      for (let r = 0; r < ROWS_LIN; r++) rowBuf[r] = dbmag[r];
      for (let b = 0; b < ROWS_BAND; b++) {
        const [a, e] = bandGroups[b]; let s = 0;
        for (let i = a; i < e; i++) s += dbmag[i];
        rowBuf[ROWS_LIN + b] = s / (e - a);
      }
      return rowBuf;
    }

    let gFrame = 0;
    function accumulate(dbmag, arr) {
      const c = Math.floor(gFrame / pool);
      if (c < dispCols) {
        const base = c * ROWS, row = hybridRow(dbmag);
        for (let r = 0; r < ROWS; r++) arr[base + r] += row[r];
      }
      if (arr === dispD) { cnt[Math.min(dispCols - 1, c)]++; gFrame++; }
    }

    for (let start = 0; start < x.length; start += chunkSamples) {
      const seg = x.subarray(start, Math.min(x.length, start + chunkSamples));
      const nFrames = Math.max(1, Math.floor((seg.length - NFFT) / HOP) + 1);
      if (seg.length < NFFT) { out.set(seg, start); continue; }

      // STFT
      const specRe = new Float32Array(nFrames * BINS);
      const specIm = new Float32Array(nFrames * BINS);
      for (let t = 0; t < nFrames; t++) {
        const o = t * HOP;
        for (let i = 0; i < NFFT; i++) re[i] = seg[o + i] * win[i];
        bs.transform(re, null, fRe, fIm);
        const base = t * BINS;
        for (let k = 0; k < BINS; k++) { specRe[base + k] = fRe[k]; specIm[base + k] = fIm[k]; }
        // 显示谱 (原图)
        const db = new Float32Array(BINS);
        for (let k = 0; k < BINS; k++) db[k] = 20 * Math.log10(Math.hypot(fRe[k], fIm[k]) + 1e-12);
        accumulate(db, dispO);
      }

      // 推理
      const data = new Float32Array(1 * 2 * nFrames * BINS);
      for (let t = 0; t < nFrames; t++) {
        const b = t * BINS;
        for (let k = 0; k < BINS; k++) {
          data[b + k] = specRe[b + k];                       // ch0 real
          data[nFrames * BINS + b + k] = specIm[b + k];      // ch1 imag
        }
      }
      const tensor = new ort.Tensor('float32', data, [1, 2, nFrames, BINS]);
      const res = await session.run({ spec: tensor });
      const oT = res[session.outputNames[0]];
      const od = oT.data; // (1,2,T,481)
      const plane = nFrames * BINS;

      // ISTFT (overlap-add, hann^2 归一)
      const segLen = (nFrames - 1) * HOP + NFFT;
      const acc = new Float32Array(segLen), accW = new Float32Array(segLen);
      const fullRe = new Float64Array(NFFT), fullIm = new Float64Array(NFFT);
      const ir = new Float64Array(NFFT), ii = new Float64Array(NFFT);
      for (let t = 0; t < nFrames; t++) {
        const b = t * BINS;
        // 从 481 bin 重建共轭对称的 960 点谱
        for (let k = 0; k < BINS; k++) { fullRe[k] = od[b + k]; fullIm[k] = od[plane + b + k]; }
        for (let k = BINS; k < NFFT; k++) { fullRe[k] = fullRe[NFFT - k]; fullIm[k] = -fullIm[NFFT - k]; }
        // irfft = real( conj(fft(conj(X)))/N )
        for (let k = 0; k < NFFT; k++) { ir[k] = fullRe[k]; ii[k] = -fullIm[k]; }
        bs.transform(ir, ii, fullRe, fullIm);
        const o = t * HOP;
        for (let i = 0; i < NFFT; i++) {
          const v = fullRe[i] * invN; // real part
          acc[o + i] += v * win[i];
          accW[o + i] += win[i] * win[i];
        }
        // 显示谱 (降噪后)
        const db = new Float32Array(BINS);
        for (let k = 0; k < BINS; k++) db[k] = 20 * Math.log10(Math.hypot(od[b + k], od[plane + b + k]) + 1e-12);
        accumulate(db, dispD);
      }
      const m = Math.min(seg.length, segLen);
      for (let i = 0; i < m; i++) {
        const w = accW[i] > 1e-8 ? accW[i] : 1e-8;
        out[start + i] = Math.max(-1, Math.min(1, acc[i] / w));
      }
      for (let i = m; i < seg.length; i++) out[start + i] = 0;

      setProgress(Math.min(1, (start + seg.length) / x.length));
      await new Promise((r) => setTimeout(r, 0));
    }

    // 池化平均
    for (let c = 0; c < dispCols; c++) {
      const n = cnt[c] || 1, base = c * ROWS;
      for (let r = 0; r < ROWS; r++) { dispO[base + r] /= n; dispD[base + r] /= n; }
    }
    return { out, dispO, dispD, dispCols };
  }

  // ── 频谱绘制 ──────────────────────────────────────────
  function specCanvas(disp, cols) {
    const cv = document.createElement('canvas');
    cv.width = cols; cv.height = ROWS;
    const g = cv.getContext('2d'), img = g.createImageData(cols, ROWS);
    const rng = VMAX - VMIN;
    for (let x = 0; x < cols; x++) {
      for (let r = 0; r < ROWS; r++) {
        let v = (disp[x * ROWS + r] - VMIN) / rng; v = v < 0 ? 0 : v > 1 ? 1 : v;
        const idx = (Math.round(v * 255)) * 3;
        const o = ((ROWS - 1 - r) * cols + x) * 4;
        img.data[o] = cmap[idx]; img.data[o + 1] = cmap[idx + 1]; img.data[o + 2] = cmap[idx + 2];
        img.data[o + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    return cv;
  }
  function drawSpec(canvas, spec) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    const g = canvas.getContext('2d');
    g.imageSmoothingEnabled = false;
    g.clearRect(0, 0, canvas.width, canvas.height);
    if (spec) g.drawImage(spec, 0, 0, spec.width, spec.height, 0, 0, canvas.width, canvas.height);
    // 轴标注 (小字, 图内)
    g.font = (11 * dpr) + 'px sans-serif'; g.textBaseline = 'top';
    g.fillStyle = 'rgba(235,235,235,.85)';
    g.fillText('24k', 5 * dpr, 3 * dpr);
    g.fillText('20k', 5 * dpr, canvas.height * (ROWS_LIN / ROWS) - 13 * dpr);
    g.fillText('0', 5 * dpr, canvas.height - 14 * dpr);
    g.fillText(VMIN + 'dB', canvas.width - 44 * dpr, canvas.height - 14 * dpr);
    g.fillText(VMAX + 'dB', canvas.width - 44 * dpr, 3 * dpr);
    g.strokeStyle = 'rgba(255,255,255,.35)';
    const y = canvas.height * (ROWS_LIN / ROWS);
    g.beginPath(); g.moveTo(0, y); g.lineTo(canvas.width, y); g.stroke();
  }
  function drawWave(canvas, data, color) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    const g = canvas.getContext('2d');
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.fillStyle = '#0d0d0d'; g.fillRect(0, 0, canvas.width, canvas.height);
    g.strokeStyle = '#2a2a2a'; g.beginPath();
    g.moveTo(0, canvas.height / 2); g.lineTo(canvas.width, canvas.height / 2); g.stroke();
    if (!data) return;
    const cols = Math.min(canvas.width, 3000), step = data.length / cols;
    g.strokeStyle = color; g.beginPath();
    for (let i = 0; i < cols; i++) {
      const a = Math.floor(i * step), b = Math.max(a + 1, Math.floor((i + 1) * step));
      let mn = 1e9, mx = -1e9;
      for (let k = a; k < b; k++) { const v = data[k]; if (v < mn) mn = v; if (v > mx) mx = v; }
      const x = i / (cols - 1) * canvas.width;
      const y1 = (1 - Math.max(-1, Math.min(1, mx))) / 2 * canvas.height;
      const y2 = (1 - Math.max(-1, Math.min(1, mn))) / 2 * canvas.height;
      g.moveTo(x, y1); g.lineTo(x, y2);
    }
    g.stroke();
    g.font = (11 * dpr) + 'px sans-serif'; g.textBaseline = 'top';
    g.fillStyle = 'rgba(235,235,235,.85)';
    g.fillText('+1', 5 * dpr, 3 * dpr); g.fillText('-1', 5 * dpr, canvas.height - 14 * dpr);
  }

  // ── WAV 导出 (16-bit PCM) ─────────────────────────────
  function encodeWav(float32, sr) {
    const n = float32.length, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
    const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
    ws(0, 'RIFF'); dv.setUint32(4, 36 + n * 2, true); ws(8, 'WAVE');
    ws(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
    dv.setUint16(22, 1, true); dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true);
    dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
    ws(36, 'data'); dv.setUint32(40, n * 2, true);
    let o = 44;
    for (let i = 0; i < n; i++) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      dv.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true); o += 2;
    }
    return new Blob([buf], { type: 'audio/wav' });
  }

  // ── 播放 ──────────────────────────────────────────────
  function stopPlay() {
    if (playSrc) { try { playSrc.onended = null; playSrc.stop(); } catch (e) { } playSrc = null; }
    playWhich = null; updatePlayBtns();
  }
  function updatePlayBtns() {
    $('#playOrig').textContent = playWhich === 'orig' ? '⏸ 原图' : '▶ 原图';
    $('#playDen').textContent = playWhich === 'den' ? '⏸ 降噪后' : '▶ 降噪后';
  }
  function play(which) {
    if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === 'suspended') actx.resume();
    const data = which === 'orig' ? srcBuf : denBuf;
    if (!data) return;
    if (playWhich === which) { stopPlay(); return; }
    stopPlay();
    const ab = actx.createBuffer(1, data.length, FS);
    ab.copyToChannel(data, 0);
    playSrc = actx.createBufferSource(); playSrc.buffer = ab; playSrc.connect(actx.destination);
    playSrc.onended = () => { playSrc = null; playWhich = null; updatePlayBtns(); };
    playSrc.start(0); playWhich = which; updatePlayBtns();
  }

  // ── 主流程 ────────────────────────────────────────────
  async function loadModel() {
    setStatus('加载模型…');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { mjs: URL.createObjectURL(new Blob([b64ToBytes('glue')], { type: 'text/javascript' })) };
    ort.env.wasm.wasmBinary = b64ToBytes('wasm');
    modelBytes = b64ToBytes('model');
    session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'] });
    setStatus('模型就绪');
  }

  async function handleFile(file) {
    if (!file) return;
    stopPlay(); srcBuf = denBuf = null;
    srcName = file.name.replace(/\.[^.]+$/, '') || 'audio';
    $('#fname').textContent = file.name;
    $('#fmeta').textContent = fmtSize(file.size);
    $('#playOrig').disabled = $('#playDen').disabled = $('#dl').disabled = true;
    setProgress(0);
    try {
      if (!session) await loadModel();
      setStatus('解码中…');
      const x = await decodeToMono48k(file);
      setStatus('降噪中…'); setProgress(0);
      await new Promise((r) => setTimeout(r, 0));
      const t0 = performance.now();
      const { out, dispO, dispD, dispCols } = await denoise(x);
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      srcBuf = x; denBuf = out;
      setStatus(`完成 · ${fmtTime(x.length / FS)} · 用时 ${dt}s`);
      setProgress(1);
      drawWave($('#waveOrig'), x, '#e06c75');
      drawWave($('#waveDen'), out, '#4ec9b0');
      drawSpec($('#specOrig'), specCanvas(dispO, dispCols));
      drawSpec($('#specDen'), specCanvas(dispD, dispCols));
      $('#playOrig').disabled = $('#playDen').disabled = $('#dl').disabled = false;
      updatePlayBtns();
    } catch (e) {
      console.error(e);
      setStatus('失败: ' + (e && e.message || e));
    }
  }

  // ── 事件绑定 ──────────────────────────────────────────
  function init() {
    const b = b64ToBytes('inferno');
    cmap = new Uint8Array(256 * 3);
    cmap.set(b.subarray(0, 256 * 3));

    const drop = $('#drop');
    ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
      e.preventDefault(); drop.classList.add('over');
    }));
    ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
      e.preventDefault(); drop.classList.remove('over');
    }));
    drop.addEventListener('drop', (e) => {
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) handleFile(f);
    });
    drop.addEventListener('click', () => $('#file').click());
    $('#file').addEventListener('change', (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); });
    $('#playOrig').addEventListener('click', () => play('orig'));
    $('#playDen').addEventListener('click', () => play('den'));
    $('#dl').addEventListener('click', () => {
      if (!denBuf) return;
      const url = URL.createObjectURL(encodeWav(denBuf, FS));
      const a = document.createElement('a');
      a.href = url; a.download = srcName + '_降噪.wav'; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
    });
    window.addEventListener('resize', () => {
      if (srcBuf) {
        drawWave($('#waveOrig'), srcBuf, '#e06c75');
        drawWave($('#waveDen'), denBuf, '#4ec9b0');
      }
    });
    setStatus('就绪 · 拖入一个音频文件');
    window.__purewav = { get src() { return srcBuf; }, get den() { return denBuf; } };
  }
  window.addEventListener('DOMContentLoaded', init);
})();
