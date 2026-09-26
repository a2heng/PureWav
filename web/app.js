/* PureWav Web — 纯前端、离线、单文件降噪
 * WebAudio 解码 → JS STFT(Bluestein, 960/480/hann) → onnxruntime-web 推理 → ISTFT → WAV 导出
 */
'use strict';
(function () {
  const $ = (s) => document.querySelector(s);
  const FS = 48000, NFFT = 960, HOP = 480, BINS = NFFT / 2 + 1; // 481（推理 STFT，模型契约，不动）
  // ── 展示谱配置（wav_browser 同款）：FFT4096 / 50%交叠 / 20Hz起 /
  // 1kHz pivot 上下各 256 行（下线性上对数）/ magma / 相对参考 -80~0dB ──
  const SP_NFFT = 4096, SP_HOP = 2048, SP_ROWS = 512;
  const SP_MAX_COLS = 2000, SP_WAVE_COLS = 3200, SP_WAVE_PAD = 10;
  const SP_FMIN = 20, SP_FMAX = 24000, SP_PIVOT = 1000, SP_PIVOT_RATIO = 0.5;
  const SP_DB_LO = -80, SP_DB_HI = 0;
  const CHUNK_SEC = 15;                          // 每段推理时长

  let session = null, modelBytes = null;
  let srcBuf = null, denBuf = null, srcName = 'audio';
  let lastView = null;   // {vO, vD, dur}：resize 只重画，不重算
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

  // ── 混合滤波器组（下线性 FMIN→pivot，上对数 pivot→fmaxEff；fmaxEff 钳到 Nyquist）──
  // 行幅度 = 三角 taper 加权平均功率（w=0.7+0.3*taper，按权重和归一）
  let _spWin = null, _spBank = null;
  function spBank(sr) {
    if (_spBank && _spBank.sr === sr) return _spBank;
    const nyq = sr / 2, maxBin = Math.max(2, SP_NFFT >> 1), rows = SP_ROWS;
    const fmaxEff = Math.min(SP_FMAX, nyq);
    const pivot = Math.min(Math.max(SP_PIVOT, SP_FMIN + 1), fmaxEff);
    const freqAt = (ratio) => {
      if (ratio <= SP_PIVOT_RATIO || pivot >= fmaxEff)
        return SP_FMIN + (pivot - SP_FMIN) * Math.min(1, ratio / SP_PIVOT_RATIO);
      const ur = (ratio - SP_PIVOT_RATIO) / (1 - SP_PIVOT_RATIO);
      return pivot * Math.exp(Math.log(fmaxEff / pivot) * ur);
    };
    const band = [];
    for (let r = 0; r < rows; r++) {
      const f0 = Math.max(1.0, freqAt(r / rows)), f1 = Math.max(f0 + 1.0, freqAt((r + 1) / rows));
      const b0 = Math.min(Math.max(Math.floor((f0 / nyq) * maxBin), 1), maxBin - 1);
      const b1 = Math.min(Math.max(Math.ceil((f1 / nyq) * maxBin), b0 + 1), maxBin);
      band.push({ start: b0, end: b1 });
    }
    _spBank = { sr, band, rows, fmaxEff };
    return _spBank;
  }

  function spMagma(v) {
    const s = [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]];
    const x = v * (s.length - 1), i = Math.min(s.length - 2, Math.floor(x)), f = x - i, a = s[i], b = s[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }

  // 整段 PCM → 波形峰值对 + 混合谱（dB，ref 为外部参考；null 则取自身峰）。
  // 原图先算（定参考），降噪后跟随同一参考，两图同量程可比。
  function spAnalyze(y, sr, ref) {
    const bank = spBank(sr), rows = bank.rows, n = y.length;
    const w = Math.min(SP_WAVE_COLS, Math.max(1, n));
    const wf = new Float32Array(w * 2);
    for (let i = 0; i < w; i++) {
      const a = Math.floor(i * n / w), b = Math.max(a + 1, Math.floor((i + 1) * n / w));
      let mn = 1e9, mx = -1e9;
      for (let k = a; k < b; k++) { const v = y[k]; if (v < mn) mn = v; if (v > mx) mx = v; }
      wf[i * 2] = mn; wf[i * 2 + 1] = mx;
    }
    const nWin = Math.max(1, Math.round(n / SP_HOP));
    const merge = Math.max(1, Math.ceil(nWin / SP_MAX_COLS));
    const cols = Math.max(1, Math.ceil(nWin / merge));
    if (!_spWin) { _spWin = new Float64Array(SP_NFFT); for (let i = 0; i < SP_NFFT; i++) _spWin[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / SP_NFFT); }
    const win = _spWin;
    const re = new Float64Array(SP_NFFT), im = new Float64Array(SP_NFFT);
    const frames = new Float32Array(cols * rows);
    for (let c = 0; c < cols; c++) {
      const acc = new Float64Array(rows), cnt = merge;
      for (let m2 = 0; m2 < merge; m2++) {
        const col = c * merge + m2;
        const centre = ((2 * col + 1) * n) / (2 * nWin);
        const s0 = Math.round(centre) - (SP_NFFT >> 1);
        re.fill(0); im.fill(0);
        for (let i = 0; i < SP_NFFT; i++) { const idx = s0 + i; if (idx >= 0 && idx < n) re[i] = y[idx] * win[i]; }
        fftPow2(re, im, false);
        for (let r = 0; r < rows; r++) {
          const bd = bank.band[r];
          const sz = Math.max(1, bd.end - bd.start), ps = 1.0 / sz;
          let e = 0, tw = 0;
          for (let b = bd.start; b < bd.end; b++) {
            const pos = sz === 1 ? 0.5 : ((b - bd.start) + 0.5) * ps;
            const taper = 1.0 - Math.abs(pos * 2.0 - 1.0);
            const wt = 0.7 + taper * 0.3;
            e += (re[b] * re[b] + im[b] * im[b]) * wt; tw += wt;
          }
          acc[r] += e / Math.max(tw, 1e-8);
        }
      }
      for (let r = 0; r < rows; r++) frames[r * cols + c] = acc[r] / cnt;
    }
    let dmax = ref;
    const db = new Float32Array(cols * rows);
    for (let i = 0; i < db.length; i++) db[i] = 10 * Math.log10(frames[i] + 1e-10);
    if (ref == null) { dmax = -1e9; for (let i = 0; i < db.length; i++) if (db[i] > dmax) dmax = db[i]; }
    const lo = dmax + SP_DB_LO, rng = Math.max(SP_DB_HI - SP_DB_LO, 1e-6);
    const data = new Uint8Array(cols * rows);
    for (let i = 0; i < db.length; i++) { let v = (db[i] - lo) / rng; v = v < 0 ? 0 : v > 1 ? 1 : v; data[i] = Math.round(v * 255); }
    return { cols, rows, data, wf, wfCols: w, ref: dmax, fmaxEff: bank.fmaxEff };
  }

  function spSpecImage(data, rows, cols) {
    const cv = document.createElement('canvas');
    cv.width = cols; cv.height = rows;
    const g = cv.getContext('2d'), img = g.createImageData(cols, rows);
    for (let y = 0; y < rows; y++) {
      const row = rows - 1 - y;
      for (let x = 0; x < cols; x++) {
        const v = data[row * cols + x] / 255, c = spMagma(v), o = (y * cols + x) * 4;
        img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    return cv;
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

  // ── 单段: STFT → 推理 → ISTFT（展示谱在外部整段另算，不在此处）──
  async function denoise(x) {
    const win = hann(NFFT);
    const bs = new Bluestein(NFFT);
    const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
    const fRe = new Float64Array(BINS), fIm = new Float64Array(BINS);

    const out = new Float32Array(x.length);
    const chunkSamples = CHUNK_SEC * FS;
    const invN = 1 / NFFT;

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

    return { out };
  }

  // ── 频谱绘制（magma；上 24k → 中 1k → 下 20Hz；右上 0dB、右下 -80dB）──
  function specCanvas(disp, cols) {
    return spSpecImage(disp, SP_ROWS, cols);
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
    g.font = (11 * dpr) + 'px sans-serif'; g.textBaseline = 'top';
    g.fillStyle = 'rgba(235,235,235,.85)';
    g.fillText('24k', 5 * dpr, 3 * dpr);
    g.fillText('1k', 5 * dpr, canvas.height * SP_PIVOT_RATIO - 13 * dpr);
    g.fillText('20', 5 * dpr, canvas.height - 14 * dpr);
    g.fillText('0dB', canvas.width - 32 * dpr, 3 * dpr);
    g.fillText('-80dB', canvas.width - 44 * dpr, canvas.height - 14 * dpr);
    g.strokeStyle = 'rgba(255,255,255,.35)';
    const y = canvas.height * SP_PIVOT_RATIO;
    g.beginPath(); g.moveTo(0, y); g.lineTo(canvas.width, y); g.stroke();
  }
  // 波形峰值跨度 + 上下留白 + 标题（整段时长）
  function drawWave(canvas, wf, wcols, color, title, duration) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    const g = canvas.getContext('2d');
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.fillStyle = '#0d0d0d'; g.fillRect(0, 0, canvas.width, canvas.height);
    g.strokeStyle = '#2a2a2a'; g.beginPath();
    g.moveTo(0, canvas.height / 2); g.lineTo(canvas.width, canvas.height / 2); g.stroke();
    if (wf) {
      const PAD = SP_WAVE_PAD * dpr, inner = Math.max(1, canvas.height - 2 * PAD);
      g.strokeStyle = color; g.beginPath();
      for (let i = 0; i < wcols; i++) {
        const x = i / Math.max(1, wcols - 1) * canvas.width;
        const a = Math.max(-1, Math.min(1, wf[i * 2])), c = Math.max(-1, Math.min(1, wf[i * 2 + 1]));
        g.moveTo(x, (1 - c) / 2 * inner + PAD); g.lineTo(x, (1 - a) / 2 * inner + PAD);
      }
      g.stroke();
    }
    g.font = (11 * dpr) + 'px sans-serif'; g.textBaseline = 'top';
    g.fillStyle = 'rgba(235,235,235,.85)';
    g.fillText(title + '  ' + fmtTime(0) + ' – ' + fmtTime(duration), 5 * dpr, 3 * dpr);
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
  // 资源两种形态：在线版是 URL（fetch），离线单文件版是 "#id"
  //（读同文档内 base64 脚本块，file:// 也能用）。
  async function assetBytes(v, what) {
    if (typeof v === 'string' && v.charAt(0) === '#') return b64ToBytes(v.slice(1));
    return fetchBytes(v, what);
  }
  async function fetchBytes(url, what) {
    let r;
    try {
      r = await fetch(url);
    } catch (e) {
      throw new Error(what + '下载失败（' + (e && e.message || e) +
        '）。本地双击 file:// 打不开，请用 `python -m http.server` 或部署到 GitHub Pages 后再开');
    }
    if (!r.ok) throw new Error(what + '下载失败：HTTP ' + r.status + ' ' + url);
    return new Uint8Array(await r.arrayBuffer());
  }

  async function loadModel() {
    setStatus('加载模型…');
    const A = window.PUREWAV_ASSETS || {};
    ort.env.wasm.numThreads = 1;
    let glue = A.glue;
    if (typeof glue === 'string' && glue.charAt(0) === '#') {
      glue = URL.createObjectURL(new Blob([b64ToBytes(glue.slice(1))],
        { type: 'text/javascript' }));
    }
    ort.env.wasm.wasmPaths = { mjs: glue };
    ort.env.wasm.wasmBinary = await assetBytes(A.wasm, '运行时');
    modelBytes = await assetBytes(A.model, '模型');
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
      const { out } = await denoise(x);
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      srcBuf = x; denBuf = out;
      // 展示谱整段另算：原图定参考，降噪后跟随同量程
      setStatus('画频谱…');
      await new Promise((r) => setTimeout(r, 0));
      const vO = spAnalyze(x, FS, null);
      const vD = spAnalyze(out, FS, vO.ref);
      const dur = x.length / FS;
      lastView = { vO, vD, dur };
      setStatus(`完成 · ${fmtTime(dur)} · 用时 ${dt}s`);
      setProgress(1);
      drawWave($('#waveOrig'), vO.wf, vO.wfCols, '#e06c75', '原图', dur);
      drawWave($('#waveDen'), vD.wf, vD.wfCols, '#4ec9b0', '降噪后', dur);
      drawSpec($('#specOrig'), specCanvas(vO.data, vO.cols));
      drawSpec($('#specDen'), specCanvas(vD.data, vD.cols));
      $('#playOrig').disabled = $('#playDen').disabled = $('#dl').disabled = false;
      updatePlayBtns();
    } catch (e) {
      console.error(e);
      setStatus('失败: ' + (e && e.message || e));
    }
  }

  // ── 导出离线版：把当前页 + 全部重资源打包成单文件 HTML ──
  // 在线版保持苗条；离线包按需在浏览器里现拼（fetch → base64 内嵌），
  // 存下来以后双击 file:// 即用。构建侧不再产第二个文件。
  function bytesToB64(u8) {
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) {
      s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    }
    return btoa(s);
  }
  async function exportOffline() {
    const btn = $('#dlOffline');
    try {
      btn.disabled = true;
      const A = window.PUREWAV_ASSETS || {};
      setStatus('打包离线版：下载运行时…');
      const ortJs = await (await fetch(A.ortJs)).text();
      if (ortJs.toLowerCase().indexOf('</scr' + 'ipt') >= 0) {
        throw new Error('运行时内嵌会截断页面，导出中止');
      }
      await new Promise((r) => setTimeout(r, 0));
      setStatus('打包离线版：下载模型(约 2MB)…');
      const wasm = new Uint8Array(await (await fetch(A.wasm)).arrayBuffer());
      await new Promise((r) => setTimeout(r, 0));
      setStatus('打包离线版：转码中…');
      const glue = new Uint8Array(await (await fetch(A.glue)).arrayBuffer());
      const model = new Uint8Array(await (await fetch(A.model)).arrayBuffer());
      setStatus('打包离线版：组装中…');
      await new Promise((r) => setTimeout(r, 0));
      let src = await (await fetch(location.href.split(/[?#]/)[0])).text();
      // 注意：'</scr'+'ipt>' 写法是故意的——字面闭合标签会提前终结
      // 当前页面内联的 script 块（HTML 解析器不认 JS 字符串）。
      const blobs =
        '<script id="glue" type="text/plain">' + bytesToB64(glue) + '</scr' + 'ipt>\n' +
        '<script id="wasm" type="text/plain">' + bytesToB64(wasm) + '</scr' + 'ipt>\n' +
        '<script id="model" type="text/plain">' + bytesToB64(model) + '</scr' + 'ipt>\n';
      // 用替换前后是否变化判定是否咬住目标（子串包含检查不可靠：
      // app.js 自身就含有同样的字面量，会永远命中误报）。
      const next1 = src.replace(/<script>window\.PUREWAV_ASSETS=\{[^}]*\};<\/script>/,
        '<script>window.PUREWAV_ASSETS={glue:"#glue",wasm:"#wasm",model:"#model"};</scr' + 'ipt>\n' + blobs);
      if (next1 === src) {
        throw new Error('页面结构对不上，导出中止（页面不是新版在线版？）');
      }
      const next2 = next1.replace(/<script src="[^"]*ort\.wasm\.min\.js"><\/script>/,
        '<script>\n' + ortJs + '\n</scr' + 'ipt>');
      if (next2 === next1) {
        throw new Error('页面结构对不上，导出中止（找不到运行时脚本位）');
      }
      src = next2;
      const url = URL.createObjectURL(new Blob([src], { type: 'text/html' }));
      const a = document.createElement('a');
      a.href = url; a.download = 'PureWavWeb.html'; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      setStatus('离线版已开始下载（PureWavWeb.html，双击即用）');
    } catch (e) {
      console.error(e);
      setStatus('导出失败: ' + (e && e.message || e));
    } finally {
      btn.disabled = false;
    }
  }

  // ── 事件绑定 ──────────────────────────────────────────
  function init() {
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
    $('#dlOffline').addEventListener('click', exportOffline);
    window.addEventListener('resize', () => {
      if (lastView) {
        const { vO, vD, dur } = lastView;
        drawWave($('#waveOrig'), vO.wf, vO.wfCols, '#e06c75', '原图', dur);
        drawWave($('#waveDen'), vD.wf, vD.wfCols, '#4ec9b0', '降噪后', dur);
        drawSpec($('#specOrig'), specCanvas(vO.data, vO.cols));
        drawSpec($('#specDen'), specCanvas(vD.data, vD.cols));
      }
    });
    setStatus('就绪 · 拖入一个音频文件');
    window.__purewav = { get src() { return srcBuf; }, get den() { return denBuf; } };
  }
  window.addEventListener('DOMContentLoaded', init);
})();
