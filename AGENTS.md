# PureWav — Embedded Audio Denoiser

Cross-platform (Windows / Linux / macOS) desktop app (`main.py` + `spectrum_viz.py`) for audio/video
noise reduction using an ONNX model. Media I/O is **pure Python via PyAV** (pip wheel that bundles
FFmpeg libraries) — no external `ffmpeg` binary is shipped or required.

## Build & Run

**Dev:** `python main.py` (requires tkinterdnd2, onnxruntime, numpy, soundfile, av, scipy, matplotlib)

**Export batch ONNX (one-time, commit result to repo):**
```bash
python export_batch.py   # -> v6_erb_skip_proj_batch.onnx
```

**Package (release):**
```
pyinstaller --clean --name "PureWav" --onefile --noconsole --icon="audio_icon.ico" --add-data "audio_icon.ico;." --add-data "v6_erb_skip_proj_batch.onnx;." --hidden-import=onnxruntime.capi._pybind_state --hidden-import=onnxruntime.capi.onnxruntime_pybind11_state --hidden-import=spectrum_viz main.py
```
Output: `dist/PureWav.exe` (PyAV ships its own PyInstaller hooks, no `--add-binary` needed)

**Package (debug, no --noconsole):**
Same command without `--noconsole` (keeps console window for debug output).

## Web 版 (瘦页面 + 共享重资源，可部署到 GitHub Pages)

- 目录 `web/`：`template.html` + `app.js` + `build_web.py`
- 构建：`python web/build_web.py` → `web/dist/index.html`（~27KB）+ `web/dist/assets/*.onnx`
  （产物目录 gitignore，手动传到 `a2heng.github.io` 仓库的 `purewav/` 下，不走工作流）。
  页面本体只含应用代码；重资源按 URL 加载：ONNX Runtime Web 三件套（UMD JS +
  glue + wasm，锁 1.29.0）默认指向已部署的 PureVox 页面共用地址
  `https://a2heng.github.io/purevox/assets/ort`（glue/wasm 与 PureVox 页字节一致，
  浏览器缓存命中，不重复下载；可用 `--ort-base` 换 CDN），模型随页面部署。
- 离线版不在构建侧产出：在线页自带「下载离线版」按钮，在浏览器里把运行时 +
  模型打包成单个 HTML 存下来（`PureWavWeb.html`，双击 file:// 即用）。
  瘦页面本身不支持 file:// 双击，本地预览：`python -m http.server --directory web/dist`
- 运行时：`ort.env.wasm.wasmPaths={mjs: url}` + fetch 来的 `wasmBinary`，
  `numThreads=1`（无 SharedArrayBuffer 也能跑）
- 流程：WebAudio 解码 → OfflineAudioContext 重采样 48k 单声道 → JS STFT(960/480/hann) →
  onnxruntime-web 推理 → ISTFT → WAV 导出；Bluestein FFT 处理非 2 的幂的 960 点
- 单文件拖拽；左侧「原图」右侧「降噪后」各上波形(±1)下频谱(Canvas 2D，无 PIL)
- 频谱色标以 audioscope (`~/audioscope/src-wasm/wasm_core/spectrogram.zig`) 为标准：
  power = |X|² / (sum(win)/2)² (满幅单音=0dB)，带内 taper 加权平均功率，
  db = 10·log10(power)，归一化到 `[VMIN, VMAX] = [-90, -10]`（桌面 spectrum_viz.py 同步此标准）

## Key Architecture

- **`main.py`** — app: dark card-style GUI (TkinterDnD, `clam` theme) + media backend + batch ONNX
  inference. Media backend is PyAV-based: `decode_mono48k` (any audio/video → 48k mono float32 in
  memory), `denoise_array` (in-memory STFT→ONNX→ISTFT), `process_audio_file` (streaming decode→write,
  no temp files), `replace_video_audio` (video stream copy + AAC re-encode, no temp files).
  ONNX sessions are cached by model path (`load_session`).
- **`spectrum_viz.py`** — spectrum visualization (matplotlib): full 0–24 kHz axis with the 20–24 kHz
  range binned into 1 kHz bands, whole-file STFT (no time segmentation). The window is a 1:5 split —
  a narrow left control column (filename + hold-to-compare button + current state) and a large right
  figure holding a ±1 waveform envelope on top and the spectrogram below. Ultra-compact: tiny 6 pt
  tick labels drawn inside the axes, no colorbar. Shows the denoised result by default; holding the
  compare button switches both waveform and spectrogram to the original (both pre-rendered, instant).
  The window decodes + denoises in memory (single source read, no temp files).
- **`v6_erb_skip_proj_batch.onnx`** — exported batch denoising model (STFT in → STFT out, ~0.52M params)
- **`models/lightweight-denoise-48k/`** — git submodule: model source code
- **`config.json`** — runtime user config (output dir, cpu cores); saved to CWD, not repo-tracked
- **`ffmpeg.exe`** — legacy, no longer used by the code; can be deleted from the repo

## Important Details

- Model expects 48kHz mono PCM; PyAV handles all decode/resample/mixdown to that target
- Batch ONNX (`v6_erb_skip_proj_batch.onnx`) must be exported from checkpoint before packaging — `export_batch.py`
- The ONNX model is STFT-domain: input `(1, 2, T, 481)`, output `(1, 2, T, 481)` — STFT/ISTFT done in numpy
- `requirements.txt`: `numpy scipy onnxruntime soundfile av tkinterdnd2 matplotlib pyinstaller`
- Video handling: decode audio to memory → denoise → `replace_video_audio` copies the video stream
  and re-encodes the denoised audio as AAC (all in-process, no temp files)
- Path resolution uses `sys._MEIPASS` when packaged, CWD when dev — both paths are in `main.py`
- Config file written to CWD as `audio_denoise_config.json` (gitignored)

## CI/CD

- GitHub Actions at `.github/workflows/ci.yml`
- Trigger: push tag `v*` or manual `workflow_dispatch`
- Flow: setup Python 3.12 → install deps → PyInstaller build → upload artifact → (on tag) create Release
- Tag naming: `v2026.08.17.2038` → artifact `PureWav-Windows-x64-2026-08-17-2038.exe`

## Conventions

- Code and UI strings are in Chinese (中文)
- No tests, no linter, no type checker configured
- Dev/test environment on this machine: `/home/a2heng/venv` (Python 3.12, onnxruntime 1.30, av 18, soundfile 0.14, matplotlib 3.11)
