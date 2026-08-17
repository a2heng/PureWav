# PureWav — Embedded Audio Denoiser

Single-file Windows desktop app (`main.py`) for audio/video noise reduction using an ONNX model. Bundles its own Python 3.8 + ffmpeg — does NOT depend on a host Python install.

## Build & Run

**Dev:** `python main.py` (requires tkinterdnd2, onnxruntime, numpy, soundfile)

**Export batch ONNX (one-time, commit result to repo):**
```bash
python export_batch.py   # -> v6_erb_skip_proj_batch.onnx
```

**Package (release):**
```
pyinstaller --clean --name "PureWav" --onefile --noconsole --icon="audio_icon.ico" --add-data "audio_icon.ico;." --add-data "v6_erb_skip_proj_batch.onnx;." --add-data "ffmpeg.exe;." --hidden-import=onnxruntime.capi._pybind_state --hidden-import=onnxruntime.capi.onnxruntime_pybind11_state main.py
```
Output: `dist/PureWav.exe`

**Package (debug, no --noconsole):**
Same command without `--noconsole` (keeps console window for debug output).

## Key Architecture

- **`main.py`** — entire app: GUI (TkinterDnD) + batch ONNX inference + ffmpeg subprocess calls
- **`v6_erb_skip_proj_batch.onnx`** — exported batch denoising model (STFT in → STFT out, ~0.52M params)
- **`models/lightweight-denoise-48k/`** — git submodule: model source code
- **`ffmpeg.exe`** — bundled; used for audio extraction, format conversion, and video audio replacement
- **`config.json`** — runtime user config (output dir, cpu cores); saved to CWD, not repo-tracked

## Important Details

- Model expects 48kHz mono PCM; ffmpeg handles all format conversion
- Batch ONNX (`v6_erb_skip_proj_batch.onnx`) must be exported from checkpoint before packaging — `export_batch.py`
- The ONNX model is STFT-domain: input `(1, 2, T, 481)`, output `(1, 2, T, 481)` — STFT/ISTFT done in numpy
- `soundfile` is used at runtime but **missing from `requirements.txt`** — must be installed manually
- `librosa` and `moviepy` are in `requirements.txt` but **never imported** in `main.py`
- Path resolution uses `sys._MEIPASS` when packaged, CWD when dev — both paths are in `main.py`
- Config file written to CWD as `audio_denoise_config.json` (gitignored)

## CI/CD

- GitHub Actions at `.github/workflows/ci.yml`
- Trigger: push tag `v*` or manual `workflow_dispatch`
- Flow: setup Python 3.8 → install deps → `compileall` → PyInstaller build → upload artifact → (on tag) create Release
- Tag naming: `v2026.08.17.2038` → artifact `PureWav-Windows-x64-2026-08-17-2038.exe`

## Conventions

- Code and UI strings are in Chinese (中文)
- No tests, no linter, no type checker configured
- `onnxruntime==1.11` is pinned — newer versions may have API changes
