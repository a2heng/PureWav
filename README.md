# PureWav

轻量级 AI 音频降噪工具，基于 [lightweight-denoise-48k](https://github.com/a2heng/lightweight-denoise-48k) 模型，支持常见音频/视频格式。

## 功能特性

- AI 降噪（~0.52M 参数，48kHz 全频带）
- 支持音频：wav、mp3、flac、ogg、m4a、wma
- 支持视频：提取音频降噪后替换原音（mp4、avi、mov、mkv、flv、wmv）
- 拖拽文件/文件夹到窗口即可处理
- 输出格式：音频 → wav；视频 → mp4
- 批量处理，进度显示

## 快速开始

```bash
git submodule update --init --recursive
pip install -r requirements.txt soundfile
python main.py
```

## 打包

Windows 下使用 PyInstaller 打包为独立 exe：

```powershell
pyinstaller --clean `
    --name "PureWav" `
    --onefile `
    --noconsole `
    --icon="audio_icon.ico" `
    --add-data "audio_icon.ico;." `
    --add-data "v6_erb_skip_proj_batch.onnx;." `
    --add-data "ffmpeg.exe;." `
    --hidden-import=onnxruntime.capi._pybind_state `
    --hidden-import=onnxruntime.capi.onnxruntime_pybind11_state `
    main.py
```

产物：`dist/AI音频文件降噪.exe`

## CI/CD

推送 `v*` tag（如 `v2026.08.17.2038`）或手动触发 `workflow_dispatch`，GitHub Actions 自动构建并创建 Release。

```bash
git tag v2026.08.17.2038
git push origin v2026.08.17.2038
```

## 项目结构

```
main.py                                      # 应用主文件（GUI + batch 推理 + ffmpeg 调用）
export_batch.py                              # batch ONNX 导出脚本（本地使用，不进 CI）
v6_erb_skip_proj_batch.onnx                  # 导出的 batch ONNX（提交到仓库）
models/lightweight-denoise-48k/              # git submodule — 降噪模型源码
ffmpeg.exe                                   # 音视频处理（bundled）
audio_icon.ico                               # 应用图标
```

## 技术栈

| 组件 | 技术 |
|---|---|
| GUI | Python + Tkinter + TkinterDnD |
| 降噪模型 | [lightweight-denoise-48k](https://github.com/a2heng/lightweight-denoise-48k)（~0.52M 参数） |
| 推理 | ONNX Runtime（batch 模式） |
| 音频处理 | soundfile + ffmpeg |
| 打包 | PyInstaller |

## 许可证

MIT
