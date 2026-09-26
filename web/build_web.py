#!/usr/bin/env python3
"""构建 PureWav Web（瘦页面 + 共享重资源，可直接部署到 GitHub Pages）。

页面本体只含应用代码（CSS/JS 内联，约几十 KB）；重资源按 URL 加载：
  - ONNX Runtime Web（UMD JS + glue + wasm，1.29.0）默认指向已部署的
    PureVox 页面共用地址（同一份文件，浏览器缓存命中，不重复下载）；
    可用 --ort-base 换成 jsDelivr 等 CDN。
  - ONNX 模型随页面部署（复制到产物 assets/ 下）。

注意：瘦页面不能再双击 file:// 打开（fetch 会被拦），本地预览用：
    python -m http.server --directory web/dist

用法:
    python web/build_web.py                          # 默认 1.29.0 + 仓库模型
    python web/build_web.py --model xxx.onnx
    python web/build_web.py --ort-base https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist
    python web/build_web.py -o /tmp/pw/index.html
"""
import argparse
import base64
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = HERE / "dist" / "index.html"
DEFAULT_ORT_BASE = "https://a2heng.github.io/purevox/assets/ort"  # 锁 1.29.0，与 PureVox 页共用
ORT_FILES = {
    "ort_js": "ort.wasm.min.js",
    "glue": "ort-wasm-simd-threaded.mjs",
    "wasm": "ort-wasm-simd-threaded.wasm",
}


def inferno_lut() -> bytes:
    import numpy as np
    try:
        import matplotlib
        cmap = matplotlib.colormaps["inferno"]
        rgb = (np.asarray(cmap(np.linspace(0, 1, 256)))[:, :3] * 255).round().astype("uint8")
    except Exception:
        # 兜底: inferno 近似控制点线性插值
        pts = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
                        [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]], float)
        x = np.linspace(0, 1, len(pts))
        xi = np.linspace(0, 1, 256)
        rgb = np.stack([np.interp(xi, x, pts[:, c]) for c in range(3)], axis=1).round().astype("uint8")
    return rgb.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT), help="输出 HTML 路径")
    ap.add_argument("--model", default=str(ROOT / "v6_erb_skip_proj_batch.onnx"))
    ap.add_argument("--ort-base", default=DEFAULT_ORT_BASE,
                    help="ORT 运行时 URL 前缀（1.29.0 三件套）")
    args = ap.parse_args()

    model = pathlib.Path(args.model)
    if not model.is_file():
        raise SystemExit(f"模型不存在: {model}")
    ort_base = args.ort_base.rstrip("/")

    out = pathlib.Path(args.out)
    assets = out.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    dst_model = assets / model.name
    if not dst_model.is_file() or dst_model.stat().st_size != model.stat().st_size:
        shutil.copyfile(model, dst_model)

    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    app_js = (HERE / "app.js").read_text(encoding="utf-8")
    if "</script" in app_js.lower():
        raise SystemExit("app.js 含 </script>，会截断内联脚本")

    html = (tpl
            .replace("__ORT_JS_URL__", f"{ort_base}/{ORT_FILES['ort_js']}")
            .replace("__GLUE_URL__", f"{ort_base}/{ORT_FILES['glue']}")
            .replace("__WASM_URL__", f"{ort_base}/{ORT_FILES['wasm']}")
            .replace("__MODEL_URL__", f"assets/{model.name}")
            .replace("__INFERNO_B64__", base64.b64encode(inferno_lut()).decode("ascii"))
            .replace("/*__APP_JS__*/", app_js))
    if "__" in html and ("__ORT_" in html or "__GLUE_" in html or "__WASM_" in html
                         or "__MODEL_" in html or "__INFERNO_" in html or "__APP_JS__" in html):
        raise SystemExit("模板占位符未替换完")

    out.write_text(html, encoding="utf-8")
    print(f"[build] {out}  ({out.stat().st_size / 1024:.1f} KB)")
    print(f"[build] 模型: assets/{model.name}  ({dst_model.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"[build] ORT:  {ort_base}/ （三件套 1.29.0，与 PureVox 页共用同一份）")
    print("[build] 本地预览: python -m http.server --directory " + str(out.parent))
    print("[build] 上 io: 手动把产物目录传到 a2heng.github.io 仓库的 purewav/ 下")


if __name__ == "__main__":
    main()
