#!/usr/bin/env python3
"""构建 PureWav Web：把 onnxruntime-web (JS + glue + wasm)、ONNX 模型、inferno 色表
全部 base64 内嵌进单个 HTML，产物完全离线、双击即用。

用法:
    python web/build_web.py                 # 自动查找/安装 onnxruntime-web
    python web/build_web.py --ort <dist>    # 指定 onnxruntime-web/dist 目录
    python web/build_web.py -o PureWavWeb.html
"""
import argparse
import base64
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = HERE / "PureWavWeb.html"
NEEDED = {
    "ort_js": "ort.wasm.min.js",
    "glue": "ort-wasm-simd-threaded.mjs",
    "wasm": "ort-wasm-simd-threaded.wasm",
}


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def find_ort_dist(explicit: str = None) -> pathlib.Path:
    cands = []
    if explicit:
        cands.append(pathlib.Path(explicit))
    env = os.environ.get("ORT_DIST")
    if env:
        cands.append(pathlib.Path(env))
    cands += [
        HERE / ".ort" / "node_modules" / "onnxruntime-web" / "dist",
        HERE / "node_modules" / "onnxruntime-web" / "dist",
        ROOT / "node_modules" / "onnxruntime-web" / "dist",
        pathlib.Path("/tmp/opencode/ort/node_modules/onnxruntime-web/dist"),
    ]
    for c in cands:
        if c.is_dir() and all((c / f).is_file() for f in NEEDED.values()):
            return c
    # 尝试 npm 安装到 web/.ort
    npm = shutil.which("npm")
    if npm:
        target = HERE / ".ort"
        print(f"[build] 未找到 onnxruntime-web，尝试 npm 安装到 {target} ...")
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run([npm, "install", "--no-audit", "--no-fund", "onnxruntime-web"],
                       cwd=str(target), check=True)
        dist = target / "node_modules" / "onnxruntime-web" / "dist"
        if dist.is_dir():
            return dist
    raise SystemExit("找不到 onnxruntime-web/dist，请用 --ort 指定，或先 npm install onnxruntime-web")


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
    ap.add_argument("--ort", default=None, help="onnxruntime-web/dist 目录")
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=str(ROOT / "v6_erb_skip_proj_batch.onnx"))
    args = ap.parse_args()

    dist = find_ort_dist(args.ort)
    model = pathlib.Path(args.model)
    if not model.is_file():
        raise SystemExit(f"模型不存在: {model}")

    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    app_js = (HERE / "app.js").read_text(encoding="utf-8")

    html = (tpl
            .replace("__ORT_JS__", (dist / NEEDED["ort_js"]).read_text(encoding="utf-8"))
            .replace("/*__APP_JS__*/", app_js)
            .replace("__GLUE_B64__", b64(dist / NEEDED["glue"]))
            .replace("__WASM_B64__", b64(dist / NEEDED["wasm"]))
            .replace("__MODEL_B64__", b64(model))
            .replace("__INFERNO_B64__", base64.b64encode(inferno_lut()).decode("ascii")))

    out = pathlib.Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"[build] {out}  ({out.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"[build] ort {dist}")


if __name__ == "__main__":
    main()
