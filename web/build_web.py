#!/usr/bin/env python3
"""构建 PureWav Web 在线版：瘦页面 + 共享重资源，可直接部署到 GitHub Pages。

页面本体只含应用代码（CSS/JS 内联，约几十 KB）；重资源按 URL 加载：
  - ONNX Runtime Web 三件套（UMD JS + glue + wasm，锁 1.29.0）默认指向已部署的
    PureVox 页面共用地址（同一份文件，浏览器缓存命中，不重复下载）；
    可用 --ort-base 换成 jsDelivr 等 CDN。
  - ONNX 模型随页面部署（复制到产物 assets/ 下）。

离线单文件版不在构建侧产出：在线页自带「下载离线版」按钮，在浏览器里把
运行时 + 模型打包成一个 HTML 存下来（双击 file:// 即用）。

注意：瘦页面不能再双击 file:// 打开，本地预览用：
    python -m http.server --directory web/dist

用法:
    python web/build_web.py
    python web/build_web.py --model xxx.onnx
    python web/build_web.py --ort-base https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist
    python web/build_web.py -o /tmp/pw/index.html
"""
import argparse
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = HERE / "dist" / "index.html"
DEFAULT_ORT_BASE = "https://a2heng.github.io/purevox/assets/ort"  # 锁 1.29.0，与 PureVox 页共用
ORT_FILES = {
    "ort_js": "ort.wasm.min.js",
    "glue": "ort-wasm-simd-threaded.mjs",
    "wasm": "ort-wasm-simd-threaded.wasm",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT), help="输出 HTML 路径")
    ap.add_argument("--model", default=str(ROOT / "v6_erb_skip_proj_batch.onnx"))
    ap.add_argument("--ort-base", default=DEFAULT_ORT_BASE,
                    help="ORT 运行时 URL 前缀（三件套 1.29.0）")
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
    for slot in ("__ORT_JS_URL__", "__GLUE_URL__", "__WASM_URL__",
                 "__MODEL_URL__", "/*__APP_JS__*/"):
        if slot not in tpl:
            raise SystemExit(f"模板缺占位符: {slot}")
    # 注意：检查只针对模板——app.js 内导出逻辑含有同样的字面量，产物里有是正常的。

    html = (tpl
            .replace("__ORT_JS_URL__", f"{ort_base}/{ORT_FILES['ort_js']}")
            .replace("__GLUE_URL__", f"{ort_base}/{ORT_FILES['glue']}")
            .replace("__WASM_URL__", f"{ort_base}/{ORT_FILES['wasm']}")
            .replace("__MODEL_URL__", f"assets/{model.name}")
            .replace("/*__APP_JS__*/", app_js))
    out.write_text(html, encoding="utf-8")
    print(f"[build] {out}  ({out.stat().st_size / 1024:.1f} KB)")
    print(f"[build]   模型 assets/{model.name}  ({dst_model.stat().st_size / 1024 / 1024:.2f} MB)")
    print("[build]   ORT %s/ （三件套 1.29.0，与 PureVox 页共用同一份）" % ort_base)
    print("[build] 本地预览: python -m http.server --directory " + str(out.parent))
    print("[build] 上 io: 手动把产物目录传到 a2heng.github.io 仓库的 purewav/ 下")


if __name__ == "__main__":
    main()
