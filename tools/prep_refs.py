"""S0 参考图预处理：把三视图整板裁成"一个视角一张、紧裁到角色外接框"。

依据《07》H-3：模型会把每张参考图压到约 384² 面积。整板直喂时单个角色
有效宽度只有约 128px，紧裁后约 384px，细节差 3 倍。所以必须裁。
同时切掉顶部标题条（否则 "05 CHIBI BLUE MAID" 有概率被画进结果）。

自动检测而非硬编码坐标：按列等分成三带，各自找"非接近白色"像素的外接框。
产出：
    assets/refs/crops/{key}_{view}.png        白底紧裁（喂模型用）
    assets/refs/crops/{key}_{view}_rgba.png   去白底（本地贴图对比用）
    assets/refs/crops/manifest.json           各视角外接框，供 QC 参考
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(HERE, "assets", "refs")
OUT_DIR = os.path.join(REF_DIR, "crops")

# 视图顺序：与设定图从左到右一致
VIEWS = ["front", "side", "back"]

SOURCES = {
    "05_chibi": "05-chibi-blue-maid.jpg",
    "04_detailed": "04-detailed-short-maid.jpg",
    "02_simple": "02-simple-short-maid.jpg",
}

WHITE_TOL = 18      # 距纯白多少以内算背景
MARGIN_RATIO = 0.03  # 紧裁后再留 3% 边，避免切到头发/尾巴末梢


def non_white_bbox(arr: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    """在给定列带内找非白像素的外接框。"""
    band = arr[y0:y1, x0:x1]
    # 距白：三通道里最暗的那个通道离 255 的差距
    dist = 255 - band.min(axis=2)
    mask = dist > WHITE_TOL
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return (x0 + int(xs.min()), y0 + int(ys.min()),
            x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)


def white_to_alpha(img: Image.Image, tol: int = 26) -> Image.Image:
    """把接近纯白的背景转成透明，边缘做软过渡（避免硬白边）。

    注意：这是原型级抠图，正式链路用 BiRefNet（见《07》许可表）。
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    dist = 255.0 - arr.min(axis=2)                 # 0=纯白
    alpha = np.clip((dist - tol) / max(tol, 1) * 255.0, 0, 255)
    rgba = np.dstack([arr, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest: dict = {}

    for key, fname in SOURCES.items():
        path = os.path.join(REF_DIR, fname)
        if not os.path.exists(path):
            print("skip (missing):", path)
            continue
        img = Image.open(path).convert("RGB")
        W, H = img.size
        arr = np.asarray(img)

        # 竖向安全区：切掉顶部标题条（约上 9%）与底部视角标签（约下 10%）
        y0 = int(H * 0.09)
        y1 = int(H * 0.90)

        # 列三等分，各自找外接框
        manifest[key] = {"source": fname, "size": [W, H], "views": {}}
        for i, view in enumerate(VIEWS):
            x0 = int(W * i / 3)
            x1 = int(W * (i + 1) / 3)
            bbox = non_white_bbox(arr, x0, x1, y0, y1)
            if bbox is None:
                print(f"  {key}/{view}: 空带，跳过")
                continue
            bx0, by0, bx1, by1 = bbox
            mx = int((bx1 - bx0) * MARGIN_RATIO)
            my = int((by1 - by0) * MARGIN_RATIO)
            crop_box = (max(0, bx0 - mx), max(0, by0 - my),
                        min(W, bx1 + mx), min(H, by1 + my))
            crop = img.crop(crop_box)
            white_bg_path = os.path.join(OUT_DIR, f"{key}_{view}.png")
            crop.save(white_bg_path)
            crop_rgba = white_to_alpha(crop)
            rgba_path = os.path.join(OUT_DIR, f"{key}_{view}_rgba.png")
            crop_rgba.save(rgba_path)
            manifest[key]["views"][view] = {
                "crop_box": list(crop_box),
                "crop_size": [crop.width, crop.height],
                "aspect": round(crop.width / crop.height, 4),
            }
            print(f"  {key}/{view}: {crop.size}  aspect={crop.width/crop.height:.3f}")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("\nmanifest ->", os.path.join(OUT_DIR, "manifest.json"))


if __name__ == "__main__":
    main()
