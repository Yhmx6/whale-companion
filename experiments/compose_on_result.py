"""把 S3 接触阴影补到 Seedream 的真实输出上，并做方向对错三对照。

这一步演示的是完整链路里最缺的一环：S2 按指令没有生成地面投影（U-10 初步验证成立），
于是"她没有影子"，看起来是贴上去的。S3 用程序化渲染补影子。

三面板对照：
  1) 无阴影（S2 原始输出）
  2) 阴影方向正确（光从左上 -> 影子向右下）
  3) 阴影方向错误（影子向左上）—— 用来验证"错的阴影比没阴影更糟"这条原则
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from s3.lighting import LightingState  # noqa: E402
from s3.s3ops import render_shadow_matte  # noqa: E402

SRC = os.path.join(ROOT, "output", "step1", "case01_0.png")
OUT = os.path.join(ROOT, "output", "step1")


def detect_character_box(img: Image.Image, k: int = 8) -> tuple[float, float, float, float]:
    """用「蓝色主导 + 最大连通域」估角色外接框。

    初版只用通道差阈值取全局 bbox，结果把天空的蓝色也算了进去（框宽到 892px）。
    改为降采样后做连通域标记，取最大的一块——角色是画面里最大的一团饱和蓝。
    不用 scipy：在 k 倍降采样图上手写 BFS，160x120 量级，毫秒级。
    """
    from collections import deque

    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.maximum(r, g)
    mask = (b - mx > 22) & (b > 50)

    H, W = mask.shape
    hh, ww = H // k, W // k
    small = mask[:hh * k, :ww * k].reshape(hh, k, ww, k).max(axis=(1, 3))

    seen = np.zeros_like(small, dtype=bool)
    best = None
    best_area = 0
    for sy in range(hh):
        for sx in range(ww):
            if not small[sy, sx] or seen[sy, sx]:
                continue
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            pts = []
            while q:
                y, x = q.popleft()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hh and 0 <= nx < ww and small[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(pts) > best_area:
                best_area = len(pts)
                ys = [p[0] for p in pts]
                xs = [p[1] for p in pts]
                best = (min(xs), min(ys), max(xs), max(ys))

    if best is None or best_area < 20:
        raise SystemExit("角色检测失败：没找到足够大的蓝色连通域")

    x0, y0, x1, y1 = [v * k for v in best]
    pad = int(0.06 * max(x1 - x0, y1 - y0))
    return (float(max(0, x0 - pad)), float(max(0, y0 - pad)),
            float(min(W, x1 + k + pad)), float(min(H, y1 + k + pad)))


def apply_shadow(img: Image.Image, box, light: LightingState) -> Image.Image:
    dark = render_shadow_matte(img.size, box, light)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    out = arr * (1.0 - dark[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def measure_feet_band(img: Image.Image, box) -> float:
    """量脚底下方条带的平均亮度，用来证明影子真的生效了。"""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32).mean(axis=2)
    w = box[2] - box[0]
    cx = (box[0] + box[2]) / 2
    y0, y1 = int(box[3]), int(min(box[3] + 0.2 * w, arr.shape[0]))
    x0, x1 = int(cx), int(min(cx + 0.8 * w, arr.shape[1]))
    if y1 <= y0 or x1 <= x0:
        return float("nan")
    return float(arr[y0:y1, x0:x1].mean())


def main() -> None:
    if not os.path.exists(SRC):
        raise SystemExit(f"缺输入: {SRC}（先跑 experiments.step1_run）")
    img = Image.open(SRC).convert("RGB")
    box = detect_character_box(img)
    w, h = img.size
    print(f"原图 {img.size} | 角色框 x{box[0]:.0f}-{box[2]:.0f} y{box[1]:.0f}-{box[3]:.0f} "
          f"| 宽{box[2]-box[0]:.0f} 高{box[3]-box[1]:.0f} "
          f"| 脚点 y={box[3]:.0f} ({box[3]/h*100:.0f}%H) 中心 x={((box[0]+box[2])/2)/w*100:.0f}%W")

    # 场景主光：太阳在左上（画面里能看到左上角的逆光），换算到我们的角度语义
    # 0=右 90=下 180=左 270=上；左上介于 180 与 270 之间 -> 225
    # intensity / hardness 按"阴天柔光"取值：初见用 0.9/0.55 渲染出来过黑过硬，
    # 与阴天场景不符，视觉验证后下调。晴天直射才该上到 0.85 以上。
    base = dict(light_azimuth_deg=225.0, elevation_deg=32.0, quality_hardness=0.32,
                intensity=0.55, confidence=0.75, gravity_deg=0.0,
                gravity_source="plane_normal")

    correct = LightingState(**base)
    wrong = LightingState(**{**base, "light_azimuth_deg": 45.0})  # 故意反向：光从右下

    print(f"正确方向: 阴影延伸角 {correct.shadow_angle_deg():.0f} 度（右下）")
    print(f"错误方向: 阴影延伸角 {wrong.shadow_angle_deg():.0f} 度（左上）")

    non = img.copy()
    ok = apply_shadow(img, box, correct)
    bad = apply_shadow(img, box, wrong)

    for name, im in (("case01_0", non), ("case01_1_shadow_ok", ok), ("case01_2_shadow_wrong", bad)):
        im.save(os.path.join(OUT, f"{name}.png"))
    print(f"脚底条带亮度：无阴影 {measure_feet_band(non, box):.1f} | "
          f"正确 {measure_feet_band(ok, box):.1f} | 错误 {measure_feet_band(bad, box):.1f}")

    # 三联对照图（整图，看氛围）
    scale = 0.5
    tw, th = int(w * scale), int(h * scale)
    grid = Image.new("RGB", (tw * 3, th + 34), (18, 18, 18))
    labels = ["A 无阴影（S2 原始输出）", "B 阴影方向正确（光左上）", "C 阴影方向错误（光影矛盾）"]
    for i, (im, lab) in enumerate(zip((non, ok, bad), labels)):
        grid.paste(im.resize((tw, th), Image.LANCZOS), (i * tw, 34))
        d = ImageDraw.Draw(grid)
        d.text((i * tw + 10, 11), lab, fill=(240, 240, 240))
    grid_path = os.path.join(OUT, "case01_shadow_compare.jpg")
    grid.save(grid_path, quality=92)
    print("整图对照 ->", grid_path)

    # 脚部局部放大：纵向三图叠放。
    # 横向三联在 1080px 显示宽度下每格只有 360px，看不清阴影方向；
    # 纵向叠放可以让每格占满宽度，只裁脚部区域，细节最清。
    bw, bh = box[2] - box[0], box[3] - box[1]
    fx0 = int(max(0, box[0] - bw * 0.35))
    fx1 = int(min(w, box[2] + bw * 1.15))
    fy0 = int(max(0, box[3] - bh * 0.30))
    fy1 = int(min(h, box[3] + bh * 0.42))
    fw, fh = fx1 - fx0, fy1 - fy0
    z = 1080 / fw
    zw, zh = 1080, int(fh * z)
    zoom = Image.new("RGB", (zw, (zh + 26) * 3), (18, 18, 18))
    for i, (im, lab) in enumerate(zip((non, ok, bad), labels)):
        crop = im.crop((fx0, fy0, fx1, fy1)).resize((zw, zh), Image.LANCZOS)
        zoom.paste(crop, (0, i * (zh + 26) + 26))
        d = ImageDraw.Draw(zoom)
        d.text((8, i * (zh + 26) + 7), lab, fill=(235, 235, 235))
    zoom_path = os.path.join(OUT, "case01_shadow_zoom.jpg")
    zoom.save(zoom_path, quality=94)
    print(f"脚部放大 -> {zoom_path}  (裁切 {fx0},{fy0}-{fx1},{fy1}  缩放 {z:.2f}x)")


if __name__ == "__main__":
    main()
