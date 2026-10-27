"""合成测试场景与占位角色 + 消融实验 runner。

为什么用合成场景而不是真实照片：这是 06 §9.3 的硬约束——
**最小可测闭环里角色要手工贴，不用模型输出**。这样测的是三项后处理
本身的价值，与模型质量完全解耦。若手工贴 + 后处理看不出前后差别，
说明参数错了，不是模型不行。

运行：
    cd prototype
    ../.venv 或系统 python 运行  python -m s3.demo
产出（prototype/output/）：
    scene.jpg          合成测试场景（含参考物体及其影子，暗示光向）
    baseline.jpg       裸贴角色（三项后处理全关）
    full.jpg           三项全开
    ablation_grid.jpg  8 组消融拼图（U-13）
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .compose import compose
from .lighting import LightingState

W, H = 1024, 768
HORIZON = int(H * 0.52)


def make_scene() -> Image.Image:
    """黄昏公园风合成场景：天空渐变 + 地面 + 参考路灯杆（影子向右暗示光从左来）。"""
    arr = np.zeros((H, W, 3), dtype=np.float32)

    # 天空：左上更亮（光从左上来）
    yy, xx = np.mgrid[0:H, 0:W]
    sky_mask = yy < HORIZON
    sky = np.zeros((H, W, 3), dtype=np.float32)
    for c, (hi, lo) in enumerate([(255, 205), (218, 160), (190, 130)]):
        t = (xx / W) * 0.6 + (1 - yy / H) * 0.4          # 左上亮
        sky[..., c] = lo + (hi - lo) * (1 - t)
    # 地面：近处暗、远处亮（透视收束），左亮右暗
    ground = np.zeros((H, W, 3), dtype=np.float32)
    depth = np.clip((yy - HORIZON) / max(H - HORIZON, 1), 0, 1)  # 0 远 1 近
    for c, (far, near) in enumerate([(150, 92), (132, 78), (108, 62)]):
        ground[..., c] = far + (near - far) * depth
    ground *= (1.08 - 0.18 * (xx / W))[..., None]         # 左侧受光略亮
    arr = np.where(sky_mask[..., None], sky, ground)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img, "RGBA")

    # 参考物体：路灯杆（在角色右侧远处），自带向右下的柔和影子 -> 暗示光从左来
    px, ptop, pbot = int(W * 0.80), HORIZON - 60, int(H * 0.86)
    d.rectangle([px - 4, ptop, px + 4, pbot], fill=(52, 50, 58, 255))
    d.ellipse([px - 10, ptop - 14, px + 10, ptop + 6], fill=(70, 66, 80, 255))
    sh = Image.new("L", (W, H), 0)
    ds = ImageDraw.Draw(sh)
    ds.rectangle([px - 6, pbot - 5, px + 180, pbot + 5], fill=255)
    ds.ellipse([px - 6, pbot - 5, px + 6, pbot + 5], fill=255)
    ds.ellipse([px + 168, pbot - 5, px + 188, pbot + 5], fill=255)
    sh = sh.rotate(-10, center=(px, pbot), resample=Image.BILINEAR)
    sh = sh.filter(ImageFilter.GaussianBlur(6))
    m = np.asarray(sh, dtype=np.float32) / 255.0 * 0.35
    base = np.asarray(img, dtype=np.float32)
    img = Image.fromarray(np.clip(base * (1 - m[..., None]), 0, 255).astype(np.uint8))

    # 轻噪声，避免死平
    n = np.asarray(img, dtype=np.float32) + np.random.default_rng(7).normal(0, 2.2, (H, W, 3))
    return Image.fromarray(np.clip(n, 0, 255).astype(np.uint8), "RGB")


def make_character() -> Image.Image:
    """占位角色剪影（深藏青身体 + 雾蓝发 + 鲸尾），显式替代模型输出。

    注意：正式链路里这一步由 S2 的 2511 输出 + BiRefNet 抠像提供。
    剪影故意做成"一眼假"的几何形状——消融对比看的是几何后处理差异，
    不是角色好不好看。
    """
    cw, ch = 240, 420
    img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    # 鲸尾（向右侧后方延伸——正是"正面物理不可见、必须侧面参考图"的那个特征）
    d.polygon([(cw * 0.52, ch * 0.62), (cw * 0.97, ch * 0.42), (cw * 0.95, ch * 0.60),
               (cw * 0.55, ch * 0.78)], fill=(70, 96, 128, 255))
    # 身体（藏青女仆裙剪影）
    d.rounded_rectangle([cw * 0.18, ch * 0.34, cw * 0.72, ch * 0.97],
                        radius=int(cw * 0.16), fill=(38, 52, 84, 255))
    # 头（大——Q 版头身比）
    d.ellipse([cw * 0.10, ch * 0.02, cw * 0.80, ch * 0.40], fill=(216, 228, 240, 255))
    # 头发（雾蓝渐变的几何近似：两个蓝色块）
    d.pieslice([cw * 0.06, ch * 0.00, cw * 0.84, ch * 0.44], 180, 360, fill=(108, 148, 190, 255))
    d.polygon([(cw * 0.10, ch * 0.18), (cw * 0.02, ch * 0.52), (cw * 0.24, ch * 0.34)],
              fill=(108, 148, 190, 255))
    # 眼睛两个点
    d.ellipse([cw * 0.28, ch * 0.20, cw * 0.36, ch * 0.26], fill=(30, 40, 60, 255))
    d.ellipse([cw * 0.50, ch * 0.20, cw * 0.58, ch * 0.26], fill=(30, 40, 60, 255))
    return img


def main() -> None:
    import os
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out, exist_ok=True)

    scene = make_scene()
    char = make_character()

    # 角色放在画面中下部。脚点定在 0.86H 而不是 0.95H —— 初版放在 95% 高度时，
    # 叠加 12 度地面透视偏移后投射阴影整条跑出画框底部，肉眼只看得到 AO 团，
    # 这是"影子没画对"和"影子被截断"的混淆源，实测发现后调整。
    box = (W * 0.30, H * 0.86 - 280, W * 0.30 + 190, H * 0.86)

    light = LightingState(
        light_azimuth_deg=180.0,   # 光从左来（与场景里灯杆影子方向一致）
        elevation_deg=42.0,
        quality_hardness=0.5,
        intensity=0.85,
        confidence=0.78,
        gravity_deg=4.0,           # 模拟轻微仰拍（手机随手拍最常见）
        gravity_source="plane_normal",
    )

    scene.save(os.path.join(out, "scene.jpg"), quality=92)

    combos = []
    for r in (False, True):
        for df in (False, True):
            for sh in (False, True):
                combos.append((r, df, sh))

    panels = []
    for r, df, sh in combos:
        img = compose(scene, char, box, light,
                      use_rotation=r, use_defocus=df, use_shadow=sh)
        tag = f"R{'+' if r else '-'} D{'+' if df else '-'} S{'+' if sh else '-'}"
        panel = img.resize((512, 384), Image.LANCZOS)
        d = ImageDraw.Draw(panel)
        d.rectangle([8, 8, 190, 34], fill=(0, 0, 0, 160))
        d.text((16, 14), tag, fill=(255, 255, 255))
        panels.append(panel)
        if not (r or df or sh):
            img.save(os.path.join(out, "baseline.jpg"), quality=92)
        if r and df and sh:
            img.save(os.path.join(out, "full.jpg"), quality=92)

    grid = Image.new("RGB", (512 * 4, 384 * 2), (24, 24, 24))
    for i, p in enumerate(panels):
        grid.paste(p, ((i % 4) * 512, (i // 4) * 384))
    grid.save(os.path.join(out, "ablation_grid.jpg"), quality=90)

    print("done ->", out)
    print("lighting:", light.as_dict())
    print("prompt fragment:", light.serialize_prompt())


if __name__ == "__main__":
    main()
