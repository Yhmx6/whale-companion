"""S3 组合流水线：场景 -> 阴影 -> 角色贴入（重力对齐 + 散焦）。

合成顺序刻意为「先画影子、再贴角色」：角色必须遮挡自己脚边的影子，
这是"站在影子上"而不是"漂在影子上"的关键。

消融开关（旋转/散焦/阴影）用于 8 组对照实验（U-13）：
产出"每项各自贡献多少"，作为 QC 权重分配的依据。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .lighting import LightingState
from .s3ops import apply_defocus, apply_gravity, render_shadow_matte


def compose(scene: Image.Image, char_rgba: Image.Image,
            box: tuple[float, float, float, float],
            light: LightingState,
            use_rotation: bool = True,
            use_defocus: bool = True,
            use_shadow: bool = True,
            focus_y_frac: float = 0.62,
            fov_compress: float = 1.0) -> Image.Image:
    """scene: RGB 图；char_rgba: 带透明通道的角色图（会被缩放到 box 尺寸）。"""
    w_img, h_img = scene.size
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    char = char_rgba.resize((max(1, int(bw)), max(1, int(bh))), Image.LANCZOS)

    # 1) 角色放到整幅透明层上（避免旋转时被自身画布裁切）
    layer = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
    layer.paste(char, (int(box[0]), int(box[1])), char)

    # 2) S3.1 重力对齐
    if use_rotation:
        layer = apply_gravity(layer, box, light.gravity_deg)

    # 3) S3.2 深度匹配散焦
    if use_defocus:
        layer = apply_defocus(layer, box, focus_y=focus_y_frac * h_img)

    # 4) S3.3 接触阴影画在场景上（先于角色）
    scene_np = np.asarray(scene.convert("RGB"), dtype=np.float32)
    if use_shadow:
        dark = render_shadow_matte((w_img, h_img), box, light, fov_compress)
        scene_np = scene_np * (1.0 - dark[..., None])

    # 5) 角色贴入（含 alpha 合成）
    base = Image.fromarray(np.clip(scene_np, 0, 255).astype(np.uint8), "RGB")
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
