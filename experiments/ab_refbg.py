"""参考图底色 A/B：确认角色边缘白色光晕的根因。

背景：5 案例 10 张图里，角色边缘全部出现一圈白/亮描边，贴纸感极重。
最可能的原因是参考图是**纯白底**——白色被一起带进了合成。

三组对照（同一张照片、同一句提示词、同 seed）：
  A 白底 PNG     —— 现状基线（已有结果，不重复调用）
  B 透明底 RGBA  —— 测"是否保留 alpha 就能消除白边"
  C 中性灰底      —— 测"把底色换成 128 灰是否更稳"（多数编辑模型对纯白敏感）

只跑 B 和 C，各 1 张，约 0.6 元。
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from api.ark import download, generate, image_path_to_data_uri  # noqa: E402
from experiments.step1_user import CASES, build_prompt, pick_size  # noqa: E402

MODEL = "doubao-seedream-5-0-pro-260628"
OUT = os.path.join(ROOT, "output", "step1", "ab_refbg")
REFS = os.path.join(ROOT, "assets", "refs", "crops")
SCENES = os.path.join(ROOT, "assets", "scenes", "user")


def variant_ref(name: str, mode: str) -> Image.Image:
    """按模式生成参考图：transparent / gray。用于本地核对实际送出的像素。"""
    src = Image.open(os.path.join(REFS, f"05_chibi_{name}_rgba.png")).convert("RGBA")
    if mode == "gray":
        bg = Image.new("RGB", src.size, (128, 128, 128))
        bg.paste(src, (0, 0), src)
        return bg
    return src


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    case = next(c for c in CASES if c["key"] == "u01_kitchen")
    scene_path = os.path.join(SCENES, case["file"])
    im = Image.open(scene_path)
    size, _ = pick_size(im)
    prompt = build_prompt(case)

    variants = {
        "B_transparent": dict(keep_alpha=True, bg=None),
        "C_gray": dict(keep_alpha=True, bg=(128, 128, 128)),
    }

    scene_uri = image_path_to_data_uri(scene_path)
    for tag, kw in variants.items():
        f = os.path.join(REFS, "05_chibi_front_rgba.png")
        s = os.path.join(REFS, "05_chibi_side_rgba.png")
        # 存一份"实际送出的参考图"便于事后核对
        variant_ref("front", "gray" if kw["bg"] else "transparent").save(
            os.path.join(OUT, f"ref_front_{tag}.png"))
        imgs = [scene_uri,
                image_path_to_data_uri(f, **kw),
                image_path_to_data_uri(s, **kw)]
        print(f"\n[{tag}] 送参考图 {len(imgs)} 张，size={size}")
        status, resp = generate(prompt, images=imgs, size=size, model=MODEL, seed=1000)
        if status != 200:
            print("  失败:", str(resp)[:400])
            continue
        out = os.path.join(OUT, f"u01_{tag}.png")
        download(resp["data"][0]["url"], out)
        print("  ->", os.path.basename(out), Image.open(out).size)


if __name__ == "__main__":
    main()
