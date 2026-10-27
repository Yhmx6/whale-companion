"""测试 Seedream 5.0 的「坐标定位编辑」能否用来单独修手。

## 背景

检索到 Seedream 5.0（Pro / Flash 同一套能力）支持在提示词里用归一化坐标指定编辑区域：
    Replace the area Image 1 120 180 640 760 with ...
    keep Image 1 700 120 920 360 unchanged
坐标归一化到 0-999。

如果这套语法在方舟直连也成立，那"修手"就变成一次便宜的局部重绘，
不需要 ControlNet、不需要换模型、不需要 GPU——成本等于一张图。

## 设计

拿第 4 批里"扯衣角"的失败图（手没捏住布料），做两件事：
1. 用差分掩码算出角色外接框，换算成 0-999 坐标；
2. 只对该区域发一次编辑指令：要求"同一角色、但她的手牢牢捏住衣角、布料被拉出褶皱"，
   并显式要求其余区域不变。

1 次调用，0.12 元。
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from api.ark import download, generate, image_path_to_data_uri  # noqa: E402
from s3.beautify import change_mask  # noqa: E402

OUT = os.path.join(ROOT, "output", "step6")
FLASH = "doubao-seedream-5-0-flash-260915"
SCENES = os.path.join(ROOT, "assets", "scenes", "selfie")

# 第 4 批 pull_hem 的结果
STEP4 = json.load(open(os.path.join(ROOT, "output", "step4", "matrix_manifest.json"),
                       encoding="utf-8"))
TARGET = next(x for x in STEP4 if x["action"] == "pull_hem" and x.get("file"))


def norm_bbox(img_path: str, scene_path: str) -> list[int]:
    """用差分掩码得到角色框，换算成 0-999 归一化坐标。"""
    res = Image.open(img_path).convert("RGB")
    scene = Image.open(scene_path).convert("RGB").resize(res.size, Image.LANCZOS)
    mask = change_mask(scene, res)
    ys = mask.any(axis=1)
    xs = mask.any(axis=0)
    import numpy as np
    y0, y1 = int(np.argmax(ys)), int(len(ys) - np.argmax(ys[::-1]))
    x0, x1 = int(np.argmax(xs)), int(len(xs) - np.argmax(xs[::-1]))
    W, H = res.size
    return [round(x0 / W * 999), round(y0 / H * 999),
            round(x1 / W * 999), round(y1 / H * 999)]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(ROOT, "output", "runs", TARGET["run"], TARGET["file"])
    scene = os.path.join(SCENES, TARGET["base"] + ".jpg")
    box = norm_bbox(src, scene)
    print(f"角色区域（0-999 归一化）: {box}")

    prompt = (
        f"只修改 Image 1 里 {box[0]} {box[1]} {box[2]} {box[3]} 这块区域。\n"
        "在这个区域里，保持画面里那个蓝发 Q 版小人完全不变——同样的发型、同样的鲸鱼耳、"
        "同样的藏青女仆裙和白色围裙、同样的短手短腿、同样的大小和位置。"
        "唯一要改的是她的**手**：让她的一只手牢牢捏住人物上衣的下摆衣角，"
        "布料被捏出一个小小的三角形褶皱，能看到布料被拉住的感觉。\n"
        "其余区域（人物、背景、地面、光线）保持完全不变，不要重绘。"
        "不要改变人物本身，不要出现多余的手指，不要出现多余的肢体。"
    )
    print("prompt 长度:", len(prompt))

    status, resp = generate(prompt, images=[image_path_to_data_uri(src)],
                            size=f"{Image.open(src).width}x{Image.open(src).height}",
                            model=FLASH)
    print("HTTP", status)
    if status != 200:
        print("失败:", str(resp)[:600])
        return
    out = os.path.join(OUT, "pull_hem_handfix.png")
    download(resp["data"][0]["url"], out)
    print("->", out, Image.open(out).size)

    meta = dict(box=box, prompt=prompt, src=src, out=out, model=FLASH,
                usage=resp.get("usage"))
    with open(os.path.join(OUT, "handfix_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
