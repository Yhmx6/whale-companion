"""第六批：新设计的 5 个动作 + 腿部位置约束验证。

按「支撑方式 × 手部呈现」重新设计的动作，全部围绕两个新思路：
1. 把手"藏起来"或"换成有自身结构的目标"（抱自己膝盖），而不是让模型去抓别人；
2. 用户要求：全身照把角色约束在人物腿部附近（VLM 定位腿部 → 归一化坐标写进提示词）。

5 个动作 × 1 次 = 5 次调用，Flash，约 0.6 元。
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image, ImageDraw  # noqa: E402

from app import pipeline  # noqa: E402
from s3.actions import get_action  # noqa: E402

SCENES = os.path.join(ROOT, "assets", "scenes", "selfie")
OUT = os.path.join(ROOT, "output", "step7")
FLASH = "doubao-seedream-5-0-flash-260915"

TRIALS = [
    ("leg_hands_behind", "s04_full"),
    ("leg_hug_knee", "s04_full"),
    ("leg_lean", "s05_walk"),
    ("foot_sit", "s05_walk"),
    ("hem_peek", "s01_mirror"),
]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    records = []
    for act_id, base in TRIALS:
        path = os.path.join(SCENES, f"{base}.jpg")
        a = get_action(act_id)
        if not os.path.exists(path) or not a:
            print(f"[跳过] {act_id}/{base}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            r = pipeline.run(
                photo_bytes=open(path, "rb").read(),
                box_norm=(0.4, 0.45, 0.14, 0.2),
                surface="ground", light_key="auto", candidates=1,
                mode="companion", action=act_id, model=FLASH,
            )
        except Exception as e:
            print(f"[ERR] {act_id}: {type(e).__name__}: {e}", flush=True)
            records.append(dict(action=act_id, error=str(e)))
            continue
        dt = time.perf_counter() - t0
        c = (r.get("candidates") or [{}])[0]
        rec = dict(action=act_id, base=base, hand_mode=a.hand_mode, support=a.support,
                   anchor=a.anchor_region, elapsed=round(dt, 1),
                   run=r["run_id"], file=c.get("file"),
                   shadow=c.get("shadow"), scores=c.get("scores"))
        records.append(rec)
        print(f"[OK] {act_id:18s} {a.hand_mode:9s} {dt:5.1f}s shadow={c.get('shadow')}", flush=True)

    with open(os.path.join(OUT, "v2_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    cells = [(r["action"], r.get("hand_mode", ""), os.path.join(ROOT, "output", "runs",
              r["run"], r["file"])) for r in records if r.get("file")]
    CW, CH, cols = 380, 507, 3
    rows = (len(cells) + cols - 1) // cols
    grid = Image.new("RGB", (CW * cols, (CH + 26) * rows), (18, 18, 18))
    for i, (act, hm, p) in enumerate(cells):
        im = Image.open(p).convert("RGB")
        im.thumbnail((CW, CH), Image.LANCZOS)
        x = (i % cols) * CW + (CW - im.width) // 2
        y = (i // cols) * (CH + 26) + 26
        grid.paste(im, (x, y))
        ImageDraw.Draw(grid).text(((i % cols) * CW + 8, (i // cols) * (CH + 26) + 8),
                                  f"{act}  [{hm}]", fill=(240, 240, 240))
    out = os.path.join(OUT, "v2_actions.jpg")
    grid.save(out, quality=92)
    print(f"\n完成 {len(cells)}/{len(TRIALS)}  花费约 {len(cells) * 0.12:.2f} 元")
    print("联系表 ->", out)


if __name__ == "__main__":
    main()
