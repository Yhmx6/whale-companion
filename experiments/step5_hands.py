"""第五批：验证"简化手"能否解决贴贴失败问题。

## 假设

12 个动作的实测里，成败与"手是否需要被清晰描绘"高度相关：
- 成功的（坐肩膀、扒肩探头）：手搭在头发/肩线上，被遮住一部分或只是平贴
- 失败的（扯衣角、扯裤脚、牵手）：必须看清手指怎么抓

所以解法不是"让模型画对手"，而是**让它不用画手**。

## 设计

三个之前偏弱/失败的动作 × 两种提示词：
  A 原描述（复用第 4 批的结果，同 seed 2000，不重复调用）
  B 原描述 + HAND_LOCK 手部简化补丁

同底图、同 seed、同引擎（Flash），唯一变量是那 206 个字符。
3 次调用，约 0.36 元。
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

from app import pipeline  # noqa: E402

SCENES = os.path.join(ROOT, "assets", "scenes", "selfie")
OUT = os.path.join(ROOT, "output", "step5")
FLASH = "doubao-seedream-5-0-flash-260915"

# 三个待验证动作（第 4 批的判定：偏弱 / 待优化）
TRIALS = [
    ("pull_hem", "s04_full", "扯衣角（原判定：偏弱，手没捏住）"),
    ("hug_calf", "s04_full", "抱小腿（原判定：接近可用，看不出环抱）"),
    ("hug_arm", "s02_park", "抱手臂（原判定：待优化，像贴在旁边）"),
]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    records = []
    for act_id, base, why in TRIALS:
        path = os.path.join(SCENES, f"{base}.jpg")
        if not os.path.exists(path):
            print(f"[跳过] 缺底图 {base}")
            continue
        t0 = time.perf_counter()
        r = pipeline.run(
            photo_bytes=open(path, "rb").read(),
            box_norm=(0.4, 0.4, 0.2, 0.3),
            surface="ground", light_key="auto", candidates=1,
            mode="companion", action=act_id, model=FLASH,
            simplify_hands=True,
        )
        dt = time.perf_counter() - t0
        c = (r.get("candidates") or [{}])[0]
        rec = dict(action=act_id, base=base, why=why, variant="hand_lock",
                   elapsed=round(dt, 1), run=r["run_id"], file=c.get("file"))
        records.append(rec)
        print(f"[OK] {act_id:12s} {dt:5.1f}s -> {c.get('file')}", flush=True)

    with open(os.path.join(OUT, "hands_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # 拼对照图：上排=原版（第4批），下排=简化手
    from PIL import Image, ImageDraw
    prev = json.load(open(os.path.join(ROOT, "output", "step4", "matrix_manifest.json"),
                          encoding="utf-8"))
    prev_map = {x["action"]: x for x in prev if x.get("file")}
    CW, CH = 400, 533
    rows = len(records)
    grid = Image.new("RGB", (CW * 2, (CH + 30) * rows), (18, 18, 18))
    for i, rec in enumerate(records):
        a = prev_map.get(rec["action"])
        pairs = [(a["file"] and os.path.join(ROOT, "output", "runs", a["run"], a["file"]), "A 原描述"),
                 (os.path.join(ROOT, "output", "runs", rec["run"], rec["file"]), "B 简化手")]
        for j, (p, lab) in enumerate(pairs):
            if not p or not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((CW, CH), Image.LANCZOS)
            x = j * CW + (CW - im.width) // 2
            y = i * (CH + 30) + 30
            grid.paste(im, (x, y))
            d = ImageDraw.Draw(grid)
            d.text((j * CW + 8, i * (CH + 30) + 8), f"{rec['action']}  {lab}", fill=(240, 240, 240))
    out = os.path.join(OUT, "hands_ab.jpg")
    grid.save(out, quality=92)
    print("对照图 ->", out)
    print(f"花费约 {len(records) * 0.12:.2f} 元")


if __name__ == "__main__":
    main()
