"""第四批：动作矩阵——把动作库里每个待验证的动作都跑一遍，做可用性分类。

用 Flash（0.12 元/张、约 14 秒），因为这一批的目的是**筛动作**，不是出终稿。
筛出来的赢家再用 Pro 复跑。

底图：
- 已有的三张自拍（镜子半身 / 公园双人 / 沙发半身）
- 新增两张全身照，否则"抱小腿、扯裤脚、抱大腿"这类动作没有测试条件
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

from PIL import Image  # noqa: E402

from api.ark import download, generate  # noqa: E402
from app import pipeline  # noqa: E402
from s3.actions import get_action  # noqa: E402

SCENES = os.path.join(ROOT, "assets", "scenes", "selfie")
OUT = os.path.join(ROOT, "output", "step4")
PRO = "doubao-seedream-5-0-pro-260628"
FLASH = "doubao-seedream-5-0-flash-260915"

NEW_BASES = {
    "s04_full": (
        "手机全身照，一位年轻女生站在秋天的公园小路上，正面朝向镜头，穿白色卫衣和牛仔裤，"
        "双手自然垂在身侧，脚穿白色运动鞋，画面从头顶到脚完整入画，"
        "背景是银杏树和小路，阳光从画面左侧照来，真实照片质感，没有文字没有水印"
    ),
    "s05_walk": (
        "手机全身照，一位年轻男生站在室内走廊里，正面朝向镜头，穿深色卫衣和休闲裤，"
        "双手插在口袋里，脚穿深色球鞋，画面从头顶到脚完整入画，"
        "背景是白色墙面和木地板，室内柔和的自然光，真实照片质感，没有文字没有水印"
    ),
}

# 动作 → 底图 的配对（选能提供所需身体部位的那张）
MATRIX = [
    ("hug_calf", "s04_full"),
    ("pull_trouser", "s04_full"),
    ("hug_thigh", "s04_full"),
    ("pull_hem", "s04_full"),
    ("hand_hold", "s04_full"),
    ("lap", "s05_walk"),
    ("shoulder_back", "s03_sofa"),
    ("on_head", "s03_sofa"),
    ("pocket", "s01_mirror"),
    ("hug_arm", "s02_park"),
    ("shoulder", "s03_sofa"),   # A 档基准，用来对照
    ("peek", "s03_sofa"),       # A 档基准
]


def ensure_bases() -> None:
    os.makedirs(SCENES, exist_ok=True)
    for key, prompt in NEW_BASES.items():
        p = os.path.join(SCENES, f"{key}.jpg")
        if os.path.exists(p):
            continue
        print(f"[底图] {key} ...", flush=True)
        st, resp = generate(prompt, size="1280x1704", model=PRO)
        if st != 200:
            print(f"  失败 {st}: {str(resp)[:200]}", flush=True)
            continue
        download(resp["data"][0]["url"], p)
        print(f"  -> {key} {Image.open(p).size}", flush=True)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ensure_bases()
    engine = os.environ.get("ENGINE", FLASH)
    records = []

    for act_id, base in MATRIX:
        act = get_action(act_id)
        path = os.path.join(SCENES, f"{base}.jpg")
        if act is None or not os.path.exists(path):
            print(f"[跳过] {act_id}/{base}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            r = pipeline.run(
                photo_bytes=open(path, "rb").read(),
                box_norm=(0.4, 0.4, 0.2, 0.3),
                surface="ground", light_key="auto", candidates=1,
                mode="companion", action=act_id, model=engine,
            )
        except Exception as e:
            print(f"[ERR] {act_id}: {type(e).__name__}: {e}", flush=True)
            records.append(dict(action=act_id, base=base, error=str(e)))
            continue
        dt = time.perf_counter() - t0
        c = (r.get("candidates") or [{}])[0]
        rec = dict(action=act_id, tier=act.tier, base=base, engine=engine,
                   elapsed=round(dt, 1), run=r["run_id"], file=c.get("file"),
                   scores=c.get("scores"))
        records.append(rec)
        print(f"[OK] {act_id:14s} tier={act.tier} {base:10s} {dt:5.1f}s {c.get('file')}", flush=True)

    with open(os.path.join(OUT, "matrix_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n完成 {len([r for r in records if r.get('file')])}/{len(MATRIX)}"
          f"  花费约 {len([r for r in records if r.get('file')]) * 0.12:.2f} 元")


if __name__ == "__main__":
    main()
