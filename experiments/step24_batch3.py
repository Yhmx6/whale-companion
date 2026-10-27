"""step24：在 10 个新场景上跑完整的布局引导，重点验避让。

## 和 step19 的差别

step19 是"单张、看效果"；这一步是"批量、看规律"：
十张一起跑，把每张的**动作 / 落点挪了多少 / 撞到了什么 / 尺寸偏多少**
摊在一张表里，才能判断这些机制是真在起作用，还是碰巧。

## 三个判据

| 判据 | 看哪 | 及格线 |
|---|---|---|
| 避让生效 | `moved_px` 与 `conflicts` | 有障碍的 7 张应当挪开、0 冲突 |
| 不误伤 | 无碍的 3 张的 `moved_px` | 应当接近 0 |
| 动作匹配 | `sprite_action` + `action_source` | 全部 `plan`，且和场景说得通 |

尺寸仍然会偏大（模型的下限，见 `GUIDED_SIZE_FLOOR`），这一步只记录不追求。

跑法：
    python experiments/step24_batch3.py            # 全部
    python experiments/step24_batch3.py c01 c05    # 挑几张
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from app.pipeline import OUT_ROOT, run_guided     # noqa: E402

SCENES = os.path.join(ROOT, "assets", "scenes", "batch3")
OUTPUT = os.path.join(ROOT, "output")
RESULT = os.path.join(OUTPUT, "batch3")

# 每张图是不是**故意**放了障碍物。用来检查"该挪的挪了、不该挪的没动"。
HAS_OBSTACLE = {
    "c01_cafe_cup": True, "c02_sofa_books": True, "c03_desk_laptop": True,
    "c04_dining_vase": True, "c05_bench_empty": False, "c06_steps_empty": False,
    "c07_windowsill_plant": True, "c08_bed_phone": True,
    "c09_bar_two_cups": True, "c10_floor_backpack": True,
}


def main() -> int:
    pick = sys.argv[1:]
    files = sorted(glob.glob(os.path.join(SCENES, "*.png")))
    if pick:
        files = [f for f in files if any(p in f for p in pick)]
    if not files:
        print(f"没有场景图：{SCENES}")
        return 1

    os.makedirs(RESULT, exist_ok=True)
    rows = []
    t_all = time.time()
    for i, path in enumerate(files, 1):
        tag = os.path.splitext(os.path.basename(path))[0]
        print(f"\n[{i}/{len(files)}] {tag}")
        with open(path, "rb") as f:
            data = f.read()
        t0 = time.time()
        res = run_guided(data, character="chibi", progress=lambda s, p: None)
        dt = time.time() - t0
        if res.get("error"):
            print(f"    !! {res['error']}")
            rows.append({"tag": tag, "error": res["error"], "sec": round(dt, 1)})
            continue

        pl = res["placement"]
        col = pl.get("collision") or {}
        dst = os.path.join(RESULT, tag)
        os.makedirs(dst, exist_ok=True)
        src = os.path.join(OUT_ROOT, res["run_id"])
        for fn in ("scene.jpg", res["guide"], "scene2d.jpg", "out_0.jpg"):
            sp = os.path.join(src, fn)
            if os.path.exists(sp):
                shutil.copy2(sp, os.path.join(dst, fn))
        with open(os.path.join(dst, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        row = {
            "tag": tag,
            "sec": round(dt, 1),
            "action": pl.get("sprite_action"),
            "src": pl.get("action_source"),
            "moved": round(col.get("moved_px") or 0, 0),
            "sits_on": (col.get("sits_on") or {}).get("name") if col.get("sits_on") else None,
            "conflicts": [c.get("name") or c.get("kind") for c in (col.get("conflicts") or [])],
            "plan_pct": pl.get("h_pct"),
            "meas_pct": pl.get("measured_pct"),
            "err": pl.get("size_err"),
            "objs": len(res.get("scene2d", {}).get("objects") or []),
            "expect_move": HAS_OBSTACLE.get(tag),
            "warnings": res.get("warnings") or [],
        }
        rows.append(row)
        flag = ""
        if row["expect_move"] and not row["moved"] and not row["conflicts"]:
            flag = "  ← 有障碍却没挪（可能没检出）"
        if row["conflicts"]:
            flag = "  ← 挪不开，会穿模"
        if row["expect_move"] is False and row["moved"] > 12:
            flag = "  ← 无碍却动了（误伤）"
        print(f"    动作 {row['action']}（{row['src']}）  物体 {row['objs']} 个")
        print(f"    避让 挪 {row['moved']:.0f}px  坐于 {row['sits_on']}"
              f"  冲突 {row['conflicts'] or '无'}")
        print(f"    尺寸 {row['plan_pct']}% → {row['meas_pct']}%（{row['err']}×）"
              f"  {dt:.0f}s{flag}")

    ok = [r for r in rows if "error" not in r]
    print("\n" + "=" * 78)
    print(f"{'场景':<22}{'动作':<19}{'挪动':>7}{'冲突':>6}"
          f"{'计划':>7}{'实测':>7}{'倍率':>6}")
    for r in ok:
        print(f"{r['tag']:<22}{str(r['action']):<19}{r['moved']:>6.0f}px"
              f"{len(r['conflicts']):>6}{r['plan_pct']:>7.1f}"
              f"{r['meas_pct'] or 0:>7.1f}{r['err'] or 0:>6.2f}")
    print("=" * 78)

    ob = [r for r in ok if r["expect_move"]]
    no = [r for r in ok if r["expect_move"] is False]
    moved = [r for r in ob if r["moved"] > 8]
    print(f"有障碍的 {len(ob)} 张里，挪开了 {len(moved)} 张；"
          f"无碍的 {len(no)} 张里，误伤 {sum(1 for r in no if r['moved'] > 12)} 张")
    hard = [r for r in ok if r["conflicts"]]
    print(f"挪不开（会穿模）：{len(hard)} 张"
          + (f"  {', '.join(r['tag'] for r in hard)}" if hard else ""))
    miss = [r for r in ob if r["moved"] <= 8 and not r["conflicts"]]
    print(f"有障碍但没挪：{len(miss)} 张"
          + (f"  {', '.join(r['tag'] for r in miss)}" if miss else ""))
    print(f"\n总用时 {time.time() - t_all:.0f}s")

    with open(os.path.join(RESULT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"明细 -> {os.path.join(RESULT, 'report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
