"""step18：验证"规划器从已有立绘里选动作"这件事到底成不成立。

## 为什么要先做这一步（不出图）

用户报的 bug 是"动作不匹配"：`guided_out2.jpg` 里规划器说"坐在长椅座面上"，
贴上去的立绘却是**站姿**，成品成了脚悬在长椅前沿外面的半空站姿。

根因不是规划器判断错，是**它自由创作动作、而引导图只能用已缓存的立绘**——
两个信号打架，模型各听一半。

所以改法是把候选清单喂给规划器，让它只能从中选。这个改动对不对，
只需要看 VLM 的输出就知道，**不必先花 0.12 元生图**。

## 要看什么

1. `action_id` 是否命中候选（`_action_source` == plan）；
2. 规划器在"有座面的照片"里是否选坐姿、"有腿的照片"里是否选腿边动作；
3. `region[3]`（框底 = 落点）是否落在依托物上——这是程序唯一会用的定位量。

跑法：
    python experiments/step18_action_match.py            # 跑全部
    python experiments/step18_action_match.py p05        # 只跑名字含 p05 的
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from app.pipeline import available_sprite_options          # noqa: E402
from s3.characters import get_character                    # noqa: E402
from s3.plan import plan_scene                             # noqa: E402

SCENES = os.path.join(ROOT, "assets", "scenes", "batch2")
MODEL = os.environ.get("ARK_IMAGE_MODEL", "doubao-seedream-5-0-flash-260915")


def main() -> int:
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(f for f in os.listdir(SCENES) if f.lower().endswith(".jpg"))
    if pick:
        files = [f for f in files if pick in f]
    if not files:
        print(f"没找到场景图：{SCENES}")
        return 1

    char = get_character("chibi")
    options = available_sprite_options(MODEL, char)
    print(f"角色 {char.id}  身高 {char.height_cm:.0f}cm  "
          f"scale={char.scale:.3f}")
    print(f"候选立绘 {len(options)} 个：{', '.join(o['id'] for o in options)}\n")

    rows = []
    for i, fn in enumerate(files, 1):
        path = os.path.join(SCENES, fn)
        t0 = time.time()
        plan = plan_scene(path, height_hint="", options=options)
        dt = time.time() - t0
        src = plan.raw.get("_action_source", "-")
        bottom = plan.region[3] if plan.region else None
        h_pct = (plan.region[3] - plan.region[1]) / 999.0 * 100 if plan.region else 0
        rows.append((fn, plan, src, bottom, h_pct, dt))
        print(f"[{i}/{len(files)}] {fn}  {dt:.1f}s")
        if not plan.ok:
            print(f"    !! 规划失败 raw={plan.raw.get('error')}")
            continue
        print(f"    action_id = {plan.action_id!r}  (来源 {src})")
        print(f"    anchor    = {plan.anchor_kind}  依托 {plan.region_name}")
        print(f"    落点 y    = {bottom}/999  → 框高 {h_pct:.1f}% 画面  区 {plan.region}")
        print(f"    动作      = {plan.action}")
        print()

    # ---- 汇总 ----
    ok = [r for r in rows if r[1].ok]
    hit = [r for r in ok if r[0] and r[1].action_id and r[2] in ("plan", "plan_name")]
    print("=" * 64)
    print(f"规划成功 {len(ok)}/{len(rows)}；action_id 直接命中候选 {len(hit)}/{len(ok)}")
    from collections import Counter
    c = Counter(r[1].action_id for r in ok)
    print("动作分布：" + "、".join(f"{k}×{v}" for k, v in c.most_common()))
    miss = [r[0] for r in ok if r[2] not in ("plan", "plan_name")]
    if miss:
        print(f"没直接命中（走了兜底）：{', '.join(miss)}")
        for fn, p, src, *_ in ok:
            if src not in ("plan", "plan_name"):
                print(f"  - {fn}: 模型给 {p.raw.get('action_id')!r} → 兜成 "
                      f"{p.action_id!r}（{src}）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
