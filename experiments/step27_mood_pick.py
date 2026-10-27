"""step27：验证「表情会不会跟着场景变」。

## 只看 VLM 的输出，不生成图

表情改得对不对，生图那一步要花钱（0.12 元 / 张）。但**规划器选得对不对**
不用花钱就能验：它输出的 `mood_id` 就是答案。

如果 10 张图它全选 `happy`，那说明提示词没起作用，后面生图全是白花钱。
所以顺序是：先验选择，再生产。

## 三个判据

| 判据 | 及格线 |
|---|---|
| 命中候选 | 全部 `_mood_source == plan`（模型直接用对了 id，不是兜底）|
| **多样性** | 10 张里至少出现 4 种不同表情。全是一个 = 没按场景选 |
| 说得通 | `mood_reason` 要提到具体的东西（光线、人物在做什么），不是套话 |

多样性和"说得通"要人看，脚本只把数据摊出来，最后打印直方图。

跑法：
    python experiments/step27_mood_pick.py            # batch3 全部
    python experiments/step27_mood_pick.py batch2     # 换成 batch2 的场景
"""

from __future__ import annotations

import glob
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from app.pipeline import available_sprite_options      # noqa: E402
from s3.characters import get_character                # noqa: E402
from s3.moods import for_scene, get_mood               # noqa: E402
from s3.plan import plan_scene                         # noqa: E402

MODEL = os.environ.get("ARK_IMAGE_MODEL", "doubao-seedream-5-0-flash-260915")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "batch3"
    files = sorted(glob.glob(os.path.join(
        ROOT, "assets", "scenes", which, "*.png"))) or \
        sorted(glob.glob(os.path.join(
            ROOT, "assets", "scenes", which, "*.jpg")))
    if not files:
        print(f"没有图：assets/scenes/{which}")
        return 1

    options = available_sprite_options(MODEL, get_character("chibi"))
    print(f"动作候选 {len(options)} 个\n")

    rows = []
    for i, path in enumerate(files, 1):
        tag = os.path.splitext(os.path.basename(path))[0]
        # 和 run_guided 完全一样的收窄逻辑：先按场景算短名单，再让规划器挑
        from PIL import Image as _I
        _im = _I.open(path)
        moods = for_scene(_im, has_person=False, objects=[],
                          action_ids=[o["id"] for o in options])
        plan = plan_scene(path, options=options, moods=moods)
        if not plan.ok:
            print(f"[{i}/{len(files)}] {tag}  规划失败 {plan.raw.get('error')}")
            continue
        mo = get_mood(plan.mood_id)
        rows.append((tag, plan.mood_id, mo.name if mo else "—",
                     plan.mood_reason))
        print(f"[{i}/{len(files)}] {tag}")
        print(f"    候选 {[m['id'] for m in moods]}")
        print(f"    动作 {plan.action_id}   表情 {plan.mood_id}"
              f"（{mo.name if mo else '未选'}，来源 {plan.raw.get('_mood_source')}）")
        print(f"    瞬间 {plan.moment or '（空）'}")
        print(f"    理由 {plan.mood_reason or '（空）'}")
        print()

    print("=" * 70)
    c = Counter(r[1] for r in rows)
    print(f"表情分布（{len(rows)} 张，{len(c)} 种）：")
    for mid, n in c.most_common():
        mo = get_mood(mid)
        print(f"  {mid:12s} {mo.name if mo else '—':6s} ×{n}")
    ok = sum(1 for r in rows if r[1])
    print(f"\n选出有效表情 {ok}/{len(rows)}")
    print("判据：≥4 种表情算通过" + ("（通过）" if len(c) >= 4 else "（不通过，需要改提示词）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
