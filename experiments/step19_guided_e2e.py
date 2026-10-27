"""step19：端到端跑一次布局引导，验证「60cm + 规划器落点 + 动作匹配」。

## 这一次要三个同时成立

用户对 `guided_out2.jpg` 的抱怨是"动作不匹配"，并追加了一条产品规定：
**她 60cm 高，写进提示词**。所以验收要同时看三件事，缺一不算过：

| 项目 | 判据 | 看哪里 |
|---|---|---|
| 尺寸 | 人物照里 `h_pct` ≈ `person_h × 60/175 / H` | 输出 meta |
| 位置 | 落点压在她该坐/该站的依托物上，不是浮空 | 看 guide 和成品 |
| 动作 | 立绘姿态 = 提示词姿态 = 照片里说得通的姿态 | `action_source` == plan |

## 产出

`output/guided2_<场景名>/` 下：
- `scene.jpg`   原照片
- `guide.jpg`   贴了立绘的引导图（**这一张先看尺寸位置对不对**）
- `out_0.jpg`   成品

引导图先看：如果引导图里她的大小和落点就不对，那问题在程序，
根本不用看成品——成品只会把同样的错画得更漂亮。

跑法：
    python experiments/step19_guided_e2e.py p05
    python experiments/step19_guided_e2e.py p05 e05     # 多张
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from app.pipeline import OUT_ROOT, run_guided     # noqa: E402

SCENES = os.path.join(ROOT, "assets", "scenes", "batch2")
OUTPUT = os.path.join(ROOT, "output")


def find(name: str) -> str | None:
    for f in os.listdir(SCENES):
        if name in f and f.lower().endswith(".jpg"):
            return os.path.join(SCENES, f)
    return None


def main() -> int:
    names = sys.argv[1:] or ["p05"]
    total_t0 = time.time()
    results = []
    for name in names:
        path = find(name)
        if not path:
            print(f"[skip] 找不到场景 {name}")
            continue
        tag = os.path.splitext(os.path.basename(path))[0]
        print(f"\n=== {tag} ===")
        with open(path, "rb") as f:
            data = f.read()
        t0 = time.time()
        res = run_guided(data, character="chibi",
                         progress=lambda s, p: print(f"  {p:3d}%  {s}"))
        dt = time.time() - t0
        if res.get("error"):
            print(f"  !! {res['error']}")
            results.append((tag, None, dt))
            continue
        pl = res["placement"]
        print(f"  用时 {dt:.1f}s  {res['size']}")
        print(f"  动作 {pl['sprite_action']}  (来源 {pl['action_source']})")
        print(f"  尺寸 {pl['target_h']}px = 画面高 {pl['h_pct']:.1f}%  "
              f"(依据 {pl['size_basis']}, 人物 {pl['person_h']}px)")
        print(f"  落点 box={pl['box']}  来源 {pl['source']}")
        print(f"  姿态 {pl['pose'][:80]}…")

        # 把这次的三张关键图单独归一份，方便人工比对
        src_dir = os.path.join(OUT_ROOT, res["run_id"])
        dst = os.path.join(OUTPUT, f"guided2_{tag}")
        os.makedirs(dst, exist_ok=True)
        for fn in ("scene.jpg", res["guide"], "scene2d.jpg", "out_0.jpg"):
            s = os.path.join(src_dir, fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, fn))
        with open(os.path.join(dst, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"  → output/guided2_{tag}/")
        results.append((tag, res, dt))

    print(f"\n总用时 {time.time() - total_t0:.1f}s")
    fail = [t for t, r, _ in results if not r]
    print(f"成功 {len(results) - len(fail)}/{len(results)}"
          + (f"  失败：{', '.join(fail)}" if fail else ""))
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
