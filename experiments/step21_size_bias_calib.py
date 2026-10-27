"""step21：标定「引导图该画多小」，让成品落在 60cm 上。

## 这一步在标什么

程序把 60cm 换算成像素是准的（`person_h × 60/175`）。不准的是**下游**：
生成模型重绘时会把她放大。实测三次分别放大 1.21 / 1.51 / 2.05 倍，
而且提示词怎么改都压不住（`GUIDED_PROMPT` 第 1、2 条都试过）。

所以改成反向补偿：引导图里把她画成 `target_h / bias`，模型放大完就正好。

`bias` 是唯一的未知量。它不是理论值，是**这个模型版本的行为参数**，
换模型就要重标。所以这一步做成可重复的实验，不把数字写死在代码里。

## 怎么判「哪个 bias 对」

用 `step20_size_measure.measure()` 客观量成品，和计划百分比比。
**不看图说话**——这个项目上目测已经错过两次了（把 +21% 看成 −6%）。

判据：`实测倍率` 最接近 1.0 的那个 bias 就是答案。
理想情况下 `实测倍率 = 放大倾向 / bias`，所以线性上最好的 bias
应该约等于无补偿时测出的倍率。

## 成本

每个 bias 一次生成。3 个点 ≈ 0.36 元 / 约 1 分钟。

跑法：
    python experiments/step21_size_bias_calib.py p01
    python experiments/step21_size_bias_calib.py p01 1.0 1.4 1.8
"""

from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from app.pipeline import OUT_ROOT, run_guided       # noqa: E402
from step20_size_measure import measure             # noqa: E402

SCENES = os.path.join(ROOT, "assets", "scenes", "batch2")
OUTPUT = os.path.join(ROOT, "output")
CALIB = os.path.join(OUTPUT, "calib")


def find(name: str) -> str | None:
    for f in os.listdir(SCENES):
        if name in f and f.lower().endswith(".jpg"):
            return os.path.join(SCENES, f)
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    name = sys.argv[1]
    biases = [float(x) for x in sys.argv[2:]] or [1.0, 1.3, 1.6, 2.0]
    path = find(name)
    if not path:
        print(f"找不到场景 {name}")
        return 1
    tag = os.path.splitext(os.path.basename(path))[0]
    with open(path, "rb") as f:
        data = f.read()

    os.makedirs(CALIB, exist_ok=True)
    rows = []
    for bias in biases:
        print(f"\n--- bias={bias} ---")
        res = run_guided(data, character="chibi", size_bias=bias,
                         progress=lambda s, p: None)
        if res.get("error"):
            print(f"  !! {res['error']}")
            continue
        pl = res["placement"]
        # 单独存一份，别让同名目录互相覆盖
        dst = os.path.join(CALIB, f"{tag}__bias{bias}")
        src = os.path.join(OUT_ROOT, res["run_id"])
        os.makedirs(dst, exist_ok=True)
        for fn in ("scene.jpg", res["guide"], "out_0.jpg"):
            s = os.path.join(src, fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, fn))
        with open(os.path.join(dst, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        m = measure(dst) or {}
        lo = m.get("loose") or {}
        tgt = m.get("target_pct")
        meas = lo.get("h_pct")
        ratio = (meas / tgt) if (meas and tgt) else None
        print(f"  动作 {pl['sprite_action']}  引导图里 {pl['guide_h']}px  "
              f"目标 {pl['target_h']}px")
        print(f"  计划 {tgt}%  实测 {meas}%  "
              f"倍率 {ratio:.2f}×" if ratio else "  量不出来")
        rows.append((bias, pl["sprite_action"], tgt, meas, ratio))

    print("\n" + "=" * 56)
    print(f"{'bias':>6} {'动作':<20} {'计划%':>7} {'实测%':>7} {'倍率':>6}")
    for bias, act, tgt, meas, ratio in rows:
        print(f"{bias:>6.2f} {act:<20} {tgt or 0:>7.1f} {meas or 0:>7.1f} "
              f"{ratio or 0:>6.2f}")
    good = [r for r in rows if r[4] and 0.85 <= r[4] <= 1.15]
    if good:
        best = min(good, key=lambda r: abs(r[4] - 1.0))
        print(f"\n最接近 1.0 的 bias = {best[0]}（倍率 {best[4]:.2f}）")
    else:
        print("\n没有一个落在 0.85~1.15 区间，需要扩大搜索范围")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
