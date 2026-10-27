"""step20：客观测量成品里她占多大。

## 为什么必须做这把尺子

"她的尺寸被放大了多少"这个问题，我一直在目测。三轮下来的目测结论
互相矛盾：有一次说"误差 -6%"，另一次说"放大 1.6 倍"，但同一双眼睛
在两张不同构图的照片上根本没有可比性——照片里人物占画面高度不同，
她的绝对像素高不同，"看起来大不大"就完全取决于参照物。

用户的要求是硬的（**60cm**），验收就不能靠眼睛。所以要有客观量。

## 怎么量

生成提示词的第 5 条是硬约束：**除她和她身下的接触阴影外，图1其他内容一律不许改动**。
所以把成品和原照片逐像素相减，差异集中的那一块就是她（加脚下的阴影）。

  1. 成品缩放到原照片尺寸（生成接口的输出尺寸可能不同：1536×1024 → 1440×960）
  2. `diff = max_channel(|out - scene|)`
  3. `mask = diff > 阈值`，阈值取 40：低反差处（树影微变、JPEG 噪声）通常在 10 以内
  4. 取 mask 的**主连通域**（最大那块），它的 bbox 高度 = 实测身高

主连通域这一步不能省：背景树叶在阳光下的细微变化会散落在整幅画面上，
直接取全 mask 的 bbox 会得到"整幅画面"。

## 已知的偏差方向

- **阴影会算进来**：她脚下的接触阴影 diff 可能超过 40，把 h 撑大 5~10%。
  所以 `--tight` 会用更高阈值（80）再量一次，两者一起看：
  低阈值给"她 + 阴影"，高阈值给"只有她本体"。
- **她挡住人物时量不准**：如果她的身体压在人裤子上，那块区域的 diff 是她造成的，
  高度是对的；但如果模型顺手改了人物，diff 会把人也圈进来，这时数字会明显超标，
  脚本会打 `?` 提示人工看。

跑法：
    python experiments/step20_size_measure.py guided2_p05_bench_sit
    python experiments/step20_size_measure.py guided2_*
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUT = os.path.join(ROOT, "output")


def _main_blob(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """最大连通域的 bbox。用行/列累加的近似连通域，避免依赖 scipy。

    做法：先把 mask 按 8×8 分块降采样，块内只要有 20% 像素命中就算"活块"，
    在粗网格上做 BFS 连通域，再把选中的块并回 bbox。
    对"测一个矩形块的高度"这个用途，8px 的粒度完全够（误差 < 1%）。
    """
    H, W = mask.shape
    k = 8
    gh, gw = H // k, W // k
    if gh < 2 or gw < 2:
        ys, xs = np.nonzero(mask)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) \
            if len(xs) else None
    blocks = mask[:gh * k, :gw * k].reshape(gh, k, gw, k).mean(axis=(1, 3))
    grid = blocks > 0.20

    seen = np.zeros_like(grid, dtype=bool)
    best: list[tuple[int, int]] = []
    for sy in range(gh):
        for sx in range(gw):
            if not grid[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            comp: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gh and 0 <= nx < gw and grid[ny, nx] \
                                and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    if not best:
        return None
    ys = [p[0] for p in best]
    xs = [p[1] for p in best]
    return (min(xs) * k, min(ys) * k, (max(xs) + 1) * k, (max(ys) + 1) * k)


def measure(folder: str) -> dict | None:
    scene_p = os.path.join(folder, "scene.jpg")
    out_p = os.path.join(folder, "out_0.jpg")
    meta_p = os.path.join(folder, "meta.json")
    if not (os.path.exists(scene_p) and os.path.exists(out_p)):
        return None
    scene = Image.open(scene_p).convert("RGB")
    out = Image.open(out_p).convert("RGB").resize(scene.size, Image.LANCZOS)
    a = np.asarray(scene, dtype=np.int16)
    b = np.asarray(out, dtype=np.int16)
    diff = np.abs(a - b).max(axis=2)
    H, W = diff.shape

    res: dict = {"folder": os.path.basename(folder), "W": W, "H": H}
    for tag, thr in (("loose", 40), ("tight", 80)):
        m = diff > thr
        bb = _main_blob(m)
        if bb is None:
            res[tag] = None
            continue
        x1, y1, x2, y2 = bb
        res[tag] = {"bbox": [x1, y1, x2, y2], "h": y2 - y1,
                    "w": x2 - x1, "h_pct": round((y2 - y1) / H * 100, 1)}

    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
        pl = meta.get("placement") or {}
        res["target_h"] = pl.get("target_h")
        res["target_pct"] = pl.get("h_pct")
        res["action"] = pl.get("sprite_action")
        res["basis"] = pl.get("size_basis")
        # 落点校验：实测底边 vs 计划底边
        if pl.get("box"):
            py, nh = pl["box"][1], pl["box"][3]
            res["plan_bottom"] = py + nh
            if res["loose"]:
                res["bottom_err"] = res["loose"]["bbox"][3] - (py + nh)
    return res


def main() -> int:
    pats = sys.argv[1:] or ["guided2_*"]
    folders: list[str] = []
    for p in pats:
        folders.extend(sorted(glob.glob(os.path.join(OUTPUT, p))))
    folders = [f for f in folders if os.path.isdir(f)]
    if not folders:
        print("没找到产物目录")
        return 1

    rows = []
    for f in folders:
        r = measure(f)
        if not r:
            continue
        rows.append(r)
        lo, ti = r.get("loose"), r.get("tight")
        print(f"\n{r['folder']}   {r['W']}x{r['H']}")
        if not lo:
            print("  量不出来（diff 全平：她可能没被画上去，或成品就是原图）")
            continue
        if r.get("target_pct") is None:
            print(f"  实测 {lo['h']}px = {lo['h_pct']}% 画面高")
            continue
        ratio = lo["h_pct"] / r["target_pct"] if r["target_pct"] else 0
        flag = ""
        if ratio > 1.25:
            flag = "  ← 偏大"
        elif 0 < ratio < 0.8:
            flag = "  ← 偏小"
        print(f"  动作 {r.get('action')}  依据 {r.get('basis')}")
        print(f"  计划 {r['target_pct']}% ({r['target_h']}px)   "
              f"实测 {lo['h_pct']}% ({lo['h']}px)"
              + (f" / 紧 {ti['h_pct']}%" if ti else ""))
        print(f"  倍率 {ratio:.2f}×{flag}")
        if "bottom_err" in r:
            print(f"  落点误差 {r['bottom_err']:+d}px"
                  f"（正=比计划低）")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
