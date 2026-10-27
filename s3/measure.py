"""成品尺寸测量：她到底被画成了多大。

## 为什么要有这个模块

用户的产品规定是「她 60cm 高」，程序也把 60cm 换算成了像素（`person_h × 60/175`）。
但**生成模型不服从这个尺寸**——三条控制手段全试过，都压不住：

| 手段 | 结果 |
|---|---|
| 提示词「外接框大小必须一模一样，不许放大」 | 无效（倍率 1.51 → 1.49） |
| 提示词补物体参照（「头顶刚过人物膝盖」） | 无效 |
| 引导图里把她缩小（21.3% → 10.6%） | **完全无视**，成品稳定 31% |
| 引导图换成纯色剪影，要求「填满形状」 | 无效（32.3%） |

所以尺寸这件事只能**测**，不能**假定**。这个模块就是那把尺子。

## 怎么量

生成提示词里有一条硬约束：除她和她身下的接触阴影外，图1其他内容不许改动。
于是把成品和原照片逐像素相减，差异集中的那一块就是她。

三个关键细节：

1. **只在规划框附近找**。整幅画面相减会把"模型顺手改了背景光线"也算进来——
   实测 e05（台阶场景）整幅 diff 的 bbox 高达 72.7% 画面高，而她在其中只占一半。
   限定在她该在的位置附近（`expand` 倍框），背景改动就被排除在外了。
2. **取主连通域**，不是 bbox of all。散落的背景噪声会把框撑大。
3. **两个阈值都报**。低阈值（40）会把她脚下的接触阴影算进来，把高度撑大 5~10%；
   高阈值（80）更接近"只有她本体"。两者一起看，差得多说明阴影很重。

## 已知不准的情况

- **她挡住人物时**：如果她压在人裤子上，那块 diff 是她造成的，高度是对的。
- **模型大改背景时**：仍可能污染。此时 `tight` 比 `loose` 可信。
- **她很小、颜色又接近背景时**：diff 可能不足 40，量不出来（返回 None）。
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _main_blob(mask: np.ndarray, k: int = 8) -> tuple[int, int, int, int] | None:
    """最大连通域的 bbox。8×8 分块 + 粗网格 BFS，避免引入 scipy。

    对"量一个矩形块的高度"这个用途，8px 粒度足够（在 1536 高的图上误差 < 1%）。
    """
    H, W = mask.shape
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


def measure_char_size(scene: Image.Image, out: Image.Image,
                      plan_box: list[int] | None = None,
                      expand: float = 1.8) -> dict:
    """量出成品里她的高度。

    `plan_box` 是 `[px, py, nw, nh]`（原图像素）。给了就把搜索范围限制在它
    扩展 `expand` 倍之后的 ROI 里——**这一步很重要**，否则模型顺手改动的
    背景光线会把测量污染到完全没法用（实测 e05 从 55% 被撑到 73%）。
    """
    scene = scene.convert("RGB")
    out = out.convert("RGB").resize(scene.size, Image.LANCZOS)
    W, H = scene.size
    a = np.asarray(scene, dtype=np.int16)
    b = np.asarray(out, dtype=np.int16)
    diff = np.abs(a - b).max(axis=2)

    x_off = y_off = 0
    roi = diff
    if plan_box:
        px, py, nw, nh = plan_box
        cx, cy = px + nw / 2.0, py + nh / 2.0
        half_w = max(nw * expand / 2.0, 24)
        half_h = max(nh * expand / 2.0, 24)
        x1 = int(max(0, cx - half_w))
        x2 = int(min(W, cx + half_w))
        y1 = int(max(0, cy - half_h))
        y2 = int(min(H, cy + half_h))
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            roi = diff[y1:y2, x1:x2]
            x_off, y_off = x1, y1

    res: dict = {"W": W, "H": H, "roi": [x_off, y_off, x_off + roi.shape[1],
                                         y_off + roi.shape[0]]}
    for tag, thr in (("loose", 40), ("tight", 80)):
        bb = _main_blob(roi > thr)
        if bb is None:
            res[tag] = None
            continue
        x1, y1, x2, y2 = bb
        res[tag] = {"bbox": [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off],
                    "h": y2 - y1, "w": x2 - x1,
                    "h_pct": round((y2 - y1) / H * 100, 1)}
    return res
