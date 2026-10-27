"""step26：把 batch3 的 10 张结果拼成对比图 + 一张场景标注图。

## 两张图解决两个不同的问题

**成品对比图**（`batch3_grid.jpg`）：十张成品 + 每张的动作、挪动量、尺寸倍率。
用来判断"这一轮到底好没好"。

**场景标注拼图**（`batch3_scene_grid.jpg`）：十张 `scene2d.jpg`（框 + 接地线 + 可坐面）。
用来判断"2.5D 场景解析准不准"——如果框就画歪了，成品里的位置再合理也没意义。

两张要分开看：成品不好可能是**解析错**（框歪、漏检），也可能是**生成没听话**。
混在一张图里会分不清是哪一层的锅。

跑法：
    python experiments/step26_grid.py
"""

from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BATCH = os.path.join(ROOT, "output", "batch3")
OUTPUT = os.path.join(ROOT, "output")
FONT = "C:/Windows/Fonts/msyh.ttc"

PAD = 10
CELL_W = 330
CELL_H = 440
HEAD = 74
COLS = 5


def _grid(files: list[tuple[str, str, dict]], out_name: str,
          title_line) -> str:
    rows = (len(files) + COLS - 1) // COLS
    W = COLS * CELL_W + (COLS + 1) * PAD
    H = rows * (CELL_H + HEAD) + (rows + 1) * PAD
    canvas = Image.new("RGB", (W, H), (250, 250, 252))
    dr = ImageDraw.Draw(canvas)
    f_lab = ImageFont.truetype(FONT, 19)
    f_sm = ImageFont.truetype(FONT, 15)

    for i, (tag, path, meta) in enumerate(files):
        r, c = divmod(i, COLS)
        x = PAD + c * (CELL_W + PAD)
        y = PAD + r * (CELL_H + HEAD + PAD)
        im = Image.open(path).convert("RGB")
        tw = int(im.width * CELL_H / im.height)
        th = int(im.height * tw / im.width)
        if tw > CELL_W:
            th = int(th * CELL_W / tw)
            tw = CELL_W
        canvas.paste(im.resize((tw, th), Image.LANCZOS),
                     (x + (CELL_W - tw) // 2, y + (CELL_H - th) // 2))
        dr.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1], outline=(220, 220, 226))
        dr.text((x + 2, y + CELL_H + 3), tag.split("_", 1)[0], font=f_lab,
                fill=(28, 28, 34))
        for j, line in enumerate(title_line(tag, meta)[:3]):
            dr.text((x + 2, y + CELL_H + 25 + j * 17), line, font=f_sm,
                    fill=(88, 88, 98))

    out = os.path.join(OUTPUT, out_name)
    canvas.save(out, quality=92)
    print("saved", out, canvas.size)
    return out


def main() -> int:
    tags = sorted(d for d in os.listdir(BATCH)
                  if os.path.isdir(os.path.join(BATCH, d)))
    made, scenes = [], []
    for tag in tags:
        out = os.path.join(BATCH, tag, "out_0.jpg")
        if not os.path.exists(out):
            continue
        meta = {}
        mp = os.path.join(BATCH, tag, "meta.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        made.append((tag, out, meta))
        sc = os.path.join(BATCH, tag, "scene2d.jpg")
        if os.path.exists(sc):
            scenes.append((tag, sc, meta))

    def line_done(tag, meta):
        pl = meta.get("placement") or {}
        col = pl.get("collision") or {}
        return [
            f"动作 {pl.get('sprite_action')}（{pl.get('action_source')}）",
            f"挪 {col.get('moved_px') or 0:.0f}px"
            f"{'  压得最少' if col.get('forced') else ''}",
            f"尺寸 {pl.get('h_pct')}% → {pl.get('measured_pct')}%"
            f"（{pl.get('size_err')}×）",
        ]

    def line_scene(tag, meta):
        objs = (meta.get("scene2d") or {}).get("objects") or []
        names = "、".join(o.get("name", "") for o in objs[:3])
        return [f"物体 {len(objs)} 个",
                names or "（无）"]

    if made:
        _grid(made, "batch3_grid.jpg", line_done)
    if scenes:
        _grid(scenes, "batch3_scene_grid.jpg", line_scene)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
