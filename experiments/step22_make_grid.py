"""step22：把本轮验证的场景拼成一张对比图。

## 为什么要有这张图

"动作匹配"和"尺寸偏大"这两件事，散在六个目录里各一张成品，没法一眼判断。
拼成一张、每格标上「计划多少 / 实测多少」，讨论才有公共对象。

## 标注读法

`动作` 是规划器从候选立绘里选中的 id（`来源 plan` 表示它直接选中了候选，
不是走了粗映射兜底）。`计划 → 实测` 是 60cm 换算出的画面百分比，
和 `s3/measure.py` 客观量出来的成品百分比。
"""

from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUT = os.path.join(ROOT, "output")
FONT = "C:/Windows/Fonts/msyh.ttc"

# (目录名, 显示名)
CASES = [
    ("guided2_p05_bench_sit", "P05 长椅·坐着的人"),
    ("guided2_p03_street_walk", "P03 街道·走路的人"),
    ("guided2_p01_full_sun_left", "P01 逆光全身"),
    ("guided2_e04_armchair", "E04 扶手椅"),
    ("guided2_e05_steps", "E05 台阶"),
    ("guided2_p04_cafe_table", "P04 咖啡桌"),
]

CELL_W = 420          # 每格宽（图按高度自适应）
PAD = 12
HEAD = 62             # 每条标注占的高度


def main() -> int:
    items = []
    for d, label in CASES:
        folder = os.path.join(OUTPUT, d)
        out = os.path.join(folder, "out_0.jpg")
        meta = os.path.join(folder, "meta.json")
        if not os.path.exists(out):
            continue
        pl = {}
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as f:
                pl = json.load(f).get("placement") or {}
        items.append((label, Image.open(out).convert("RGB"), pl))
    if not items:
        print("没有可拼的成品")
        return 1

    # 统一到同一格高：按高度缩放、水平居中，避免竖图把整行撑高
    cell_h = 460
    cols = 3
    rows = (len(items) + cols - 1) // cols
    w = cols * CELL_W + (cols + 1) * PAD
    h = rows * (cell_h + HEAD) + (rows + 1) * PAD
    canvas = Image.new("RGB", (w, h), (250, 250, 252))
    dr = ImageDraw.Draw(canvas)
    f_label = ImageFont.truetype(FONT, 22)
    f_small = ImageFont.truetype(FONT, 17)

    for i, (label, im, pl) in enumerate(items):
        r, c = divmod(i, cols)
        x = PAD + c * (CELL_W + PAD)
        y = PAD + r * (cell_h + HEAD + PAD)
        tw = int(im.width * cell_h / im.height)
        th = int(im.height * tw / im.width)
        thumb = im.resize((tw, th), Image.LANCZOS)
        canvas.paste(thumb, (x + (CELL_W - tw) // 2, y))
        dr.rectangle([x, y, x + CELL_W - 1, y + cell_h - 1],
                     outline=(222, 222, 228))
        dr.text((x + 2, y + cell_h + 4), label, font=f_label, fill=(28, 28, 34))
        plan_pct = pl.get("h_pct")
        meas = pl.get("measured_pct")
        err = pl.get("size_err")
        act = pl.get("sprite_action") or "-"
        src = pl.get("action_source") or "-"
        dr.text((x + 2, y + cell_h + 30), f"动作 {act}（{src}）",
                font=f_small, fill=(90, 90, 100))
        info = f"尺寸 计划 {plan_pct}% → 实测 {meas}%（{err}×）" \
            if plan_pct and meas else "尺寸 未测出"
        dr.text((x + 2, y + cell_h + 50), info, font=f_small,
                fill=(180, 60, 40) if err and err > 1.25 else (60, 120, 70))

    out = os.path.join(OUTPUT, "guided_round1_grid.jpg")
    canvas.save(out, quality=92)
    print("saved", out, canvas.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
