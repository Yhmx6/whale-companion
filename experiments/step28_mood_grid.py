"""step28：四种表情并排对比。

## 这张图要说明什么

「不同场景不同表情」这件事，散在各处看不出来——每张单独看，都是"她在笑"。
四个表情并排、把**面部描述**写在下面，才能一眼判断：
模型是照着描述画的脸，还是无论怎么描述都画同一张。

所以要放**四种差异最大的**：雀跃（月牙眼+张嘴）/ 好奇（圆睁眼+O 形嘴）/
害羞（脸红+低头）/ 得意（半眯眼+翘嘴角）。如果这四张看起来是同一张脸，
那这套表情机制就是假的。

跑法：
    python experiments/step28_mood_grid.py
"""

from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUT = os.path.join(ROOT, "output")
FONT = "C:/Windows/Fonts/msyh.ttc"

# (标题, 图路径, meta 路径)
CASES = [
    ("雀跃 happy", "batch3/c05_bench_empty/out_0.jpg",
     "batch3/c05_bench_empty/meta.json"),
    ("好奇 curious", "batch3/c07_windowsill_plant/out_0.jpg",
     "batch3/c07_windowsill_plant/meta.json"),
    ("害羞 shy", "mooddemo/e04_armchair_shy/out_0.jpg",
     "mooddemo/e04_armchair_shy/meta.json"),
    ("得意 smug", "mooddemo/e02_bench_smug/out_0.jpg",
     "mooddemo/e02_bench_smug/meta.json"),
]

CELL_W = 400
CELL_H = 470
HEAD = 96
PAD = 12


def main() -> int:
    items = []
    for title, rel_img, rel_meta in CASES:
        p = os.path.join(OUTPUT, rel_img)
        if not os.path.exists(p):
            print("缺图:", rel_img)
            continue
        face, reason = "", ""
        mp = os.path.join(OUTPUT, rel_meta)
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                mo = (json.load(f).get("mood") or {})
            face = mo.get("face") or ""
            reason = mo.get("reason") or ""
        items.append((title, Image.open(p).convert("RGB"), face, reason))
    if not items:
        return 1

    cols = len(items)
    W = cols * CELL_W + (cols + 1) * PAD
    H = CELL_H + HEAD + 2 * PAD
    canvas = Image.new("RGB", (W, H), (250, 250, 252))
    dr = ImageDraw.Draw(canvas)
    f_t = ImageFont.truetype(FONT, 24)
    f_s = ImageFont.truetype(FONT, 15)

    for i, (title, im, face, reason) in enumerate(items):
        x = PAD + i * (CELL_W + PAD)
        y = PAD
        tw = int(im.width * CELL_H / im.height)
        th = CELL_H
        if tw > CELL_W:
            th = int(th * CELL_W / tw)
            tw = CELL_W
        canvas.paste(im.resize((tw, th), Image.LANCZOS),
                     (x + (CELL_W - tw) // 2, y + (CELL_H - th) // 2))
        dr.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1], outline=(220, 220, 226))
        dr.text((x + 2, y + CELL_H + 4), title, font=f_t, fill=(28, 28, 34))
        # 面部描述折行，每行约 26 字
        line_y = y + CELL_H + 34
        for j in range(0, min(len(face), 104), 26):
            dr.text((x + 2, line_y), face[j:j + 26], font=f_s, fill=(96, 96, 108))
            line_y += 18
            if line_y > H - 20:
                break

    out = os.path.join(OUTPUT, "mood_grid.jpg")
    canvas.save(out, quality=93)
    print("saved", out, canvas.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
