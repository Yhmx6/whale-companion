"""Step 16：10 个新场景跑一遍 Q 版鲸鱼娘，看表现。

## 这批场景是为什么造的

之前所有验证都集中在 8 张自拍 / 3 张物件图上，而那 3 张物件图是我为了让某个动作
成立专门挑的。等于说"在已知能用的场景上验证过"——这不叫泛化。

这一批刻意做成**没见过的构图**，并且成对覆盖：

| 组 | 场景 | 想压测什么 |
|---|---|---|
| 带人物 | 黄昏侧光全身 / 室内半身沙发 / 傍晚逆光行走 / 咖啡馆半身+桌沿 / 正午顶光坐长椅 | 全身 vs 半身、顺光 vs 逆光 vs 顶光、人物+物件同时存在 |
| 空场景 | 客厅沙发 / 公园长椅 / 餐桌前沿 / 扶手椅 / 石台阶 | 物件路线的召回与尺度，五种不同家具 |

## 每张记录什么

- **路线**（person / object / none）与**自动选中的动作**——这是最可能出问题的一环，
  它同时依赖 framing、光向、物件框三个字段；
- 她占画面高的**比例**（尺度是否离谱）；
- `shadow` / `relight` 的实际分支，用来确认光照链路有没有真的跑起来；
- 耗时与错误。

## 判据

没有"正确动作"这种唯一答案，所以不设自动 pass/fail。看三件事：

1. 有没有**报错**（这条是硬的）；
2. 尺度是否落在人眼能接受的区间（手办 12%~30% 画面高，坐姿可略小）；
3. 动作和场景是否**物理说不通**（比如蹲在半空、坐在没有座面的地方）。

用法：
    python experiments/step16_batch_scenes.py
"""

from __future__ import annotations

import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def _route_of(prof, res) -> str:
    """路线由"有没有人 / 有没有可用座面"推出来，`vlm_match_action` 不返回它。"""
    if prof.has_person:
        return "person"
    return "object" if any(p.usable for p in prof.props) else "none"

from app import pipeline  # noqa: E402
from s3.actions import enabled_actions, get_action, vlm_match_action  # noqa: E402
from s3.profile import filter_by_framing, vlm_profile  # noqa: E402

OUT = os.path.join(ROOT, "output", "step16")
os.makedirs(OUT, exist_ok=True)
SCENES = sorted(glob.glob(os.path.join(ROOT, "assets", "scenes", "batch2", "*.jpg")))
F = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)


def main() -> None:
    rows = []
    tiles = []
    print("=" * 96)
    for path in SCENES:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as f:
            raw = f.read()
        W, H = Image.open(path).size

        t0 = time.time()
        # **只在立绘路线的动作里挑**。
        #
        # 用 `pick_action_v2` 会挑到 `peek` / `shoulder` / `hem_peek`——那几个是
        # 场景内路线（和场景一起生成 + 肩线锚点定位），`run_sprite` 做不了，
        # 硬跑出来是她站在地上。而且会静默花 0.12 元去生成一张这个动作的立绘。
        #
        # 这一批要测的是**主路径（立绘合成）**，所以先把候选集收窄到 sprite。
        prof = vlm_profile(path)
        pool = [a for a in enabled_actions() if a.render == "sprite"]
        pool = filter_by_framing(pool, prof.framing, prof.selfie) or pool
        # 景别对不上就没得选了（所有地面动作都要求 full），直接记为"无可用动作"
        if not pool:
            print(f"[{name:18s}] 无可用立绘动作（framing={prof.framing}）")
            rows.append((name, "-", "-", "-", "-",
                         f"framing={prof.framing}，无可用立绘动作"))
            continue
        pick = vlm_match_action(path, actions=pool)
        if "error" in pick:
            print(f"[{name:18s}] 选动作失败：{pick['error'][:60]}")
            rows.append((name, "-", "-", "-", "-", f"选动作失败: {pick['error'][:30]}"))
            continue
        action = pick["action_id"]
        res = pipeline.run_sprite(raw, action, character="chibi")
        dt = time.time() - t0
        if "error" in res:
            print(f"[{name:18s}] 合成失败：{res['error'][:60]}")
            rows.append((name, pick.get("route", "?"), action, "-", f"{dt:.1f}s",
                         f"合成失败: {res['error'][:30]}"))
            continue

        c = res["candidates"][0]
        pl = c["placement"]
        h_pct = pl["target_h"] / H * 100
        rel = (c.get("relight") or {})
        rows.append((
            name,
            _route_of(prof, res),
            action,
            f'{pl["target_h"]}px ({h_pct:.1f}%)',
            f"{dt:.1f}s",
            f'shadow={c["shadow"]}  relight={"on" if rel.get("applied") else "off"}',
        ))
        print(f'[{name:18s}] route={_route_of(prof, res):6s} action={action:16s} '
              f'她高 {pl["target_h"]:4d}px ({h_pct:4.1f}%)  {dt:4.1f}s  '
              f'shadow={c["shadow"]:18s} relight={"on" if rel.get("applied") else "off"}')

        im = Image.open(os.path.join(ROOT, "output", "runs", res["run_id"],
                                     c["file"])).convert("RGB")
        tw = 360
        im = im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, tw, 40], fill=(0, 0, 0))
        d.text((6, 4), f"{name}", fill=(255, 240, 120), font=F)
        d.text((6, 22), f"{action} · {h_pct:.0f}%", fill=(210, 210, 210), font=F)
        tiles.append(im)

    print("=" * 96)
    if tiles:
        cols = 5
        rows_n = (len(tiles) + cols - 1) // cols
        th = max(t.height for t in tiles)
        sheet = Image.new("RGB", (360 * cols, th * rows_n), (18, 18, 18))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 360, (i // cols) * th))
        out = os.path.join(OUT, "batch2_sheet.jpg")
        sheet.save(out, quality=92)
        print(f"对比图: {out}")

    with open(os.path.join(OUT, "batch2_report.md"), "w", encoding="utf-8") as f:
        f.write("# 10 场景 · Q 版鲸鱼娘实测\n\n")
        f.write("| 场景 | 路线 | 动作 | 她高 | 耗时 | 链路 |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(f"报告: {os.path.join(OUT, 'batch2_report.md')}")


if __name__ == "__main__":
    main()
