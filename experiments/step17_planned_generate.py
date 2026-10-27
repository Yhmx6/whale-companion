"""Step 17：规划优先的生成路线 —— 三种做法对比。

## 用户提出的方向

"每张图首先由视觉模型判断鲸鱼娘该在什么位置、做什么动作，再调用生图大模型，
而不是把单一的动作贴在原图上。"

这个方向解决立绘路线的两个硬伤：动作是枚举的（半身照没有候选）、她永远贴在最前面
（手不会真的搭在人身上，也不会被挡住）。

## 三种做法，实测结论

| 做法 | 位置/动作 | 尺寸 | 遮挡与互动 |
|---|---|---|---|
| A 纯规划→生成 | 好 | **不可控（实测 1.7~2.8 倍）** | 自然 |
| B 规划 + 尺度锚点 | 好 | 仍不可控（甚至更差） | 自然 |
| C 布局引导生成 | 好 | **精确** | 自然 |

### 为什么 A/B 的尺度救不回来

`build_companion_prompt` 里的位置框是**画面百分比约束**，而
`s2_prompt.py` 里早躺着一条 `_DEPRECATED_PERCENT_SCALE = "（已证伪）角色高度约占画面高度的 N%"`
——这一类指令模型不服从，之前 A/B 就证过。我又试了**物体相对锚点**
（"头顶只到人物膝盖，整个人不超过小腿一半高"），仍然不服从，而且 p05 那张
反而从 39.8% 涨到 53.1%。

**结论：这个生成模型对"把角色画小"这件事没有可控性。** 这正是当初转向立绘的原因。

### 为什么 C 成立

先把立绘按**精确尺寸**贴到照片上（这一步用现有的 `_sprite_placement`，
位置和大小都由程序算），把这张"带小人的照片"当图1 送给模型，要求：

- 她的**外接框位置和大小必须和贴上去的一模一样**（对比参照物就在图里，不是抽象百分比）；
- 除她和她身下的接触阴影外，**其他一律不许改动**。

实测（p05_bench_sit，目标 241px = 画面高 23.5%）：
- 出来的人物**就是那个尺寸**，头顶在人物大腿位置；
- **男士的腿挡在她前面**——真遮挡，由扩散过程生成；
- 背景、人物、树、光线全部没动。

这也解释了为什么"百分比"不行而"图里已有的框"行：**参照物从抽象变成了具体**。

## 用法

    python experiments/step17_planned_generate.py            # 只跑规划（免费，只花 VLM）
    python experiments/step17_planned_generate.py --gen 4    # 跑前 4 张的纯规划生成（每张 0.12 元）
    python experiments/step17_planned_generate.py --guided p05_bench_sit   # 布局引导，每张 0.12 元
"""

from __future__ import annotations

import base64
import glob
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from app.pipeline import run_planned  # noqa: E402
from s3.characters import get_character  # noqa: E402
from s3.plan import plan_scene  # noqa: E402
from s3.profile import vlm_profile  # noqa: E402

OUT = os.path.join(ROOT, "output", "step17")
os.makedirs(OUT, exist_ok=True)
SCENES = sorted(glob.glob(os.path.join(ROOT, "assets", "scenes", "batch2", "*.jpg")))

GUIDED_PROMPT = (
    "图1是一张真实照片，照片里已经贴了一个卡通小人。图2、图3是同一个二次元角色的设定图。\n"
    "这个角色的形象是：{char}。\n"
    "请把图1里那个贴上去的卡通小人**重绘**成图2图3的角色形象，并让她与场景自然融合：\n"
    "1. 【最重要】她的**外接框位置和大小必须和那个贴上去的小人一模一样**——"
    "她的头顶、脚底、左右边界都不要变，不许放大，不许移动；\n"
    "2. 姿势保持：{pose}；\n"
    "3. 在她身下画出压出的接触阴影；画面里的受光方向就是她的受光方向；\n"
    "4. 除她和紧挨着的接触阴影外，**图1其他内容一律不许改动**：人物的脸和衣服、"
    "家具、背景、光线、清晰度全部保持原样；\n"
    "5. 她必须是图2图3里的那个角色；\n"
    "6. 画面里恰好一个角色；不要文字、水印、边框。"
)


def plan_only() -> None:
    for p in SCENES:
        prof = vlm_profile(p)
        hint = (f"画面里有{prof.framing or '一个'}景别人物" if prof.has_person
                else "画面里没有人物")
        plan = plan_scene(p, height_hint=hint)
        name = os.path.basename(p)[:18]
        if not plan.ok:
            print(f"{name:20s} ✗ {plan.reason or plan.raw.get('error')}")
            continue
        print(f"{name:20s} {plan.anchor_kind:14s} {plan.region} 她占{plan.height_pct:4.1f}%")
        print(f'{"":22s}动作：{plan.action}')


def plain_generate(n: int) -> None:
    for p in SCENES[:n]:
        r = run_planned(open(p, "rb").read(), character="chibi")
        name = os.path.basename(p)[:18]
        if "error" in r:
            print(f"{name:20s} ✗ {r['error'][:60]}")
            continue
        me = r.get("measured", {})
        print(f"{name:20s} 计划 {r['plan']['height_pct']:.1f}%  实测 {me.get('h_pct')}%  "
              f"偏差 x{me.get('h_ratio')}")
        im = Image.open(os.path.join(ROOT, "output", "runs", r["run_id"], "out_0.jpg"))
        im.resize((im.width // 2, im.height // 2)).save(
            os.path.join(OUT, f"plain_{name}.jpg"), quality=92)


def guided(name: str, pose: str = "坐在座面上，双腿垂在座沿外侧，双手自然放在膝上") -> None:
    """布局引导：先把立绘按精确尺寸贴好，再让模型只负责重绘与融合。"""
    from app.pipeline import _sprite_placement
    from api.ark import download, generate, image_path_to_data_uri
    from s3.actions import get_action
    from s3.sprite import load_sprite

    rel = os.path.join(ROOT, "assets", "scenes", "batch2", name + ".jpg")
    photo = Image.open(rel).convert("RGB")
    W, H = photo.size
    prof = vlm_profile(rel)
    props = prof.props
    route = "person" if prof.has_person else "object"
    action = "seat_edge_sit" if route == "object" else "leg_hands_behind"

    char = get_character("chibi")
    sp = load_sprite(action, "doubao-seedream-5-0-flash-260915", char)
    if sp is None:
        print("缺立绘缓存"); return
    th, cx, by, meta = _sprite_placement(get_action(action), prof, props, route, W, H,
                                         char, sprite=sp)
    nw = max(1, int(sp.width * th / sp.height))
    px = max(2, min(W - nw - 2, int(cx - nw / 2)))
    py = int(by - th)
    s = sp.resize((nw, th), Image.LANCZOS)
    guide = photo.copy()
    guide.paste(s, (px, py), s)
    guide.save(os.path.join(OUT, f"guide_{name}.jpg"), quality=94)
    print(f"{name}: 贴图框 ({px},{py},{nw},{th})  她占画面高 {th / H * 100:.1f}%")

    prompt = GUIDED_PROMPT.format(char=char.identity, pose=pose)
    buf = io.BytesIO(); guide.save(buf, format="JPEG", quality=94)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    t = time.time()
    st, resp = generate(prompt,
                        images=[uri] + [image_path_to_data_uri(x) for x in char.refs],
                        size="1280x1024", model="doubao-seedream-5-0-flash-260915", seed=4002)
    print(f"HTTP {st}  {time.time() - t:.0f}s")
    if st != 200:
        print(str(resp)[:300]); return
    p = os.path.join(OUT, f"guided_{name}.png")
    download((resp.get("data") or [{}])[0]["url"], p)
    im = Image.open(p)
    im.resize((im.width // 2, im.height // 2)).save(
        os.path.join(OUT, f"guided_{name}.jpg"), quality=92)
    print("saved", p)


def main() -> None:
    if "--guided" in sys.argv:
        guided(sys.argv[sys.argv.index("--guided") + 1])
    elif "--gen" in sys.argv:
        plain_generate(int(sys.argv[sys.argv.index("--gen") + 1]))
    else:
        plan_only()


if __name__ == "__main__":
    main()
