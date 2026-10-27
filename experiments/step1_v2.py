"""提示词 v2：加死身高/体型约束，解决"角色被放大成正常人"和"Q版腿被拉长"。

## 问题诊断（来自用户反馈 + 10 张实测）

u03 客厅那张：角色比沙发靠背还高。真实世界里一个 Q 版小人不可能是这个尺度。
u04 卧室那张：从 Q 版漂成 4-5 头身，腿被拉长。

根因有两条：
1. **提示词用的是"占画面高度 44%"这种相对量**，模型照着画面百分比画，完全
   不理会"她在真实空间里应该多大"。一个 Q 版玩偶坐在双人沙发上，只该占画面
   很小一块。
2. **只描述了服装配色，没锁头身比**。模型默认按"好看的女孩子"画，而好看的
   默认比例就是 6-7 头身的少女——腿自然就长了。

## v2 的三处改动

1. **真实尺寸锚点**：给她一个真实世界的绝对尺寸（约 40cm，一个抱枕）。
2. **参照物比较**：每个案例给一句具体的"她只到某某位置"，比百分比可靠得多。
3. **头身比锁死 + 腿部量化**："大腿加小腿不超过头部高度的 1.5 倍"，
   把"腿要短"变成一个模型能对照的数字。

## A/B 设计

V2a = 体型 + 身高锚点
V2b = V2a + 禁白描边（另一个已知缺陷，顺带验证能否一起解决）

两组都在 u03_living（用户点名的那张）上跑，与基线 u03_living_0 对比。
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from api.ark import download, generate, image_path_to_data_uri  # noqa: E402
from experiments.step1_user import CASES, CHAR_DESC, pick_size  # noqa: E402

MODEL = "doubao-seedream-5-0-pro-260628"
OUT = os.path.join(ROOT, "output", "step1", "v2")
SCENES = os.path.join(ROOT, "assets", "scenes", "user")
REFS = os.path.join(ROOT, "assets", "refs", "crops")

# 每个案例一句"她只到某某位置"的参照物锚点
ANCHORS = {
    "u01_kitchen": "她站在白色地砖上，头顶大约只到深红色橱柜柜门中间那道横缝的高度",
    "u02_dining": "她坐在橙色坐垫上，头顶还不到椅背的一半高，整体比坐垫上的灰色荷叶边大不了多少",
    "u03_living": "她坐在紫色沙发上，头顶大约只到沙发靠背高度的一半，明显比旁边那个花抱枕小",
    "u04_bedroom": "她坐在床沿，头顶只比床面高一点点，整体比被子上的褶皱大不了多少",
    "u05_desk": "她坐在桌面右前缘，头顶大约只到机械键盘按键的高度，比旁边的鼠标大不了多少",
}

# 体型锁：把"Q版"翻译成模型能对照的量化描述
BODY_LOCK = (
    "【角色体型，这条最重要】这是一个 Q 版（chibi）小人，只有 2 到 2.5 头身："
    "头非常大，身体很小，四肢短而圆润。腿必须短——"
    "大腿加小腿的总长度不超过头部高度的 1.5 倍，坐着垂下来只有一点点，膝盖离地面很近。"
    "她的真实身高很娇小，大约 40 厘米，相当于一个抱枕或一个毛绒玩偶。"
    "绝对不要把她画成正常比例的少女或成年人，不要拉长腿部，不要用写实的四肢比例，"
    "不要把她放大到与家具同高。"
)

# 描边锁：另一个已知系统性问题（10/10 复现），顺带验证能否一起治
EDGE_LOCK = (
    "【边缘处理，这条也很重要】不要给角色添加白色描边、亮色轮廓光、外发光、"
    "阴影描边或贴纸式的白边。角色的轮廓必须与背景自然衔接、融为一体，"
    "看起来像是本来就在这个空间里，而不是后期贴上去的一张图。"
)


def build_prompt_v2(case: dict, with_edge_lock: bool) -> str:
    anchor = ANCHORS.get(case["key"], "")
    parts = [
        "图1是一张真实的室内照片。图2、图3是同一个二次元角色的设定图。\n",
        f"这个角色的形象是：{CHAR_DESC}。\n",
        f"{BODY_LOCK}\n",
        f"请只把这个角色放进图1里：{case['placement']}。{case['pose_desc']}。\n",
        f"【尺度锚点，按这句定大小，不要按画面百分比放大】{anchor}。\n",
        "硬性要求：\n",
        "1. 角色形象必须与设定图完全一致：蓝色渐变长发、深蓝色鲸鱼耳、"
        "藏青女仆裙、白色荷叶边围裙及围裙上的鲸鱼图案、身后深蓝色鲸尾、深蓝色小皮鞋；\n",
        f"2. 角色必须与{case['contact']}真实接触，不得悬浮，也不得陷进去；\n",
        f"3. 光照：{case['light']}，角色身上的受光方向必须与这个环境光一致；\n",
        "4. 不要生成角色投射在地面上的影子（投影由后期单独合成），"
        "只保留角色自身受光以及身体与接触面之间的轻微暗部；\n",
        "5. 保持图1原有的背景、家具、物品、构图、透视、色调与清晰度完全不变，"
        "不得重绘背景，不得移动、添加或删除任何原有物品；\n",
        "6. 画面中只出现这一个角色，不要文字、水印、边框、多个角色，不要三视图并排；\n",
        "7. 角色在画面里应该显得小巧，允许周围留出大片环境。\n",
    ]
    if with_edge_lock:
        parts.append(f"{EDGE_LOCK}\n")
    return "".join(parts)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    keys = os.environ.get("CASES", "u03_living").split(",")
    variants = os.environ.get("VARIANTS", "a,b").split(",")

    front = os.path.join(REFS, "05_chibi_front.png")
    side = os.path.join(REFS, "05_chibi_side.png")
    records = []

    for key in keys:
        case = next((c for c in CASES if c["key"] == key), None)
        if case is None:
            print(f"[跳过] 未知案例 {key}")
            continue
        scene_path = os.path.join(SCENES, case["file"])
        im = Image.open(scene_path)
        size, _ = pick_size(im)
        scene_uri = image_path_to_data_uri(scene_path)
        ref_uris = [image_path_to_data_uri(front), image_path_to_data_uri(side)]

        for v in variants:
            with_edge = (v == "b")
            prompt = build_prompt_v2(case, with_edge)
            tag = f"{key}_v2{v}"
            print(f"\n=== {tag} | 禁描边={with_edge} | {case['title']} ===")
            t0 = time.perf_counter()
            status, resp = generate(prompt, images=[scene_uri] + ref_uris,
                                    size=size, model=MODEL, seed=1000)
            dt = time.perf_counter() - t0
            if status != 200:
                print("  失败:", str(resp)[:400])
                records.append({"tag": tag, "status": status, "error": str(resp)[:400]})
                continue
            out = os.path.join(OUT, f"{tag}.png")
            download(resp["data"][0]["url"], out)
            print(f"  HTTP 200  {dt:.1f}s  {Image.open(out).size} -> {os.path.basename(out)}")
            records.append({"tag": tag, "status": 200, "latency_s": round(dt, 1),
                            "file": os.path.basename(out), "prompt": prompt,
                            "case": case["key"], "edge_lock": with_edge, "size": size})

    path = os.path.join(OUT, "manifest_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    ok = len([r for r in records if r.get("status") == 200])
    print(f"\n完成 {ok} 张，约 {ok * 0.30:.2f} 元 -> {OUT}")


if __name__ == "__main__":
    main()
