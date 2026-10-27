"""Step 1 用户真实照片实验：5 案例 × 2 候选。

与首轮（case01）的差别：
1. 用**真实手机照片**，不再是 AI 生成的干净底图
2. 每张照片按实际空间单独设计落点（站立 / 坐），对应 PRD 的典型场景
3. 每案例出 2 个候选（不同 seed），为后面统计"单张可用率"攒数据

落位策略（来自首轮实测的一条经验）：**提示词里的百分比是软约束**——
首轮请求脚点 78% 实际落到 83%、请求高度 28% 实际 38%。所以每个案例
都用「定性描述 + 参照物」为主、百分比为辅。
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

MODEL = "doubao-seedream-5-0-pro-260628"
OUT = os.path.join(ROOT, "output", "step1", "user")
SCENES = os.path.join(ROOT, "assets", "scenes", "user")
REFS = os.path.join(ROOT, "assets", "refs", "crops")

CHAR_DESC = (
    "蓝色渐变长发（发根深蓝、发梢浅蓝）、深蓝色的鲸鱼耳、"
    "藏青色的女仆裙配白色荷叶边围裙、围裙正面有鲸鱼图案、"
    "身后有一条粗大的深蓝色鲸尾、脚穿深蓝色小皮鞋"
)

# 每案例的落点设计：贴身描述参照物，百分比只做辅助
CASES = [
    dict(
        key="u01_kitchen", file="u01_kitchen.jpg",
        title="站在厨房地砖上（地面/站立）",
        placement="站在画面左下方的白色地砖上，在深红色橱柜前方的空地上，位于燃气灶的左前侧",
        pose_desc="全身站直，正面朝向镜头，双脚踩在地砖上",
        contact="地砖",
        px=28, py=90, height=34,
        light="自然光从画面左侧的窗帘透进来，柔和偏冷白，左侧略亮右侧略暗",
        azimuth=200.0, elevation=35.0, hardness=0.32, intensity=0.5,
    ),
    dict(
        key="u02_dining", file="u02_dining.jpg",
        title="坐在橙色坐垫的椅子上（椅子/坐）",
        placement="坐在画面右侧那把有橙色方形坐垫、灰色荷叶边裙边的椅子上，身体正面朝向镜头",
        pose_desc="端坐在橙色坐垫上，双腿自然垂在椅子前方，脚踝并拢",
        contact="椅子的橙色坐垫",
        px=58, py=88, height=40,
        light="室内顶灯照明，光线均匀柔和，没有明显的方向性硬影",
        azimuth=270.0, elevation=70.0, hardness=0.15, intensity=0.32,
    ),
    dict(
        key="u03_living", file="u03_living.jpg",
        title="坐在紫色沙发上（沙发/坐）",
        placement="坐在画面右侧的紫色布艺沙发上，位置在沙发靠左那一段，身体正面朝向镜头",
        pose_desc="坐在沙发坐垫上，双腿自然垂在沙发前缘，双脚落到地面上",
        contact="沙发坐垫",
        px=70, py=86, height=44,
        light="画面左侧窗帘透入的自然光为主，光线柔和，右侧略暗",
        azimuth=185.0, elevation=40.0, hardness=0.2, intensity=0.36,
    ),
    dict(
        key="u04_bedroom", file="u04_bedroom.jpg",
        title="坐在床沿（床/坐）",
        placement="坐在画面左侧白色床铺的床沿上，身体正面朝向镜头",
        pose_desc="坐在床沿，双腿自然垂在床边，双手放在膝盖附近",
        contact="床沿",
        px=30, py=86, height=40,
        light="室内吸顶灯照明，光线柔和均匀，略微偏暖",
        azimuth=270.0, elevation=65.0, hardness=0.15, intensity=0.30,
    ),
    dict(
        key="u05_desk", file="u05_desk.jpg",
        title="坐在桌面右前角（桌面/坐）",
        placement="坐在画面右下方桌面的前缘，位置在粉色鼠标垫的左侧、键盘的右前方",
        pose_desc="坐在桌沿上，双腿自然垂在桌子外侧，双手撑在桌面",
        contact="桌面",
        px=80, py=86, height=36,
        light="显示器与室内灯混合照明，偏冷的白光，光线比较均匀",
        azimuth=250.0, elevation=50.0, hardness=0.28, intensity=0.34,
    ),
]


def build_prompt(c: dict) -> str:
    return (
        "图1是一张真实的室内照片。图2、图3是同一个二次元角色的设定图。\n\n"
        f"这个角色的形象是：{CHAR_DESC}。\n\n"
        f"请只把这个角色放进图1里：{c['placement']}。{c['pose_desc']}。"
        f"角色高度约占画面高度的 {c['height']}%，水平位置大约在画面 {c['px']}% 处。\n\n"
        "硬性要求：\n"
        "1. 角色形象必须与设定图完全一致：蓝色渐变长发、深蓝色鲸鱼耳、"
        "藏青女仆裙、白色荷叶边围裙及围裙上的鲸鱼图案、身后深蓝色鲸尾、深蓝色小皮鞋，"
        "不得改动配色、服装结构与发型；\n"
        f"2. 角色必须与{c['contact']}真实接触，不得悬浮在空中，也不得陷进去；\n"
        f"3. 光照：{c['light']}，角色身上的受光方向必须与这个环境光一致；\n"
        "4. 不要生成角色投射在地面上的影子（投影由后期单独合成），"
        "只保留角色自身受光以及身体与接触面之间的轻微暗部；\n"
        "5. 保持图1原有的背景、家具、物品、构图、透视、色调与清晰度完全不变，"
        "不得重绘背景，不得移动、添加或删除任何原有物品；\n"
        "6. 画面中只出现这一个角色，不要出现文字、水印、边框、多个角色，"
        "不要输出三视图并排。\n"
        "7. 如果给定位置的空间不足以容纳完整角色，可以适当缩小角色以保证完整入画。"
    )


def pick_size(im: Image.Image) -> tuple[str, float]:
    """按原图比例算输出尺寸，总像素压在 260 万以内（0.30 元档）。"""
    ratio = im.height / im.width
    w = 1280 if ratio >= 1 else 1440
    h = w * ratio
    scale = min(1.0, (2_600_000 / (w * h)) ** 0.5)
    w, h = int(w * scale / 8) * 8, int(h * scale / 8) * 8
    return f"{w}x{h}", w * h / 1e6


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    front = os.path.join(REFS, "05_chibi_front.png")
    side = os.path.join(REFS, "05_chibi_side.png")
    for p in (front, side):
        if not os.path.exists(p):
            raise SystemExit(f"缺参考图: {p}（先跑 tools/prep_refs.py）")

    per_case = int(os.environ.get("PER_CASE", "2"))
    manifests = []

    for c in CASES:
        scene_path = os.path.join(SCENES, c["file"])
        if not os.path.exists(scene_path):
            print(f"[跳过] 缺场景 {scene_path}")
            continue
        im = Image.open(scene_path)
        size, mp = pick_size(im)
        prompt = build_prompt(c)
        imgs = [image_path_to_data_uri(scene_path),
                image_path_to_data_uri(front),
                image_path_to_data_uri(side)]

        print(f"\n=== {c['key']} | {c['title']} ===")
        print(f"  原图 {im.size} -> 请求 {size} ({mp:.2f}MP)  每案例 {per_case} 候选")

        rec = {**{k: c[k] for k in ("key", "title", "px", "py", "height", "contact")},
               "scene": c["file"], "scene_size": list(im.size), "request_size": size,
               "prompt": prompt, "model": MODEL, "candidates": []}

        for i in range(per_case):
            t0 = time.perf_counter()
            status, resp = generate(prompt, images=imgs, size=size,
                                    model=MODEL, seed=1000 + i * 7)
            dt = time.perf_counter() - t0
            if status != 200:
                print(f"  [{i}] HTTP {status} 失败: {str(resp)[:300]}")
                rec["candidates"].append({"index": i, "status": status,
                                          "error": str(resp)[:500], "latency_s": round(dt, 1)})
                continue
            item = (resp.get("data") or [{}])[0]
            out = os.path.join(OUT, f"{c['key']}_{i}.png")
            download(item["url"], out)
            got = Image.open(out)
            print(f"  [{i}] HTTP 200  {dt:.1f}s  {got.size} -> {os.path.basename(out)}")
            rec["candidates"].append({"index": i, "status": 200, "latency_s": round(dt, 1),
                                      "size": item.get("size"), "file": os.path.basename(out),
                                      "usage": resp.get("usage")})
        manifests.append(rec)

    path = os.path.join(OUT, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifests, f, ensure_ascii=False, indent=2)
    n_ok = sum(len([x for x in r["candidates"] if x.get("status") == 200]) for r in manifests)
    print(f"\n完成：{len(manifests)} 案例，成功 {n_ok} 张，约 {n_ok * 0.30:.2f} 元")
    print("manifest ->", path)


if __name__ == "__main__":
    main()
