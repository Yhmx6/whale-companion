"""第 1 步实验：用真实照片跑一次 Seedream 多图编辑（Adapters 的 Seedream 通路）。

本次目的**不是测成功率**，而是打通链路并验证四件事：
1. 三张图（场景 + 正面 + 侧面）能否一次传入并被正确理解
2. 模型是否遵从框位与比例约束
3. 是否能听懂"不要生成地面投影阴影"（对应《07》U-10）
4. 输出宽高比是否跟随原照片（对应《07》U-07 的 Seedream 侧对照）

真正测成功率需要 30 张用户真实照片，见 PRD 与《07》U-04。
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from api.ark import download, generate, image_path_to_data_uri  # noqa: E402

OUT = os.path.join(ROOT, "output", "step1")
MODEL = "doubao-seedream-5-0-pro-260628"   # 账号实际有权限的模型


def fetch_scene(path: str) -> str:
    """准备一张写实场景底图。

    首选外网免费图库（Lorem Picsum，CC0）；本机访问境外站点超时，
    因此退化方案是**用 Seedream 自己生成一张写实照片当底图**。
    两者都只用于打通链路；正式实验必须用用户真实手机照片，
    否则测出来的可用率会偏乐观。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return path

    # 首选：外网图库（失败就跳过，不阻塞）
    try:
        import urllib.request
        url = "https://picsum.photos/seed/whalegirl01/1280/960"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r, open(path, "wb") as f:
            f.write(r.read())
        print("底图来源：Lorem Picsum（图库照片）")
        return path
    except Exception as e:
        print(f"图库不可达（{type(e).__name__}），改用 Seedream 生成写实底图")

    # 退化：用文生图造一张写实场景
    prompt = (
        "一张写实的手机随手拍照片：公园里的石板小路，左侧有树和灌木，"
        "右侧是空旷的草地，画面中下部是大片平整的地面，阴天柔和的自然光从左上方照来，"
        "地面上有淡淡的树影。画面里没有任何人。真实的照片质感，轻微的手机镜头畸变，"
        "自然色彩，没有文字，没有水印。"
    )
    status, resp = generate(prompt, size="1280x960", model=MODEL)
    if status != 200:
        raise SystemExit(f"生成底图失败: {status} {str(resp)[:600]}")
    download(resp["data"][0]["url"], path)
    print("底图来源：Seedream 文生图（合成照片，仅用于链路验证）")
    return path


def build_prompt(px: float, py: float, height_frac: float) -> str:
    return (
        "图1是一张真实环境的照片。图2、图3是同一个角色的设定图（Q版，"
        "蓝色渐变长发、深蓝色鲸鱼耳、藏青女仆裙配白色荷叶边围裙、"
        "围裙上有鲸鱼图案、身后有一条粗大的深蓝色鲸尾）。\n\n"
        f"任务：只把图2/图3中的这个角色完整地画进图1里。角色站在画面水平约 {px:.0f}% 处，"
        f"脚底落在画面垂直约 {py:.0f}% 的位置，角色高度约占画面高度的 {height_frac*100:.0f}%。\n\n"
        "硬性要求：\n"
        "1. 角色形象必须与设定图一致：蓝色渐变长发、深蓝色鲸鱼耳、藏青女仆裙、"
        "白色荷叶边围裙及其上的鲸鱼图案、身后的深蓝色鲸尾，不得改动配色与服装结构；\n"
        "2. 角色双脚落地并与地面接触，身体竖直轴与画面竖直方向一致，不得悬浮、不得陷入地面；\n"
        "3. 光源来自画面左侧（主光方位约 180 度、仰角约 42 度、柔光），"
        "角色左亮右暗，角色受光必须与环境光照一致；\n"
        "4. 不要生成角色投射在地面上的影子（投影由后期单独合成），"
        "只保留角色自身受光以及脚部与地面接触处的轻微暗部；\n"
        "5. 保持图1原有的背景、构图、透视、色调与清晰度完全不变，不得重绘背景；\n"
        "6. 画面中只出现这一个角色，不要出现文字、边框、多个角色、不要输出三视图并排。"
    )


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    scene_path = fetch_scene(os.path.join(ROOT, "assets", "scenes", "picsum_01.jpg"))

    refs = os.path.join(ROOT, "assets", "refs", "crops")
    front = os.path.join(refs, "05_chibi_front.png")
    side = os.path.join(refs, "05_chibi_side.png")
    for p in (scene_path, front, side):
        if not os.path.exists(p):
            raise SystemExit(f"缺文件: {p}")

    im = Image.open(scene_path)
    print(f"场景尺寸: {im.size}  宽高比 {im.width/im.height:.3f}")

    prompt = build_prompt(px=42, py=78, height_frac=0.28)
    imgs = [image_path_to_data_uri(scene_path),
            image_path_to_data_uri(front),
            image_path_to_data_uri(side)]

    # 输出尺寸按原照片等比、总像素控制在 261 万以内（0.30 元档）
    w, h = 1280, int(round(1280 * im.height / im.width / 8) * 8)
    size = f"{w}x{h}"
    print(f"请求 size={size}  像素={w*h/1e4:.0f}万  模型={MODEL}")

    t0 = time.perf_counter()
    status, resp = generate(prompt, images=imgs, size=size, model=MODEL)
    dt = time.perf_counter() - t0
    print(f"HTTP {status}  耗时 {dt:.1f}s")

    if status != 200:
        print("失败：", str(resp)[:1500])
        return

    data = resp.get("data") or []
    print(f"返回 {len(data)} 张 | usage={resp.get('usage')}")
    for i, item in enumerate(data):
        url = item.get("url")
        out = os.path.join(OUT, f"case01_{i}.png")
        download(url, out)
        got = Image.open(out)
        print(f"  [{i}] {item.get('size')}  实际 {got.size}  -> {os.path.basename(out)}")

    with open(os.path.join(OUT, "case01_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(f"model={MODEL}\nsize={size}\nlatency={dt:.1f}s\n\n{prompt}\n")


if __name__ == "__main__":
    main()
