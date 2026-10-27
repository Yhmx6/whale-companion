"""step23：造 10 个新测试场景（本批场景是**生成出来的素材**，不是真实照片）。

## 为什么要造，而不是找

外网图库在这台机器上不通（picsum 返回 000、loremflickr 500），
而且随机图库没法**针对性构图**——这一轮要验的是"落点会不会撞到东西"，
需要的是"桌面正中正好摆着一杯咖啡"这种刁钻构图，随机图碰不到。

所以用生图模型自己造。代价是这批素材带 AI 痕迹，VLM 解析可能比真实照片
容易或难，结论要打个折。**它们只用来验流程，不能当质量基线。**

## 构图设计

5 张**故意放障碍物**在"她会想坐的位置"上，专门考验 {@link s3.scene3d} 的避让；
5 张没有障碍物，当对照——如果避让逻辑误判，对照组的 `moved_px` 会不为 0。

| # | 场景 | 障碍物 | 期望 |
|---|---|---|---|
| c01 | 咖啡厅木桌 | 正中一杯拿铁 + 托盘 | 必须挪开 |
| c02 | 客厅灰沙发 | 坐垫上一本书 | 必须挪开 |
| c03 | 书桌 | 笔记本 + 马克杯 | 必须挪开 |
| c04 | 餐桌 | 正中白花瓶 | 必须挪开 |
| c05 | 公园长椅 | 无 | 不该挪 |
| c06 | 石阶 | 无 | 不该挪 |
| c07 | 窗台 | 一盆绿植 | 必须挪开 |
| c08 | 床沿 | 一部手机 | 必须挪开 |
| c09 | 吧台 | 两个玻璃杯 | 必须挪开 |
| c10 | 木地板 | 一个背包 | 必须挪开 |

跑法：
    python experiments/step23_make_scenes.py
    python experiments/step23_make_scenes.py c01 c02     # 只造几张
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "assets", "scenes", "batch3")
MODEL = os.environ.get("ARK_IMAGE_MODEL", "doubao-seedream-5-0-flash-260915")

# 统一的风格前缀：手机随手拍的生活照，不是产品图。
# 不这么写，模型会给"影棚布光 + 完美构图"，VLM 反而不好判断真实接地线。
STYLE = ("手机随手拍的生活照，自然光，轻微手抖感，真实的家居/街景环境，"
         "不过度修饰，画面里没有文字和水印。")

SCENES = [
    ("c01_cafe_cup", "1024x1536",
     "一张咖啡厅的木质方桌，镜头正对桌子。"
     "**桌子正中央摆着一杯白色的拿铁咖啡，杯子底下有陶土色的托盘**。"
     "桌子左边和右边都是空的桌面。背景是虚化的咖啡厅内墙和暖色灯光。没有人物。"),
    ("c02_sofa_books", "1536x1024",
     "一张浅灰色的布艺三人沙发，正面拍摄。沙发上靠着两个米色抱枕，"
     "**坐垫正中间放着一本摊开的书**。沙发左右两边的坐垫是空的。"
     "背景是客厅的白色墙面和一盏落地灯。没有人物。"),
    ("c03_desk_laptop", "1536x1024",
     "一张原木色书桌，侧面平视拍摄。**桌面正中间放着一台打开的银色笔记本电脑，"
     "电脑右边是一个白色的马克杯**。桌面左侧是空的。"
     "背景是白墙和一扇窗。没有人物。"),
    ("c04_dining_vase", "1536x1024",
     "一张胡桃木餐桌，镜头略低于桌面平视。"
     "**桌子正中间摆着一个白色的陶瓷花瓶，花瓶里插着几枝干花**。"
     "桌子左右两端都是空桌面。背景是餐厅的浅色墙面。没有人物。"),
    ("c05_bench_empty", "1536x1024",
     "一条公园里的木质长椅，正面略侧的角度拍摄，整条长椅完整入画。"
     "**长椅的座面上什么都没有，完全是空的**。后面是绿树和灌木，"
     "前面是水泥步道。晴天，阳光从左边照过来。没有人物。"),
    ("c06_steps_empty", "1536x1024",
     "一段户外的石阶，从下往上拍，能看到五六级台阶。"
     "**每一级台阶上都空无一物**。台阶两边是草地和灌木。"
     "傍晚的暖色阳光从左前方照来。没有人物。"),
    ("c07_windowsill_plant", "1024x1536",
     "一扇白色窗台，正面拍摄，窗台板很宽。"
     "**窗台正中间放着一盆绿萝，花盆是灰白色的陶瓷盆**。"
     "窗台左右两端是空的。窗外是虚化的树影和天空。没有人物。"),
    ("c08_bed_phone", "1536x1024",
     "一张铺着白色床单的床，拍摄床沿一角，床沿完整可见。"
     "**床沿正中间放着一部黑色的手机，屏幕朝上**。床沿左右都是空的。"
     "背景是卧室的浅灰色墙面。没有人物。"),
    ("c09_bar_two_cups", "1536x1024",
     "一段木质吧台，正面平视拍摄，吧台面是长条形的。"
     "**吧台中间摆着两个透明的玻璃水杯，两个杯子挨在一起**。"
     "吧台左右两边都是空的。背景是虚化的酒架和暖光。没有人物。"),
    ("c10_floor_backpack", "1024x1536",
     "一片浅色的木地板，俯视略带倾斜的角度。"
     "**地板正中间放着一个深灰色的帆布背包**。背包周围的地板是空的。"
     "光线从右上方照进来，背包在地上投出影子。没有人物。"),
]


def main() -> int:
    pick = [a for a in sys.argv[1:] if not a.startswith("-")]
    todo = [s for s in SCENES if not pick or any(p in s[0] for p in pick)]
    os.makedirs(OUT_DIR, exist_ok=True)
    from api.ark import download, generate

    fail = []
    for i, (name, size, desc) in enumerate(todo, 1):
        path = os.path.join(OUT_DIR, name + ".png")
        if os.path.exists(path) and "--force" not in sys.argv:
            print(f"[{i}/{len(todo)}] {name} 已存在，跳过")
            continue
        prompt = desc + STYLE
        st, resp = generate(prompt, size=size, model=MODEL, seed=7000 + i)
        if st != 200:
            print(f"[{i}/{len(todo)}] {name} 失败 HTTP {st}: {str(resp)[:120]}")
            fail.append(name)
            continue
        download(resp["data"][0]["url"], path)
        print(f"[{i}/{len(todo)}] {name}  {size}  -> {os.path.relpath(path, ROOT)}")
    if fail:
        print(f"\n失败：{', '.join(fail)}")
        return 1
    print(f"\n完成，共 {len(todo) - len(fail)}/{len(todo)} 张，在 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
