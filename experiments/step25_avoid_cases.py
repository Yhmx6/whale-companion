"""step25：避让规则的回归用例（纯离线，0 成本、0.1 秒）。

## 为什么要有这个文件

`avoid_collision` 的规则里有四个方向相反的判断（谁更近、谁挡谁、
哪个是面、哪个是面上的东西）。这一轮我在里面栽了两次：

- 把 `base < foot` 读成「物体更近」，于是「修正」成了相反的判据，
  结果 c10 里背包明明在她前面、她直接穿了过去；
- 给「小物件」设了 35% 尺寸门槛，c01 那张特写的杯盘（各占 44% / 54% 画面宽）
  被判成「大家具」放过了。

两次都不是靠读代码发现的，是跑完 10 张图（8 分钟、1.2 元）看结果才发现的。

所以把走过的 case 全固化在这里。改任何一条规则，先跑这个：
0.1 秒出结果，比跑一轮批量快几千倍。

## 用例怎么来的

每条物体的数据都原样抄自真实运行时的 VLM 输出
（`output/batch3/*/meta.json` 的 `scene2d.objects`），不是编的。
编的数据不会暴露真实框的怪样子——比如桌子 `box=[0,855,999,999]`
横跨整个画面、`base_y=999` 顶到画面底边。

跑法：
    python experiments/step25_avoid_cases.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from s3.scene3d import (Scene2D, SceneObject, avoid_collision,  # noqa: E402
                        snap_to_surface)


def O(**kw) -> SceneObject:
    return SceneObject(**kw)


# (名字, 场景物体, W, H, 落点 x 比, 落点 y 比, 期望挪动?, 期望坐于, 说明)
CASES = [
    ("p04 咖啡桌·杯子在正中", [
        O(kind="table", box=[0, 855, 999, 999], base_y=855, sit=True, sit_y=855),
        O(kind="cup", box=[306, 787, 608, 947], base_y=947),
        O(kind="plate", box=[329, 863, 671, 971], base_y=969),
    ], 1024, 1536, 0.457, 0.858, True, "桌子",
     "用户报的那张：她原本坐在杯口上。杯子 base 比脚线还低，"
     "按深度判据该放过，靠「不是家具就躲」拦住"),

    ("c01 咖啡厅·杯盘占半幅", [
        O(kind="table", box=[0, 515, 999, 999], base_y=999, sit=True, sit_y=805),
        O(kind="cup", box=[316, 482, 755, 700], base_y=695),
        O(kind="plate", box=[226, 605, 766, 753], base_y=748),
    ], 1024, 1536, 0.60, 0.79, True, "桌子",
     "特写：杯盘各占画面 44% 和 54% 宽。真放不下，应走 forced 取最小重叠"),

    ("c05 长椅·空的", [
        O(kind="bench", box=[142, 190, 880, 935], base_y=935, sit=True, sit_y=637),
        O(kind="plant", box=[0, 275, 999, 785], base_y=785),
    ], 1536, 1024, 0.50, 0.53, False, "长椅",
     "对照组。灌木框横跨全宽是背景面，不能当障碍；"
     "落点要靠 snap 吸到座面，否则长椅反被判成障碍"),

    ("c06 台阶·空的", [
        O(kind="step", box=[0, 300, 999, 999], base_y=999, sit=True, sit_y=640),
    ], 1536, 1024, 0.55, 0.55, False, "台阶",
     "对照组：只有一件横跨全宽的台阶，不该有任何挪动"),

    ("c08 床沿·手机在中间", [
        O(kind="bed", box=[0, 443, 999, 999], base_y=999, sit=True, sit_y=777),
        O(kind="phone", box=[432, 451, 577, 564], base_y=564),
    ], 1536, 1024, 0.50, 0.61, True, "床",
     "手机 base 比脚线高（在更远处），深度判据会判合法，"
     "靠「不是家具就躲」拦住"),

    ("c09 吧台·两个杯子", [
        O(kind="counter", box=[0, 605, 999, 999], base_y=999, sit=True, sit_y=841),
        O(kind="cup", box=[395, 430, 505, 680], base_y=680),
        O(kind="cup", box=[505, 430, 610, 680], base_y=680),
    ], 1536, 1024, 0.50, 0.61, True, None,
     "台面横跨全宽但她坐在上面，筛子 3 跳过家具，只躲两个杯子"),

    ("c10 木地板·背包", [
        O(kind="bag", box=[226, 191, 813, 726], base_y=726),
    ], 1024, 1536, 0.53, 0.33, True, None,
     "关键回归：背包 base 比她脚线低得多（更近），深度判据判冲突。"
     "我一度把方向改反，这张就穿模了"),

    ("c04 餐桌·花瓶在左", [
        O(kind="table", box=[0, 635, 999, 999], base_y=999, sit=True, sit_y=853),
        O(kind="vase", box=[445, 488, 565, 723], base_y=720),
        O(kind="chair", box=[820, 580, 999, 999], base_y=999, sit=True),
    ], 1536, 1024, 0.64, 0.85, False, "桌子",
     "花瓶在画面中间偏左，她本就在右边，不该挪。"
     "用来抓「过度避让」这种反向错误"),
]


def main() -> int:
    bad = 0
    for (name, objs, W, H, cx_r, by_r, expect_move,
         expect_sits, why) in CASES:
        scene = Scene2D.build(objs)
        target_h = H * 0.264
        target_w = target_h * 0.70
        cx, by = cx_r * W, by_r * H
        snap = snap_to_surface(scene, cx, by, W, H, target_h)
        res = avoid_collision(scene, cx, snap["y"], target_w, target_h, W, H)
        moved = res["moved_px"]
        sits = (res["sits_on"] or {}).get("name")
        ok = (moved > 8) == expect_move
        if expect_sits is not None and sits != expect_sits:
            ok = False
        if res.get("conflicts"):
            ok = False                    # 任何「整条横扫都撞」都算失败
        bad += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        print(f"      挪 {moved:.0f}px（期望{'挪' if expect_move else '不动'}）"
              f"  坐于 {sits}  snap={snap['snapped']}"
              f"{'  forced' if res.get('forced') else ''}")
        if not ok:
            print(f"      !! {why}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
