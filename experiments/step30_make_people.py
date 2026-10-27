"""step30：生成 10 张**带人物**的测试照片，姿势 / 状态 / 场景各不同。

## 为什么要专门造这一批

现有两批测试素材各有短板：

| 批次 | 内容 | 短板 |
|---|---|---|
| `batch2` | 10 张真实照片 | 有人的只有 5 张，姿势重复（站/坐） |
| `batch3` | 10 张 AI 生成的**空家具** | 一个活人都没有 |

空场景的问题在表情那轮暴露得很清楚：**batch3 只有 4 种表情，batch2 有 7 种**。
原因不是提示词，是空房间里本来就没有情绪线索——模型只能靠光线猜。
所以要测"有人在旁边"这条主场景，就得有真正带人的素材。

## 覆盖维度

生产上会遇到的是"人在做某件事，她凑过去"。所以每张都要有**明确的活动状态**，
而不是"站着看镜头"：

| # | 姿势 | 状态 | 场景 | 光线 |
|---|---|---|---|---|
| h01 | 坐 | 低头看手机、端着咖啡 | 咖啡厅靠窗 | 窗外自然光 + 室内暖灯 |
| h02 | 坐 | 看书 | 公园长椅 | 午后树影斑驳 |
| h03 | 站 | 双手插兜等车 | 地铁站台 | 冷白站台灯 |
| h04 | 侧躺 | 刷手机 | 客厅沙发 | 傍晚暖黄 |
| h05 | 走 | 回头、长发被吹起 | 街道斑马线 | 阴天散射 |
| h06 | 站 | 仰头找书、手扶书架 | 图书馆 | 偏黄顶灯 |
| h07 | 站 | 低头切菜 | 厨房料理台 | 窗外日光 |
| h08 | 靠 | 搭着栏杆看远处 | 阳台 | 黄昏逆光 |
| h09 | 坐/弯腰 | 系鞋带 | 户外石阶 | 上午侧光 |
| h10 | 盘腿坐 | 野餐 | 草地 | 晴天直射 |

## 三条硬要求

1. **人物必须完整入画**，尤其是脚——尺度换算靠"人物全身高"当尺子，
   脚被裁掉就量不出比例（这是 `profile.py` 里写死的约束）。
2. **不用真实人物**，提示词里写明"普通的年轻女性，不指向任何真实的人"，
   避免生成出来像某个具体的人。
3. **真实感优先**：手机随手拍、自然光、不过度修饰。影棚布光会让 VLM
   判断接地线和光向更容易出错，反而不适合当测试素材。

跑法：
    python experiments/step30_make_people.py
    python experiments/step30_make_people.py h01 h05     # 只造几张
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "assets", "scenes", "batch4")
MODEL = os.environ.get("ARK_IMAGE_MODEL", "doubao-seedream-5-0-flash-260915")

# 统一后缀。`不指向任何真实的人` 这句是必要的——不写的话模型容易
# 往明星脸上靠，生成出来的素材既不适合公开演示，也没必要。
STYLE = ("手机随手拍的生活照，真实感，自然光，不过度修饰，"
         "画面里没有文字和水印。一位普通的年轻女性，不指向任何真实的人。")

# (名字, 尺寸, 描述)
SCENES = [
    ("h01_cafe_phone", "1024x1536",
     "一位年轻女性坐在咖啡厅靠窗的木桌旁，侧身对着镜头，一只手端着白色马克杯，"
     "低头看着桌上的手机。**全身完整入画，能看到她的鞋子和椅子腿**。"
     "窗外自然光从左前方照进来，室内有暖色吊灯。"),
    ("h02_bench_read", "1536x1024",
     "一位年轻女性坐在公园木质长椅上看书，双腿并拢，**双脚踩在地面上**。"
     "全身完整入画，整条长椅都看得见。午后阳光穿过树叶，在地上洒下斑驳光影。"),
    ("h03_subway_wait", "1024x1536",
     "一位年轻女性站在地铁站台边等车，双手插在米色外套口袋里，侧脸看向轨道方向。"
     "**全身完整入画，双脚踩在站台地面上**。冷白色的站台照明，地面有反光。"),
    ("h04_sofa_phone", "1536x1024",
     "一位年轻女性侧躺在浅灰色客厅沙发上刷手机，头靠在扶手上，双腿曲起。"
     "能看到整张沙发和木地板。傍晚的暖黄色灯光从窗帘缝里透进来。"),
    ("h05_street_walk", "1024x1536",
     "一位年轻女性走在城市街道上，回头看向镜头，长发被风带起，一只手拎着帆布包。"
     "**全身完整入画，双脚踩在斑马线上**。阴天的柔和散射光，没有硬阴影。"),
    ("h06_library_stand", "1024x1536",
     "一位年轻女性站在图书馆的高书架前，仰头看着高处的一本书，一只手扶着书架边缘。"
     "**从头顶到膝盖都完整入画**。室内偏黄的顶灯，背景是一排排书。"),
    ("h07_kitchen_cook", "1536x1024",
     "一位年轻女性在厨房里切菜，侧身站在料理台前，低头看着砧板，一只手按着蔬菜。"
     "**上半身到大腿入画**。厨房窗外的日光斜射进来。"),
    ("h08_balcony_look", "1024x1536",
     "一位年轻女性靠在阳台栏杆上看着远处，双手搭在栏杆上，侧背对镜头。"
     "**全身完整入画，能看到她的脚和栏杆底部**。黄昏的暖橘色逆光，天空有云。"),
    ("h09_steps_shoelace", "1536x1024",
     "一位年轻女性坐在户外的石阶上系鞋带，弯腰低头，一只脚踩在下一级台阶上。"
     "**全身完整入画，能看清台阶和她踩着的鞋**。上午的侧光，台阶上有斜影。"),
    ("h10_grass_picnic", "1536x1024",
     "一位年轻女性盘腿坐在草地上野餐，面前铺着一块格纹野餐垫，旁边放着藤编篮子。"
     "**全身完整入画，能看清她的腿和草地**。晴天的明亮阳光。"),
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
        st, resp = generate(desc + STYLE, size=size, model=MODEL, seed=9100 + i)
        if st != 200:
            print(f"[{i}/{len(todo)}] {name} 失败 HTTP {st}: {str(resp)[:120]}")
            fail.append(name)
            continue
        download(resp["data"][0]["url"], path)
        print(f"[{i}/{len(todo)}] {name}  {size}  -> "
              f"{os.path.relpath(path, ROOT)}")
    if fail:
        print(f"\n失败：{', '.join(fail)}")
        return 1
    print(f"\n完成 {len(todo) - len(fail)}/{len(todo)} 张，在 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
