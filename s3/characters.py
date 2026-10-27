"""角色注册表：把"是哪个角色"从代码里抽出来。

## 为什么要抽

在这之前，角色是**写死在代码里的**：
`_encode_refs()` 固定读 `05_chibi_front.png` / `05_chibi_side.png`，
`build_sprite_prompt()` 固定写"蓝色渐变长发、Q 版 2 到 2.5 头身"，
立绘缓存固定存在 `assets/sprites/`。加第二个角色就得改这三处，
而且两套立绘会撞名。

## 一个角色由什么定义

| 字段 | 作用 | 错了会怎样 |
|---|---|---|
| `refs` | 生成立绘时的参考图 | 换脸、换衣服 |
| `identity` | 提示词里的外形描述 | 同上 |
| `body` | 体型描述（头身比） | Q 版和成人比例串味 |
| `scale` | 相对人物身高的比例 | 大小全错 |
| `sprite_dir` | 立绘缓存目录 | 两个角色的立绘互相覆盖 |

`scale` 是**相对人物身高**，不是绝对厘米：
- `chibi` 0.26 → 175cm 的人旁边约 45cm，读作"手办/玩偶"
- `mature` 0.92 → 约 160cm，读作"站在旁边的人"

## 体型和姿态描述是两件事

动作库里的 `desc` 描述的是**姿态**（坐在哪、手放哪、朝哪看），
它在不同体型之间基本通用；但有些词是 Q 版专用的（"短腿"、"圆圆的小团子"）。
所以 `Action` 保留 `desc` 作为默认，需要时用 `desc_by_body` 覆盖——
不把两份文案硬塞进同一个字段。

## 关于等身角色的已知缺口

- **没有遮挡**：她永远画在最前面。站/坐在人物旁边没问题，
  "从背后探头"这类需要被人物手臂挡住的就不能用（立绘路线的固有限制）。
- **物件路线的尺度常量不适用**：`SEAT_H_FRAC` 是按玩偶定的（占画面高 12~42%），
  等身角色坐上去会偏小，而且 `SEAT_H_RANGE` 会把她夹到 42% 以内。
  等身角色目前只走人物路线。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHAR_ROOT = os.path.join(ROOT, "assets", "characters")


# 参照人物的假定身高。照片里量不出绝对厘米数，只能如此：
# "她 60cm 高" → 占人物身高的 60/175 ≈ 0.343。换一批人群（儿童照片）会偏，
# 但比"拍脑袋给个百分比"有物理含义，至少改身高时只改一个数。
REF_PERSON_CM = 175.0


@dataclass(frozen=True)
class Character:
    id: str
    name: str
    refs: tuple[str, ...]          # 参考图（绝对路径），顺序传给生成接口
    identity: str                  # 提示词里的外形描述
    body: str                      # 体型 / 头身比
    height_cm: float               # **真实身高**，产品的硬规定
    scale: float                   # 相对人物身高
    sprite_dir: str = ""           # 立绘缓存目录
    note: str = ""

    def sprite_dir_or_default(self) -> str:
        return self.sprite_dir or os.path.join(CHAR_ROOT, self.id, "sprites")


def _ref(*parts: str) -> str:
    return os.path.join(ROOT, "assets", *parts)


CHIBI = Character(
    id="chibi",
    name="Q 版鲸鱼娘",
    # 老仓是 assets/refs/crops/05_chibi_front.png，不动它——已经在用
    refs=(_ref("refs", "crops", "05_chibi_front.png"),
          _ref("refs", "crops", "05_chibi_side.png")),
    # 权威形象见 `s3/s2_prompt.py::CHAR_DESC`（用户提供的定稿描述）。
    # 这里保留简短版：立绘生成走参考图，短句 + 两张设定图比长句更稳，
    # 长句会稀释"三头身/小巧"这类结构约束。
    identity=("三头身 Q 版，深海蓝到海蓝的渐变长卷发、头顶弧形呆毛、白色蕾丝女仆发箍配"
              "亮蓝蝴蝶结、鲸鱼鳍状耳、深蓝女仆裙配白色荷叶边围裙（正中印小蓝鲸）、"
              "身后深蓝鲸尾、白色花边短袜配深蓝玛丽珍鞋"),
    body="Q 版 2 到 2.5 头身，头大身子小，四肢短圆",
    # 用户定死的产品规格：她真实身高 60 厘米。
    # 之前这里写 0.26（=45cm），是按"40cm 玩偶 + 一点余量"拍的。
    # 现在改成由身高反推：60 / 175 ≈ 0.343，改身高只需改 `height_cm`。
    height_cm=60.0,
    scale=60.0 / REF_PERSON_CM,
    # 老的 6 张立绘就在 assets/sprites/，直接沿用，不搬文件
    sprite_dir=os.path.join(ROOT, "assets", "sprites"),
    note="原型期角色，动作库和尺度常量都是按它标的",
)

MATURE = Character(
    id="dafeiyu_mature",
    name="鲸鱼娘（成人形态）",
    refs=(_ref("characters", "dafeiyu_mature", "front.png"),
          _ref("characters", "dafeiyu_mature", "side.png"),
          _ref("characters", "dafeiyu_mature", "face.png")),
    identity=("深蓝色长卷发（发梢渐变到浅蓝）、蓝色鲸鱼鳍状的耳、藏青与宝蓝拼接的女仆裙、"
              "白色荷叶边围裙（左下摆有鲸鱼刺绣）、胸前白底黑边束腰马甲、"
              "白色长袜、深蓝色玛丽珍小皮鞋、身后蓝色鲸尾"),
    # 这条必须改：拿"Q 版 2 头身"去生成成人设定图，模型会给她画成大头娃娃
    body="成年女性比例，约 7 头身，肩窄腰细、腿长",
    height_cm=160.0,
    scale=160.0 / REF_PERSON_CM,
    note="等身；没有遮挡处理，只能用于不与人重叠的动作",
)

REGISTRY: dict[str, Character] = {c.id: c for c in (CHIBI, MATURE)}
DEFAULT_ID = "chibi"


def get_character(cid: str | None = None) -> Character:
    """按 id 取角色；未知 id 直接退到默认，不抛异常。

    退到默认而不是报错：角色 id 会从 API 参数、前端下拉框一路传进来，
    一个拼错不该让整张图失败——用错的角色至少还能出图，报错就是白跑。
    """
    return REGISTRY.get(cid or DEFAULT_ID, REGISTRY[DEFAULT_ID])


def list_characters() -> list[dict]:
    return [{"id": c.id, "name": c.name, "scale": c.scale, "note": c.note}
            for c in REGISTRY.values()]
