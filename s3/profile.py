"""照片画像（PhotoProfile）：一次 VLM 调用，把生成前后都要用的信息一次取回。

## 为什么要有这一层

改造前，一张照片在两条互不相干的地方被 VLM 问了两次：

1. `build_companion_prompt` 之前 —— 问腿部框，为了往提示词里写位置约束；
2. 后处理之前 —— 又问了一遍人物框和腿部框，为了当尺度尺子、并顺手问光向。

两次问的是同一张图上同样的事，白花一次调用、还多 1.5 秒。更糟的是两次结果可能
不一致（框略有差异），导致"提示词里说在 A、后处理按 B 算"。

现在统一成一次 `vlm_profile()`，产出 `PhotoProfile`，供三处消费：

| 字段 | 谁用 | 干什么 |
|---|---|---|
| framed | 动作路由 | 硬过滤：半身照不许选腿部动作 |
| person | S2.5 尺度归一 | 当尺子，目标高 = 人物身高 × scale_ratio |
| legs | S2.5 定位 | 定水平位置和落地线（底边就是鞋底） |
| light | S3 阴影 | 选 sun_left / left / auto |
| selfie | 动作路由 | 自拍照脸占比大，优先上半身动作 |
| has_prop | 预留 | 有可交互物件时以后走物件路线 |

## 为什么不让它一次把东西全找出来

`has_prop` 只做"有没有"的布尔判断，不要求给出物件的位置和类别。

实测经验：让 lite 级 VLM 同时做"多目标检测 + 类别判断 + 遮挡判断"，
漏检和错框会明显变多，而它做"两个框 + 几个单选判断"非常稳（框误差 <2%）。
需要物件框时另外发一次专门的问题，失败也只损失物件分支，不影响人物分支。

## framing 的判定口径

这是硬过滤的依据，口径必须写死，否则同一个词不同模型理解不一样：

- `closeup`：脸占画面高度一半以上，或只能看到头肩
- `half`   ：能看到腰部以上（含手），看不到膝盖
- `full`   ：能看到膝盖以下，或整个人完整入画
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from s3.actions import Action, enabled_actions

PROFILE_SYS = (
    "你是照片构图分析助手。给你一张照片，回答几个单选问题和两个外接框。\n"
    "坐标归一化到 0 到 999：左上角是 (0,0)，右下角是 (999,999)。\n"
    "\n"
    "framing 是景别，只能取以下三个值之一：\n"
    '  "closeup" 脸占画面高度一半以上，或只能看到头肩\n'
    '  "half"    能看到腰部以上（含手），但看不到膝盖\n'
    '  "full"    能看到膝盖以下，或整个人完整入画\n'
    "\n"
    "person 是「主要那个人」的外接框：上边取头发最高点，下边取鞋底最低点。\n"
    "  画面里有多个人的时候，只框离镜头最近、最居中的那一个，不要把其他人框进来。\n"
    "  景别是 closeup 时，如果画面里没有完整的头或脚，就按画面内实际能看到的部分给框。\n"
    "legs 是主要那个人双腿的外接框：上边取胯部，下边取鞋底最低点；看不到腿就给 null。\n"
    "\n"
    "light 是主光来向。**不依赖画面里有没有人**——看物体的亮暗面、看投影朝哪边倒：\n"
    '  "left"  亮面在左、影子朝右（光从左边来）\n'
    '  "right" 亮面在右、影子朝左（光从右边来）\n'
    '  "top"   光从正上方来，影子缩在物体脚下，桌面地面最亮\n'
    '  "flat"  阴天或室内散射光，各个面亮度差不多、看不出投影方向\n'
    "  画面里没有人时，照样根据家具、墙面、地面、树木的明暗和投影来判断，必须回答。\n"
    "\n"
    "selfie 是布尔：这张照片是不是本人手持设备自拍（手臂长度、脸离镜头近、"
    "视角略微从上往下）。\n"
    "has_prop 是布尔：画面里有没有清晰可坐或可依靠的物体。"
    "范围放宽——长椅、沙发、椅子、床、桌子、餐桌、书桌、台面、吧台、台阶、"
    "树干、栏杆都算。只要画面里有一件坐得上去或靠得上去的东西，就填 true。"
    "只判断有没有，不用给出位置。\n"
    "\n"
    "props 是这些可坐物件的列表（最多 3 个，按可坐性从高到低）。没有就输出空列表。\n"
    "  kind 只能取：sofa / bench / chair / bed / table / counter / step\n"
    "  桌子、台面的边缘同样可以坐，桌面台面也要报；近景只看到一段桌沿也要报。\n"
    "  不要因为\"这张照片看起来不像能坐人\"就不报，一件都找不到才输出空列表。\n"
    "  seat 框住的是**坐垫朝外那一面**：就是你正对着看到的那块竖直的布面/板面。\n"
    "    x1 / x2 是它的左右沿；y1 是这一面的**上沿**，也就是人坐上去时臀部落到的那条线；\n"
    "    y2 是下沿。\n"
    "  常见错误：不要框整件沙发（含靠背和扶手）；不要框裙边、底座、地面；"
    "不要把沙发前面的茶几算进去。\n"
    "  拿不准时，找坐垫上「水平的面」和「竖直的面」相交的那条棱，它就是 y1。\n"
    "  ✗ 不要把朝上的水平面当成这一面。桌面尤其容易错：从侧面看一张餐桌，"
    "你会看到一大块朝上的桌面，**不要**把它整个框进来（那会得到 y1≈0、x1=0、x2=999）；"
    "要的是桌面**前沿那条棱**所在的高度，seat 的上下两条边都贴着那条棱附近。\n"
    "\n"
    "objects 是画面里**所有占据空间的实体东西**，用来判断把小人物放上去会不会撞到。\n"
    "  这和 props 不是一个用途：props 只关心「能不能坐」，objects 关心「占了哪块地方」。\n"
    "  所以小东西也要报：杯子、盘子、碗、书、手机、笔记本电脑、花瓶、花盆、包、"
    "  台灯、遥控器，以及人。最多 6 个，按在画面里占的面积从大到小排。\n"
    "  kind 只能取：sofa / bench / chair / bed / table / counter / step / cup / plate / "
    "book / phone / laptop / vase / plant / bag / lamp / person / other\n"
    "  每项四个字段：\n"
    "    box     外接框 [x1, y1, x2, y2]，框住它**整个可见轮廓**；\n"
    "    base_y  它**底部与支撑面接触**的那条线的 y。杯子放在桌上 = 杯底那条线；\n"
    "            椅子放在地上 = 椅脚那条线；挂在墙上的画 = 它的下沿。\n"
    "    support 它底下撑着什么：\"floor\" 地面 / \"table\" 桌面或台面 / \"wall\" 墙 / \"unknown\"\n"
    "    solid   它是不是**实心的、人会撞上去**的东西。杯子、书、电脑、家具都是 true；\n"
    "            地毯、影子、窗户上的反光、远处的风景是 false。\n"
    "  只要能看出是独立一件东西就报，哪怕很小。桌上那杯咖啡如果漏了，"
    "小人就会被放到杯子里面去。\n"
    "\n"
    "只输出 JSON，不要解释、不要代码块标记。格式必须是：\n"
    '{"framing": "full", "person": [x1, y1, x2, y2], "legs": [x1, y1, x2, y2], '
    '"light": "left", "selfie": false, "has_prop": true, '
    '"props": [{"kind": "sofa", "seat": [x1, y1, x2, y2]}], '
    '"objects": [{"kind": "table", "box": [x1, y1, x2, y2], "base_y": 880, '
    '"support": "floor", "solid": true}, '
    '{"kind": "cup", "box": [x1, y1, x2, y2], "base_y": 860, '
    '"support": "table", "solid": true}]}\n'
    "看不到的部分写 null。\n"
    "画面里根本没有人时：framing 写 \"none\"、person 和 legs 写 null，"
    "但 light、has_prop、props、objects **仍然必须回答**。例如：\n"
    '{"framing": "none", "person": null, "legs": null, "light": "left", '
    '"selfie": false, "has_prop": true, '
    '"props": [{"kind": "bench", "seat": [120, 560, 880, 700]}], '
    '"objects": [{"kind": "bench", "box": [100, 480, 900, 700], "base_y": 700, '
    '"support": "floor", "solid": true}]}'
)

FRAMINGS = ("closeup", "half", "full")
LIGHTS = ("left", "right", "top", "flat")

# 动作的 anchor_region -> 给 VLM 的部位描述。只问当前动作需要的那一个，
# 不问"所有部位"：多问一个部位就多一份错框风险，而不需要的部位问了也没用。
ANCHOR_ASK = {
    "shoulder": "人物的双肩。要一条横向的窄框：上边取左右肩峰的最高点，"
                "下边取肩膀与上臂交界处，左右覆盖两侧肩膀外缘。不要框到脖子和头",
    "torso": "人物上衣的下摆边缘。要一条横向的窄框：只框衣服最下缘那一圈布料，"
             "上边不要超过下摆以上 3% 画面高度，不要框整件衣服",
    "head": "人物的头顶。框住头发最高处到眉骨这一段（含头顶头发）",
    "back": "人物的后背。左肩到右肩、肩线到腰线之间的区域",
}


@dataclass
class PhotoProfile:
    framing: str = ""                  # closeup / half / full / none
    person: list[int] | None = None
    legs: list[int] | None = None
    anchor: list[int] | None = None    # 当前动作的贴附部位框，见 ANCHOR_ASK
    anchor_part: str = ""
    light: str = ""
    selfie: bool = False
    has_prop: bool = False
    # 可坐物件。以前这是**第二次 VLM 调用**（`vlm_props`）拿的，
    # 现在并进同一次画像——物件路线因此从两次调用降到一次（5~7s → 2~3s）。
    # 类型用字符串是因为 `Prop` 定义在本类之后。
    props: "list[Prop]" = field(default_factory=list)
    # 2.5D 场景的原料：画面里所有占空间的实体（含杯子、书这类小物件），
    # 每条带外接框 + 接地线 + 支撑面 + 是否实体。建场景见 `s3/scene3d.py`。
    # `props` 回答"能不能坐"，`objects` 回答"占了哪块地方"——两个不同的问题，
    # 所以是两个字段，不是把 props 扩成 objects（那会让"可坐性"和"占位"互相稀释）。
    objects: "list" = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def has_person(self) -> bool:
        return bool(self.person) and self.framing not in ("", "none")

    def person_h_px(self, h_px: int) -> float | None:
        """人物框的**垂直高度**（像素）。没有人物框就返回 None。

        ⚠️ 这不是"人物身高"。站着时两者近似相等，但躺着时她身体的高度
        只有躯干那么厚，身高跑到了横向去。要算"她该多大"请用
        `body_len_px()`——这条坑见它的注释。
        """
        if not self.person:
            return None
        return (self.person[3] - self.person[1]) / 999.0 * h_px

    def body_len_px(self, w_px: int, h_px: int) -> float | None:
        """人物**身体长轴**的像素长度——也就是"身高"该占多少像素。

        ## 为什么不能直接用框高

        `person_h_px * scale` 这个公式隐含假设"人站着，框高就是身高"。
        躺着的时候完全不成立。

        实测用户那张照片（1400×933，人侧躺在沙发上）：
        人物框 `[203, 340, 816, 647]`，
        **框高只有 287px**（她身体的厚度），**框宽 859px**（她躺着的长度）。
        按框高算出她 60cm 只有 98px = 画面高 10.5%，
        而真实该是 295px = 31.6%——**小了 3 倍**。
        用户的原话是"尺寸太小了，不符合 60cm"。

        ## 判据

        取框的**较长边**作为身体长度，但只在"明显是横躺"时才用宽度：
        `宽 > 高 × 1.2` 才判为躺/卧。

        为什么留 1.2 的余量而不是 1.0：坐着的人框宽可能接近框高
        （比如盘腿坐、抱膝坐，人是一个近似方形的团），那种情况身高仍然
        是纵向的。只有明显横向铺开（侧躺、趴着）才该改用宽度。

        ## 这个近似的边界

        斜躺着时框宽只是水平跨度，比真实身长略短（对角线更长）。
        但差得不多，而且比"用框高"那种差 3 倍的情况好得多。
        正对镜头的躺姿最准；侧向透视强的会偏小一些。
        """
        if not self.person:
            return None
        x1, y1, x2, y2 = self.person
        w = (x2 - x1) / 999.0 * w_px
        h = (y2 - y1) / 999.0 * h_px
        return max(w, h) if w > h * 1.2 else h

    def legs_h_px(self, h_px: int) -> float | None:
        if not self.legs:
            return None
        return (self.legs[3] - self.legs[1]) / 999.0 * h_px

    def foot_y_px(self, h_px: int) -> float | None:
        """脚底线（地面线）。贴地动作的落点。

        优先取**人物框**底边而不是腿框底边：两者的定义都写了"下边取鞋底最低点"，
        但实测腿框经常被截断（s04_full 里腿框底边 916、人物框 986，差了 7% 画面高）。
        用腿框底边会让她凭空浮起 120 像素，用人物框底边才落在鞋底上。
        """
        if self.person:
            return self.person[3] / 999.0 * h_px
        if self.legs:
            return self.legs[3] / 999.0 * h_px
        return None

    def describe(self) -> str:
        bits = [f"景别={self.framing or '未知'}"]
        if self.selfie:
            bits.append("自拍")
        bits.append(f"光={self.light or '未知'}")
        if self.has_prop:
            bits.append("有可交互物件")
        return "，".join(bits)


def _bbox(v, min_w: int = 40, min_h: int = 60) -> list[int] | None:
    """宽松的框校验：只挡退化框。

    比 `actions._checked` 宽松，因为 closeup 照片里人物框天然会贴满画面，
    用那条 `w > 850 判为不可用` 的规则会把正常结果判死。
    """
    if not isinstance(v, list) or len(v) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(t) for t in v[:4]]
    except (TypeError, ValueError):
        return None
    if x2 - x1 < min_w or y2 - y1 < min_h:
        return None
    return [x1, y1, x2, y2]


def _user_text(anchor_part: str) -> str:
    base = ("请给出这张照片的画像。其中 props 要标的是**坐垫朝外竖立面**那一条前沿线，"
            "不是朝上的水平面——桌子的桌面朝上的那一大块不算。")
    ask = ANCHOR_ASK.get(anchor_part)
    if not ask:
        return base
    return (base + f"\n另外，请在 anchor 字段里给出{ask}。"
                   "这个部位接下来用来安放小人，位置偏差会直接看出来，请尽量准确。")


def _validate_anchor(person: list[int] | None,
                     anchor: list[int] | None) -> list[int] | None:
    """锚点框校验。

    主要挡两种错：
      1. VLM 把"双肩"理解成"框住整个人"，返回一个和人物框一样大的框。
         那样把小人搬过去就坐在头顶上了。
      2. 部位根本不在画面里时，VLM 会返回贴着画面边缘的横条
         （实测 s01_mirror 问衣摆，回的是 [257, 970, 576, 999]，
          因为那张照片拍到腰就被裁掉了）。局部部位不该贴边框。

    判据：
      a. 四边都不许贴画面边（留 2 个归一化单位的余量）；
      b. 不比人物框更宽；
      c. 高度不超过人物身高的 40%。
    """
    if not anchor:
        return None
    ax0, ay0, ax1, ay1 = anchor
    if min(ax0, ay0) <= 2 or max(ax1, ay1) >= 997:
        return None
    if not person:
        return anchor if (ax1 - ax0) > 60 else None
    px0, py0, px1, py1 = person
    if (ax1 - ax0) > (px1 - px0) * 1.05:
        return None
    if (ay1 - ay0) > (py1 - py0) * 0.40:
        return None
    return anchor


def vlm_profile(photo_path: str, anchor_part: str = "",
                timeout: int = 60) -> PhotoProfile:
    """一次调用拿到整张照片的画像。失败时返回一个空画像（不抛异常）。

    `anchor_part` 传动作的 `anchor_region`（shoulder / torso / head / back）时，
    额外返回该部位的锚点框。这是给上半身动作的重定位用的——实测 `peek` 会把
    小人画成浮在肩膀外侧的空气里，"扒着肩膀"这类描述约束不住，必须拿到肩膀框
    再程序化搬过去。只问当前动作需要的那一个部位，多问一个就多一份错框风险。
    """
    from api.ark import ENV, _post, image_path_to_data_uri

    payload = {
        "model": ENV.get("ARK_VLM_MODEL", "doubao-seed-2-1-lite-260915"),
        "messages": [
            {"role": "system", "content": PROFILE_SYS},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": image_path_to_data_uri(photo_path, max_side=640)}},
                {"type": "text", "text": _user_text(anchor_part)},
            ]},
        ],
        "temperature": 0.0,
        # 520 不够了：多了 objects 数组（最多 6 条 × 框 + 接地线 + 支撑面 + 实体）。
        # 截断的后果不是"少几个字段"而是**整段 JSON 坏掉**，只能靠正则兜底，
        # 而兜底解析不出 objects，避让就静默失效了。
        "max_tokens": 980,
        "thinking": {"type": "disabled"},      # 关思考：39 秒 -> 约 1.5 秒
    }
    st, resp = _post("/chat/completions", payload, timeout=timeout)
    if st != 200:
        return PhotoProfile(raw={"error": f"HTTP {st}: {str(resp)[:160]}"})
    try:
        txt = resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return PhotoProfile(raw={"error": "bad_response"})
    txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data = None
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # 兜底：JSON 拼坏时按 key 逐个挖
        data = {}
        fm = re.search(r'"?framing"?\s*:\s*"([a-z]+)"', txt)
        if fm:
            data["framing"] = fm.group(1)
        for key in ("person", "legs"):
            m = re.search(key + r'"?\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', txt)
            if m:
                data[key] = [int(g) for g in m.groups()]
        lm = re.search(r'"?light"?\s*:\s*"([a-z]+)"', txt)
        if lm:
            data["light"] = lm.group(1)
        data["props"] = _props_from_text(txt)

    if not isinstance(data, dict):
        return PhotoProfile(raw={"error": "bad_json", "text": txt[:160]})

    framing = str(data.get("framing") or "").strip().lower()
    if framing not in FRAMINGS + ("none",):
        framing = ""
    light = str(data.get("light") or "").strip().lower()
    if light not in LIGHTS:
        light = ""

    person = _bbox(data.get("person"))
    legs = _bbox(data.get("legs"), min_w=30, min_h=50)
    legs = _validate_legs(person, legs)
    anchor = None
    if anchor_part:
        anchor = _bbox(data.get("anchor"), min_w=60, min_h=10)
        anchor = _validate_anchor(person, anchor)

    return PhotoProfile(
        framing=framing,
        person=person,
        legs=legs,
        anchor=anchor,
        anchor_part=anchor_part if anchor else "",
        light=light,
        selfie=bool(data.get("selfie")),
        has_prop=bool(data.get("has_prop")),
        props=_parse_props(data),
        objects=_parse_objects(data),
        raw=data,
    )


def _validate_legs(person: list[int] | None,
                   legs: list[int] | None) -> list[int] | None:
    """用人体比例校验腿框，挡掉"贴着画面底边的整幅横条"。

    半身照里腿本来就看不见，VLM 会退化成返回 [0, 910, 998, 999] 这种横条
    （实测 s03_sofa）。这种框拿去做位置约束会把角色按到画面最底下，
    拿去做落地线会落在一个凭空捏造的位置上。

    两条判据，都来自人体测量学：
      1. 腿高 / 身高 应落在 28%~75%（正常约 45%~58%，留足余量给
         坐姿、蹲姿、以及 VLM 把胯部标高标低的误差）；
      2. 腿框底边 = 鞋底 = 人物框底边，两者相差不应超过身高的 12%。
    """
    if not legs:
        return None
    if not person:
        # 没有人物框可对照时，至少挡掉"贴着底边的整幅横条"
        return None if (legs[1] > 900 and legs[3] - legs[1] < 120) else legs

    ph = person[3] - person[1]
    lh = legs[3] - legs[1]
    if ph <= 0:
        return None
    if not (0.28 * ph <= lh <= 0.75 * ph):
        return None
    if abs(legs[3] - person[3]) > 0.12 * ph:
        return None
    return legs


def filter_by_framing(actions: list[Action], framing: str,
                      selfie: bool = False) -> list[Action]:
    """按景别硬过滤动作。这是 framing 字段第一次真正参与逻辑。

    改造前 `framing` 只是透传给前端展示的装饰字段，导致"半身照匹配到腿部动作"
    这类错配只能靠 VLM 自己自觉——它经常不自觉地。

    两条规则：
    1. 动作声明了 framing 就必须包含当前景别；
    2. 自拍照额外偏向"不需要看到腿"的动作（自拍几乎都是 closeup/half）。

    过滤后为空时**退回全集**：宁可给一个不完美的动作，也不要因为规则写死而拒绝
    一张本来能用的照片。真正的拒绝路径在 `available_actions` 里，按锚点判。
    """
    if not framing or framing == "none":
        return list(actions)
    hit = [a for a in actions if not a.framing or framing in a.framing]
    if selfie:
        # 自拍的手臂/人脸一定在画面里，优先挑贴着上半身的动作
        pref = [a for a in hit if a.anchor_region in ("shoulder", "torso", "head", "back")]
        if pref:
            hit = pref + [a for a in hit if a not in pref]
    return hit or list(actions)


def pick_action(photo_path: str) -> dict:
    """自动匹配的总入口：画像 -> 按景别过滤 -> 让 VLM 从过滤后的候选里挑。

    返回 `{action_id, confidence, reason, visible_parts, framing, filtered}`。

    两处优化：
    1. 画像已经给了景别，候选集从 5 个缩到 2~3 个，VLM 的选择题更简单、更准；
    2. 过滤后只剩一个候选时**直接返回**，省掉第二次 VLM 调用（约 1.5 秒）。
    """
    prof = vlm_profile(photo_path)
    acts = filter_by_framing(enabled_actions(), prof.framing, prof.selfie)
    n_all = len(enabled_actions())
    if len(acts) == 1:
        a = acts[0]
        return {"action_id": a.id, "confidence": 0.9,
                "reason": f"照片景别是 {prof.framing}，只有这个动作适配",
                "visible_parts": [], "framing": prof.framing,
                "filtered": [x.id for x in acts], "n_all": n_all}

    from s3.actions import vlm_match_action
    res = vlm_match_action(photo_path, actions=acts)
    res["framing"] = prof.framing
    res["filtered"] = [x.id for x in acts]
    res["n_all"] = n_all
    return res


# ---------------------------------------------------------------------------
# 二级检测：可交互物件
# ---------------------------------------------------------------------------
#
# 为什么单独发一次问，而不是塞进 vlm_profile：
# 「两个框 + 几个单选判断」是 lite 级 VLM 做得很稳的任务（实测框误差 <2%），
# 而「找出画面里所有可坐/可依靠的物体并分类」是多目标检测，难度高一个量级。
# 混在一次问里会拖累本来准确的框。
#
# 所以分两级：第一级只判断 has_prop（有/没有），确认有才发第二级。
# 第二级失败只损失物件分支，不影响人物分支。

PROP_SYS = (
    "你是照片场景分析助手。找出画面里可坐的物件，并定位它**坐垫朝外那一面**。\n"
    "坐标归一化到 0 到 999：左上角是 (0,0)，右下角是 (999,999)。\n"
    "\n"
    "kind 只能取以下之一：\n"
    '  "sofa" 沙发 / "bench" 长椅 / "chair" 椅子 / "bed" 床 / "table" 桌子 / '
    '"counter" 台面 / "step" 台阶\n'
    "只报清晰可见、没被完全挡住的。最多 3 个，按可坐性从高到低排。\n"
    "注意：桌子、台面的边缘同样可以坐，桌面/台面也要报（kind=table 或 counter）。\n"
    "即使是近景特写、只看到桌面一角或一段桌沿，也要报出来——只要能看到那条前沿线。\n"
    "如果画面里一件都找不到，才输出空列表；不要因为\"这张照片看起来不像能坐人\"就不报。\n"
    "\n"
    "每个物件给外接框 [x1, y1, x2, y2]，框住的是**坐垫朝外的那一面**：\n"
    "  就是你正对着看到的那一块竖直的布面/板面（沙发上通常是条纹或纯色的那一圈立面）；\n"
    "  y1 = 这一面的上沿，也就是人坐上去时臀部落到的那条线。\n"
    "  y2 = 这一面的下沿。\n"
    "\n"
    "常见错误，务必避免：\n"
    "  ✗ 不要把整个沙发（靠背、扶手）框进来；\n"
    "  ✗ 不要把沙发前面挂着的裙边、沙发底座、地面当成这一面；\n"
    "  ✗ 不要把沙发前面的茶几算进去。\n"
    "拿不准时，找坐垫上「水平的面」和「竖直的面」相交的那条棱，它就是 y1。\n"
    "\n"
    "只输出 JSON，不要解释、不要代码块标记。格式必须是：\n"
    '{"props": [{"kind": "sofa", "seat": [x1, y1, x2, y2]}]}\n'
    '画面里没有这类物件时输出 {"props": []}。'
)

# 物件中文名，用于把动作描述里的 {anchor} 占位符换成实际名词
KIND_CN = {
    "sofa": "沙发", "bench": "长椅", "chair": "椅子", "bed": "床",
    "table": "桌子", "counter": "台面", "step": "台阶",
}


@dataclass
class Prop:
    kind: str
    x1: int
    x2: int
    y: int
    usable: bool = True
    why_not: str = ""

    @property
    def name_cn(self) -> str:
        return KIND_CN.get(self.kind, "边缘")

    @property
    def edge(self) -> list[int]:
        """兼容旧接口：把三个数展开成一个退化的框 [x1, y, x2, y]。"""
        return [self.x1, self.y, self.x2, self.y]


def _parse_props(data) -> list[Prop]:
    """从 `{"props": [...]}` 里解析可坐物件，逐条过可用性门槛。

    单独抽出来是因为现在有**两个入口**：合并后的 `vlm_profile` 和遗留的
    `vlm_props`。两边的解析、门槛、排序必须完全一致，否则"同一次分析"
    在两条路上会给出不同答案。
    """
    out: list[Prop] = []
    if not isinstance(data, dict):
        return out
    for item in (data.get("props") or [])[:3]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").lower()
        if kind not in KIND_CN:
            continue
        seat = item.get("seat")
        bb = _bbox(seat, min_w=60, min_h=0) if isinstance(seat, list) else None
        if not bb:
            continue
        x1, y, x2, _y2 = bb
        p = Prop(kind=kind, x1=x1, x2=x2, y=y)
        _gate_prop(p)
        out.append(p)
    out.sort(key=lambda p: (not p.usable,))
    return out


_PROP_RE = (r'"kind"\s*:\s*"(\w+)"[^}]*?"seat"\s*:\s*\[\s*(\d+)\s*,?\s*'
            r'(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)')


def _props_from_text(txt: str) -> list[dict]:
    """JSON 拼坏时的兜底：按 key 用正则挖。

    逗号可省——模型会吐出 `[373, 483 934 985]` 这种缺逗号的坏 JSON
    （实测 u03_living 中过一次），数字本身是对的，不能因此丢掉结果。
    """
    return [{"kind": k, "seat": [int(a1), int(b1), int(a2), int(b2)]}
            for k, a1, b1, a2, b2 in re.findall(_PROP_RE, txt)]


def vlm_props(photo_path: str, timeout: int = 60) -> list[Prop]:
    """第二级检测：找出可坐物件及其坐垫前沿。失败返回空列表（不抛异常）。

    ## 为什么问"三个数"而不是"一个框"

    第一版问的是"坐垫顶面的外接框"，实测拿不到能用的结果：
    - u03_living 第一次框到了沙发 + 前面的茶几（高宽比 0.77），按那个框坐下去
      人坐在**地板上**；
    - 改成强调"只要坐垫顶面"之后，模型把沙发拆成两半，两块都还是太方；
    - 加宽高比校验后所有结果都被判死，整条物件路线用不了。

    根因是"框住一个透视中的水平面"本身就难：俯视角下坐垫顶面会拉得很高，
    模型没法用矩形稳定表达。而"前沿那条线在哪"是一个**单值**问题——
    只要 x1/x2/y 三个数，歧义小得多。
    """
    from api.ark import ENV, _post, image_path_to_data_uri

    payload = {
        "model": ENV.get("ARK_VLM_MODEL", "doubao-seed-2-1-lite-260915"),
        "messages": [
            {"role": "system", "content": PROP_SYS},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": image_path_to_data_uri(photo_path, max_side=640)}},
                {"type": "text", "text": "请找出画面里可坐的物件和它们的坐垫前沿。"},
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": 260,
        "thinking": {"type": "disabled"},
    }
    st, resp = _post("/chat/completions", payload, timeout=timeout)
    if st != 200:
        return []
    try:
        txt = resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return []
    txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        data = {"props": _props_from_text(txt)}
    return _parse_props(data)


def _gate_prop(p: Prop, min_span: float = 0.10) -> None:
    """可用性门槛。三道，全部来自"她能不能真的坐上去"这个物理问题。

    1. 座面可见宽度 ≥ 画面宽的 10%：再窄就放不下一个比例合理的角色。
    2. 前沿线不能贴画面边缘：家具被裁掉的照片，坐上去会出画。
    3. 前沿线的竖直位置要在 25%~93% 之间：
       - 太高（<25%）说明标到了靠背或墙上的东西；
       - 太低（>93%）说明标到了地面，坐上去等于坐在地上，没有垂腿空间。
    """
    span = (p.x2 - p.x1) / 999.0
    if span < min_span:
        p.usable, p.why_not = False, "座面太窄，放不下比例合理的角色"
        return
    # 座面横跨整个画面**不是**错误：公园长椅、长餐桌经常从左沿一路到右沿，
    # 人坐在中间某一段完全合理。所以这里不再因为"贴边"就否决，
    # 改成在放置时把落点夹进画面内（见 pipeline 的 clamp）。
    # 只有"整个座面都在画面外"才算错，那种情况 x2-x1 会很小，第一道门槛已经拦住。
    if p.y < 250:
        p.usable, p.why_not = False, "前沿线位置过高，可能标到了靠背而不是坐垫"
        return
    if p.y > 930:
        p.usable, p.why_not = False, "前沿线位置过低，缺少垂腿空间"
        return


# ---------------------------------------------------------------------------
# 路由：人物优先，其次物件，都没有就拒绝
# ---------------------------------------------------------------------------

def available_actions(prof: PhotoProfile, props: list[Prop],
                      actions: list["Action"] | None = None) -> tuple[list, str]:
    """返回 (候选动作, 路由原因)。

    路由顺序按用户设定：**人物优先**。有人物就只走人物路线——
    哪怕画面里同时有沙发，也优先让小人贴人，因为"贴贴"本身就是产品核心，
    而且人身上的锚点已经被验证过，可靠度高于家具。
    没有人物才退到物件路线；两者都没有就是"这张照片不可用"。

    framing 过滤在选定的路线上再做一次，避免半身照选中腿部动作。
    """
    acts = actions if actions is not None else enabled_actions()

    if prof.has_person:
        pool = [a for a in acts if a.anchor_kind == "person_part"]
        pool = filter_by_framing(pool, prof.framing, prof.selfie)
        return pool, "person"

    usable = [p for p in props if p.usable]
    if usable:
        pool = [a for a in acts if a.anchor_kind == "seat_plane"]
        return pool, "object"

    if props:
        why = props[0].why_not or "物件不可用"
        return [], f"no_anchor:画面里没有人物，可交互物件也不可用（{why}）"
    return [], "no_anchor:画面里没有人物，也没有找到可坐的物件或可依靠的物体"


def pick_action_v2(photo_path: str) -> dict:
    """自动匹配：画像 -> 物件 -> 路由 -> 让 VLM 挑，失败自动换路线再试一次。

    ## 为什么要"换路线重试"

    `vlm_profile` 判"有没有人物"偶尔会假阳性——实测一张空沙发照被报成有半身人物
    （4 次里出现 1 次）。一旦走错路线，候选集全是"贴在人身上"的动作，
    模型看一眼就回答"没有可依附的部位"。

    这个信号很有价值：它其实在说"路线选错了"。所以收到 `no_fit` 就换另一条路线
    再问一次。重试只在失败时发生，正常情况下不会多花调用。
    """
    prof = vlm_profile(photo_path)
    # 座面信息已经并进同一次画像（`prof.props`），正常路径没有第二次调用。
    # 但合并版偶尔会把座面前沿标到画面顶端（把朝上的桌面整个框进来），
    # 那种情况在这里补问一次——只有"确实没有可用物件"时才多花调用。
    props: list[Prop] = prof.props
    if not prof.has_person and not any(p.usable for p in props):
        try:
            extra = vlm_props(photo_path)
        except Exception:
            extra = []
        if any(p.usable for p in extra):
            props = extra

    pool, route = available_actions(prof, props)
    if not pool:
        return {"error": route.split("：", 1)[-1] if "：" in route else route}

    if len(pool) == 1:
        a = pool[0]
        return {"action_id": a.id, "confidence": 0.85,
                "reason": f"走{('人物' if route == 'person' else '物件')}路线，只有这个动作适配",
                "visible_parts": [], "framing": prof.framing, "route": route,
                "filtered": [x.id for x in pool], "n_all": len(enabled_actions())}

    from s3.actions import vlm_match_action
    res = vlm_match_action(photo_path, actions=pool)

    if res.get("error") == "no_fit":
        # 换一条路线再试
        alt_pool = [x for x in enabled_actions()
                    if x.anchor_kind == ("seat_plane" if route == "person" else "person_part")]
        if route == "person":
            alt_pool = filter_by_framing(alt_pool, prof.framing, prof.selfie) or alt_pool
        if alt_pool:
            alt = alt_pool[0] if len(alt_pool) == 1 else None
            alt = alt or vlm_match_action(photo_path, actions=alt_pool)
            if "action_id" in alt:
                alt["framing"] = prof.framing
                alt["route"] = "object" if route == "person" else "person"
                alt["retried"] = True
                alt["filtered"] = [x.id for x in alt_pool]
                alt["n_all"] = len(enabled_actions())
                return alt
        return {"error": "换了两条路线都找不到能成立的动作。"
                         "换一张人或家具更完整的照片会好很多。"}

    res["framing"] = prof.framing
    res["route"] = route
    res["filtered"] = [x.id for x in pool]
    res["n_all"] = len(enabled_actions())
    return res


def _parse_objects(data) -> list:
    """解析 `objects` 字段，产出 `SceneObject` 列表。

    放在这里而不是 `scene3d.py` 里，是因为它和 `_parse_props` 是同一件事的两个分支：
    都在把 VLM 的 JSON 翻译成数据结构。放一起，改字段时不会漏掉一边。

    `scene3d` 用局部 import 避开循环依赖——`scene3d` 顶层不 import `profile`，
    但 `profile` 往下走时要用它的类型。
    """
    from s3.scene3d import parse_objects
    if not isinstance(data, dict):
        return []
    return parse_objects(data.get("objects"))
