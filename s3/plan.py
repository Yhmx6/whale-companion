"""场景规划器：让视觉模型先决定"她在哪、做什么"，再交给生图模型。

## 为什么要多这一步

现在的立绘路线是"从 9 个预制动作里挑一个，按固定规则贴到某个位置"。
两个硬伤都来自这个结构：

1. **动作是枚举的**。半身自拍里没有一个动作的 `framing` 匹配得上，
   候选集直接为空（实测 5 张自拍里有 2 张只能退回"坐沙发扶手 / 坐桌沿"）。
2. **她永远是"贴上去"的**。立绘是独立生成再合成的，所以她的手不会真的搭在人身上，
   也不会被人挡住——只能靠程序补阴影和遮挡来假装。

如果让生图模型把她**画进画面**，这两点自然解决：动作可以针对这张照片现编，
遮挡和接触由同一个扩散过程完成。

但生图模型不知道自己该把她放多大、放哪儿。所以需要先有一个"分镜"：
把照片读一遍，决定位置框、动作、依托物，再把这些写成生成提示词里的硬约束。

## 设计要点

- **一次调用**：位置、动作、依托、理由一起出。分成两次问会互相矛盾。
- **位置框就是尺度约束**：提示词里"她必须出现在 x1 y1 x2 y2 这个区域内"
  同时限定了位置和大小。这是唯一能压住"模型把她画成 41%"的手段。
- **动作必须可执行**：她得有个东西可坐/可站/可靠。没有依托就不要写"跳起来"。
- **主动回避手**：这个项目的实测结论是手部抓握画不好，且"不要画手指"这类
  否定式指令遵从度很低。所以规划阶段就偏向"手被遮住 / 平贴 / 抱自己"的姿态。
- **不压脸、不贴边**：压脸会毁掉照片主体，贴边没法判断她有没有出画。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

HERE_ANCHOR_KINDS = ("person_part", "seat_plane", "ground_plane",
                     "vertical_prop", "horizontal_edge", "none")


@dataclass
class ScenePlan:
    where: str = ""              # 给生成模型看的位置描述（一句话）
    action: str = ""             # 给生成模型看的姿态描述（一句话）
    action_id: str = ""          # 从候选立绘里选中的动作 id
    mood_id: str = ""            # 从表情库里选中的表情 id
    mood_face: str = ""          # 该表情的画面特征（agent 用不到，诊断看）
    mood_reason: str = ""        # 为什么选这个表情
    moment: str = ""             # 这一瞬间她在想什么（情绪判断的依据）
    region_name: str = ""        # 依托物的名字，如"人物的左小腿"
    region: list[int] = field(default_factory=list)   # 0-999 归一化框
    anchor_kind: str = ""        # person_part / seat_plane / ...
    height_pct: float = 0.0      # 她应占画面高的百分比（由 region 反算）
    reason: str = ""             # 为什么这么选（诊断用）
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.action and len(self.region) == 4)

    def as_dict(self) -> dict:
        return {"where": self.where, "action": self.action,
                "action_id": self.action_id,
                "mood_id": self.mood_id, "moment": self.moment,
                "mood_reason": self.mood_reason,
                "region_name": self.region_name, "region": self.region,
                "anchor_kind": self.anchor_kind,
                "height_pct": round(self.height_pct, 1), "reason": self.reason}


PLAN_SYS = (
    "你是照片与二次元角色的合成导演。给你一张真实照片，你要决定："
    "这个角色应该出现在画面里的哪个位置，以及她在做什么。\n"
    "坐标归一化到 0 到 999：左上角 (0,0)，右下角 (999,999)。\n"
    "\n"
    "## 她是谁\n"
    "一个 Q 版小人，2 到 2.5 头身，头大身子小、四肢短圆，"
    "**真实身高 60 厘米**，像一个能带在身上的玩偶。所以她在画面里**很小**：\n"
    "站在 175 厘米的成年人旁边，她的头顶大约刚过对方的膝盖。\n"
    "\n"
    "## 位置怎么定\n"
    "- 优先让她和画面里**已经存在的东西**发生关系：人的腿 / 脚 / 肩 / 手、"
    "家具的座面或边缘、地面、树干、台阶。凭空站在空地上是最差的选择。\n"
    "- `region` 是她的**完整外接框**（含头发和尾巴），不是依托物的框。\n"
    "- 框的**底边**（region 的第 4 个数）就是她的落脚点或落座点——"
    "这是程序唯一会用的定位信息，所以它必须正好压在座面 / 地面 / 人物腿线上。"
    "框的高度程序不采用（大小由 60 厘米算出），但请按 60 厘米该占的比例画，"
    "否则你自己的比例感会带着底边漂。\n"
    "- **不要**把框放在人物脸上（会毁掉照片主体），**不要**贴着画面四边。\n"
    "\n"
    "## 动作怎么定\n"
    "- **必须**从下面「可选动作」里选一个 `id`，不许自创动作。"
    "原因：每个候选动作都有一张已经画好的立绘，引导图里贴的就是那一张；"
    "你要是自创一个，立绘和文字对不上，画面会拧巴。\n"
    "- 选定 id 之后，`action` 字段请按**这张照片的实际情况**把姿态重写一遍："
    "保留候选动作的姿势骨架（坐 / 站 / 蹲 / 抱膝盖 / 手在哪），"
    "把「依托物」换成画面里真实的那件东西（长椅的木条座面、男士的左小腿…），"
    "可以补一句她在看什么。别改姿势，只改场景名词。\n"
    "- 挑那个**和画面最贴**的：有水平座面就选坐的，能看到人的腿和脚就选腿边的，"
    "没有人物但有座面就选坐边缘的。\n"
    "\n"
    "## 表情怎么定\n"
    "- 动作定「她在做什么」，表情定「她脸上是什么」，**两件事要分开想**。"
    "同一个动作在不同照片里应该有不同的表情——她坐在黄昏的窗台上和坐在"
    "热闹的咖啡厅里，不会是一个神态。\n"
    "- **第一步：先在心里写一句话**，描述「这一瞬间她在想什么」。"
    "要具体到这张照片——时间、光是什么颜色、周围安静还是热闹、"
    "有没有人在她旁边、她刚做了什么。**不要写「她很高兴」这种结论**，"
    "要写「傍晚的窗台很静，她刚坐下，阳光是橘色的」。\n"
    "- **第二步：把这句话对应到一个 mood_id**。上面那句话如果换成别人来读，"
    "读到的情绪是什么，就选哪个。\n"
    "- 判断的时候，这几样各占一份权重，不要只看光线：\n"
    "  ① 周围有没有人、在不在看她；② 是白天还是傍晚、光是什么颜色；\n"
    "  ③ 这个空间是家、是店铺、还是户外；④ 她刚找到位置还是已经待了很久。\n"
    "- **不要看动作选表情。** 同一个 `seat_edge_sit`，在卧室床沿、"
    "在热闹吧台、在黄昏台阶上，应该是三种不同的脸。"
    "如果这次和上一张选了同一个表情，先问自己「这两张照片真的没有区别吗」。\n"
    "- **下面这几句不许用来当理由**，它们对每张照片都成立，等于没选：\n"
    "  ✗「她刚到这里，正在好奇地打量周围」——任何一张都能这么说；\n"
    "  ✗「她刚坐下来，心情很雀跃」——坐着就是坐着，不代表高兴；\n"
    "  ✗「光线柔和、氛围安静」——生活照的光几乎都柔和。\n"
    "  选之前先问：**这张照片和别的照片，到底哪一点不一样？**"
    "把那个不一样的点写进 `moment`，再从它推表情。\n"
    "\n"
    "## 输出\n"
    "只输出 JSON，不要解释、不要代码块标记。格式：\n"
    '{"where": "一句话，她在画面中的位置", '
    '"action_id": "从可选动作里选的 id", '
    '"action": "一句话，她在做什么（按这张照片重写，会被直接写进生图提示词）", '
    '"moment": "一句话，这一瞬间她在想什么（写具体的场景感受，不写情绪结论）", '
    '"mood_id": "从可选表情里选的 id", '
    '"mood_reason": "一句话，从上面那句 moment 怎么推到这个表情", '
    '"region_name": "依托物名字，如\\"长椅的木条座面\\"", '
    '"region": [x1, y1, x2, y2], '
    '"anchor_kind": "person_part | seat_plane | ground_plane | vertical_prop | horizontal_edge", '
    '"reason": "一句话，为什么选这个位置和动作"}\n'
    "如果这张照片实在找不到任何能放她的依托（纯墙、纯天空），"
    'anchor_kind 写 "none"、region 写 []，并在 reason 里说明。'
)


def _mood_block(moods: list[dict] | None) -> str:
    """把表情清单排成提示词里的一段。

    只给 `when`（什么时候适合）不给 `face`（五官怎么画）——`face` 是要写进
    **生图**提示词的，规划器只需要判断"这个场景适合哪种情绪"。
    把 `face` 也塞进来会让清单长一倍，而模型并不会因此选得更准。
    """
    if not moods:
        return ""
    lines = ["\n## 可选表情（**必须**从中选一个 mood_id）"]
    for m in moods:
        extra = ""
        if m.get("avoid_with"):
            extra = f"（和 {'、'.join(m['avoid_with'])} 的姿态不搭，尽量别配）"
        lines.append(f"- `{m['id']}` {m['name']}：{m.get('when', '')}{extra}")
    return "\n".join(lines) + "\n"


def _options_block(options: list[dict]) -> str:
    """把候选立绘动作排成提示词里的一段清单。

    为什么要限定在候选里选：引导图是**一张已经画好的立绘**，贴进照片里当占位符。
    生成模型同时收到两个信号——"照这张小人的姿势重画"和"按文字改成另一个姿势"。
    两者不一致时它会各听一半，产出的姿态介于两者之间（实测：规划说"坐"、
    立绘是站姿，成品就成了"脚悬在长椅前沿的半空站姿"）。
    把候选清单交给规划器，最省事、也最彻底地消掉了这个矛盾。
    """
    if not options:
        return ""
    lines = ["\n## 可选动作（**必须**从中选一个 id，不许自创）"]
    for o in options:
        kind = o.get("anchor_kind") or ""
        lines.append(f"- `{o['id']}` {o.get('name', '')}"
                     f"（依托：{kind or '不限'}）：{o.get('desc', '')}")
    return "\n".join(lines) + "\n"



def _parse_bbox(v) -> list[int]:
    if not isinstance(v, list) or len(v) != 4:
        return []
    try:
        x1, y1, x2, y2 = (int(round(float(x))) for x in v)
    except (TypeError, ValueError):
        return []
    # 夹进 0-999 并保证有序
    x1, x2 = sorted((max(0, min(999, x1)), max(0, min(999, x2))))
    y1, y2 = sorted((max(0, min(999, y1)), max(0, min(999, y2))))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return []
    return [x1, y1, x2, y2]


def _parse(text: str) -> dict:
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        # 兜底：JSON 拼坏时按 key 挖。模型偶发会漏逗号或把 region 写成空格分隔。
        out: dict = {}
        for key in ("where", "action", "action_id", "mood_id", "mood_reason",
                    "moment", "region_name", "anchor_kind", "reason"):
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
            if m:
                out[key] = m.group(1)
        m = re.search(r'"region"\s*:\s*\[\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)',
                      text)
        if m:
            out["region"] = [int(g) for g in m.groups()]
        return out


def _bigrams(s: str) -> set[str]:
    """中文二字组。中文没有空格分词，二字组是最省事又够稳的近似。"""
    return set(re.findall(r"[\u4e00-\u9fa5]{2}", s.replace("{anchor}", "座面")))


def _pick_action_id(raw_id: str, action: str,
                    options: list[dict]) -> tuple[str, str]:
    """校验规划器选的 id 真的在候选里；不在就按文字相似度兜一个。

    返回 `(action_id, source)`，source 是 `plan` / `plan_name` / `fallback` / `none`，
    方便在 meta 里看出"这次是不是没听话"。

    为什么要兜：模型会漏写 `action_id`，也会把 id 写成中文名、写成 `坐边缘`。
    这时不能整张图失败——得兜一个最像的。

    ## 打分为什么把 name 和 desc 分开算

    一开始把两者的大字集合合在一起算"命中率"，结果 `"站在他腿边仰头看他的脸"`
    匹配不到 `leg_hands_behind`（name="站在腿边仰头"）——因为 desc 有 30 多个
    二字组，命中 3 个只有 0.09，被长描述稀释掉了。
    分开算就对了：name 的二字组只有 5 个，命中 3 个是 0.6。

    **name 是动作的核心语义，desc 只是细节**，所以分开算、取较大者，
    并且 name 的门槛低（0.3）、desc 的门槛高（0.18）——
    desc 重合说明"细节像"，那必须像得比较多才算数。
    """
    ids = [o["id"] for o in options]
    rid = (raw_id or "").strip().strip("`'\"")
    if rid in ids:
        return rid, "plan"
    # 模型有时会漏 id 但把 name 填进 action_id
    for o in options:
        if rid and rid == (o.get("name") or "").strip():
            return o["id"], "plan_name"
    if not action or not options:
        return "", "none"

    best, best_score = "", 0.0
    for o in options:
        nk = _bigrams(o.get("name", ""))
        dk = _bigrams(o.get("desc", ""))
        n_hit = sum(1 for k in nk if k in action) / len(nk) if nk else 0.0
        d_hit = sum(1 for k in dk if k in action) / len(dk) if dk else 0.0
        score = max(n_hit if n_hit >= 0.3 else 0.0,
                    d_hit if d_hit >= 0.18 else 0.0)
        if score > best_score:
            best, best_score = o["id"], score
    return (best, "fallback") if best_score > 0 else ("", "none")


def plan_scene(photo_path: str, *, height_hint: str = "",
               options: list[dict] | None = None,
               moods: list[dict] | None = None,
               timeout: int = 60) -> ScenePlan:
    """一次 VLM 调用，产出这张照片的分镜。失败返回一个 ok=False 的空计划。

    `options` 是**当前已缓存立绘的动作清单**（见 `available_sprite_options`）。
    传了就把它写进提示词，让规划器从中选一个 `action_id`——这是"动作匹配"的关键：
    引导图只能是一张已存在的立绘，规划器若要自由创作姿态，两个信号就会打架。
    不传则退回旧行为（自由描述动作），仅用于还没有立绘缓存的冷启动。

    `moods` 是表情候选（见 `s3/moods.py::options_for`）。传了就要它选一个
    `mood_id`。表情和动作是两件独立的事：动作定"她在哪、手脚怎么放"，
    表情定"她脸上是什么"。同一个动作在黄昏窗台上和在热闹咖啡厅里
    不该是同一个神态。
    """
    from api.ark import ENV, _post, image_path_to_data_uri

    user = "请给这张照片出分镜。"
    if height_hint:
        user += f"\n补充信息：{height_hint}"

    sys_prompt = PLAN_SYS + _options_block(options or []) + _mood_block(moods)

    payload = {
        "model": ENV.get("ARK_VLM_MODEL", "doubao-seed-2-1-lite-260915"),
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": image_path_to_data_uri(photo_path, max_side=640)}},
                {"type": "text", "text": user},
            ]},
        ],
        "temperature": 0.2,
        # 多了 action_id、mood_id、mood_reason 和两段候选清单，560 会被截断
        "max_tokens": 700,
        "thinking": {"type": "disabled"},
    }
    st, resp = _post("/chat/completions", payload, timeout=timeout)
    if st != 200:
        return ScenePlan(raw={"error": f"HTTP {st}: {str(resp)[:160]}"})
    try:
        txt = resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return ScenePlan(raw={"error": "bad_response"})

    data = _parse(txt)
    region = _parse_bbox(data.get("region"))
    kind = str(data.get("anchor_kind") or "").strip().lower()
    action = str(data.get("action") or "").strip()
    act_id, act_src = _pick_action_id(str(data.get("action_id") or ""),
                                      action, options or [])
    # 表情：只做"在不在候选里"的校验，不做文字兜底。
    # 动作可以按描述兜（姿态是连续的量，相近就能用），表情不行——
    # "害羞"和"雀跃"在五官上是两套完全不同的画法，兜错了不如不指定。
    mood_id = str(data.get("mood_id") or "").strip().strip("`'\"")
    mood_src = "plan"
    if moods:
        valid = {m["id"] for m in moods}
        if mood_id not in valid:
            # 模型可能把中文名填进 mood_id
            by_name = {m["name"]: m["id"] for m in moods}
            if mood_id in by_name:
                mood_id, mood_src = by_name[mood_id], "plan_name"
            else:
                mood_id, mood_src = "", "none"
    from s3.moods import get_mood
    mo = get_mood(mood_id)

    plan = ScenePlan(
        where=str(data.get("where") or "").strip(),
        action=action,
        action_id=act_id,
        mood_id=mood_id,
        mood_face=mo.face if mo else "",
        mood_reason=str(data.get("mood_reason") or "").strip(),
        moment=str(data.get("moment") or "").strip(),
        region_name=str(data.get("region_name") or "").strip(),
        region=region,
        anchor_kind=kind if kind in HERE_ANCHOR_KINDS else "",
        reason=str(data.get("reason") or "").strip(),
        raw=data,
    )
    plan.raw["_action_source"] = act_src
    plan.raw["_mood_source"] = mood_src
    if region:
        plan.height_pct = (region[3] - region[1]) / 999.0 * 100.0
    return plan

