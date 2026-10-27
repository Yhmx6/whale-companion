"""表情库：把「她在想什么」从「她在做什么」里拆出来。

## 为什么要拆

改造前表情是**焊死在动作描述里**的。扫一遍 9 个启用动作，只有 3 个写了表情词，
而且都是「笑」——`seat_edge_sit` 是"笑得很开心"、`jump_air` 是"笑得非常开心"、
`shoulder` 是"得意地笑"。剩下 6 个动作连表情都没提。

后果是：不管她在空沙发上、在逆光的街边、还是坐在人对面，脸上永远是同一个笑。
用户的原话是「不同场景不同动作表情也不一样」，这句话有两层：
**不同动作要不同表情**，**同一动作在不同场景也要不同表情**。第二层靠动作
描述是做不到的——动作就那么几个，场景却千变万化。

所以把表情独立成 `Mood`，由规划器**看着这张照片**选一个。

## face 为什么写得这么细

这个项目的实测结论反复指向同一条：**模型对「关系」的遵从度低，对「画面上
能看见什么」的遵从度高**。

- 写"她很害羞"→ 出来的还是那个笑；
- 写"脸通红、眼睛往下看、下巴微微收着、嘴角抿着往上"→ 才画得对。

同一条教训出现在 `leg_hug_knee`（写"小腿折在大腿下面"没用，改成"正面看不到
小腿和脚"才画对）和 `simplify_hands`。所以 `face` 一律写成**五官的具体动作**：
眼睛是睁是眯、眉毛抬还是皱、嘴角朝哪、脸颊红不红。
不用"开心""难过"这类情绪词——那是给人读的，不是给模型画的。

## body 是可选的身体语言

表情不只在脸上。害羞会缩肩、惊讶会后仰、得意会抬下巴。写进 `body` 里，
和 `face` 一起进提示词。留空表示只改脸。

## 和动作的关系

两者是**正交**的：动作定"她在哪、手脚怎么放"，表情定"她脸上是什么"。
所以拼起来用：`action.desc` + `mood.face`。

唯一要避免的是**动作自带的表情和选中的表情打架**。比如 `jump_air` 的 desc
里写着"笑得非常开心"，这时选 `sleepy`（困倦）就会拧巴。
`conflicts` 字段用来声明这种冲突，规划器选的时候会看到。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mood:
    id: str
    name: str
    face: str          # 五官的具体动作，必须可画
    body: str = ""     # 身体语言（可选）
    when: str = ""     # 什么场景适合——给规划器看
    conflicts: tuple[str, ...] = ()   # 和哪些动作天然冲突


MOODS: tuple[Mood, ...] = (
    Mood(
        id="happy", name="雀跃",
        face="两只眼睛弯成两道向上翘的月牙、几乎看不到瞳孔，嘴角大幅上扬露出小虎牙，"
             "两颊泛着粉红，嘴巴微微张开像在笑出声",
        body="肩膀轻轻耸起，整个人往上掂了一点",
        when="她正处在兴头上——刚抢到好位置、正要开始玩、看见了喜欢的东西。"
             "明亮的户外、有动感的构图、人物也在笑的时候选它",
    ),
    Mood(
        id="shy", name="害羞",
        face="两颊红得很明显、红晕一直漫到耳朵，眼睛往下看不敢直视，"
             "下巴微微收着，嘴角抿起来往上翘一点点",
        body="肩膀往里收，两只手不自觉地在身前绞在一起",
        when="**有人在看她**——人物正对着她、镜头离得近、或者她刚做完一件小事被发现。人离得越近越该选它",
    ),
    Mood(
        id="curious", name="好奇",
        face="两只眼睛睁得很大、瞳孔又黑又圆，眉毛往上抬，"
             "嘴巴微微张成一个小小的圆形，头歪向一边",
        body="身体往前探一点点，像在打量什么",
        when="**有事正在发生**——画面里有她没见过的东西、人物在走路/倒水/看书，或者她刚到这个陌生地方",
    ),
    Mood(
        id="smug", name="得意",
        face="一条眉毛挑起来、另一条压着，眼睛半眯着，"
             "嘴角只有一边往上翘，下巴微微抬起",
        body="双手抱在胸前或者叉着腰，站得笔直",
        when="**她占了上风**——坐得比人物高、抢到了最中间最舒服的位置、刚把什么事办成了",
        conflicts=("leg_hug_knee", "crouch_by_leg"),   # 蜷成一团时做不出这个姿态
    ),
    Mood(
        id="sleepy", name="困倦",
        face="眼皮垂下来、眼睛只睁开一条缝，眼睛下面有一点点阴影，"
             "嘴巴小小地张着像刚打完哈欠，脸颊软软的",
        body="身体有点软、重心往一边倒，头微微往下垂",
        when="**她累了或环境让人犯困**——天快黑了、暖黄色台灯、床和沙发、或者她在这个位置等了很久",
        conflicts=("jump_air",),                       # 跳起来还困着就矛盾了
    ),
    Mood(
        id="surprised", name="惊讶",
        face="两只眼睛圆睁、眼白露出来，眉毛高高抬起，"
             "嘴巴张成一个明显的 O 形，脸颊绷着",
        body="身体微微往后仰，肩膀抬起来",
        when="**有什么突然出现**——画面里有意外的东西、人物突然转头看她、或者构图本身很冲（大特写、仰拍）",
        conflicts=("leg_hug_knee", "sit_cross_legged"),  # 抱膝/盘腿时后仰不成立
    ),
    Mood(
        id="tender", name="温柔",
        face="眼睛柔和地半垂着、眼神软软的，嘴角是很轻的一点上扬，"
             "整个人看起来很安静",
        body="头微微侧着，视线落在人物身上",
        when="**她只是安静地陪着，什么都不做**。这是最容易被滥用的选项——光线柔和不算理由，每张生活照的光都柔和。只有当其他表情都不合适、画面上确实没有任何事件在发生时，才选它",
    ),
    Mood(
        id="determined", name="鼓劲",
        face="眉毛往中间压、眉心有一点点皱，嘴唇抿紧成一条线，"
             "眼睛直直地看着前方",
        body="两只手在身侧握成小拳头，身体绷直",
        when="**她准备使劲或跟上**——人物在往前走她要跟、或者她要搬/推什么东西",
    ),
    Mood(
        id="pout", name="嘟嘴",
        face="嘴巴嘟起来、下唇往前凸，眉毛微微皱起，"
             "眼睛斜斜地看向一边不看镜头",
        body="双臂抱在身前，头扭向一侧",
        when="**她吃了亏**——被挤到角落、位置被人占了、想要什么没得到",
        conflicts=("jump_air", "shoulder"),
    ),
)

REGISTRY: dict[str, Mood] = {m.id: m for m in MOODS}
DEFAULT_ID = "happy"


def get_mood(mid: str | None) -> Mood | None:
    """按 id 取表情。未知 id 返回 None（不是退到默认）——

    这里和 `characters.get_character` 的选择相反，因为语义不同：
    角色 id 错了必须给个能用的（否则整张图失败），
    表情 id 错了只该"这次不指定表情"，让立绘/动作自带的表情生效。
    硬塞一个 happy 会把"温柔"的场景改成傻笑。
    """
    return REGISTRY.get((mid or "").strip())


def options_for(action_ids: list[str] | None = None) -> list[dict]:
    """给规划器的候选清单。`action_ids` 传了就过滤掉和这些动作冲突的表情。"""
    out = []
    for m in MOODS:
        if action_ids is not None and m.conflicts and \
                any(a in action_ids for a in m.conflicts):
            # 不是完全禁用——毕竟规划器可能先选了表情再选动作。
            # 但要在清单里标出来，让它自己权衡。
            blocked = [a for a in m.conflicts if a in action_ids]
            out.append({"id": m.id, "name": m.name, "when": m.when,
                        "face": m.face, "avoid_with": blocked})
            continue
        out.append({"id": m.id, "name": m.name, "when": m.when, "face": m.face})
    return out


def describe(mid: str | None) -> str:
    """拼成可以直接写进生成提示词的一句。取不到表情就返回空串。"""
    m = get_mood(mid)
    if m is None:
        return ""
    s = f"她的表情是「{m.name}」：{m.face}"
    if m.body:
        s += f"；{m.body}"
    return s


def as_dicts() -> list[dict]:
    return [{"id": m.id, "name": m.name, "face": m.face, "body": m.body,
             "when": m.when, "conflicts": list(m.conflicts)} for m in MOODS]


# ---------------------------------------------------------------------------
# 按场景收窄候选
# ---------------------------------------------------------------------------
#
# ## 为什么必须由程序收窄
#
# 让规划器在 9 个表情里自由选，实测会塌缩到一两个最通用的选项
# （第一版 tender 占 5/10，改完变 happy 占 9/10）。原因是它总能给最通用的那个
# 找到理由——"她刚坐下，正处在兴头上"对每张照片都成立。
#
# 所以改成：**程序先算出这张照片适合哪几种，再把短名单交给规划器**。
# 好处是"场景差异"变成了可计算、可测试的量，而不是指望 VLM 自己注意到。
#
# ## 用哪几个量
#
# 全部是像素级的、确定性的：
#
# | 量 | 怎么算 | 对应什么情绪 |
# |---|---|---|
# | 亮度 | 灰度均值 | 暗 → 困倦/害羞；亮 → 雀跃/惊讶 |
# | 色温偏暖 | R 均值 − B 均值 | 暖（夕阳/暖灯）→ 温柔/困倦；冷 → 好奇/鼓劲 |
# | 对比度 | 灰度标准差 | 硬光高对比 → 惊讶/得意 |
# | 饱和度 | HSV 的 S 均值 | 高饱和 → 雀跃；低饱和 → 温柔/困倦 |
#
# 再加两个**语义**信号：画面里有没有人、主体是什么家具（床/吧台/户外）。
# 前四个是像素统计，后两个来自 `objects`。
#
# ## 这层的边界
#
# 它是**启发式**，不是真在理解情绪。参数（0.42 / 0.68 这些）是拍的，
# 换一批素材可能要调。所以它只做"收窄"，不做"决定"——最终选哪个还是规划器。
# 收窄错了的代价是"表情不够贴"，不是"整张图坏掉"。

EMOTION_RULES_VERSION = "v1"


def scene_stats(photo) -> dict:
    """照片的情绪相关统计量。全部归一到 0~1（除了 warmth 是 -1~1）。

    `photo` 传 PIL Image。缩到短边 256 再算——这些量都是全局统计，
    分辨率对结果没有影响，但快 20 倍。
    """
    import numpy as np
    im = photo.convert("RGB")
    w, h = im.size
    if min(w, h) > 256:
        k = 256 / min(w, h)
        im = im.resize((max(8, int(w * k)), max(8, int(h * k))))
    a = np.asarray(im, dtype=np.float32) / 255.0
    gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        # 暖度：红蓝均值差，乘 2 拉到 -1~1 的量级（实测生活照多在 ±0.25 内）
        "warmth": float((a[:, :, 0].mean() - a[:, :, 2].mean()) * 2.0),
        "saturation": float(sat.mean()),
    }


def narrow(stats: dict, has_person: bool = False,
           kinds: list[str] | None = None,
           limit: int = 3) -> list[str]:
    """按场景统计收窄表情候选。返回 id 列表，按得分从高到低。

    ## 为什么用打分而不是"按规则顺序收集"

    第一版是"命中哪条规则就往列表里追加"，结果**先命中的永远赢**：
    `sleepy` 同时出现在四条规则里（暖光、偏暗、床沙发），于是 7/10 张它排第一。
    换一批素材会换成别的表情垄断——本质是这套结构没有让不同信号互相竞争。

    改成打分：每条规则给若干个表情加不同权重，最后按总分排序。
    这样一条规则可以同时推两个表情、也可以只轻推一个，
    强弱关系写在权重里，而不是写在列表顺序里。

    ## 权重是怎么定的

    - 语义信号（床、吧台）给 2 分：它比"亮度 0.67"更说明她此刻在什么状态。
    - 色温给 2 分：区分度最大的像素量。
    - 明暗/对比/饱和各 1~2 分：辅助。
    - 有没有人 1 分：只做微调，不该单独决定情绪。
    """
    b = stats.get("brightness", 0.5)
    w = stats.get("warmth", 0.0)
    c = stats.get("contrast", 0.15)
    s = stats.get("saturation", 0.15)
    kinds = kinds or []

    score: dict[str, float] = {m.id: 0.0 for m in MOODS}

    def give(mid: str, pts: float) -> None:
        if mid in score:
            score[mid] += pts

    # ---- 1. 语义：这是什么地方 ----
    if any(k in ("bed", "sofa") for k in kinds):
        give("sleepy", 2); give("shy", 1)
    if any(k in ("counter", "table") for k in kinds):
        give("smug", 2); give("curious", 1)
    if any(k in ("bench", "step", "plant") for k in kinds):
        give("curious", 2); give("determined", 1)
    if any(k in ("cup", "plate", "book", "phone", "laptop") for k in kinds):
        give("curious", 1)

    # ---- 2. 色温 ---- 
    #
    # 暖光再按亮度分流：**暗的暖光像深夜的台灯（困），亮的暖光像黄昏（温柔）**。
    # 不分开的话"暖 → 温柔+困倦"两个一起加，它们在每张暖光图里都并列，
    # 等于没区分。
    if w > 0.12:
        if b < 0.45:
            give("sleepy", 2); give("shy", 1)
        elif b > 0.60:
            give("tender", 2); give("happy", 1)
        else:
            give("tender", 2); give("sleepy", 1)
    elif w < -0.02:
        give("curious", 1); give("determined", 2)
    else:
        give("happy", 2); give("curious", 1)

    # ---- 3. 明暗 / 对比 / 饱和 ----
    if b < 0.35:
        give("sleepy", 2)
    elif b > 0.65:
        give("happy", 2); give("surprised", 1)
    if c > 0.22:
        give("surprised", 2); give("smug", 1)
    if c < 0.12:
        give("tender", 1)
    if s > 0.35:
        give("happy", 2)
    elif s < 0.08:
        give("tender", 2)

    # ---- 4. 有没有人 ----
    if has_person:
        give("shy", 1); give("tender", 1)
    else:
        give("curious", 1); give("smug", 1)

    # 得分并列时按 MOODS 的定义顺序取，保证结果确定（同样的输入永远同样的输出）
    order = {m.id: i for i, m in enumerate(MOODS)}
    ranked = sorted(score, key=lambda k: (-score[k], order[k]))
    top = [k for k in ranked if score[k] > 0][:limit]
    if len(top) < 3:
        for k in ("happy", "tender", "curious", "curious"):
            if k not in top:
                top.append(k)
            if len(top) >= 3:
                break
    return top


def for_scene(photo, has_person: bool = False, objects=None,
              action_ids: list[str] | None = None) -> list[dict]:
    """一条龙的入口：算统计量 → 收窄 → 排出给规划器的候选清单。

    调用方只需要给照片和场景信息，不用关心内部是打分还是排序。
    **顺序有意义**：得分高的排前面，规划器更容易选到它，
    但后面的仍然可选——收窄是给它划范围，不是替它决定。
    """
    kinds = [getattr(o, "kind", "") for o in (objects or [])]
    stats = scene_stats(photo)
    short = narrow(stats, has_person=has_person, kinds=kinds)
    base = {m["id"]: m for m in options_for(action_ids)}
    out = []
    for mid in short:
        m = base.get(mid)
        if m is None:            # 和动作冲突被过滤掉了
            continue
        out.append(m)
    # 全被冲突过滤掉时退回通用清单，不能给空列表（规划器会乱选）
    return out or options_for(action_ids)[:4]


def pick_random(action_id: str = "", avoid_recent: list[str] | None = None) -> str:
    """完全随机挑一个表情。

    ## 为什么随机也要排除冲突

    "随机"要的是**多样**，不是"出戏"。`smug`（挑眉得意）和 `leg_hug_knee`
    （蜷成一团抱膝）在姿态上不共存——脸可以得意，但那个动作里她连下巴都
    埋在膝盖上了，画出来会拧巴。`Mood.conflicts` 已经声明了这类组合，
    随机时按当前动作过滤掉。

    除此之外不做任何"场景匹配"——用户在界面上选的就是"随机"，
    不该偷偷按光线再挑一遍。

    `avoid_recent` 传上一次用过的 id，能进一步减少连续重复。
    只是**降权**不是排除：如果过滤完只剩那一个，还是会用它。
    """
    import random
    cands = [m for m in MOODS
             if not (action_id and action_id in m.conflicts)]
    if not cands:
        cands = list(MOODS)
    if avoid_recent:
        fresh = [m for m in cands if m.id not in avoid_recent]
        if fresh:
            cands = fresh
    return random.choice(cands).id
