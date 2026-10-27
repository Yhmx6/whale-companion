"""流水线编排：一张照片进，N 张成品出。

S0 预处理 → S2 并发生成 → S3 后处理（去白边 / 环境色 / 接触阴影）→ S4 轻量质检

设计原则：
1. **生成器可插拔**：`generate_fn` 由调用方注入，默认用方舟 Seedream。
   换 Qwen / 换自建只改这一处（《05》§6 的适配器思路）。
2. **后处理是模型无关资产**：去白边、环境色、阴影全部纯 CPU，
   换任何生成器都要用，所以独立成层。
3. **失败不阻塞**：某个候选失败只标记该候选，不影响其他。
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from s3.lighting import LightingState  # noqa: E402
from s3.s2_prompt import (COMPANION_ACTIONS, anchor_for, build_companion_prompt,  # noqa: E402
                          build_prompt)
from s3.s3ops import (contact_core, occlude, render_shadow_matte,  # noqa: E402
                      shadow_under)
from s3.relight import amp_for, relight, scene_tint, side_light  # noqa: E402
from s3.beautify import (add_ai_label, change_mask, character_mask, defringe,  # noqa: E402
                         match_ambient)
from s3.rescale import _bbox, rescale_to_box, rescale_to_height  # noqa: E402
from s3.actions import enabled_actions, get_action  # noqa: E402
from s3.characters import get_character, list_characters  # noqa: E402
from s3.profile import filter_by_framing, vlm_profile, vlm_props  # noqa: E402

REFS_DIR = os.path.join(ROOT, "assets", "refs", "crops")
OUT_ROOT = os.path.join(ROOT, "output", "runs")

# 每种接触面：姿势描述 + 接触对象 + 接触点相对框高的位置
# contact_ratio=1.0 表示接触点在框底边（站姿）；坐姿接触面在框底边之上。
# 这个比例是启发式估值：坐姿的接触线（臀底/椅面）大约在框高的 82% 至 88% 处。
# 首轮真实验证发现：用框底边当接触点，影子会落到地上而她其实坐在垫子上。
SURFACE_SPEC = {
    "ground": dict(pose="全身站直，正面朝向镜头，双脚踩在地面上", contact="地面",
                   contact_ratio=1.0),
    "chair": dict(pose="端坐在坐垫上，双腿自然垂在椅子前方", contact="坐垫",
                  contact_ratio=0.86),
    "bed": dict(pose="坐在床沿，双腿自然垂在床边", contact="床沿",
                contact_ratio=0.88),
    "desk": dict(pose="坐在桌沿上，双腿自然垂在桌子外侧", contact="桌面",
                 contact_ratio=0.84),
    "wall": dict(pose="背靠墙面站立，身体轻微侧向镜头", contact="墙面",
                 contact_ratio=1.0),
}

# 主光方向选项：只给"光从哪边来"这种知觉判断，不给专业参数（见 06 §8.4）
# reach 是投射距离倍率：室内散射光影子短（1.0），晴天低角度阳光影子长得多（1.9）
# 物件路线的尺度：**她画出来的高度 / 画面高度**。
#
# ## 为什么不用"座面宽度 × 系数"
#
# 试过。第一次框到整条沙发（可见宽 1198px），算出 186px；第二次同一个模型
# 只框了沙发的一段（425px），算出 66px——小到看不见。
# **VLM 给的座面横向范围不稳定**，"框多长算一段坐垫"没有客观边界。
# 拿不稳定的量当尺子，结果就会忽大忽小。
#
# ## 为什么可以接受"按画面高度取经验比例"
#
# 照片里没有人物时，本来就没有任何绝对尺度参照：一面墙、一张沙发，
# 不知道它有多远、多大，烧不出真实厘米数。所以这里不假装能算出绝对尺寸，
# 而是给一个**经验比例**——"一个玩偶坐在这个家具上，大概占画面高度的多少"。
# 这个量稳定、可调，而且不受 VLM 框抖动影响。
#
# 取值依据：家庭室内照里，一张沙发通常占画面高度 55%~75%，
# 40cm 的玩偶坐上去约是沙发高度的 1/4~1/3 → 13%~25% 画面高。取中段。
#
# 真正的修法是给一个物理参照（比如 VLM 顺便判断"这是单人椅还是三人沙发"，
# 或者画面里恰好有一件已知尺寸的东西）。等有了人物分支之外的可靠参照再改。
SEAT_H_FRAC = {
    "sofa": 0.20, "bench": 0.19, "chair": 0.22, "bed": 0.18,
    "table": 0.20, "counter": 0.18, "step": 0.16,
}
# 第二把尺子：**座面可见宽度 × 系数**。和上面的画面比例取较大值。
#
# 为什么要两把尺子：
#   - 只有"画面高度比例"：近景特写会严重偏小。实测一张桌面特写（桌子占满画面），
#     按 0.20 算她只有 193px，而桌面厚度就有 90px——她只比桌板厚两倍，
#     读起来像贴纸，完全看不出"坐"。
#   - 只有"座面宽度"：远景或座面被截断时，宽度会算小，角色缩到看不见
#     （实测空沙发两次调用分别算出 186px 和 66px）。
#   取两者的较大值，正好把两个失效方向都兜住：不会小到看不见，也不会在近景里缩成纸片。
SEAT_W_RATIO = {
    "sofa": 0.19, "bench": 0.19, "chair": 0.30, "bed": 0.19,
    "table": 0.23, "counter": 0.16, "step": 0.26,
}
# 兜底夹紧：无论下面算出什么，都不许超出这个区间
SEAT_H_RANGE = (0.12, 0.42)


LIGHT_PRESETS = {    "auto": dict(azimuth=None, hardness=0.25, intensity=0.42, confidence=0.35, reach=1.0),
    "left": dict(azimuth=180.0, hardness=0.45, intensity=0.62, confidence=0.75, reach=1.0),
    "right": dict(azimuth=0.0, hardness=0.45, intensity=0.62, confidence=0.75, reach=1.0),
    # 晴天户外：影子更硬、更深、更长。8 组实测里户外光照与室内差异太大，
    # 用室内参数渲出来像"没影子"，所以单列两档。
    "sun_left": dict(azimuth=180.0, hardness=0.78, intensity=0.82, confidence=0.88, reach=1.9),
    "sun_right": dict(azimuth=0.0, hardness=0.78, intensity=0.82, confidence=0.88, reach=1.9),
}


def _encode_refs(character=None) -> list[str]:
    """把角色的参考图编码成 data URI。参考图由角色注册表给，不再写死。"""
    from api.ark import image_path_to_data_uri
    from s3.characters import get_character
    character = character or get_character()
    return [image_path_to_data_uri(p) for p in character.refs]


def _pick_size(im: Image.Image, max_mp: float = 2.55) -> str:
    ratio = im.height / im.width
    w = 1280 if ratio >= 1 else 1440
    h = w * ratio
    s = min(1.0, (max_mp * 1e6 / (w * h)) ** 0.5)
    return f"{int(w * s / 8) * 8}x{int(h * s / 8) * 8}"


def build_case_prompt(surface: str, box_norm: tuple[float, float, float, float],
                      anchor_override: str | None, light_key: str) -> str:
    """把归一化框 + 接触面 + 锚点 + 光照，组装成提示词。"""
    spec = SURFACE_SPEC.get(surface, SURFACE_SPEC["ground"])
    x, y, w, h = box_norm
    cx, cy = (x + w / 2) * 100, (y + h) * 100
    placement = (
        f"放在画面水平约 {cx:.0f}%、垂直约 {cy:.0f}% 的位置，"
        f"整个人占画面宽度约 {w * 100:.0f}%"
    )
    light_desc = {
        "auto": "环境柔和的自然光，没有强烈的方向性",
        "left": "主光从画面左侧照进来，左侧偏亮、右侧偏暗",
        "right": "主光从画面右侧照进来，右侧偏亮、左侧偏暗",
    }.get(light_key, "环境柔和的自然光")
    return build_prompt(
        placement=placement,
        pose_desc=spec["pose"],
        contact=spec["contact"],
        light_desc=light_desc,
        anchor=anchor_override or anchor_for(surface),
    )


def _qc_scores(scene: Image.Image, out: Image.Image,
               box_px: tuple[float, float, float, float],
               mask: "np.ndarray | None" = None, work: int = 400) -> dict:
    """S4 轻量质检：接触暗区 + 轮廓白边强度 + 蓝色占比。只做记录，不做拦截。"""
    from s3.beautify import change_mask

    # 全部在缩小图上算。这几个数只是**记录**，不参与任何判断
    #（注释原文就是"只做记录，不做拦截"），却原来在 1280×1600 上跑了两轮
    # MaxFilter/MinFilter，实测吃 1.0 秒 —— 占整条 CPU 链路的 26%。
    w = max(64, min(int(work), out.width))
    h = max(64, int(round(out.height * w / out.width)))
    a_s = np.asarray(scene.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32)
    b_s = np.asarray(out.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32)

    m_full = change_mask(scene, out) if mask is None else mask.astype(bool)
    res = {"changed_px_ratio": round(float(m_full.sum()) / m_full.size, 4)}
    m_img = Image.fromarray((m_full.astype(np.uint8) * 255), "L").resize(
        (w, h), Image.BILINEAR)
    m_s = np.asarray(m_img) > 127

    # 接触暗区：框底边下方条带是否比同图地面基线更暗
    k = w / float(out.width)
    x0, y0, x1, y1 = [int(v * k) for v in box_px]
    yy0 = min(max(y1, 0), h - 2)
    yy1 = min(yy0 + max(3, int((x1 - x0) * 0.18)), h)
    xx0, xx1 = min(max(x0, 0), w - 2), min(max(x1, 0), w)
    res["contact_band_delta"] = round(
        float(a_s[yy0:yy1, xx0:xx1].mean() - b_s[yy0:yy1, xx0:xx1].mean()), 1)

    # 白边强度：轮廓环内亮像素占比
    lum = b_s.mean(axis=2)
    from PIL import ImageFilter
    ring = (np.asarray(m_img.filter(ImageFilter.MaxFilter(7))) > 127) & ~(
        np.asarray(m_img.filter(ImageFilter.MinFilter(7))) > 127)
    res["rim_bright_ratio"] = round(float((lum[ring] > 205).mean()) if ring.any() else 0.0, 3)

    # 蓝色占比（身份特征代理指标）
    r_, g_, bl_ = b_s[..., 0], b_s[..., 1], b_s[..., 2]
    blue = (bl_ - np.maximum(r_, g_) > 18) & (bl_ > 45)
    res["blue_ratio"] = round(float(blue[m_s].mean()) if m_s.any() else 0.0, 3)
    return res



# ---------------------------------------------------------------------------
# 立绘合成路线
# ---------------------------------------------------------------------------
#
# 与"场景内生成"的根本差别：**不需要生成**。
# 立绘是提前做好并缓存的（tools/build_sprites.py），与照片无关；
# 每张照片只是把缓存立绘缩放、定位、贴上去，全程 CPU，亚秒级。
#
# 这也是为什么这条路的抠图质量好一个量级：立绘在纯白背景上生成，
# 抠图判据是"近白且与画面边界连通"，实测内部空隙 1.8~4.5%，
# 而在真实照片上抠最好也有 7.6%。

# 景别 -> "画面里能看到的这截占全身多少"。只给"按头对齐"用（半身照背后探头），
# 不是精确人体测量：她的下半身会被画面裁掉，系数偏一点只影响露出来那部分多大。
_VISIBLE_FRAC = {"full": 1.0, "half": 0.58, "closeup": 0.30}

# 放不下时最少缩到原比例的多少。低于这个值"等身"就不成立了，
# 与其继续缩，不如承认这张照片放不下、在结果里标出来。
_FIT_FLOOR = 0.5


def _sprite_placement(a, prof, props, route, W: int, H: int, char=None,
                      sprite=None, fg=None):
    """算出 (目标高度, 中心 x, 底边 y, 诊断信息)。

    和场景内路线的落点规则完全一致（人物身高 / 座面宽度两把尺子），
    只是这里不需要"抠出来再缩放"，直接拿数值去 resize 立绘。
    """
    if route == "object":
        # 等身角色不能走物件路线：这条路的尺度是按玩偶标的
        # （`SEAT_H_FRAC` 占画面高 12~42%），而且 `SEAT_H_RANGE` 会把她夹到 42% 以内。
        # 更要紧的是——没人物的照片里**没有可靠的尺寸参照**，
        # 家具的远近、大小都只能靠经验比例猜，猜出来的"等身"不是等身。
        # 要做得对，得引入物理参照（家具真实尺寸 ÷ 画面里家具的高度 = cm/px），
        # 那是另一件事，先不做。
        if float(getattr(char, "scale", 0) or 0) >= 0.6:
            raise ValueError(
                "等身角色需要照片里有完整的人来当尺寸参照。"
                "这张照片没有人物，请改用手办尺度的角色。")
        # 座面检测是 VLM 输出，会偶发抽风（同一张长椅图，之前几次都给出可用框，
        # 某次返回 0 个可用）。这里必须给出可读的失败，而不是 IndexError：
        # 崩溃栈会一路冒到调用方，前端拿到的是一段 traceback 而不是"这张用不了"。
        usable = [p for p in props if p.usable]
        if not usable:
            raise ValueError("这张照片没找到可用的座面，换一张或改用人物照片。")
        prop = usable[0]
        seat_w = (prop.x2 - prop.x1) / 999.0 * W
        by_frame = SEAT_H_FRAC.get(prop.kind, 0.20) * H
        span = seat_w / max(W, 1)
        # 特写时用"座面宽度"兜画面比例。但**只在座面没被画面裁掉时才有意义**：
        #
        # 座面横跨整个画面（span≈1.0）时，`seat_w` 就是画面宽度本身——
        # 它量的不是"家具多大"，而是"画面被家具占满了"。
        # 拿它当尺子，结果只跟画幅成正比。实测五张新场景：
        #   沙发 0.54 / 长椅 0.72 / 扶手椅 0.64 → 20~22%（正常）
        #   餐桌 1.00 → 34.5%、台阶 1.00 → 39.0%（凭空大 1.7~1.9 倍）
        # 所以 0.9 以上判"被裁掉，没有宽度信息"，退回画面比例。
        if 0.6 < span < 0.9:
            target_h = max(by_frame, seat_w * SEAT_W_RATIO.get(prop.kind, 0.19))
        else:
            target_h = by_frame
        lo, hi = SEAT_H_RANGE[0] * H, SEAT_H_RANGE[1] * H
        target_h = max(lo, min(hi, target_h))
        center_x = (prop.x1 + (prop.x2 - prop.x1) * 0.32) / 999.0 * W
        sink = (1.0 - a.contact_ratio) * target_h
        bottom_y = prop.y / 999.0 * H + sink
        return target_h, center_x, bottom_y, {"mode": "seat", "kind": prop.kind,
                                              "seat_y": prop.y, "span": round(span, 2)}

    # `body_len_px` 而不是 `person_h_px`：前者会按姿态取框的长边，
    # 躺着的人身高在横向上（见 `profile.body_len_px` 的注释）。
    person_h = prof.body_len_px(W, H) or 0.0
    # 尺度改由**角色**决定，不再用 `a.scale_ratio`。
    # `scale_ratio` 是 Q 版时代的字段（所有动作统一 0.26），
    # 混着角色和动作两件事；等身角色进来之后必须拆开：
    #   chibi 0.26（≈45cm 手办）  /  mature 0.92（≈160cm，站在旁边的人）
    ratio = float(getattr(char, "scale", None) or a.scale_ratio)

    # 等身角色只有"看得见脚"的照片才算得出比例。
    #
    # `body_len_px` 半身照里量的只是**画面里看得到的那半截**
    # （实测 s02_park 只有上身，1308px）。拿它乘 0.92 会把她画得跟半身人一样高，
    # 而且脚底线也是画面底边，落点跟着一起错。
    # 玩偶尺度（0.26）受影响小得多——差一半也就是从 45cm 变成 25cm，还能看；
    # 等身差一半就完全不是"等身"了，所以这里直接拒绝而不是猜一个系数。
    if (ratio >= 0.6 and (prof.framing or "") != "full"
            and not getattr(a, "behind", False)):
        raise ValueError(
            "这张照片看不到人物的脚，算不出等身比例。请换一张全身照，"
            "或者改用手办尺度的角色。")

    target_h = person_h * ratio
    foot_y = prof.foot_y_px(H) or 0.0
    lb = prof.legs
    if lb:
        leg_cx = (lb[0] + lb[2]) / 2.0 / 999.0 * W
        leg_w = (lb[2] - lb[0]) / 999.0 * W
    else:
        pb = prof.person or [0, 0, 999, 999]
        leg_cx = (pb[0] + pb[2]) / 2.0 / 999.0 * W
        leg_w = (pb[2] - pb[0]) / 999.0 * W * 0.5
    # 她站在人物腿旁边的横向让位。等身角色宽度大得多，固定 6px 的间隙
    # 会让她的手臂压在人腿上，所以间隙随她的尺寸放大。
    # 宽度按立绘**真实宽高比**算。原来写死 0.78（Q 版的宽高比），
    # 换到成人形态（793×1536 ≈ 0.52）就完全不对——影响横向让位、
    # 也影响"被挡住多少"这种要拿宽度算的东西。
    aspect = (sprite.width / sprite.height) if sprite is not None else 0.78
    target_w = target_h * aspect
    gap = max(6.0, target_h * 0.03)
    margin = W * 0.025

    # ---- 放不下就缩小（等身角色才会触发）----
    #
    # 玩偶尺度（0.26）时她只有 200px 宽，随便放都放得下，这条从不生效。
    # 等身尺度的实测：`s04_full` 里她 699px 宽，人腿左侧只剩约 250px，
    # 硬贴的后果是 px 被夹到 2——**左边贴死在画面边缘**，而且她的肩压在人手臂上。
    #
    # 所以按可用宽度反算一个比例。下限 0.5 是防止极端构图把她缩成玩偶——
    # 那时"等身"这个设定已经不成立，不如保留一个能认出来的尺寸并在 meta 里标出来。
    avail = leg_cx - leg_w * 0.5 - gap - margin
    fit_k = 1.0
    # `behind` 动作不参与这条：她站在人身后，宽度大半被人挡住，
    # 根本不需要那 700px 空地。不排除的话会被白缩一轮——
    # 实测正是如此：target_h 从 1355 被压到 918，她的头只到人物胯部。
    if not getattr(a, "behind", False) and avail > 40 and target_w > avail:
        fit_k = max(_FIT_FLOOR, avail / target_w)
        target_h *= fit_k
        target_w = target_h * aspect
        gap = max(6.0, target_h * 0.03)
    # ---- 站在人物身后（被挡住）----
    #
    # 前面那套是"人旁边的空地"：要求她完整可见，于是要 `target_w` 那么宽的空。
    # 这里反过来——她只从人物轮廓的一侧**露出 visible_ratio 那么宽的缝**，
    # 其余部分被人物盖住。等身角色在竖构自拍里因此也放得下。
    lift = a.lift_ratio * target_h
    if getattr(a, "behind", False) and fg is not None:
        from s3.s3ops import silhouette_edge
        if (prof.framing or "") == "full" or not prof.person:
            bottom = foot_y - lift
            band_lo, band_hi = 0.15, 0.94
        elif (prof.framing or "") != "full":
            # 半身照的越肩构图**暂不支持**，用明确的失败挡住，不出废图。
            #
            # 卡点不在定位，在**掩码**：`s02_park`（两人坐长椅）里 MODNet 把长椅
            # 和两个人连成一块，连通域过滤按"是否穿过人物框中心区"分不开它们。
            # 遮挡阶段一盖，她整片被擦掉，成品只剩几条蓝边。
            #
            # 单人的全身照没这个问题（实测 s04_full 掩码干净）。
            # 要支持半身构图，得换更干净的人物掩码（例如按人分别实例分割），
            # 或者改用"肩线锚点 + 只遮挡头部那一小块"的做法。
            raise ValueError(
                "「在身后探头」目前只支持全身照。半身照里人物和家具容易连成一片，"
                "遮挡会误伤。请换一张全身照，或改用其他动作。")
        else:
            # 半身照没有脚可以对齐，就**按头对齐**。
            #
            # 这是"从身后探头"这类构图的正解：她的头和人物的头在同一高度，
            # 身体往下延伸到画面外——本来就该出画，不需要脚。
            #
            # 她有多高要靠景别反推：`person_h` 只是"画面里看得到的那一截"。
            # `_VISIBLE_FRAC` 是"半身/特写各占全身多少"的经验值，
            # 只用于"头对齐"这个用途——她的下半身反正会被画面裁掉，
            # 系数偏一点只影响她露出来的那部分多大，不影响位置。
            frac = _VISIBLE_FRAC.get(prof.framing, 0.75)
            target_h = person_h * ratio / frac
            top_px = (prof.person[1] / 999.0) * H + person_h * 0.03
            bottom = top_px + target_h
            # 取边只在**人物头部到肩**这一带：再往下就是长椅和背景，
            # 掩码在那里不可靠（实测 y>700 之后左右边界直接是整幅宽）。
            band_lo, band_hi = 0.02, (person_h * 0.30) / max(target_h, 1.0)
        top = bottom - target_h
        edges = silhouette_edge(fg, int(top + target_h * band_lo),
                                int(top + target_h * band_hi))
        if edges is not None:
            left_edge, right_edge = edges
            want = float(getattr(a, "visible_ratio", 0.55))
            # 露多少由**剩余空间**反算，不是常量。
            #
            # 实测 `s02_park`（两人自拍）：两个人的头只占 x 204~1091，
            # 左边只剩 172px；而她要露出 55% 需要 526px。写死比例的结果是
            # 她被推出画面，成品里只剩几条蓝边。
            #
            # 反过来想：这种构图本来就是"从人缝里探出一点点"，
            # 露 18% 才是对的样子。
            room_left = left_edge - margin
            room_right = W - right_edge - margin
            side, room = ("left", room_left) if room_left >= room_right else ("right", room_right)
            vis = max(0.16, min(want, room / max(target_w, 1.0)))
            px = (left_edge - target_w * vis) if side == "left" \
                else (right_edge - target_w * (1.0 - vis))
            px = max(2, min(W - target_w - 2, px))
            meta = {"mode": "behind", "person_h": int(person_h), "foot_y": int(foot_y),
                    "lift_px": int(lift), "scale_ratio": round(ratio, 3),
                    "fit_k": 1.0, "space_limited": False,
                    "behind_side": side, "visible_ratio": round(vis, 2),
                    "vis_limited": vis < want - 1e-3,
                    "occluder_left": left_edge, "occluder_right": right_edge,
                    "character": getattr(char, "id", "")}
            return target_h, px + target_w / 2.0, bottom, meta
    center_x = leg_cx - (leg_w * 0.5 + target_w * 0.5 + gap)
    return target_h, center_x, foot_y - lift, {"mode": "person", "person_h": int(person_h),
                                               "foot_y": int(foot_y), "lift_px": int(lift),
                                               "scale_ratio": round(ratio, 3),
                                               "fit_k": round(fit_k, 3),
                                               "space_limited": fit_k < 0.995,
                                               "character": getattr(char, "id", "")}


def _validate_action_vs_scene(act, scene2d, options: list[dict]) -> tuple:
    """动作和场景对不上时，换一个能成立的。

    ## 为什么需要

    实测 c10（俯视木地板 + 正中间一个背包）：规划器选了 `seat_edge_sit`
    （坐在边缘垂腿），因为它把**背包当成了可坐的边缘**。成品里她坐在背包顶上。

    规划器只看图说话，它不知道"背包不能坐"这条常识——而 `objects` 里有
    `kind=bag, sit=False`，这个信息是现成的，程序一查就知道它选错了。

    ## 判据

    坐姿动作（`anchor_kind == "seat_plane"`）**必须真的有可坐的面**。
    场景里一件 `sit=True` 的物体都没有时，降级成地面动作优先，其次蹲姿。

    只在 `objects` 非空时校验：空的时候分不清"真没有座面"和"VLM 漏检"，
    误降级会把本来能坐的场景改成站着。
    """
    from s3.actions import get_action

    if act is None or act.anchor_kind != "seat_plane":
        return act, ""
    if not scene2d.objects:
        return act, ""
    if any(o.sit for o in scene2d.objects):
        return act, ""

    avail = {o["id"] for o in options}
    # 地面动作的优先顺序：站着最稳（手背身后零风险），其次蹲着，最后盘腿坐地上。
    # 都要在手上有兜底的前提下选——手部抓握在这个模型上画不好。
    for cand in ("leg_hands_behind", "crouch_by_leg", "leg_hug_knee",
                 "sit_cross_legged"):
        if cand in avail:
            nb = get_action(cand)
            if nb is not None:
                return nb, (f"规划器选了「{act.name}」，但画面里没有任何可坐的面"
                            f"（物体 {len(scene2d.objects)} 件，全都不 sit），"
                            f"换成「{nb.name}」")
    return act, ""


def _desc_overlap(text: str, desc: str) -> float:
    """两段中文姿态描述的词元重合度（二字组命中率）。

    用来判断"规划器重写的 action 是不是还在说同一个姿势"。
    中文不像英文有空格分词，按二字组切是最省事又够稳的近似——
    "坐在长椅的木条座面上" 和 "坐在座面的前边缘上" 会共享
    "坐在""座面"这些组，重合度明显，而 "站在腿边仰头" 就不会。
    """
    import re
    keys = set(re.findall(r"[\u4e00-\u9fa5]{2}", desc.replace("{anchor}", "座面")))
    if not keys:
        return 0.0
    return sum(1 for k in keys if k in text) / len(keys)


def available_sprite_options(model_key: str, character=None) -> list[dict]:
    """扫一遍立绘缓存目录，返回**已经有立绘**的动作候选清单。

    ## 为什么规划器必须从这里选

    布局引导这条路线的引导图，是一张**已经画好的立绘**贴进照片当占位符。
    生成模型因此同时收到两个信号：图里那个小人的姿势、提示词里的姿势描述。
    两者不一致时它各听一半——实测就是"规划器说坐长椅、立绘是站姿"，
    成品成了脚悬在长椅前沿外面的半空站姿。

    所以候选集必须是"有立绘的那些动作"，而不是"动作库里所有的动作"。
    动作库有 23 条，缓存的只有 10 张；让规划器在 23 里挑，它有一半概率
    挑到没立绘的，那就又没有引导图了。

    `{anchor}` 占位符在这里换成中性的「座面」，因为规划器还不知道
    具体是哪件家具——它自己会在 `action` 里写实际名词。
    """
    from s3.actions import ACTION_LIBRARY
    from s3.characters import get_character
    from s3.sprite import sprite_path

    char = character or get_character()
    out: list[dict] = []
    for a in ACTION_LIBRARY:
        if not os.path.exists(sprite_path(a.id, model_key, char)):
            continue
        out.append({"id": a.id, "name": a.name,
                    "desc": a.desc.replace("{anchor}", "座面"),
                    "anchor_kind": a.anchor_kind,
                    "needs": list(a.needs), "framing": list(a.framing)})
    return out


def _ensure_sprite(a, model_key: str, progress=None, character=None):
    """取缓存立绘；没有就现生成一张并缓存（一次性成本，之后复用）。

    缓存按角色分目录，否则两个角色的同名姿态互相覆盖。
    """
    from s3.characters import get_character
    from s3.sprite import (build_sprite_prompt, key_white, load_sprite,
                           save_sprite, trim)
    char = character or get_character()
    sp = load_sprite(a.id, model_key, char)
    if sp is not None:
        return sp, "cached"
    if progress:
        progress("生成立绘（首次）", 30)
    from api.ark import download, generate
    st, resp = generate(build_sprite_prompt(a.desc, char),
                        images=_encode_refs(char),
                        size="1024x1536", model=model_key, seed=3001)
    if st != 200:
        raise RuntimeError(f"立绘生成失败 HTTP {st}: {str(resp)[:200]}")
    tmp = os.path.join(OUT_ROOT, "_sprite_tmp.png")
    os.makedirs(OUT_ROOT, exist_ok=True)
    download((resp.get("data") or [{}])[0]["url"], tmp)
    sp = trim(key_white(Image.open(tmp).convert("RGB")))
    save_sprite(a.id, model_key, sp, char)
    return sp, "generated"


def _size_anchor(prof, plan, height_cm: float = 60.0) -> str:
    """给她一句"只到某某位置"的尺寸参照。

    这是这个项目里唯一被证有效的尺度约束形式：
    - **物体相对参照**（"只到人物膝盖""只有坐垫宽度三分之一"）——模型能理解，
      因为它是在和画面里已经存在的东西做比较；
    - **画面百分比**（"她占画面高度 20%"）——实测不服从，是 `s2_prompt.py`
      里 `_DEPRECATED_PERCENT_SCALE` 记录的被证伪写法。

    人物照片用人物身上的部位当尺子（膝盖、小腿中部）；没人物的照片用家具的
    局部尺寸当尺子（坐垫宽度、桌沿厚度）。

    ## 为什么布局引导路线也要这一段

    引导图里已经有一个尺寸正确的小人，理论上"照着重画"就够了。实测不够：
    p05 的引导图里她是 31.1% 画面高，成品出来约 45%，放大了 1.4 倍。
    提示词里光写"不许放大"压不住——那是**否定式指令**，而这个模型对否定式
    的遵从度本来就低（`docs/08` 第七节）。
    改成给一个**可对照的实物**（"头顶刚过人物膝盖"），它就有了正向的落点。
    """
    cm = f"{height_cm:.0f}"
    if prof.has_person and prof.framing == "full":
        # 60cm 对 175cm 的人 ≈ 到膝盖（膝高约 45~50cm，加上头发到膝盖上方）。
        # 绝对厘米 + 物体参照一起给：前者是产品规定，后者是模型能对照的实物。
        return (f"她的头顶大约刚过人物的膝盖，整个人不超过人物大腿的高度"
                f"（她真实身高 {cm} 厘米）")
    if prof.has_person:
        return ("她整个人大约只有人物头部那么大，"
                "只占人物肩膀到肘部这段的空间")
    kind = (plan.anchor_kind or "").strip()
    if kind in ("seat_plane", "horizontal_edge"):
        # 座面支路用一条**通用物理关系**当参照：她坐着时臀到头顶约 51cm
        # （60cm 的 85%），而常见座面高 45cm——两者本来就几乎相等。
        # 所以"她坐着的高度 ≈ 座面到地面的那段高度"在任何有座面的照片里都成立，
        # 而且座面高度是画面里直接看得见的量。
        #
        # 原来这里写的是"头顶远低于座面后方的靠背"。那句在沙发、椅子上对，
        # 在**台阶**上就完全错了——台阶没有靠背（实测 e05 就是这么翻的，
        # 模型拿不到参照，把她放大到计划的 1.8 倍）。
        return ("她坐着的时候，整个人的高度大约等于**座面到地面的那一段高度**："
                "头顶只比座面高一点点，远不到座面后方靠背的高度")
    if kind == "ground_plane":
        return "她整个人大约只有一块地砖那么高，像落在地上的玩偶"
    return ("她整个人只有半个抱枕那么大，"
            "在画面里所占的面积远小于画面里的家具")


# 规划器给的 anchor_kind -> 用哪张立绘当"引导图里的小人"。
#
# **现在只是兜底**。真正的路径是规划器从 `available_sprite_options()` 里选
# `action_id`——那才是"动作匹配"的正解，因为引导图只能是一张已存在的立绘，
# 而规划器自由创作姿态时，两个信号会打架（实测：规划说坐、立绘是站，
# 成品成了脚悬在长椅前沿的半空站姿）。
# 这张表只在规划器漏写 id、且按文字也兜不出来时才用，粒度很糙。
_ANCHOR_TO_SPRITE = {
    "seat_plane": "seat_edge_sit",
    "horizontal_edge": "seat_edge_sit",
    "ground_plane": "leg_hands_behind",
    "person_part": "leg_hands_behind",
    "vertical_prop": "leg_hands_behind",
}

GUIDED_PROMPT = (
    "图1是一张真实照片，照片里已经贴了一个卡通小人。图2、图3是同一个二次元角色的设定图。\n"
    "这个角色的形象是：{identity}。\n"
    "请把图1里那个贴上去的卡通小人**重绘**成图2图3的角色形象，并让她与场景自然融合：\n"
    "1. 【最重要】成品里她的**全部轮廓**必须落在图1那个小人已经占据的范围之内。"
    "这个范围是：**从头顶呆毛的最高点到鞋底**，**从尾巴最末端到头发最外侧**——"
    "注意它已经把她所有的头发、呆毛、尾巴、裙摆都算进去了。"
    "所以她的头发、尾巴、裙子都不许越过那个范围，也不许为了把她画得更可爱、"
    "更清楚而把任何部位放大；\n"
    "2. 【尺寸复核】画完之后，她的高度必须正好等于图1里那个小人的高度："
    "{size_anchor}。"
    "如果她看起来比这个更高或更大，就画错了，必须缩小到那个高度；"
    "她真实身高 {height_cm:.0f} 厘米，不要画成与人物同高的成年人；\n"
    "3. 姿势改成：{pose}；\n"
    "4. 【表情与朝向】{mood_clause}；\n"
    "5. 在她身下和紧贴她的地方画出压出的接触阴影；画面里的受光方向就是她的受光方向；\n"
    "6. 除她和紧挨着的接触阴影外，**图1其他内容一律不许改动**：人物的脸和衣服、"
    "家具、背景、光线、清晰度全部保持原样；\n"
    "7. 她必须是图2图3里的那个角色：同样的发型、发箍、耳朵、裙子、围裙、尾巴、鞋子；\n"
    "8. 画面里恰好一个角色；不要文字、水印、边框。"
)


# 剪影引导：贴在引导图上的是一个**纯色形状**，不是立绘。
#
# ## 为什么要这个变体
#
# 实测发现生成模型对"图里那个小人多大"这件事**不服从**：把引导图里的小人
# 从 42.8% 缩到 10.6%，成品稳定在 31~34% 不变（见 `GUIDED_SIZE_LIMIT`）。
# 它对"一个卡通小人"有内在的尺寸先验。
#
# 但"填满一个给定的形状"是另一类任务——inpainting 式的语义，模型没有
# 自由发挥尺寸的余地。所以试这个：引导图里放品红色剪影（就是立绘的 alpha），
# 让它把剪影原样替换成角色。
#
# 品红是选的：照片里几乎不会出现这个色，模型不会把它误认成场景内容；
# 又不至于像纯黑那样被当成"阴影/洞口"。
GUIDED_SILHOUETTE_PROMPT = (
    "图1是一张真实照片，照片里有一个**品红色的纯色剪影**。图2、图3是同一个二次元角色的设定图。\n"
    "这个角色的形象是：{identity}。\n"
    "请把图1里那个品红色剪影**替换**成图2图3的角色：\n"
    "1. 【最重要】她必须**完全填满那个剪影的形状**：她的头顶正好到剪影的最高点，"
    "鞋底正好到剪影的最低点，头发和尾巴的外缘正好与剪影的左右边缘重合。"
    "不许超出剪影一个像素，也不许比剪影小一圈——剪影的每一条边都是她的轮廓；\n"
    "2. 剪影被替换之后，**品红色必须完全消失**，画面里不许残留任何品红色边缘；\n"
    "3. 姿势：{pose}；\n"
    "4. 【表情与朝向】{mood_clause}；\n"
    "5. 在她身下和紧贴她的地方画出压出的接触阴影；画面里的受光方向就是她的受光方向；\n"
    "6. 除她和紧挨着的接触阴影外，**图1其他内容一律不许改动**：人物的脸和衣服、"
    "家具、背景、光线、清晰度全部保持原样；\n"
    "7. 她必须是图2图3里的那个角色：同样的发型、发箍、耳朵、裙子、围裙、尾巴、鞋子；"
    "她真实身高 {height_cm:.0f} 厘米，不要画成与人物同高的成年人；\n"
    "8. 画面里恰好一个角色；不要文字、水印、边框。"
)

SILHOUETTE_RGB = (255, 0, 170)

# 这个生成模型不肯把角色画到画面高的 26% 以下。
#
# ## 怎么测出来的
#
# 四张场景，程序把 60cm 换算成画面百分比（`person_h × 60/175`），
# 生成后用 `s3/measure.py` 客观量成品（不是目测）：
#
# | 场景 | 人物占画面 | 60cm 应占 | 实测 | 倍率 |
# |---|---|---|---|---|
# | p01 站姿 | 62% | 21.3% | 31.8% | 1.49 |
# | p03 站姿 | 55% | 18.7% | 26.0% | 1.39 |
# | p05 抱膝坐 | 90.6% | 31.1% | 36.7% | 1.18 |
# | e05 坐台阶 | 无人物 | 26.4% | 42.2% | 1.60 |
#
# 关键不是"偏大"，是**实测值全部 ≥ 26%，没有一个更低**。
# 而 p01 上专门做了反向验证：把引导图里的小人从 21.3% 缩到 14.2%、10.6%，
# 成品照样是 33.9%、31.2%（[`step21_size_bias_calib.py`]）。缩一半都不动。
#
# 所以这是个**下限**，不是随机误差：它对"卡通小人"有内在的尺寸先验，
# 小于这个尺度就当作"贴纸"重画成它认为合理的大小。
#
# ## 这个常量怎么用
#
# 不是用来"补偿"的（补偿已证无效），是用来**提前告诉调用方这次做不做得成**。
# `target_pct < 0.26` 就是做不到，与其让用户拿眼睛猜，不如直接标出来
# 并建议改用立绘路线（那条路尺寸 100% 精确，代价是没有遮挡互动）。
GUIDED_SIZE_FLOOR = 0.30


def _make_guide(photo: Image.Image, sprite, px: int, py: int,
                nw: int, nh: int, style: str = "sprite") -> Image.Image:
    """把立绘（或它的纯色剪影）贴到照片上，得到给生成模型的引导图。

    style="sprite"     贴立绘原图——模型看得见她长什么样，但会按自己的先验改尺寸
    style="silhouette" 只贴形状——尺寸由形状锁死，长相交给图2图3的设定图
    """
    guide = photo.copy()
    s = sprite.resize((nw, nh), Image.LANCZOS)
    if style == "silhouette":
        flat = Image.new("RGB", s.size, SILHOUETTE_RGB)
        guide.paste(flat, (px, py), s)
    else:
        guide.paste(s, (px, py), s)
    return guide


def run_guided(photo_bytes: bytes, character: str | None = None,
               model: str | None = None, progress=None,
               size_bias: float | None = None,
               guide_style: str = "sprite",
               mood: str | None = None,
               action: str | None = None) -> dict:
    """布局引导生成：规划器定「在哪 + 做什么」，程序定「多大」，生成模型只负责「重绘 + 融合」。

    ## 为什么是这三段式

    实测三条路，只有这条同时满足三个要求：

    | 做法 | 位置动作 | 尺寸 | 遮挡互动 |
    |---|---|---|---|
    | 纯规划 -> 生成 | 好 | 失控 1.7~2.8 倍 | 自然 |
    | 规划 + 文字尺度锚点 | 好 | 仍失控 | 自然 |
    | **布局引导** | 好 | **精确** | 自然 |

    纯生成救不回尺度，是因为这个模型对"把角色画小"没有可控性：画面百分比
    （`s2_prompt._DEPRECATED_PERCENT_SCALE` 早就记为已证伪）和物体相对锚点都不服从。
    但**图里已经存在一个尺寸正确的小人**时它照做——参照物从抽象变成了具体。

    ## 落点用谁的

    **用规划器的**，不用物件路线的。物件路线按"VLM 给的座面框 + x1+width*0.32"算，
    实测在 p05 把她放到了长椅前面的地上（因为座面框的横向范围不稳定）。
    规划器给的是语义位置（"坐在长椅座面上、男士左侧"），配一个它自己定的框，
    可靠得多。大小仍然不听它的——它对厘米到百分比的换算系统性偏差 1.7 倍以上。
    """
    from s3.characters import get_character
    from s3.plan import plan_scene
    from s3.sprite import load_sprite

    def tick(stage: str, pct: int):
        if progress:
            progress(stage, pct)

    char = get_character(character)
    model_key = model or os.environ.get("ARK_IMAGE_MODEL", "doubao-seedream-5-0-flash-260915")
    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    tick("读取照片", 6)
    photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    W, H = photo.size
    scene_path = os.path.join(run_dir, "scene.jpg")
    photo.save(scene_path, quality=95)

    tick("理解照片", 14)
    prof = vlm_profile(scene_path)
    hint = (f"画面里有{prof.framing or '一个'}景别人物" if prof.has_person
            else "画面里没有人物")

    # ---- 物品解析为空时重试一次 ----
    #
    # 实测 `objects` 会偶发整段缺失：同一张 c09 吧台图，一轮解析出 4 个物体
    # （台面 + 两个杯子），下一轮一个都没有。挂了 `objects` 的避让就**静默失效**——
    # 不报错、不告警，只是她悄悄地又坐回了杯子上面。
    #
    # 这类"静默失效"比报错危险得多，所以宁可多花一次调用（约 1.5 秒）重试。
    #
    # 触发条件不能用 `has_prop`：实测 c09 / c02 的 `has_prop` 是 **false**，
    # 但 `props` 里明明有 2 件和 1 件——模型自己前后矛盾。
    # 只看 `has_prop` 的话重试永远不触发，`objects` 空着就一路空到底。
    # 所以三个信号只要有任何一个说有东西，就重试。
    if not prof.objects and (prof.has_prop or prof.props):
        prof2 = vlm_profile(scene_path)
        if prof2.objects:
            prof = prof2
            prof.raw["_retried"] = True

    # ---- 简易 2.5D 场景：谁占了哪块地方、谁在前谁在后 ----
    #
    # 和画像**同一次 VLM 调用**产出（`prof.objects`）。分开问会多 1.5 秒，
    # 而且两次的框可能对不上——场景图和画像打架时没法判断谁对。
    #
    # 有它才能回答"这个落点上有没有东西"，没它就只能像以前那样
    # 把中心 x 算出来直接贴（p04 就是这么坐到咖啡杯里去的）。
    from s3.scene3d import (Scene2D, avoid_collision, render_scene_overlay,
                            snap_to_surface)
    scene2d = Scene2D.build(prof.objects)

    # ---- 候选动作：只给**已经有立绘**的那些 ----
    # 这一步是"动作匹配"的关键。引导图是一张贴好的立绘，规划器必须从中选，
    # 否则文字和立绘两个信号打架（详见 available_sprite_options 的注释）。
    options = available_sprite_options(model_key, char)
    if not options:
        return {"error": f"角色 {char.id} 还没有任何立绘缓存，先生成一张再走这条路线",
                "character": char.id}

    # 调用方指定了动作（前端下拉框选的）→ 候选收窄成这一个，规划器只填位置和表情。
    # 不能直接把 `action` 塞给 `plan.action` 就完事：那样规划器仍会自由描述姿态，
    # 而文字姿态必须和贴上去的立绘一致，否则又回到"两个信号打架"。
    forced_action = ""
    if action:
        hit = [o for o in options if o["id"] == action]
        if hit:
            options = hit
            forced_action = action

    tick("出分镜", 22)
    # 表情候选：**程序先按场景收窄**，再交给规划器。
    #
    # 不能直接把 9 个表情全丢给 VLM——实测它会塌缩到最通用的那个
    # （第一版 tender 占 5/10，改完 happy 占 9/10），因为它总能给最通用的
    # 选项找到理由："她刚坐下，正处在兴头上"对每张照片都成立。
    #
    # `for_scene` 用像素统计（亮度/色温/对比/饱和）+ 语义（床、吧台…）
    # 先算出一份 4 个的短名单，规划器在短名单里挑。这样"场景差异"
    # 变成了程序可计算、可测试的量，而不是指望 VLM 自己注意到。
    from s3.moods import describe as mood_desc
    from s3.moods import for_scene as moods_for_scene
    mood_list = moods_for_scene(photo, has_person=prof.has_person,
                               objects=prof.objects,
                               action_ids=[o["id"] for o in options])
    plan = plan_scene(scene_path, height_hint=hint, options=options,
                      moods=mood_list)
    # ---- 表情：默认**随机**，也可以指定或按场景挑 ----
    #
    # 三档，由 `mood` 参数区分：
    #
    # | mood | 行为 |
    # |---|---|
    # | 不传 / `""` / `"random"` | **随机挑一个**（默认） |
    # | `"auto"` | 按场景挑——用规划器看完照片后的判断 |
    # | 具体 id（`shy` 等） | 指定 |
    #
    # 默认是随机，因为用户明确要"表情全随机"：同一个动作连着出几张，
    # 希望脸不一样，而不是每张都按光线算出同一个结论。
    #
    # 随机也不是完全不管不顾：`pick_random` 会按当前动作过滤掉姿态上
    # 不共存的表情（`Mood.conflicts`），比如"蜷成一团抱膝"时不会随机到
    # "挑眉得意"。
    from s3.moods import get_mood as _gm
    from s3.moods import pick_random as _pick_random
    mood_norm = (mood or "").strip().lower()
    # 背视角动作没有脸，整套表情逻辑直接跳过——
    # 随机一个"害羞"出来但提示词里写"看不到脸"，只会自相矛盾。
    _act_mood = get_action(plan.action_id) if plan.action_id else None
    if _act_mood is not None and not getattr(_act_mood, "face_visible", True):
        plan.mood_id, plan.mood_face = "", ""
        plan.mood_reason = "背视角——看不到脸，表情不适用"
        plan.raw["_mood_source"] = "none_back"
        mood_norm = "_skip_"
    if mood_norm == "_skip_":
        pass
    elif mood_norm == "auto":
        # 保留模型按场景选的结论，什么都不做
        plan.raw["_mood_source"] = "scene"
    elif mood_norm and _gm(mood_norm) is not None:
        plan.mood_face = _gm(mood_norm).face
        plan.mood_id = mood_norm
        plan.raw["_mood_source"] = "forced"
        plan.mood_reason = ""
    else:
        # 不传 / random / 拼错的 id → 随机
        mid = _pick_random(plan.action_id)
        mo = _gm(mid)
        plan.mood_id = mid
        plan.mood_face = mo.face if mo else ""
        plan.mood_reason = "随机挑选" + (
            f"（已排除与「{plan.action_id}」姿态不搭的表情）" if plan.action_id else "")
        plan.raw["_mood_source"] = "random"
    if not plan.ok:
        return {"error": f"规划失败：{plan.reason or plan.raw.get('error') or '没有可用分镜'}",
                "plan": plan.as_dict()}

    # ---- 动作：规划器选的优先，选不出来才回落到粗映射 ----
    act_id = plan.action_id
    act_src = plan.raw.get("_action_source", "plan")
    avail_ids = {o["id"] for o in options}
    if act_id and act_id not in avail_ids:      # 防御：不该发生
        act_id, act_src = "", "reject"
    if not act_id:
        # 粗映射。只在这里当兜底——它的粒度太糙，撑不起"动作匹配"。
        guard = _ANCHOR_TO_SPRITE.get(plan.anchor_kind, "")
        act_id = guard if guard in avail_ids else ""
        act_src = "anchor_map"
    if not act_id:
        # 再兜一层：按规划器给的依托类型找同类动作，最后才拿第一个。
        same = [o["id"] for o in options
                if o.get("anchor_kind") == plan.anchor_kind]
        act_id = same[0] if same else options[0]["id"]
        act_src = "anchor_kind" if same else "first"
    act = get_action(act_id)
    if act is None:                       # 理论上不会，兜一层
        act = get_action("leg_hands_behind") or get_action(options[0]["id"])
        act_id, act_src = act.id, "default"

    # ---- 动作与场景一致性：坐姿动作必须真的有可坐的面 ----
    # 放在这里（而不是选动作时）是因为要等场景图建出来才知道有没有座面。
    # **用户手动指定的动作不校验**——他明确选了"坐在边缘"，就算场景里没检出座面，
    # 也该按他说的来（可能是 VLM 漏检），而不是偷偷换成站姿。
    if forced_action:
        scene_check_note = ""
    else:
        act2, why = _validate_action_vs_scene(act, scene2d, options)
        if act2 is not act:
            act = act2
            act_id, act_src = act.id, "scene_check"
            scene_check_note = why
        else:
            scene_check_note = ""

    # ---- 大小：由角色真实身高算成厘米，不听规划器的 ----
    #
    # `char.scale` 就是 `height_cm / REF_PERSON_CM`。所以 `person_h * scale`
    # 读作"人物 175cm 时，她 60cm"，这是照片里唯一能落到厘米上的换算。
    # 物件路线（没人物的照片）没有这把尺子——照片里量不出家具的真实尺寸，
    # 也就烧不出 cm/px，那条支路退回经验比例并在 meta 里标 `size_basis`。
    #
    # **用 `body_len_px` 而不是框高**：躺着的人框高只是身体厚度，
    # 实测差 3 倍（用户那张沙发照：框高 287px、框宽 859px，
    # 按框高算她只有画面高 10.5%，按身体长度算是 31.6%）。
    person_h = prof.body_len_px(W, H) or 0.0
    if person_h:
        target_h = person_h * char.scale
        size_basis = "person_cm"
    else:
        target_h = H * 0.20 * char.scale / 0.26
        size_basis = "percent_estimate"
    target_h = float(max(24.0, min(H * 0.42, target_h)))

    # ---- 位置：听规划器的框（中心 x + 底边 y）----
    if plan.region:
        x1, y1, x2, y2 = plan.region
        center_x = (x1 + x2) / 2.0 / 999.0 * W
        bottom_y = y2 / 999.0 * H
        place_src = "plan"
    else:
        target_h, center_x, bottom_y, _m = _sprite_placement(
            act, prof, prof.props,
            "person" if prof.has_person else "object", W, H, char)
        place_src = "fallback"

    sprite = load_sprite(act_id, model_key, char)
    if sprite is None:
        return {"error": f"缺立绘缓存：{act_id}", "plan": plan.as_dict()}

    # ---- 落点吸附：规划器的 y 经常悬在座面上方 ----
    #
    # 实测 c05：规划器给 y=548，长椅座面在 654，差 106 像素（约 10% 画面高）。
    # 人眼看着是"坐在座面上方一点点"，但程序差 40 像素就匹配不上 `sit_target`，
    # 于是长椅从"我的座位"变成"挡路的障碍"，整个避让跟着跑偏。
    # c08 床沿更极端：差 171 像素。
    #
    # 只在她身高的 75% 范围内吸，再远就说明规划器指的不是这个面。
    target_w = target_h * (sprite.width / sprite.height)
    snap = snap_to_surface(scene2d, center_x, bottom_y, W, H, target_h)
    if snap.get("snapped"):
        bottom_y = snap["y"]

    # ---- 横向避让：别把她放到咖啡杯里 ----
    #
    # 上面算出来的 `center_x` 只来自规划器给的框中心。**规划器不知道那儿有什么**，
    # 它说的是"坐在桌面边缘"，而桌面正中正好摆着一杯咖啡——实测 p04 就是这样，
    # 成品里她坐在了杯口上。
    #
    # 现在用 2.5D 场景图查一遍：如果她的框压到了实体（杯子、托盘、书…），
    # 就沿横向找最近的空位。只挪 x 不挪 y——y 是"坐在什么高度"，
    # 动了它等于换动作；横向挪动几乎不改变"她在哪、在做什么"。
    avoid = avoid_collision(scene2d, center_x, bottom_y, target_w, target_h,
                            W, H)
    center_x = avoid["x"]

    # ---- 姿态文字：以**立绘本身**的姿势为准，规划器的措辞只作场景补充 ----
    #
    # 不能直接用 plan.action 当姿势：它可能被规划器改写过头。姿势的定义权
    # 归立绘（那才是模型要照着重画的东西），规划器负责的是"依托物叫什么"。
    # 词元重合够高说明它确实在改写同一段姿态，这时才把它的措辞接上去。
    pose = act.desc.replace("{anchor}", plan.region_name or "座面").rstrip("。")
    if plan.action:
        pa = plan.action.rstrip("。")
        if pa and _desc_overlap(pa, act.desc) >= 0.2:
            pose = f"{pose}；{pa}"

    # ---- 引导图里画多高：目标高度 ÷ 生成模型的放大倾向 ----
    #
    # **这段补偿实测无效**，留着是因为它是一个可复现的事实记录，
    # 不是因为它有用：p01 上把引导图里的小人从 21.3% 缩到 14.2%、10.6%，
    # 成品仍然是 33.9%、31.2%——模型对"小人多大"根本不看。
    # 反过来放大到 42.8%，成品跟到 64.1%，说明它只在"够大"时才按比例跟。
    #
    # 也就是说尺寸这件事**控制不住**，只能测（见 `s3/measure.py`）
    # 并提前告警（`GUIDED_SIZE_FLOOR`）。完整实验见 docs/11 §二十四。
    bias = float(size_bias if size_bias is not None
                 else os.environ.get("GUIDE_SIZE_BIAS", "1.0"))
    bias = max(0.5, min(2.5, bias))
    guide_h = target_h / bias

    nw = max(1, int(sprite.width * guide_h / sprite.height))
    nh = max(1, int(guide_h))
    px = int(center_x - nw / 2)
    py = int(bottom_y - nh)
    px = max(2, min(W - nw - 2, px))
    if nh <= H - 4:
        py = max(2, min(H - nh - 2, py))

    guide = _make_guide(photo, sprite, px, py, nw, nh, guide_style)
    # 剪影必须存 PNG：JPEG 会在品红边缘留下色度抽样伪影，
    # 那些淡淡的粉边正是"品红要完全消失"这条要求的反面。
    if guide_style == "silhouette":
        guide_path = os.path.join(run_dir, "guide.png")
        guide.save(guide_path, format="PNG")
    else:
        guide_path = os.path.join(run_dir, "guide.jpg")
        guide.save(guide_path, quality=94)

    # 场景标注图：把 2.5D 场景叠回照片，和引导图并排存下来。
    # 出问题时第一眼看这张——如果避让后的位置在这张图上就压着杯子，
    # 那说明解析错了（框歪了/杯子漏了），跟生成模型无关。
    try:
        scene_img = render_scene_overlay(scene2d, photo)
        scene_img.save(os.path.join(run_dir, "scene2d.jpg"), quality=92)
    except Exception as e:              # 诊断图失败不该毁掉成品
        scene_img = None
        avoid["render_error"] = f"{type(e).__name__}: {e}"

    tick("生成（重绘与融合）", 45)
    from api.ark import download, generate, image_path_to_data_uri
    size_anchor = _size_anchor(prof, plan, char.height_cm)
    tpl = (GUIDED_SILHOUETTE_PROMPT if guide_style == "silhouette"
           else GUIDED_PROMPT)
    # 表情这一段按**动作有没有脸**分成两种写法。
    #
    # 背视角（`face_visible=False`，如扶着栏杆看远处）里压根看不到五官，
    # 把"眼睛眉毛嘴角都要改到位"塞进提示词，模型只有两种反应：
    # 要么在看不见的地方瞎编，要么**干脆把她转过来面对镜头**——
    # 后者直接毁掉"她在看远方"这件事。
    #
    # 所以背视角要反过来写：明确说"看不到脸、不许转过来"，
    # 把情绪表达挪到头部角度和肩膀姿态上。
    if not getattr(act, "face_visible", True):
        mood_clause = ("**她背对镜头，看不到脸**。不要画她的正脸，"
                       "也不要为了表现情绪而把她的身体或头转到面向镜头。"
                       "情绪只通过头抬起或低下的角度、肩膀的松紧、"
                       "以及身体前倾的程度来表达")
        # 背视角不需要表情，清掉以免在 meta / 前端里显示一个不存在的表情
        plan.mood_id = ""
        plan.mood_face = ""
        plan.mood_reason = "背视角——看不到脸，表情不适用"
    else:
        # 表情没选出来时给一句中性兜底，不能留空
        _m = mood_desc(plan.mood_id) or \
            "她脸上是自然放松、带一点笑意的神情：眼睛微微弯着、嘴角轻轻上扬。"
        mood_clause = (f"{_m}。图2图3里的她是没有表情的基准形象，"
                       f"**这张图里她的脸必须按上面这段来画**，"
                       f"不要沿用设定图上的表情；眼睛、眉毛、嘴角、脸颊都要改到位")
    prompt = tpl.format(identity=char.identity,
                        height_cm=char.height_cm,
                        size_anchor=size_anchor,
                        mood_clause=mood_clause,
                        pose=pose)
    st, resp = generate(prompt,
                        images=[image_path_to_data_uri(guide_path)] + _encode_refs(char),
                        size=_pick_size(photo), model=model_key, seed=4002)
    if st != 200:
        return {"error": f"生成失败 HTTP {st}: {str(resp)[:200]}", "plan": plan.as_dict()}
    raw_path = os.path.join(run_dir, "raw_0.png")
    download((resp.get("data") or [{}])[0]["url"], raw_path)

    tick("导出", 92)
    raw = Image.open(raw_path).convert("RGB")

    # ---- 量一下她实际被画成多大 ----
    #
    # 这不是可选项。尺寸是产品硬规定（60cm），而这个模型不服从任何一条
    # 尺寸控制指令（三条都试过，见 `s3/measure.py` 的模块注释）。
    # 既然控制不住，至少要**测出来**，让调用方知道这次准不准，
    # 而不是让用户自己拿眼睛猜。
    # 优先用**低阈值**（40）那一个。
    #
    # 一开始用的是 tight（80），理由是"阴影 diff 小，高阈值能把它排除掉"。
    # 实测反了：她的浅色部分（白围裙、白袜、浅蓝发梢）diff 也只有 40~60，
    # 高阈值框不到它们，于是 bbox 只剩深色裙子那一块——p05 报出 24.2%，
    # 而同图的 loose 是 35.2%，差 1.45 倍。**测量工具本身在系统性低估。**
    #
    # 低阈值偏高估（把脚下的接触阴影算进去，约 +5~10%），
    # 但那个方向比"只框到局部"安全得多：判断"有没有偏大"时宁可高估。
    from s3.measure import measure_char_size
    try:
        meas = measure_char_size(photo, raw, [px, py, nw, nh])
    except Exception as e:                      # 测量失败不该毁掉成品
        meas = {"error": f"{type(e).__name__}: {e}"}
    got = (meas.get("loose") or meas.get("tight") or {})
    got_pct = got.get("h_pct")
    size_err = round(got_pct / (target_h / H * 100), 2) if got_pct else None

    # 这次能不能做成，提前说清楚。
    target_pct = target_h / H
    size_limited = target_pct < GUIDED_SIZE_FLOOR
    warnings: list[str] = []
    if size_limited:
        warnings.append(
            f"她要占画面高 {target_pct * 100:.0f}%，低于生成模型能画的最小值"
            f"（约 {GUIDED_SIZE_FLOOR * 100:.0f}%），所以成品里她会偏大约 "
            f"{GUIDED_SIZE_FLOOR / target_pct:.1f} 倍。"
            f"要精确的 60cm，请改用立绘路线（尺寸精确，但没有遮挡互动）；"
            f"或者让人物在画面里再大一些（拍得更近）。")
    if avoid.get("conflicts"):
        names = "、".join(c.get("name") or c.get("kind") for c in avoid["conflicts"])
        warnings.append(
            f"落点压到了「{names}」，而且整条横向都找不到空位，"
            f"只能原位输出——这张大概率会穿模。换一张更空的照片，"
            f"或者让她换一个依托物。")
    elif avoid.get("moved_px", 0) >= 8:
        warnings.append(
            f"原落点会撞到东西，已横向挪动 {avoid['moved_px']:.0f} 像素"
            f"（约画面宽的 {avoid['moved_px'] / W * 100:.0f}%）。")
    if scene_check_note:
        warnings.append(scene_check_note)

    final = add_ai_label(raw)
    out_path = os.path.join(run_dir, "out_0.jpg")
    final.save(out_path, quality=93)

    return {
        "run_id": run_id, "mode": "guided", "model": model_key,
        "scene": "scene.jpg", "guide": os.path.basename(guide_path),
        "guide_style": guide_style,
        "size": f"{raw.width}x{raw.height}",
        "character": char.id, "height_cm": char.height_cm,
        "plan": plan.as_dict(),
        "mood": {"id": plan.mood_id,
                 "source": plan.raw.get("_mood_source", ""),
                 "face": plan.mood_face,
                 "reason": plan.mood_reason,
                 "options": len(mood_list)},
        "placement": {"box": [px, py, nw, nh], "target_h": int(target_h),
                      "h_pct": round(target_h / H * 100, 1),
                      "sprite_action": act_id, "action_source": act_src,
                      "size_basis": size_basis,
                      "person_h": int(person_h) if person_h else 0,
                      "pose": pose, "size_anchor": size_anchor,
                      "size_bias": round(bias, 3), "guide_h": int(guide_h),
                      # 实测：成品里她占画面高多少、相对目标偏了多少倍。
                      # `size_err` 是唯一能回答"60cm 到底做到了没有"的数字。
                      "measured_pct": got_pct, "size_err": size_err,
                      "measured": meas,
                      "size_limited": size_limited,
                      # 避让诊断：`moved_px` 说明她被挪了多远，
                      # `sits_on` 说明她最终坐在哪件东西上，
                      # `conflicts` 非空说明**整条横扫都撞**、挪不开，
                      # `forced` 说明没找到空位、退到了"压得最少"的位置。
                      "collision": avoid,
                      "snap": snap,
                      "source": place_src},
        "scene2d": scene2d.as_dict(),
        "profile": {"framing": prof.framing, "light": prof.light,
                    "has_person": prof.has_person, "has_prop": prof.has_prop,
                    "n_objects": len(prof.objects),
                    "n_props": len(prof.props),
                    "retried": bool(prof.raw.get("_retried"))},
        "warnings": warnings,
        "candidates": [{"index": 0, "file": "out_0.jpg", "raw": "raw_0.png"}],
        "errors": [], "elapsed_hint": "规划 + 生成：约 20~35 秒",
    }


def run_planned(photo_bytes: bytes, character: str | None = None,
                model: str | None = None, progress=None) -> dict:
    """规划优先的生成路线：先由视觉模型出分镜，再让生图模型把她画进去。

    ## 和立绘路线的根本差别

    立绘路线是"从 9 个预制动作里挑一个、按固定规则贴上去"。两个硬伤都来自这个结构：
    动作是枚举的（半身照直接没有候选），而且她永远是贴在最前面的（手不会真的搭在人身上，
    也不会被人挡住）。

    这条路线让生图模型把她**画进画面**：动作针对这张照片现编，遮挡与接触由同一个
    扩散过程完成。代价是 0.12~0.30 元 / 20~40 秒，而且尺度需要靠提示词里的
    位置框去压——这是它历史上翻车的地方（实测过 41%）。

    ## 为什么做完不把她在抠出来重新贴一遍

    旧的场景路线生成完会"把她摘出来、按目标框重新缩放、贴回干净底图"。
    那样尺度确实准了，但**把生成出来的互动全丢了**：重新贴回去的她一定在最前面，
    人物挡不住她，她的手也压不到人身上——等于绕一圈又回到立绘路线的短板上。

    所以这里只**测量**不**改写**：把实测尺度和计划尺度一起返回，让调用方知道
    这次生成偏了多少。要不要重生成，是产品层的决定。
    """
    from s3.characters import get_character
    from s3.plan import plan_scene
    from s3.s2_prompt import build_companion_prompt

    def tick(stage: str, pct: int):
        if progress:
            progress(stage, pct)

    char = get_character(character)
    model_key = model or os.environ.get("ARK_IMAGE_MODEL", "doubao-seedream-5-0-flash-260915")
    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    tick("读取照片", 6)
    photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    scene_path = os.path.join(run_dir, "scene.jpg")
    photo.save(scene_path, quality=95)

    tick("理解照片", 16)
    prof = vlm_profile(scene_path)

    # 把画像里的信息当"量纲"喂给规划器：它只有知道画面里有没有人、人多大，
    # 才可能把"她该占画面多少"估对。
    hints = []
    if prof.has_person:
        ph = prof.body_len_px(photo.width, photo.height) or 0
        hints.append(f"画面里有{prof.framing or '一个'}景别人物，"
                     f"人物身高约占画面高度的 {ph / photo.height * 100:.0f}%。"
                     f"她应该约是人物身高的 26%，也就是画面高度的 "
                     f"{ph / photo.height * 100 * 0.26:.0f}% 左右。")
    else:
        hints.append("画面里没有人物。")
    if prof.props:
        k = [p for p in prof.props if p.usable]
        if k:
            hints.append("画面里可用的座面：" + "、".join(p.name_cn for p in k[:2]) + "。")

    tick("出分镜", 26)
    plan = plan_scene(scene_path, height_hint=" ".join(hints))
    if not plan.ok:
        return {"error": f"规划失败：{plan.reason or plan.raw.get('error') or '模型没有给出可执行的分镜'}",
                "plan": plan.as_dict()}

    # ---- 尺度：位置听规划器的，大小听程序的 ----
    #
    # 实测规划器的尺度估计系统性不准：p01 给 7.5%（该 16%）、p02 11%（该 16%）、
    # p03 9%（该 14%）、p05 35.5%（该 24%）。它擅长的是"她在什么位置、和什么发生关系"，
    # 不擅长把"40 厘米"换算成画面百分比——这需要知道画幅、人物身高、透视，
    # 而这些都是我们手上有精确值的量。
    #
    # 所以：保留规划器框的**横向中心**和**底边**（那是接触线，语义信息），
    # 只把高度换成算出来的目标值。这样"在哪"用语义判断，"多大"用测量。
    _bl = prof.body_len_px(photo.width, photo.height)
    if prof.has_person and _bl:
        want_pct = _bl / photo.height * 100 * 0.26
    else:
        want_pct = 20.0
    want_pct = max(12.0, min(30.0, want_pct))
    plan_raw_region = list(plan.region)
    plan_h_pct = plan.height_pct
    if plan.region and plan.height_pct > 0:
        x1, y1, x2, y2 = plan.region
        cx = (x1 + x2) / 2.0
        h = want_pct / 100.0 * 999.0
        w = (x2 - x1) * (h / (y2 - y1))
        nx1 = max(0, int(round(cx - w / 2)))
        nx2 = min(999, int(round(cx + w / 2)))
        ny1 = max(0, int(round(y2 - h)))
        plan.region = [nx1, ny1, nx2, y2]
        plan.height_pct = (y2 - ny1) / 999.0 * 100

    # 尺度锚点：用"她只到某某位置"这种**物体相对参照**，不用画面百分比
    # （百分比写法已被 A/B 证伪，见 s2_prompt._DEPRECATED_PERCENT_SCALE）。
    # 身高传角色的真实值，别让锚点里的厘米数和提示词里的对不上。
    anchor = _size_anchor(prof, plan, char.height_cm)

    light_desc = {"left": "光从画面左侧来", "right": "光从画面右侧来",
                  "top": "正上方顶光", "flat": "柔和的散射光"}.get(prof.light, "环境已有的光线")
    prompt = build_companion_prompt(
        action=plan.action,
        person_desc="画面中的人物",
        light_desc=light_desc,
        region_bbox=plan.region,
        region_name=plan.region_name or "画面里的依托物",
        empty_scene=not prof.has_person,
        anchor=anchor,
    )

    tick("生成", 42)
    from api.ark import download, generate, image_path_to_data_uri
    refs = _encode_refs(char)
    st, resp = generate(prompt, images=[image_path_to_data_uri(scene_path)] + refs,
                        size=_pick_size(photo), model=model_key, seed=2000)
    if st != 200:
        return {"error": f"生成失败 HTTP {st}: {str(resp)[:200]}", "plan": plan.as_dict()}
    raw_path = os.path.join(run_dir, "raw_0.png")
    download((resp.get("data") or [{}])[0]["url"], raw_path)

    tick("测量", 88)
    raw = Image.open(raw_path).convert("RGB")
    scene_at_raw = photo.resize(raw.size, Image.LANCZOS)
    measured = {}
    try:
        cm = character_mask(scene_at_raw, raw)
        bb = _bbox(cm)
        if bb:
            h = (bb[3] - bb[1]) / raw.height * 100
            measured = {"bbox": list(bb), "h_pct": round(h, 1),
                        "planned_h_pct": round(plan.height_pct, 1),
                        "h_ratio": round(h / plan.height_pct, 2) if plan.height_pct else None}
    except Exception as e:
        measured = {"error": str(e)[:80]}

    final = add_ai_label(raw)
    out_path = os.path.join(run_dir, "out_0.jpg")
    final.save(out_path, quality=93)

    return {
        "run_id": run_id, "mode": "planned", "model": model_key,
        "scene": "scene.jpg", "size": f"{raw.width}x{raw.height}",
        "character": char.id,
        "plan": {**plan.as_dict(), "region_raw": plan_raw_region,
                 "height_pct_raw": round(plan_h_pct, 1)},
        "measured": measured,
        "candidates": [{"index": 0, "file": "out_0.jpg", "raw": "raw_0.png"}],
        "errors": [], "elapsed_hint": "规划 + 生成：约 25~45 秒",
    }


def _props_or_retry(prof, scene_path: str) -> list:
    """取座面框；合并版给不出可用结果时，单独补问一次。

    ## 为什么要这条兜底

    把两组问题塞进一次调用，实测 prompt 一变就可能在某个字段上退化。
    最典型的一次：餐桌那张，模型把"桌面朝上的那一大块"整个框进来
    （`seat = [0, 0, 999, 485]`），前沿线 y 从 395 掉到 0，
    于是"这张照片用不了"——而它本来是好用的。

    提示词已经针对这一点改过（在 PROFILE_SYS 里给了反例），但 VLM 的行为
    不会有任何保证。所以再加一层：**只在合并结果确实不可用时**才补问，
    正常路径仍然只有一次调用。最坏情况也只是退回老的两段式，不会更差。
    """
    props = prof.props
    if prof.has_person:
        return props
    if any(p.usable for p in props):
        return props
    try:
        extra = vlm_props(scene_path)
    except Exception:
        return props
    return extra if any(p.usable for p in extra) else props


def run_sprite(photo_bytes: bytes, action: str, light_key: str = "auto",
               model: str | None = None, progress=None,
               force_route: str | None = None,
               character: str | None = None) -> dict:
    """立绘路线主流程：无生成、纯 CPU 合成。

    返回结构与 `run()` 对齐，前端不用区分两条路。
    `character` 是角色 id（见 `s3/characters.py`），不传走 Q 版。
    """
    def tick(stage: str, pct: int):
        if progress:
            progress(stage, pct)

    a = get_action(action)
    if a is None:
        return {"error": f"未知动作: {action}"}
    # 只会做立绘路线。收到场景路线的动作（shoulder / peek / hem_peek）必须挡住：
    # 那几个动作靠"和场景一起生成 + 肩线锚点定位"成立，拿通用落点硬套出来的结果
    # 是她站在地上，跟"扒着肩膀探头"毫无关系——而且会**静默**花 0.12 元
    # 去生成一张这个动作的立绘（实测踩过，peek__ 和 shoulder__ 两张）。
    if a.render != "sprite":
        return {"error": f"动作「{a.id}」走场景内路线，不能走立绘合成；"
                         "请改用 pipeline.run() 或选择立绘路线的动作。"}
    char = get_character(character)

    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    tick("读取照片", 8)
    photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    scene_path = os.path.join(run_dir, "scene.jpg")
    photo.save(scene_path, quality=95)

    tick("分析照片", 22)
    # 座面框已经并进同一次画像，正常路径只有一次调用。
    # 物件路线原来是 profile + props 两次（实测 5.2~7.4 秒），第二次就为了问一句"哪个能坐"。
    prof = vlm_profile(scene_path)
    props = _props_or_retry(prof, scene_path)
    usable = [p for p in props if p.usable]
    route = "person" if prof.has_person else ("object" if usable else "none")
    if force_route in ("person", "object"):
        route = force_route
    if route == "none":
        return {"error": "这张照片暂时用不了：画面里没有人物，也没找到可用的座面。",
                "route": "none"}

    tick("取立绘", 45)
    model_key = model or os.environ.get("ARK_IMAGE_MODEL", "doubao-seedream-5-0-flash-260915")
    try:
        sprite, how = _ensure_sprite(a, model_key, progress, char)
    except Exception as e:
        return {"error": f"立绘不可用: {e}"}

    # ---- 前景（人物）轮廓：只有"被挡住"的动作才需要 ----
    #
    # 0.17s 的中等成本，且会软失败（模型/onnxruntime 缺失时返回 None，
    # 退回"不遮挡"的老行为）。所以只在需要时算，不让常规路径白付这笔钱。
    fg = None
    if getattr(a, "behind", False):
        tick("分析遮挡", 55)
        from s3.seg import person_alpha
        fg = person_alpha(photo, keep_box=prof.person)

    # ---- 落点 ----
    W, H = photo.size
    try:
        target_h, center_x, bottom_y, meta = _sprite_placement(a, prof, props, route, W, H,
                                                          char, sprite=sprite, fg=fg)
    except ValueError as e:
        return {"error": str(e), "route": route}
    if target_h < 8:
        return {"error": "算出的目标尺寸异常，已放弃。"}
    nw = max(1, int(sprite.width * target_h / sprite.height))
    nh = max(1, int(target_h))
    px = int(center_x - nw / 2)
    py = int(bottom_y - nh)
    px = max(2, min(W - nw - 2, px))
    # 竖向上不硬夹：`behind` 的半身照构图里她的下半身本来就该出画
    #（按头对齐后 target_h 可能比画面还高）。PIL 的 paste 会自己裁掉越界部分。
    if nh <= H - 4:
        py = max(2, min(H - nh - 2, py))

    tick("合成", 65)
    step = photo.copy()
    s = sprite.resize((nw, nh), Image.LANCZOS)
    step.paste(s, (px, py), s)

    # 她的精确轮廓：立绘自带 alpha，贴上去的位置又是我们算的，
    # 所以这里能拿到**逐像素准确**的掩码——不需要任何启发式。
    # 照片路线只能靠 character_mask（蓝色主导 ∩ 差分），她的白肚皮和白鞋底
    # 会被漏掉；接触到阴影那一步就会把那些像素当背景压暗。
    # 立绘路线没有这个短板，掩码越准，阴影才越敢贴着轮廓画。
    alpha_px = np.asarray(s, dtype=np.uint8)[..., 3] > 96
    cm_exact = np.zeros((H, W), dtype=bool)
    # 同一份 alpha 的浮点版：重打光要按它羽化，硬掩码会把轮廓砍出亮度台阶
    alpha_f = np.zeros((H, W), dtype=np.float32)
    ax0, ay0 = max(0, px), max(0, py)
    ax1, ay1 = min(W, px + nw), min(H, py + nh)
    if ax1 > ax0 and ay1 > ay0:
        cm_exact[ay0:ay1, ax0:ax1] = alpha_px[ay0 - py:ay1 - py, ax0 - px:ax1 - px]
        alpha_f[ay0:ay1, ax0:ax1] = (
            np.asarray(s, dtype=np.float32)[ay0 - py:ay1 - py, ax0 - px:ax1 - px, 3] / 255.0)

    # ---- 主光：先定下来，重打光和阴影都要用同一份 ----
    light = LIGHT_PRESETS.get(light_key, LIGHT_PRESETS["auto"])
    if light_key == "auto" and prof.light:
        dirn = prof.light
        if dirn == "left":
            light = LIGHT_PRESETS["sun_left"] if a.support == "ground" else LIGHT_PRESETS["left"]
        elif dirn == "right":
            light = LIGHT_PRESETS["sun_right"] if a.support == "ground" else LIGHT_PRESETS["right"]

    # ---- 环境色匹配（整体色偏，不改空间分布）----
    step, amb = match_ambient(photo, step, mask=cm_exact)

    # ---- 重打光（沿主光方向补一层明暗，解决"她是平的"）----
    #
    # 顺序不能反：先改分布再挪整体色，否则梯度会被后面的直流平移削掉一部分；
    # 也不能放到阴影之后——阴影是"她挡住了光"，得在她被照亮之后才说得通。
    #
    # 方向：**画面自己的横向亮度差**，不是 VLM 的 left/right。
    #   实测 s04_full 那张，VLM 说"光从左边来"，但树影和她的投影都朝左倒、
    #   牛仔裤和球鞋右侧更亮 —— 光是从右边来的。方向判反了。
    #   同一张图的横向亮度差是 +4.9%（向右变亮），和实拍一致。
    #   VLM 仍然有用：它给的 `hardness` 决定梯度**幅度**，那是个不会反号的量。
    #
    # 只处理有侧光的情况。顶光/散射光不处理是有意的：
    #   立绘自带"正面略偏上"的烘焙明暗，顶光这一维本来就大致对，
    #   再叠一层就是重复打光；没有侧光时硬造一道斜坡只会更像滤镜。
    #
    # `sl` 同时给下面的投射阴影用——**影子和明暗必须是同一个光源**。
    # 否则会出现"她朝向左边变亮、影子却朝右躺"这种一眼假的组合，
    # 而这正是修好重打光之后的实际状态（实测影子质心偏 +62.9px，方向反了）。
    relight_info = {"applied": False, "reason": "no_side_light"}
    sl = side_light(photo)
    if sl is not None:
        tint = scene_tint(photo, alpha_f)
        step, relight_info = relight(step, alpha_f, sl["azimuth"], tint=tint,
                                     amp=amp_for(light["hardness"], sl["contrast"]))
        relight_info["source"] = "scene_slope"
        relight_info["scene_contrast"] = sl["contrast"]

    # ---- 接触阴影 ----

    bb = (px, py, px + nw, py + nh)
    bb_h = max(bb[3] - bb[1], 1)
    lift = int(meta.get("lift_px", 0) or 0)
    if a.support == "ground":
        contact_y = float(meta.get("foot_y", bb[3]))
    else:
        contact_y = bb[3] - (1.0 - a.contact_ratio) * bb_h

    # 投影方向和重打光共用同一个光向。画面量得出侧光就用画面的，
    # 量不出（顶光/散射）才退回预设值——那时 confidence=0，渲染器只出 AO，
    # 方向本来就用不上，所以这个退回是安全的。
    shadow_az = sl["azimuth"] if sl is not None else light["azimuth"]
    ls = LightingState(
        light_azimuth_deg=(shadow_az if shadow_az is not None else 180.0),
        quality_hardness=light["hardness"], intensity=light["intensity"],
        confidence=(light["confidence"] if light["azimuth"] is not None else 0.0))
    shadow_box = (bb[0], bb[1], bb[2], max(bb[3], contact_y))
    dark = render_shadow_matte(step.size, shadow_box, ls, contact_y=contact_y,
                              reach=light.get("reach", 1.0))
    # 阴影落在**她下面**：她遮挡自己的影子，影子只压背景那一半。
    # 别写成 `step * (1 - dark)`——那是把她也一起压暗了，
    # 桌沿那张实测 48.7% 的身体被 AO 染黑。
    step = shadow_under(step, photo, alpha_f, dark)
    shadow_kind = "sprite_" + ("ao" if light["azimuth"] is None else "full")
    meta["shadow_azimuth"] = round(float(shadow_az), 1) if shadow_az is not None else None
    # 注意：VLM 判顶光/散射时 confidence=0，渲染器只出 AO，此时这个方向**没被用到**。
    # 单独标出来，免得诊断信息看起来像"用了方向"。
    if light["azimuth"] is None:
        meta["shadow_dir_source"] = "unused_ao_only"
    else:
        meta["shadow_dir_source"] = "scene_slope" if sl is not None else "vlm_label"

    if lift <= 0:
        try:
            core = contact_core(step.size, cm_exact, contact_y, float(bb_h))
            if core is not None:
                arr = np.asarray(step, dtype=np.float32) * (1 - core[..., None])
                step = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
                shadow_kind += "+core"
        except Exception as e:
            meta["shadow_core_error"] = str(e)[:80]

    # ---- 让前景把她挡住（必须放在最后）----
    #
    # 放最后的原因：她的接触阴影不该压到前面的人身上，重打光只作用在她自己
    # 的 alpha 内（被挡住的部分白算，可忽略）。这一步把人物像素原样还原，
    # 于是她"被挡住"这件事在数学上等价于"把她贴在人物图层下面"。
    if fg is not None:
        step = occlude(step, photo, fg)
        meta["occluded"] = True

    tick("导出", 92)
    final = add_ai_label(step)
    out_path = os.path.join(run_dir, "out_0.jpg")
    final.save(out_path, quality=93)

    return {
        "run_id": run_id, "mode": "companion", "model": model_key,
        "scene": "scene.jpg", "size": f"{W}x{H}",
        "route": route, "render": "sprite", "sprite_source": how,
        "profile": {"framing": prof.framing, "selfie": prof.selfie, "light": prof.light,
                    "has_prop": prof.has_prop, "person": prof.person, "legs": prof.legs},
        "candidates": [{
            "index": 0, "file": "out_0.jpg", "support": a.support,
            "shadow": shadow_kind, "ambient": amb, "relight": relight_info,
            "placement": {"target_h": int(target_h), "box": [px, py, nw, nh],
                          "contact_y": int(contact_y), **meta},
            "scores": _qc_scores(photo, final, (px, py, px + nw, py + nh),
                                 mask=cm_exact),
        }],
        "errors": [], "elapsed_hint": "立绘路线：无生成，亚秒级",
    }

def run(photo_bytes: bytes, box_norm: tuple[float, float, float, float],
        surface: str, light_key: str = "auto", candidates: int = 2,
        anchor_override: str | None = None, progress=None,
        generate_fn=None, mode: str = "solo",
        action: str | None = None, model: str | None = None,
        simplify_hands: bool = False, force_route: str | None = None,
        character: str | None = None, route: str = "guided",
        mood: str | None = None) -> dict:
    """主流程。返回 job 结果字典（含每张产物路径与质检分）。

    ## 一键出图：`route` 默认 `auto`，不需要任何参数

    用户只要给一张照片，剩下的全部由 AI 决定——位置、动作、表情、多大、
    甚至"这张照片里根本没有她能待的地方"这种判断。
    这四个参数**都有合理的默认值**，调用方一个都不用传：

    | 参数 | 不传时的行为 |
    |---|---|
    | `route` | `auto` → 走引导生成（唯一一个"什么都不用选"就能出好图的路） |
    | `mood` | 空 → 按场景的光线/空间/有没有人自动挑 |
    | `action` | 空 → 让规划器从已缓存立绘里选 |
    | `model` | Flash（0.12 元） |

    ## 四条路线（只有 `sprite` 需要额外参数）

    | route | 走哪条 | 需要传什么 | 特点 |
    |---|---|---|---|
    | **`auto` / `guided`（默认）** | `run_guided` | **什么都不用** | 位置/动作/表情全自动；有遮挡、会避障；**但 60cm 做不到**，0.12 元 / 16~22 秒 |
    | `sprite` | `run_sprite` | **`action` 必填** | 程序化合成，**尺寸 100% 精确**、0 元 / 3~5 秒；但无遮挡、表情只有一个 |
    | `planned` | `run_planned` | 无需 | 纯规划直出（不贴引导图），尺寸最不可控，仅作对照 |
    | `solo` | 老的"画进场景"路径 | 需要 `box` | 早期形态，现在只在明确要"不贴贴"时用 |

    `force_route`：强制走 "person" 或 "object" 路线，绕过自动判定。
    只对 `guided`/`planned` 有意义（`sprite` 有自己的一套路由）。
    """
    def tick(stage: str, pct: int):
        if progress:
            progress(stage, pct)

    # ---- 路线分派 ----
    #
    # `auto` 和 `guided` 等价。默认就是 auto，所以**前端可以一个参数都不传**。
    #
    # 这一版把 `mode` 从分派条件里拿掉了：原来只有 `mode=="companion"` 才分派，
    # 于是"一键出图"时如果前端忘了传 mode，就会静默掉进老的 solo 路径——
    # 那是完全不同的东西（把角色画进场景 vs 贴贴），而且**不会报错**。
    # 现在只要没显式要 solo，就一律走引导生成。
    r = (route or "auto").strip().lower()
    if r in ("auto", "guided"):
        return run_guided(photo_bytes, character=character, model=model,
                          progress=progress, mood=mood, action=action)
    if r == "planned":
        # run_planned 是早期"纯规划直出不贴引导图"的对照实现，
        # 没有表情/避让/测量，签名也不接 mood。留着是为了能复现
        # "为什么它不行"（尺寸 1.7~2.8 倍）这段结论。
        return run_planned(photo_bytes, character=character, model=model,
                           progress=progress)
    if r == "sprite":
        if not action:
            return {"error": "sprite 路线必须指定 action（立绘是按动作缓存的）。"
                             "想要一键出图请用 route=auto（默认）。",
                    "route": "sprite"}
        a0 = get_action(action)
        if a0 is None:
            return {"error": f"没有这个动作：{action}", "route": "sprite"}
        if a0.render == "scene":
            return {"error": f"动作「{a0.name}」需要和场景一起生成，"
                             f"不能走 sprite 路线；请传 route=\"guided\"。",
                    "route": "sprite"}
        return run_sprite(photo_bytes, action, light_key=light_key,
                          model=model, progress=progress,
                          force_route=force_route,
                          character=character)
    if r != "solo":
        return {"error": f"未知的 route：{route}"
                         f"（可选 auto / guided / sprite / planned / solo）"}

    # ---- 以下都是 solo 模式的老路径（route="solo"） ----
    # 动作库里的 `render` 字段决定走哪条：
    #   sprite —— 她是独立人形，没有东西挡她 → 立绘 + 程序化合成，**不生成**
    #   scene  —— 她必须被场景里的东西挡住一部分 → 只能和场景一起生成再抠
    # 分派必须在建 run_dir 之前做，因为 run_sprite 自己会建。
    if action:
        a0 = get_action(action)
        if a0 is not None and a0.render == "sprite":
            return run_sprite(photo_bytes, action, light_key=light_key,
                              model=model, progress=progress,
                              force_route=force_route,
                              character=character)
        # 场景内路线是**模型把角色画进照片**，角色形象由提示词决定，
        # 换角色等于换提示词——现在只有 Q 版那套。不拦的话会静默拿 Q 版的
        # 提示词去画新角色，出来一个四不像。
        _cid = (character or "").strip()
        if a0 is not None and _cid and _cid != "chibi":
            return {"error": f"动作「{a0.id}」走的是场景内路线，暂不支持换角色；"
                             "请选立绘路线的动作（蹲 / 坐 / 跳 / 站腿边 / 坐边缘）。"}

    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    tick("读取照片", 5)
    photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    scene_path = os.path.join(run_dir, "scene.jpg")
    photo.save(scene_path, quality=95)
    size = _pick_size(photo)
    w_px, h_px = [int(v) for v in size.split("x")]
    # 原图坐标下的框（按输出尺寸缩放）
    x, y, bw, bh = box_norm
    box_px = (x * w_px, y * h_px, (x + bw) * w_px, (y + bh) * h_px)

    tick("构建提示词", 12)
    # 一次画像搞定生成前后都要用的信息（景别 / 人物框 / 腿框 / 锚点 / 光向）。
    # 改造前这里要调两次 VLM：先 vlm_locate 拿腿框写提示词，后处理里
    # vlm_locate_pair 再问一遍拿尺子和光向。同一张图同样的事问两遍，
    # 白花一次调用，还可能两次结果不一致。
    #
    # 锚点部位按动作类型传：地面动作要腿框、上半身动作要肩/衣摆框，
    # 只问当前动作需要的那一个。
    act_for_anchor = get_action(action) if (mode == "companion" and action) else None
    anchor_part = ""
    if act_for_anchor and act_for_anchor.support == "person":
        anchor_part = act_for_anchor.anchor_region
    prof = vlm_profile(scene_path, anchor_part=anchor_part) if mode == "companion" else None

    props: list = []
    route = "solo"
    if mode == "companion":
        # 路由：人物优先，其次物件，都没有就**不生成**。
        # 提前退出能省一次图像费用（15~70 秒 + 0.12~0.30 元）和一次失望。
        # 没人物的照片**一律**去查一次物件，不再先看 has_prop。
        # 原因：has_prop 这个布尔会误判为 false——实测同一张长椅照、
        # 同一张餐桌特写，都有过"明明有座面却答没有"。而 has_prop=False
        # 会让整张照片直接被判"不可用"，代价太大。
        # 多花一次约 1.5 秒的轻量调用，换掉一个"整张照片报废"的失败模式，划算。
        want_obj = (force_route == "object") or (not prof.has_person)
        props = prof.props if want_obj else []
        usable = [p for p in props if p.usable]
        if force_route in ("person", "object"):
            route = force_route
            if route == "object" and not usable:
                return {"error": "没有可用的座面，无法走物件路线。",
                        "route": "none", "profile": {"framing": prof.framing,
                                                     "has_prop": prof.has_prop}}
        else:
            route = "person" if prof.has_person else ("object" if usable else "none")
        if route == "none":
            why = ("画面里没有人物，"
                   + (f"可交互物件也不可用（{props[0].why_not}）" if props
                      else "也没找到可坐的物件或可依靠的物体"))
            return {"error": f"这张照片暂时用不了：{why}。"
                             "换一张能看清人、或者有沙发/长椅/桌子的照片会好很多。",
                    "route": "none", "profile": {"framing": prof.framing,
                                                 "has_prop": prof.has_prop}}

        # 动作与路线不匹配时自动换一个，而不是报错——用户选的是"贴贴"，
        # 不是"必须用这个动作"。换过之后在结果里注明。
        action_switched = None
        a_probe = get_action(action) if action else None
        want_kind = "person_part" if route == "person" else "seat_plane"
        if not a_probe or a_probe.anchor_kind != want_kind:
            pool = [x for x in enabled_actions() if x.anchor_kind == want_kind]
            pool = filter_by_framing(pool, prof.framing, prof.selfie) or pool
            if pool:
                action_switched = {"from": action, "to": pool[0].id, "route": route}
                action = pool[0].id
                act_for_anchor = get_action(action)

    if mode == "companion":
        a = get_action(action) if action else None
        act_desc = a.desc if a else (action or "")
        region, region_name = None, "人物"
        if route == "object" and props:
            # 物件路线：把 {anchor} 换成实际名词（沙发/长椅/桌子…），
            # 同一段动作描述能覆盖所有水平坐面，不用为每种家具写一遍。
            prop = [p for p in props if p.usable][0]
            act_desc = act_desc.replace("{anchor}", prop.name_cn)
            region_name = f"{prop.name_cn}的座面"
            # 给她一个大概的范围，别让她画到画面另一头去；
            # 精确位置由后处理按座面前沿线搬，这里只求"别差太远"。
            region = [prop.x1, max(0, prop.y - 160), prop.x2, min(999, prop.y + 90)]
            prof.anchor = [prop.x1, prop.y, prop.x2, prop.y]
            prof.anchor_part = "seat"
        elif a and a.anchor_region in ("legs", "feet") and prof.legs:
            region = prof.legs
            region_name = "人物的小腿和脚"
        prompt = build_companion_prompt(action=act_desc, simplify_hands=simplify_hands,
                                        region_bbox=region, region_name=region_name,
                                        empty_scene=(route == "object"))
    else:
        prompt = build_case_prompt(surface, box_norm, anchor_override, light_key)

    tick("生成候选", 20)
    refs = _encode_refs()
    photo_b64 = photo.resize(
        (min(photo.width, 1536), int(photo.height * min(photo.width, 1536) / photo.width)),
        Image.LANCZOS)

    def default_generate(i: int):
        from api.ark import ENV, download, generate, image_path_to_data_uri
        buf = io.BytesIO()
        photo_b64.save(buf, format="JPEG", quality=92)
        scene_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        st, resp = generate(prompt, images=[scene_uri] + refs, size=size,
                            model=model or ENV.get("ARK_IMAGE_MODEL"), seed=2000 + i * 13)
        if st != 200:
            raise RuntimeError(f"HTTP {st}: {str(resp)[:300]}")
        item = (resp.get("data") or [{}])[0]
        p = os.path.join(run_dir, f"raw_{i}.png")
        download(item["url"], p)
        return p

    gen = generate_fn or default_generate
    raw_paths: list[str | None] = [None] * candidates
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(candidates, 4)) as pool:
        futs = {pool.submit(gen, i): i for i in range(candidates)}
        done = 0
        for f in as_completed(futs):
            i = futs[f]
            done += 1
            tick(f"生成候选 {done}/{candidates}", 20 + int(45 * done / candidates))
            try:
                raw_paths[i] = f.result()
            except Exception as e:  # 单个候选失败不阻塞
                errors.append(f"候选{i}: {e}")

    tick("后处理", 70)
    light = LIGHT_PRESETS.get(light_key, LIGHT_PRESETS["auto"])
    spec = SURFACE_SPEC.get(surface, SURFACE_SPEC["ground"])
    # 接触点：坐姿在框底边之上（接触面是坐垫/椅面），站姿就是框底边
    contact_y = box_px[1] + (box_px[3] - box_px[1]) * spec["contact_ratio"]

    # 贴贴模式：把 VLM 定位提到循环外，只调一次（原来放在候选循环里，
    # 候选数 2 就会白调两次）。
    act_main = get_action(action) if (mode == "companion" and action) else None

    # auto 档没有光照方向，只能画 AO，"贴在地上"的感觉很弱——户外照尤其明显。
    # 画像里已经带了主光来向，直接把 auto 升级成方向档。
    # 用户显式选了左/右/晴天档时不覆盖（那是他的判断优先）。
    if light_key == "auto" and prof and prof.light:
        dirn = prof.light
        if dirn == "left":
            light = LIGHT_PRESETS["sun_left"] if surface == "ground" else LIGHT_PRESETS["left"]
        elif dirn == "right":
            light = LIGHT_PRESETS["sun_right"] if surface == "ground" else LIGHT_PRESETS["right"]
        elif dirn == "top":
            light = dict(azimuth=90.0, hardness=0.6, intensity=0.55,
                         confidence=0.6, reach=0.6)
        # flat 保持 auto（只出 AO，不赌方向）

    results = []
    for i, p in enumerate(raw_paths):
        if not p:
            continue
        raw = Image.open(p).convert("RGB")
        # S2.5 尺度归一：模型压不住方差，用掩码把角色摘出来按框精确缩放。
        # 底图用原照片（干净、无重编码损失），角色按框底边对齐贴回。
        scene_at_out = photo.resize(raw.size, Image.LANCZOS)
        if mode == "companion":
            # 贴贴模式的尺度不能靠提示词（"约40厘米"是软约束，实测偏大 1.5 到 2 倍），
            # 也不能靠框（贴贴本来就没有框）。正确做法：**用人物身体当尺子**。
            #
            # 尺子选「人物全身高」而不是「腿长」：40cm / 175cm ≈ 0.23 是纯人体测量学
            # 比例，不依赖任何估算；而腿长取决于 VLM 把胯部标在哪，胯部一偏尺度就漂。
            # 人的框还会被截断（实测腿框底边比人物框底边高 7% 画面高），
            # 所以落地线取人物框底边。
            step, scale_info = raw, {"applied": False, "reason": "no_ruler"}
            a0 = act_main
            if a0 and a0.support == "ground" and prof and prof.person:
                lb, pb = prof.legs, prof.person
                W_o, H_o = raw.size
                person_h = prof.body_len_px(W_o, H_o) or 0.0
                foot_y = prof.foot_y_px(H_o)
                if person_h and foot_y is not None:
                    target_h = person_h * a0.scale_ratio
                    # 水平位置：有腿框就用腿框（更准），没有就用人物框的中段近似
                    if lb:
                        leg_cx = (lb[0] + lb[2]) / 2.0 / 999.0 * W_o
                        leg_w = (lb[2] - lb[0]) / 999.0 * W_o
                    else:
                        leg_cx = (pb[0] + pb[2]) / 2.0 / 999.0 * W_o
                        leg_w = (pb[2] - pb[0]) / 999.0 * W_o * 0.5
                    target_w = target_h * 0.78          # Q版宽高比
                    # 放在腿的外侧（左），避免压在腿上；贴边夹紧在函数内部处理
                    center_x = leg_cx - (leg_w * 0.5 + target_w * 0.5 + 6)
                    # 掩码在 rescale_to_height 内部用 character_mask（蓝 ∩ 差分）。
                    # 单用蓝色主导会把蓝天算进来、单用差分会把人的鞋算进来，
                    # 两者都会让包围盒贴边，缩放因此被**静默**拒绝——这正是
                    # 前两轮"角色过大"却没报错的原因。
                    # 悬空姿态（跳）：bbox 底边要抬到地面线以上，
                    # 抬多少由 lift_ratio 决定（相对她自身高度）。
                    # 站着/坐着时 lift=0，等式退化成"底边贴地面线"。
                    lift = a0.lift_ratio * target_h
                    step, scale_info = rescale_to_height(
                        scene_at_out, raw, target_h=target_h,
                        center_x=center_x, bottom_y=foot_y - lift)
                    scale_info["ruler"] = {"person_h": int(person_h),
                                           "foot_y": int(foot_y),
                                           "lift_px": int(lift),
                                           "framing": prof.framing}
            elif a0 and a0.support == "person" and prof and prof.person:
                # 上半身动作（坐肩 / 探头 / 扒衣摆）：搬到锚点上。
                #
                # 一开始我打算只纠大小、不动位置，理由是"肩膀和衣摆是显眼的强锚点，
                # 模型自己能摆对"。真实生成一次就把这个假设打翻了：`peek` 把她画成
                # 一整个人浮在人物肩膀外侧的空气里，连"扒着肩膀"的意思都没表达出来，
                # 而且位置偏到了墙面上。光纠大小等于把她的尺寸改对、仍然贴在错的地方。
                #
                # 所以和地面动作走同一套：先拿锚点框，再把她的 bbox 底边钉在锚点边上。
                #   anchor_edge="top"    -> 底边 = 锚点上沿（坐在肩线上）
                #   anchor_edge="bottom" -> 底边 = 锚点下沿（扒在下摆上、身体压住布边）
                W_o, H_o = raw.size
                person_h = prof.person_h_px(H_o) or 0.0
                if prof.anchor and person_h:
                    ax0, ay0, ax1, ay1 = prof.anchor
                    anchor_w = (ax1 - ax0) / 999.0 * W_o
                    # 坐在肩膀的**外侧三分之一**而不是两肩中间：两肩中间是脖子，
                    # 坐在那里会盖住人物的脸和脖子，视觉上不成立。
                    center_x = ax0 / 999.0 * W_o + anchor_w * 0.22
                    edge_y = ay0 if a0.anchor_edge == "top" else ay1
                    target_h = person_h * a0.scale_ratio
                    inset = (ay1 - ay0) / 999.0 * H_o * 0.08
                    # contact_ratio 决定 bbox 底边要沉到锚点线以下多少：
                    # 站立/坐地时臀或脚就在 bbox 底边（1.0），不沉；
                    # 坐肩时臀在 62% 处，底边要再往下 38% 身高，臀才落在肩上。
                    # 少了这项，"坐肩"会变成"站在肩上"（实测鞋底正好压肩线）。
                    sink = (1.0 - a0.contact_ratio) * target_h
                    bottom_y = (edge_y / 999.0 * H_o) + inset + sink
                    step, scale_info = rescale_to_height(
                        scene_at_out, raw, target_h=target_h,
                        center_x=center_x, bottom_y=bottom_y)
                    scale_info["ruler"] = {"person_h": int(person_h),
                                           "framing": prof.framing,
                                           "mode": "anchor_placed",
                                           "anchor": prof.anchor,
                                           "sink_px": int(sink)}
                else:
                    # 锚点不可用时只纠大小（比如照片没拍到衣摆）。
                    # 位置交给模型——它画歪了也只能忍，总比搬到一个算错的位置强。
                    if person_h:
                        step, scale_info = rescale_to_height(
                            scene_at_out, raw,
                            target_h=person_h * a0.scale_ratio, place=False)
                        scale_info["ruler"] = {"person_h": int(person_h),
                                               "framing": prof.framing,
                                               "mode": "scale_only"}
            elif a0 and a0.support == "object" and route == "object" and props:
                # 物件路线：照片里没人，她是"坐在家具上"的唯一主角。
                #
                # 尺子换成**座面可见宽度**（人物身高这把尺子在这里不存在），
                # 按家具种类取系数，结果再夹进图像高度的合理区间。
                # 落点：臀底在座面前沿线上，双脚垂到线以下 —— contact_ratio=0.62。
                prop = [p for p in props if p.usable][0]
                W_o, H_o = raw.size
                seat_w = (prop.x2 - prop.x1) / 999.0 * W_o
                by_frame = SEAT_H_FRAC.get(prop.kind, 0.20) * H_o
                # 第二把尺子只在**特写**时启用：座面几乎占满画面宽度。
                # 这种构图下"占画面高度比例"必然偏小（桌面特写实测只有 193px，
                # 而桌板本身就有 90px 厚，她读起来像贴纸）；
                # 而长椅/沙发那些中远景，画面比例已经验证过是对的，不要拿
                # 不稳定的座面宽度去动它（同一张空沙发两次调用宽 425/1198）。
                span_ratio = seat_w / max(W_o, 1)
                if span_ratio > 0.6:
                    by_seat = seat_w * SEAT_W_RATIO.get(prop.kind, 0.19)
                    target_h = max(by_frame, by_seat)
                    scale_info_extra = {"span_ratio": round(span_ratio, 2),
                                        "mode": "closeup"}
                else:
                    target_h = by_frame
                    scale_info_extra = {"span_ratio": round(span_ratio, 2),
                                        "mode": "frame_ratio"}
                lo, hi = SEAT_H_RANGE[0] * H_o, SEAT_H_RANGE[1] * H_o
                target_h = max(lo, min(hi, target_h))
                # 坐在座面左侧约三分之一处：居中的话把整条座面挡死，
                # 家具就白拍了；靠边一点才看得出她是"坐在这个家具上"。
                center_x = (prop.x1 + (prop.x2 - prop.x1) * 0.32) / 999.0 * W_o
                # 座面横跨整个画面是常见的（公园长椅、长餐桌），此时按 32% 处
                # 落点会贴到左边缘、一半在画外。这里夹一下，保证整只都画得下。
                half = target_h * 0.78 / 2.0
                center_x = max(half + 6, min(W_o - half - 6, center_x))
                sink = (1.0 - a0.contact_ratio) * target_h
                bottom_y = prop.y / 999.0 * H_o + sink
                step, scale_info = rescale_to_height(
                    scene_at_out, raw, target_h=target_h,
                    center_x=center_x, bottom_y=bottom_y,
                    # 物件路线把缩小下限放宽到 0.15：照片里没有人物，模型完全
                    # 没有尺度参照，实测会把小人画到 828px（目标 186px，比值 0.22），
                    # 被默认的 0.35 拦掉就整条路线失效。缩小对画质无损，
                    # 真正的风险是"掩码圈错了"，那由 character_mask 的贴边检查兜。
                    max_down=0.15)
                scale_info["ruler"] = {"kind": prop.kind, "seat_w": int(seat_w),
                                       "seat_y": prop.y, **scale_info_extra}
        else:
            step, scale_info = rescale_to_box(scene_at_out, raw, box_px)
        info: dict = {"index": i, "scale": scale_info}
        if scale_info.get("applied"):
            tick(f"尺度归一 x{scale_info['ratio']:.2f}", 76)

        # 1) 去白边（提示词已证伪，只能程序化）
        scene_ref = scene_at_out          # 与 step 同尺寸，避免坐标空间错位
        step, rim_fixed = defringe(scene_ref, step)
        info["defringe_replaced_ratio"] = round(rim_fixed, 3)

        # 2) 环境色匹配
        step, amb = match_ambient(scene_ref, step)
        info["ambient"] = amb

        # 3) 接触阴影
        #    贴贴模式要按"支撑方式"分开处理，不能一律当挂着：
        #      support=person 挂在人身上 -> 影子落在人物衣服上，由模型负责，我们不画
        #      support=ground 站在地上   -> 必须画地面阴影，否则会浮空
        #    初版漏了这个区分，导致"扯衣角"这类站姿动作全部浮空（用户实测反馈）。
        if mode == "companion":
            act = get_action(action) if action else None
            support = act.support if act else "person"
            info["support"] = support
            # 阴影必须画在**她现在的实际位置**上。若尺度归一执行了，她已经被
            # 挪到腿边/肩膀上，此时若还按 raw 的差分框画影，影子会留在旧位置，
            # 画面上就是"人在左边、影子在右边"。所以优先用 scale_info.placed。
            bb = None
            placed = scale_info.get("placed") if scale_info.get("applied") else None
            if placed:
                bb = (placed[0], placed[1], placed[0] + placed[2], placed[1] + placed[3])
            else:
                # 未缩放时用**未后处理**的 raw 取掩码：defringe 会把轮廓亮边
                # 填成内部颜色、match_ambient 会把角色色偏往背景拉，两者都会
                # 削弱与原图的差分，导致后处理后再取掩码检测失败。
                try:
                    bb = _bbox(character_mask(scene_ref, raw))
                except Exception:
                    bb = None
            if bb is None:      # 兜底：退回用户框，总比不画影子强
                bb = (int(box_px[0]), int(box_px[1]), int(box_px[2]), int(box_px[3]))

            # 接触线：她到底"碰到哪儿"。
            #
            # 原来不论什么姿态都用 bb[3]（包围盒底边）。对站姿对，
            # 但坐肩/坐椅面这类 contact_ratio=0.62 的姿态，脚是悬空的，
            # 接触点在身体 62% 处——阴影画在 bb[3] 等于画在她脚下的半空里。
            cr = act.contact_ratio if act else 1.0
            bb_h = max(bb[3] - bb[1], 1)
            lift = int((scale_info.get("ruler") or {}).get("lift_px", 0) or 0)
            ground_y = (scale_info.get("ruler") or {}).get("foot_y")
            if support == "ground":
                contact_y = float(ground_y if ground_y is not None else bb[3])
            else:
                contact_y = float(bb[3] - (1.0 - cr) * bb_h)

            if support == "ground":
                # 地面动作：方向性投影 + AO，砸实"踩在地上"
                ls = LightingState(
                    light_azimuth_deg=(light["azimuth"] if light["azimuth"] is not None else 180.0),
                    quality_hardness=light["hardness"],
                    intensity=light["intensity"],
                    confidence=(light["confidence"] if light["azimuth"] is not None else 0.0),
                )
                # 悬空姿态下她的底边离地还有一段，阴影必须落在地面线上；
                # 阴影盒也要延伸到那条线，否则 matte 的绘制区域盖不到。
                shadow_box = (bb[0], bb[1], bb[2], max(bb[3], contact_y))
                dark = render_shadow_matte(
                    step.size, shadow_box, ls,
                    contact_y=contact_y, reach=light.get("reach", 1.0))
                arr = np.asarray(step, dtype=np.float32) * (1 - dark[..., None])
                step = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
                info["shadow"] = "ground_ao" if light["azimuth"] is None else "ground_full"
            elif scale_info.get("applied"):
                # 上半身/物件动作被**搬动过**时，模型原来画的接触暗部留在旧位置，
                # 我们贴的是干净底图，新位置一点阴影都没有、看着像贴纸。
                # 补一层 AO。强度比原来提高：实测 0.55×0.85 在浅色桌面上几乎看不出来。
                ls = LightingState(confidence=0.0,
                                   intensity=min(light["intensity"] + 0.15, 0.75))
                dark = render_shadow_matte(step.size, bb, ls, contact_y=contact_y)
                arr = np.asarray(step, dtype=np.float32) * (1 - dark[..., None])
                step = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
                info["shadow"] = "person_ao"
            else:
                # 没搬动过：模型自己画的接触暗部还在原位，不要叠第二层
                info["shadow"] = "model_handled"

            # 接触暗核：贴着轮廓的那条细暗带，让"压在上面"被读出来。
            # 只在真正接触时画——跳起来时她没碰到任何东西，硬画一条暗线会假。
            if lift <= 0 and info["shadow"] != "model_handled":
                try:
                    cm = character_mask(scene_ref, step)
                    # 照片路线拿不到她的精确轮廓（她由模型画进图里，没有 alpha），
                    # 只能给两层：`cm` 决定暗带的横向范围，`change_mask` 用来兜底清零。
                    # change_mask 是"她 + 她脚下的影子"那个连通域，比 cm 大一圈
                    # （她的白围裙、白鞋不在 cm 里，但在 change 里）。
                    # 万一两层都没盖住，后果只是暗带画不出来，不会横穿她的身体。
                    protect = change_mask(scene_ref, step)
                    core = contact_core(step.size, cm, contact_y, float(bb_h),
                                        protect=protect)
                    if core is not None:
                        arr = np.asarray(step, dtype=np.float32) * (1 - core[..., None])
                        step = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
                        info["shadow"] += "+core"
                    else:
                        info["shadow"] += "+core_skip"
                except Exception as e:
                    info["shadow_core_error"] = str(e)[:80]
            final = add_ai_label(step)
            out_path = os.path.join(run_dir, f"out_{i}.jpg")
            final.save(out_path, quality=93)
            info["file"] = f"out_{i}.jpg"
            info["scores"] = _qc_scores(scene_ref, final, box_px)
            results.append(info)
            continue
        if light["azimuth"] is not None:
            ls = LightingState(light_azimuth_deg=light["azimuth"],
                               quality_hardness=light["hardness"],
                               intensity=light["intensity"],
                               confidence=light["confidence"])
            dark = render_shadow_matte(raw.size, box_px, ls, contact_y=contact_y,
                                       reach=light.get("reach", 1.0))
            arr = np.asarray(step, dtype=np.float32) * (1 - dark[..., None])
            step = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
            info["shadow"] = ls.shadow_tier()
        else:
            # auto：只用 AO 层（不赌方向），置信度压到 0 让渲染器只出 AO
            ls = LightingState(confidence=0.0, intensity=light["intensity"])
            dark = render_shadow_matte(raw.size, box_px, ls, contact_y=contact_y)
            arr = np.asarray(step, dtype=np.float32) * (1 - dark[..., None] * 0.7)
            step = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
            info["shadow"] = "ao_only"

        # 4) AI 标识（合规，必须烧进导出文件）
        final = add_ai_label(step)
        out_path = os.path.join(run_dir, f"out_{i}.jpg")
        final.save(out_path, quality=93)
        info["file"] = f"out_{i}.jpg"
        info["scores"] = _qc_scores(scene_ref, final, box_px)
        results.append(info)

    tick("完成", 100)
    return {
        "run_id": run_id,
        "mode": mode,
        "model": model or os.environ.get("ARK_IMAGE_MODEL", ""),
        "scene": "scene.jpg",
        "prompt": prompt,
        "size": size,
        "candidates": results,
        "errors": errors,
        "profile": ({
            "framing": prof.framing,
            "selfie": prof.selfie,
            "light": prof.light,
            "has_prop": prof.has_prop,
            "person": prof.person,
            "legs": prof.legs,
        } if prof else None),
        "elapsed_hint": "单张约 60-80 秒，候选并行",
    }
