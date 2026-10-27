"""S2.5 尺度归一：把模型画出来的角色，按框精确缩放。

## 为什么需要这一步

用户实测发现：同一提示词、不同 seed，角色大小能差一倍。提示词只能把**均值**拉对，
**方差**压不住——模型对绝对尺度没有稳定控制能力。

## 为什么"生成偏小"是安全的

关键观察：**她原来的位置可以从原照片恢复**，因为我们一直保留原图。
所以"把角色从原来的位置取走"是免费的，不需要 inpaint。

于是流程成立：
    生成一张偏小的角色
      → 用掩码把她摘出来（带 alpha）
      → 按框的目标尺寸缩放
      → 贴到框的位置
      → 底图直接用原照片（干净、无重编码损失）

尺度从此不受模型影响，100% 由框决定。代价只有两个：
1. 放大后边缘会软一点（2 倍以内可接受，超过 3 倍放弃）
2. 模型在角色周围做的环境色渗透会有一部分留在原位置——但原位置本来就被丢弃了

## 什么情况不缩放

- 掩码贴到画面边框（说明模型把她画出画外了，缺失的部分无法重建）→ 跳过并标记
- 缩放倍率超过 max_up（放大过头会糊）→ 跳过
- 缩放倍率在容差内（本来就基本对）→ 跳过，避免无谓的重采样
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from s3.beautify import _largest_component, change_mask, character_mask


def _bbox(mask) -> tuple[int, int, int, int] | None:
    ys = mask.any(axis=1)
    xs = mask.any(axis=0)
    if not ys.any():
        return None
    y0, y1 = int(np.argmax(ys)), int(len(ys) - np.argmax(ys[::-1]))
    x0, x1 = int(np.argmax(xs)), int(len(xs) - np.argmax(xs[::-1]))
    return x0, y0, x1, y1


def _convex_hull_mask(mask) -> "np.ndarray":
    """掩码点集的凸包（Andrew 单调链），返回同尺寸的填充布尔图。

    不引 scipy/OpenCV：先抽稀到 4000 个点再做，耗时几毫秒。
    """
    from PIL import ImageDraw

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros_like(mask)
    pts = np.stack([xs, ys], 1)
    step = max(1, pts.shape[0] // 4000)
    pts = sorted(map(tuple, pts[::step]))

    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return np.zeros_like(mask)
    canvas = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
    ImageDraw.Draw(canvas).polygon(hull, fill=255)
    return np.asarray(canvas) > 127


def _fill_enclosed(m: "np.ndarray", add: "np.ndarray", sw: int = 200) -> "np.ndarray":
    """只把**被她的轮廓围住**的低反差区域补进来。

    ## 为什么必须是"封闭"而不只是"在她那一行之内"

    先用过"行内跨度填充"（把每行最左到最右之间都补上），修好了她腿上的洞，
    但把身子底下的长椅木板也一起补了——因为她就坐在长椅上，她那一行里
    装的全是长椅，而长椅与画面边缘是连通的，不是洞。
    实测长椅那张贴回去以后，她腰部前后多出几条横木和几个螺栓。

    ## 做法

    从裁切框的**四条边界**泛洪（不是四个角，四个角太容易漏），
    泛到的就是"外部"；泛不到的空白区域就是被她围住的内部洞。
    只在洞内用低阈值补，洞外一切照旧。

    降采样到 200 宽再做：她身上的洞都是大块，缩小不影响判断，
    而 PIL 的 floodfill 是纯 Python 循环，全分辨率太慢。
    """
    h, w = m.shape
    sh = max(1, int(h * sw / w))
    small = Image.fromarray((m.astype("uint8") * 255), "L")
    small = small.resize((sw, sh), Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)
    for x in range(sw):
        for y in (0, sh - 1):
            if small.getpixel((x, y)) == 0:
                ImageDraw.floodfill(small, (x, y), 128, thresh=0)
    for y in range(sh):
        for x in (0, sw - 1):
            if small.getpixel((x, y)) == 0:
                ImageDraw.floodfill(small, (x, y), 128, thresh=0)
    # 注意：flood 的填充值是 128，而角色本体是 255。一定要先二值化成
    # "是否 128"再放大，直接拿 >64 比较会把角色本体也当成外部（踩过）。
    outside_small = (np.asarray(small) == 128).astype("uint8") * 255
    outside = np.asarray(
        Image.fromarray(outside_small, "L").resize((w, h), Image.BILINEAR)) > 64
    holes = (~outside) & (~m)
    return m | (holes & add)


def _alpha_in_box(scene: Image.Image, result: Image.Image,
                  box: tuple[int, int, int, int],
                  core: "np.ndarray | None" = None,
                  thr: float = 45.0,
                  thr_low: float = 22.0) -> Image.Image | None:
    """在给定盒子里抽出角色的不透明轮廓（用于缩放后重新贴回）。

    ## alpha 和包围盒的要求正好相反

    包围盒要"最紧"——只能用「蓝色主导 ∩ 差分」的核心，多一点高度就错。
    alpha 要"最全"——她的脸是肤色、围裙和袜子是白色、鞋是深蓝，都不满足
    "蓝色主导"，所以**不能**直接拿核心掩码当 alpha（实测会把她抠成半透明、
    脸和下半身消失）。

    ## 但核心掩码的**凸包**是最好的边界

    试过四种边界，只有凸包能同时满足"包住她"和"不漏背景"：

    - 阈值 14 的差分：模型每次都会把地面重画一遍，结果是**裁切框会显形**，
      她身后一块发暗的矩形，脚下还粘着人物球鞋的碎片。
    - 阈值 45 的差分：她完整了，但边界顺着她画在地上的投影往外扩，
      投影被裁切框切断，贴回去就是一条带直边的暗带。
    - 行扫描填充 / 四边泛洪补洞：都会把周围地面判成"内部"，出一大团带直边灰块。
    - **核心的凸包 ∩ 阈值 45 差分**：凸包把她的投影挡在外面（投影落在包外），
      差分保住细腻边缘和内部细节，两边的好处都要到了。

    最后用形态学闭运算（先膨胀后腐蚀，核 5）补掉她深色区域留下的麻点：
    深藏青裙摆在阴影里与暗色路面的差异可能不足 45，不补会显得半透明。
    """
    cx0, cy0, cx1, cy1 = box
    a = np.asarray(result.crop(box), dtype=np.float32)
    s = np.asarray(scene.crop(box), dtype=np.float32)
    m = np.abs(a - s).max(axis=2) > thr
    if m.sum() < 50:
        return None
    h, w = m.shape

    solid = m
    if core is not None:
        hull = _convex_hull_mask(core)
        if hull.any():
            # ## 内部补洞：这才是"她身上有透明部分"的根因
            #
            # 阈值 45 是为了保证**外轮廓**不沾到背景（背景重绘幅度 14~30）。
            # 但她身体内部有些地方与底图的差异不足 45：
            #   - 深藏青裙摆压在阴影里
            #   - 白色围裙压在亮地面上
            #   - 蓝色头发的高光过渡区
            # 这些像素 alpha 归零，贴回干净底图时背景就透出来了。
            # 实测蹲姿那张：**14.1% 的不透明像素是内部空洞**，
            # 表现为她"下半身半透明、腿和鞋能看见地板"。
            #
            # 修法：外轮廓仍用高阈值 m，内部用低阈值的 low 补上，
            # 但 **只补被她轮廓围住的封闭区域**。
            #
            # 两个走过的弯路：
            #   "low ∩ 凸包"——凸包在凹形处过宽，呆毛到右侧头发之间整片三角都在包内，
            #     只要有轻微重绘就被吸进来，她头顶右边多出一条横贯灰线。
            #   "low ∩ 行内跨度"——修好了腿上和裙摆的洞，但她就坐在长椅上，
            #     她那一行里装的全是长椅，于是把木板条和螺栓一起贴了过来。
            # 封闭空洞是唯一正确的口径：她的腿、裙摆、围裙都是被自己围住的，
            # 而长椅、地面、背景都与画面边缘连通。
            low = np.abs(a - s).max(axis=2) > thr_low
            filled = _fill_enclosed(m, low)
            # 安全阀：补得太多说明 low 连成片了（比如她坐在强反光面上），
            # 宁可留一点洞，也不要把背景糊进来。
            if filled.sum() <= m.sum() * 2.5:
                solid = filled
            else:
                solid = m & hull
            if solid.sum() < m.sum() * 0.45:
                # 凸包切掉了太多（她伸出了包，或核心掩码不完整），退回原掩码
                solid = m

    # 只保留最大的连通域。
    # 模型偶尔会画**两个**小人（实测 u03_living 一次生成吐了一大小两只，
    # 差分掩码把两只都算进来，贴回去画面上就有两个鲸鱼娘）。
    # 两个是彼此不连通的两块，留面积最大的那块即可。
    # 若最大块占比不到一半，说明她的身体本身被切成了几块，这时候不能丢，
    # 保留原掩码。
    lc = _largest_component(solid)
    if lc.sum() >= solid.sum() * 0.5:
        solid = lc

    im = Image.fromarray((solid.astype("uint8") * 255), "L")
    im = im.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))   # 闭运算
    im = im.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.2))
    return im


def rescale_to_box(original: Image.Image, result: Image.Image,
                   box_px: tuple[float, float, float, float],
                   tol: float = 0.12,
                   max_up: float = 3.0,
                   max_down: float = 0.45) -> tuple[Image.Image, dict]:
    """把 result 里的角色按 box_px 目标尺寸重排到 original 上。

    返回 (新图, 诊断信息)。诊断信息里 applied=False 表示没动，原因写在 reason。
    """
    mask = change_mask(original, result)
    if mask.sum() < 200:
        return result, {"applied": False, "reason": "mask_too_small"}

    bbox = _bbox(mask)
    if bbox is None:
        return result, {"applied": False, "reason": "empty_mask"}
    x0, y0, x1, y1 = bbox
    W, H = result.size

    # 贴边判定：角色被画出画外了，缺的部分补不回来
    if x0 <= 2 or y0 <= 2 or x1 >= W - 2 or y1 >= H - 2:
        return result, {"applied": False, "reason": "touches_border",
                        "bbox": [x0, y0, x1, y1]}

    h_act, w_act = y1 - y0, x1 - x0
    bx, by, bw, bh = box_px
    h_tgt = bh
    ratio = h_tgt / max(h_act, 1)

    info = {
        "applied": False,
        "ratio": round(float(ratio), 3),
        "actual_h": int(h_act),
        "target_h": int(round(h_tgt)),
        "bbox": [x0, y0, x1, y1],
    }
    if abs(ratio - 1.0) <= tol:
        info["reason"] = "within_tolerance"
        return result, info
    if ratio > max_up:
        info["reason"] = "ratio_too_large"
        return result, info
    if ratio < max_down:
        info["reason"] = "ratio_too_small"
        return result, info

    # 1) 摘出角色：裁到 bbox（外扩 2px 保住边缘），alpha 由掩码羽化得到
    pad = 2
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
    crop = result.crop((cx0, cy0, cx1, cy1)).convert("RGBA")
    m = Image.fromarray((mask[cy0:cy1, cx0:cx1].astype("uint8") * 255), "L")
    alpha = m.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.0))
    crop.putalpha(alpha)

    # 2) 缩放
    nw, nh = max(1, int(crop.width * ratio)), max(1, int(crop.height * ratio))
    crop = crop.resize((nw, nh), Image.LANCZOS)

    # 3) 贴到框上：底边对齐框底边，水平居中于框
    canvas = original.convert("RGB").copy()
    px = int(bx + bw / 2 - nw / 2)
    py = int(by + bh - nh)
    canvas.paste(crop, (px, py), crop)

    info.update(applied=True, reason="rescaled", placed=[px, py, nw, nh],
                new_h=int(nh))
    return canvas, info


def rescale_to_height(original: Image.Image, result: Image.Image,
                      target_h: float,
                      center_x: float | None = None,
                      bottom_y: float | None = None,
                      place: bool = True,
                      tol: float = 0.12, max_up: float = 3.0,
                      max_down: float = 0.35) -> tuple[Image.Image, dict]:
    """按"绝对高度"重排角色，而不是按框。

    贴贴模式专用：她的大小应该由**人物的身体**决定（比如"只到小腿的一半高"），
    而人物的身体尺寸可以从 VLM 给的身体部位框换算成像素。所以这里接的不是框，
    是一个由"部位框高度 × 动作比例"算出来的目标像素高度。

    初版 companion 模式干脆跳过了缩放，结果尺度完全放飞——用户实测"角色过大"。

    ## place=False：只纠正大小，不动位置

    人的上半身动作（坐肩、探头、扒衣摆）是模型**真的能画对**的一类：
    肩膀和衣摆是画面里显眼的强锚点，模型对"小人在肩上"有大量先验，
    实测遮挡和接触暗部都成立。这类动作如果强行重定位，反而会把她从
    一个正确的位置搬到一个算出来的位置，把已经对的遮挡关系搞坏。

    所以这类动作用 `place=False`：只有幅度超过容差才重采样，
    缩放后**底边和中心保持原位**，位置一个字不动。
    `center_x` / `bottom_y` 传 None 时自动取她当前包围盒的中心和底边。
    """
    # 掩码用 character_mask（蓝 ∩ 差分），不能用单一判据：
    #   - 只用蓝色主导：户外照的蓝天被算进来，bbox 贴到顶边/右边 → touches_border
    #   - 只用差分：人物被重绘的鞋和衣服一起算进来 → 同样贴边
    # 两者都会让缩放被**静默**放弃，症状是"角色过大"，但日志里看不到报错。
    from s3.beautify import character_mask
    mask = character_mask(original, result)
    if mask.sum() < 200:
        return result, {"applied": False, "reason": "mask_too_small"}

    bbox = _bbox(mask)
    if bbox is None:
        return result, {"applied": False, "reason": "empty_mask"}
    x0, y0, x1, y1 = bbox
    W, H = result.size
    if x0 <= 2 or y0 <= 2 or x1 >= W - 2 or y1 >= H - 2:
        return result, {"applied": False, "reason": "touches_border",
                        "bbox": [x0, y0, x1, y1]}

    h_act = y1 - y0
    ratio = target_h / max(h_act, 1)
    info = {"applied": False, "ratio": round(float(ratio), 3),
            "actual_h": int(h_act), "target_h": int(round(target_h)),
            "bbox": [x0, y0, x1, y1], "place": place}
    if ratio > max_up or ratio < max_down:
        info["reason"] = "ratio_out_of_range"
        return result, info
    if abs(ratio - 1.0) <= tol:
        if not place:
            # 只纠正大小时，容差内**什么都不做**：位置由模型负责，它已经摆对了
            info["reason"] = "within_tolerance"
            return result, info
        # 要重定位时，尺度对也得走一遍安放——她可能画在了离腿很远的地方。
        # 只跳过重采样，不跳过定位。初版这里直接 return，导致"大小对了但位置乱"。
        info["reason"] = "within_tolerance_placed"
        ratio = 1.0

    # place=False 时保持原位：中心不变、底边不变（脚还踩在同一条线上）
    if not place:
        if center_x is None:
            center_x = (x0 + x1) / 2.0
        if bottom_y is None:
            bottom_y = float(y1)

    pad = 2
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
    crop = result.crop((cx0, cy0, cx1, cy1)).convert("RGBA")
    # alpha 与包围盒用不同判据，原因见 _alpha_in_box 的文档
    alpha = _alpha_in_box(original, result, (cx0, cy0, cx1, cy1),
                          core=mask[cy0:cy1, cx0:cx1])
    if alpha is None:                    # 盒子里几乎没有高幅差异，退回核心掩码
        alpha = Image.fromarray((mask[cy0:cy1, cx0:cx1].astype("uint8") * 255), "L")
    crop.putalpha(alpha)

    nw, nh = max(1, int(crop.width * ratio)), max(1, int(crop.height * ratio))
    crop = crop.resize((nw, nh), Image.LANCZOS)

    canvas = original.convert("RGB").copy()
    px = int(center_x - nw / 2)
    py = int(bottom_y - nh)
    px = max(2, min(W - nw - 2, px))      # 贴边夹紧
    py = max(2, min(H - nh - 2, py))
    canvas.paste(crop, (px, py), crop)
    info.update(applied=True, reason="rescaled", placed=[px, py, nw, nh], new_h=int(nh))
    return canvas, info
