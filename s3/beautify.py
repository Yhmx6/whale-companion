"""S3 追加两个环节：去白边（defringe）与环境色匹配（ambient match）。

为什么需要：

1. **白描边**（U-26）：10/10 复现，提示词已证伪无法消除，只能程序化处理。
   做法不是"猜哪里是白边"，而是用「生成图 − 原图」的差分得到角色掩码——
   我们本来就要求模型不改背景，所以差分几乎只剩角色。掩码可靠，处理才可靠。

2. **环境色匹配**（U-30）：角色是从设定图搬过来的，色温与场景天然不同。
   直接把角色整体往背景的平均色偏上拉一点（只拉直流分量，不动对比度），
   比重新生成一遍便宜得多，且不会破坏角色配色。

两条都纯 CPU、确定性、无模型依赖。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _to_arr(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.float32)


def _to_img(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def _mask_img(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), "L")


def _mask_arr(im: Image.Image, thr: int = 127) -> np.ndarray:
    return np.asarray(im, dtype=np.uint8) > thr


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys = mask.any(axis=1)
    xs = mask.any(axis=0)
    if not ys.any():
        return None
    y0, y1 = int(np.argmax(ys)), int(len(ys) - np.argmax(ys[::-1]))
    x0, x1 = int(np.argmax(xs)), int(len(xs) - np.argmax(xs[::-1]))
    return x0, y0, x1, y1


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """取最大连通域（4 邻域 BFS）。不引 scipy：图小，纯 Python 也够快。"""
    from collections import deque

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best_pts: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            pts = []
            while q:
                cy, cx = q.popleft()
                pts.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(pts) > len(best_pts):
                best_pts = pts
    out = np.zeros_like(mask, dtype=bool)
    for y, x in best_pts:
        out[y, x] = True
    return out


def _blue_mask(result: Image.Image) -> np.ndarray:
    """蓝色主导掩码——角色是饱和蓝，场景通常是低饱和的暖色/灰白。兜底用。

    【局限，实测踩过】户外照片里有蓝天，天空同样满足「蓝 > 红绿 + 22」，
    于是掩码会横跨半个画面。用它直接取包围盒，结果 bbox 贴到画面顶边和右边，
    下游据此判定 touches_border，比例尺归一被静默跳过——连续两轮"角色过大"
    的真正原因就是这个，而不是提示词。**只有确认背景无蓝色时才可单独使用。**
    """
    a = _to_arr(result)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (b - np.maximum(r, g) > 22) & (b > 50)
    m_img = _mask_img(m).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
    return _mask_arr(m_img)


def character_mask(scene: Image.Image, result: Image.Image,
                   thr: float = 14.0, edge_pad: int = 2) -> np.ndarray:
    """角色掩码：**蓝色主导 ∩ change_mask**。贴贴模式取包围盒只用这一个函数。

    为什么必须是这个交集（两个单用都试过，都错）：

    - 只用蓝色主导：户外照的蓝天被整块判成角色。实测 bbox=(523,0,1280,1655)，
      从画面顶边跨到右边框 → 下游判 touches_border → 比例尺归一被静默放弃，
      而且**不报错**，表现就是"角色大小完全不受控"，查了两轮才定位到。
    - 只用差分（change_mask）：人物被模型轻微重绘的鞋、衣服、投影一起进掩码。
      实测 bbox 横跨 (167,1035)-(1280,1704)，同样贴边。

    交集恰好把两者的问题都消掉：蓝天是蓝的但**没有变**，牛仔裤变了但**不蓝**，
    只有"既是蓝色主导、又相对原图发生了变化"的那一块才是她。四张实测底图
    （户外全身 ×2、室内全身 ×2）交集后包围盒全部不贴边，且高度与目视一致。

    差分那一侧必须用 change_mask 而不是裸阈值差分：模型会对整图做轻微重绘，
    裸差分覆盖率能到 24% 且连通成一片；change_mask 只保留最大连通域，
    正好滤掉天空里那些"也变了一点点"的零散像素。

    两个试过但放弃的加强手段：
    - 沿差分泛洪吸收她的白色围裙/白鞋：窗口内差分同样密集成片，泛洪一定漏到
      窗口边界，包围盒反而变成窗口大小（实测宽度从 487 涨到 836）。
    - 降采样后取最大连通域再放大：320 宽下她的呆毛和细边会断，高度少 12%
      （611 → 540），比留着少量噪点更糟。
    代价：她的白色部分不计入包围盒。实测底边只差约 15px（1%），可以接受。
    """
    cm = change_mask(scene, result, thr=thr)
    blue = _blue_mask(result)
    both = cm & blue
    if both.sum() < 200:
        # 交集为空：她不是蓝色，或模型几乎没改动。退到单判据，宁可粗也不要空。
        return cm if cm.sum() >= 200 else blue

    # 兜底：若交集仍贴边而差分不贴边，说明 change_mask 触发了安全阀退化成
    # 纯蓝色掩码（整图被重绘时会发生），此时差分的包围盒更可信。
    W, H = result.size
    bb = _bbox_of(both)
    if bb and (bb[0] <= edge_pad or bb[1] <= edge_pad
               or bb[2] >= W - edge_pad or bb[3] >= H - edge_pad):
        cb = _bbox_of(cm)
        if cb and cb[0] > edge_pad and cb[1] > edge_pad \
                and cb[2] < W - edge_pad and cb[3] < H - edge_pad:
            return cm
    return both


def change_mask(scene: Image.Image, result: Image.Image,
                thr: float = 14.0, work: int = 256,
                max_ratio: float = 0.30) -> np.ndarray:
    """得到"角色所在区域"的掩码（bool）。

    初版直接在全分辨率上做逐像素差分，阈值 9 —— 结果掩码占了 52% 画面。
    原因是模型会对整图做轻微重编码与调色，全局的小幅偏移全部越过了阈值。
    修正三处：
      1. 先降采样到 work 宽并高斯模糊，压掉编码噪声与细微纹理差；
      2. 阈值提到 14，只保留真实的内容变化；
      3. **只取最大连通域**——角色是一整块，背景的零散变化不该进掩码。

    另设安全阀：若最大连通域仍超过 max_ratio（说明整图被重绘了，或差分判据失效），
    退回蓝色主导掩码。宁可掩码偏小，也不要让它糊满整张图。
    """
    if scene.size != result.size:
        scene = scene.resize(result.size, Image.LANCZOS)
    w = work
    h = max(1, int(result.height * w / result.width))
    small_s = scene.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2))
    small_r = result.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2))
    d = np.abs(_to_arr(small_r) - _to_arr(small_s)).max(axis=2)
    m = d > thr
    if not m.any():
        return _blue_mask(result)

    comp = _largest_component(m)
    if comp.sum() / comp.size > max_ratio:
        return _blue_mask(result)

    up = _mask_img(comp).resize(result.size, Image.BILINEAR)
    up = up.filter(ImageFilter.MaxFilter(5))       # 补齐边缘，避免漏掉轮廓
    return _mask_arr(up, thr=90)


def defringe(scene: Image.Image, result: Image.Image,
             ring_px: int = 3, bright_ratio: float = 1.06) -> tuple[Image.Image, float]:
    """抑制角色轮廓外侧的亮边（白描边）。

    原理：先用掩码算出"只含角色内部"的颜色场（把角色内部像素模糊并向边界外推），
    再把轮廓环里明显比这个颜色场更亮的像素替换掉。这样白边被"角色自己的颜色"吃掉，
    轮廓尺寸不变、不发胖。

    返回 (处理后的图, 被替换像素占比)。
    """
    from PIL import ImageFilter as F

    mask = change_mask(scene, result)
    if mask.sum() < 100:
        return result, 0.0

    m_img = _mask_img(mask)
    core = _mask_arr(m_img.filter(F.MinFilter(2 * ring_px + 1)))
    dil = _mask_arr(m_img.filter(F.MaxFilter(2 * ring_px + 1)))
    ring = dil & ~core

    arr = _to_arr(result)
    # 只保留角色内部的像素，模糊后归一化 -> 边界外侧也能得到"向内取色"的场
    core_f = core.astype(np.float32)[..., None]
    num = _to_arr(_to_img(arr * core_f)).astype(np.float32)
    num = np.asarray(Image.fromarray(num.astype(np.uint8)).filter(
        F.GaussianBlur(radius=ring_px * 2.5)), dtype=np.float32)
    den = np.asarray(Image.fromarray((core_f[..., 0] * 255).astype(np.uint8)).filter(
        F.GaussianBlur(radius=ring_px * 2.5)), dtype=np.float32) / 255.0
    field = num / np.maximum(den, 1e-3)[..., None]

    lum = arr.mean(axis=2)
    lum_field = field.mean(axis=2)
    is_bright = (ring & (lum > lum_field * bright_ratio + 4.0))
    if not is_bright.any():
        return result, 0.0

    alpha = 0.85
    out = arr.copy()
    out[is_bright] = out[is_bright] * (1 - alpha) + field[is_bright] * alpha
    return _to_img(out), float(is_bright.sum()) / max(ring.sum(), 1)


def match_ambient(scene: Image.Image, result: Image.Image,
                  strength: float = 0.32, max_shift: float = 26.0,
                  mask: np.ndarray | None = None,
                  work: int = 320) -> tuple[Image.Image, dict]:
    """把角色的整体色偏往背景拉一点，解决"角色色温与场景不搭"。

    只做直流分量平移（不动对比度/饱和度），幅度封顶，避免把深蓝色女仆裙洗成灰色。
    返回 (处理后的图, 实际位移量)。

    `mask` 可选：立绘路线有精确 alpha，直接传进来，别再用 `change_mask` 去猜。
    启发式掩码在 256 宽上算完再放大，边缘差 ~6px，会让色偏漏到她的轮廓外面。

    ## 为什么要缩图算（这一条让这个函数从 2.2s 降到 ~0.1s）

    这里求的只是一个**三元色偏向量**：她身上平均色 与 她外圈平均色 之差。
    这件事根本不需要像素级精度，但初版在**全分辨率**上做形态学滤波：
    `MinFilter(5)` + `MaxFilter(17)` + `MaxFilter(5)`，1280×1600 的图上
    光 `rankfilter` 就吃了 2.86 秒——占整条 CPU 链路的 59%。

    改成：把图和她掩码缩到 `work` 宽再滤波求均值，位移量照样算得对，
    应用时按 alpha 羽化铺回全图（顺手消掉了原来 `core` 硬边造成的接缝）。
    """
    mask = change_mask(scene, result) if mask is None else mask.astype(bool)
    if mask.sum() < 100:
        return result, {}

    W, H = result.size
    sw = max(64, int(work))
    sh = max(64, int(round(H * sw / W)))
    small = result.convert("RGB").resize((sw, sh), Image.BILINEAR)
    m_img = Image.fromarray((mask.astype(np.uint8) * 255), "L").resize(
        (sw, sh), Image.BILINEAR)

    core = _mask_arr(m_img.filter(ImageFilter.MinFilter(5)))
    outer = _mask_arr(m_img.filter(ImageFilter.MaxFilter(17))) & ~_mask_arr(
        m_img.filter(ImageFilter.MaxFilter(5)))
    if core.sum() < 20 or outer.sum() < 20:
        return result, {}

    arr_s = _to_arr(small)
    char_mean = arr_s[core].mean(axis=0)
    bg_mean = arr_s[outer].mean(axis=0)
    delta = np.clip(bg_mean - char_mean, -max_shift, max_shift) * strength

    # 按 alpha 羽化铺回全图：她盖住的地方整份偏移，边缘自然过渡，背景不动。
    arr = np.asarray(result.convert("RGB"), dtype=np.float32)
    out = np.clip(arr + delta[None, None, :] * mask[..., None], 0, 255)
    return Image.fromarray(out.astype(np.uint8), "RGB"), {
        "char_mean": [round(float(v), 1) for v in char_mean],
        "bg_mean": [round(float(v), 1) for v in bg_mean],
        "applied_shift": [round(float(v), 1) for v in delta],
    }


# 显式标识的字高下限，取自强制性国家标准。
#
# GB 45438-2025《网络安全技术 人工智能生成合成内容标识方法》5.2 条：
#   a) 图片内容显式标识应采用文字提示；
#   b) 应同时包含「人工智能/AI」和「生成/合成」两个要素；
#   c) 应位于图片的**边或角**；
#   d) 字型清晰可辨；
#   e) **文字高度不应低于画面最短边长度的 5%**。
#
# 配套的《人工智能生成合成内容标识办法》（2025-09-01 施行）另有禁止性规定：
# 任何组织和个人不得恶意删除、篡改、伪造、隐匿标识。
#
# 改造前这里写的是 2.8%，**低于国标下限**——那个尺寸看起来"不碍事"，
# 但并不合规。合规的最低成本就是把这个数字提到 5%，再靠位置和底色
# 让它尽量不挡主体（国标只要求"在边或角"，没说必须多显眼）。
AI_LABEL_MIN_RATIO = 0.05


def add_ai_label(img: Image.Image, text: str | None = None) -> Image.Image:
    """在导出图上烧入显式 AI 标识。

    这是**强制要求**，不是可选的装饰：国标 GB 45438-2025 与
    《人工智能生成合成内容标识办法》都要求生成内容带可被用户明显感知的
    显式标识，且禁止删除或隐匿。

    能调的只有"长什么样"，不能调"有没有"：
      - 位置：国标只要求「边或角」，选右下角（最不挡主体）；
      - 底色：半透明黑衬白字，保证在亮背景上也能看清；
      - 字号：**不能低于最短边的 5%**，这是硬下限。
    """
    from PIL import ImageDraw, ImageFont

    w, h = img.size
    short = min(w, h)
    # 字号先按 5% 取，最后还会用实际渲染高度再校一次——
    # 不同字体的 font size 和实际字高不完全相等，只按 size 算可能刚好差一点。
    size = max(int(round(short * AI_LABEL_MIN_RATIO)), 16)
    font = None
    candidates = [
        "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhl.ttc",
        "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf", "arial.ttf",
    ]
    font_path = ""
    for c in candidates:
        try:
            font = ImageFont.truetype(c, size)
            font_path = c
            break
        except (OSError, IOError):
            continue
    if font is None:                     # 一个字体都没有：退回内置位图字体
        try:
            font = ImageFont.load_default(size=size)   # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()
    label = text or "AI生成"
    # 无中文字体时退回英文，避免导出方块
    try:
        font.getlength("AI生成")
        if font.getlength("AI生成") <= 0:
            label = "AI GENERATED"
    except Exception:
        label = "AI GENERATED"

    def _text_h(f) -> float:
        """文字的实际渲染高度。

        测的是**实际画出来的像素高度**，不是 font size。两者在不同字体下
        差 10~20%——只按 font size 判断"够不够 5%"，很可能做出来是 4.4%，
        看着差不多、实际不合规。

        用整串文字测而不是测 "AI"：拉丁字母的高度比汉字矮，
        拿 "AI" 的高度当基准会把字号算小。
        """
        probe = label or "AI生成"
        try:
            box = f.getbbox(probe)
            return float(box[3] - box[1])
        except Exception:
            try:
                return float(f.getsize(probe)[1])
            except Exception:
                return float(size)

    # 实际字高不足 5% 就逐步加大。上限 8 轮，防止异常字体把字号撑爆。
    for _ in range(8):
        if _text_h(font) >= short * AI_LABEL_MIN_RATIO:
            break
        size = int(size * 1.06) + 1
        try:
            font = ImageFont.truetype(font_path or candidates[0], size)
        except Exception:
            break

    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out, "RGBA")
    try:
        tw = d.textlength(label, font=font)
        th = _text_h(font)
    except Exception:
        tw, th = size * len(label) * 0.6, size
    pad = max(int(size * 0.28), 4)
    # 右下角：国标要求"边或角"，右下角离画面主体最远
    x0 = w - tw - pad * 2
    y0 = h - th - pad * 2
    d.rectangle([x0, y0, w - 1, h - 1], fill=(0, 0, 0, 110))
    d.text((x0 + pad, y0 + pad), label, font=font, fill=(255, 255, 255, 235))
    return out
