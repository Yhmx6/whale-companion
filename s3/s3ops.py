"""S3 三项程序化几何后处理：重力对齐 / 深度匹配散焦 / 接触阴影。

依据：《06-管线设计与失败形态映射》§2 §3 §4，参数表见 §2.2。
全部纯 CPU（numpy + PIL），单张渲染目标 < 5ms 量级（不含磁盘 IO）。
本文件是初始实现（team-lead 抢在限频窗口写的骨架），待后端工程师复核。

坐标约定：box = (x1, y1, x2, y2) 像素坐标；接触点 contact = (cx, y2)。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .lighting import LightingState

# ---------------------------------------------------------------------------
# S3.3 接触阴影（06 §2.2 参数表）
# ---------------------------------------------------------------------------

# 三段投射层：累计长度 / 模糊 sigma / 不透明度（乘 intensity）
_SEGMENTS = [
    # (end_len, sigma, opacity)   长度单位均为 w（角色底边宽）
    (0.8, 0.05, 0.35),
    (1.6, 0.12, 0.18),
    (2.4, 0.25, 0.08),
]
_SEG_HARD = [  # hardness > 0.85 时的 4 段方案（06 §1.3）
    (0.6, 0.04, 0.42),
    (1.1, 0.08, 0.26),
    (1.7, 0.15, 0.13),
    (2.6, 0.28, 0.06),
]

# 各段胶囊厚度（画面上影子的横向宽度），单位 w。可调参数（U-12）
_SEG_THICK = [0.20, 0.32, 0.46, 0.60]

AO_A = 0.42         # AO 椭圆长半轴 = 0.42w（全宽 ≈ 0.84w，略窄于角色）
AO_B_RATIO = 0.30   # 短/长轴比，由相机俯角决定（不能写死，见 06 §2.2）
AO_SIGMA = 0.03     # AO 模糊 sigma = 0.03w
AO_OPACITY = 0.60

# 坐姿补偿：接触面在角色包围盒**内部**时，AO 椭圆会被她自己挡住大半。
#
# 这个缺陷一直存在，只是以前看不出来——因为阴影盖在她身上，
# 那块"压在她裙摆上的暗部"冒充了接触感。把阴影挪到她图层下面之后
# （`shadow_under`），长椅那张的接缝带只剩 16.7% 的像素有可见压暗，
# 她看起来像浮在座面上。
#
# 原因很直接：AO 半轴 0.42w → 全宽 0.84w，比她本人（1.0w）还窄，
# 中心又正好在被她挡住的位置 → 可见部分约等于零。
#
# 所以按"接触面以下还剩多少她的身体"来放宽：
#   站姿 hidden≈0    → 仍是 0.42（AO 整个落在地面上，放宽只会变成一团大暗斑）
#   坐姿 hidden≈0.38 → 0.42 + 0.28×0.76 ≈ 0.63（全宽 1.26w，两侧各溢出约一成）
AO_SPILL_K = 0.28
AO_SPILL_REF = 0.5   # hidden 到这个比例就算"完全被挡住"，补偿吃满

# 参数修正记录（2026-10-26，真实照片上视觉验证）：
# 06 号 §2.2 原文写 AO 椭圆 a=0.6w / b=0.22w。按半长轴理解时全宽达 1.2w，
# 在真实石板路上渲染出来是一团比角色还宽的暗斑，明显失真。
# 改为 0.42w / 0.30，全宽约 0.84w、纵深约 0.25w，与角色脚部尺度相符。
# 这两个数是"贴地感"最敏感的参数，正式实现需在 50-100 张金标准集上回归（U-11）。


def _dilate1(m: "np.ndarray") -> "np.ndarray":
    """3×3 二值膨胀（纯 numpy，不引 scipy）。

    只用于把角色掩码往外推 1px，吸收抗锯齿的过渡像素——那些像素是"半她半背景"，
    压暗了会在她轮廓上留一圈描边。
    """
    import numpy as np

    m = m.astype(bool)
    h, w = m.shape
    p = np.pad(m, 1, mode="constant", constant_values=False)
    acc = m.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            acc |= p[dy:dy + h, dx:dx + w]
    return acc


def contact_core(size: tuple[int, int], mask: "np.ndarray",
                  contact_y: float, height_px: float,
                  strength: float = 0.45,
                  thickness_ratio: float = 0.035,
                  spread_ratio: float = 2.5,
                  protect: "np.ndarray | None" = None) -> "np.ndarray | None":
    """接触暗核：在"她与支撑面之间"压一条贴合轮廓的窄暗带。

    ## 为什么现有的 AO 不够

    `render_shadow_matte` 的 AO 层是一个**又大又软的椭圆**（长半轴 0.42w、
    sigma 0.03w、透明度 0.60）。它表达的是"环境光遮蔽"这一层低频信息，
    方向上是对的，但它**不区分"她碰到哪儿"**——椭圆覆盖了她脚下很大一片，
    而真正让物体看起来"压在上面"的，是接触处那一条又窄又深的暗线。

    真实照片里，任何物体与支撑面接触的地方都会有一条几乎纯黑的窄带
    （几何上接近零距离遮蔽），宽度只有几像素。缺了它，即使 AO 在，
    人眼也会读出"贴纸"。

    ## 暗带落在哪儿：**背景上，不是她身上**（这个函数的全部要害）

    第一版把高斯剖面沿**整列**施加（`dark[:, cols] = prof[:, None]`），
    没有排除角色自己的像素。坐姿的接触线恰好就是臀底，于是暗带横贯她的身体，
    三张物件路线图（长椅 / 餐桌 / 沙发）里她身上都多了一条灰线。
    这是很典型的"作用域写错"：`cols` 表达的是"她在接触线附近占了哪些列"，
    但被当成了"这些列整列都该压暗"。

    正确的语义是：**接缝处在背景一侧变暗，物体本身不变**。
    物理上也成立——被遮蔽的是受光面，不是遮挡物。

    所以这里做三件事：
      1. 只在接触线**下方**衰减（受光面在这一侧），上方只留 ~1px 的抗锯齿过渡；
      2. 横向范围从她的轮廓向两侧各外扩 `spread_ratio * t`，让暗带"溢"到座面上；
      3. 最后把（膨胀 1px 的）角色掩码整个清零 —— 她身上一个暗像素都不留。

    ## 为什么跟着轮廓走，而不是画一条直线

    她接触面的形状取决于姿态：站着是两小块鞋底、坐着是臀底一长条、
    跳起来是零（调用方会跳过）。用她自己的掩码在接触线附近的**水平占据范围**
    来生成暗带，站/坐/蹲都能自动对上，不需要为每种姿态写参数。

    `mask` 必须是角色掩码（同尺寸），`height_px` 用她的高度来定暗带厚度，
    这样不同景别下暗带的视觉粗细一致。

    ## mask 的精度直接决定效果

    照片路线只能用 `character_mask`（蓝色主导 ∩ 差分）这个启发式，
    她的白肚皮、白鞋底不在里面 —— 那些像素会被当成背景压暗。
    立绘路线不存在这个问题：立绘自带 alpha，传进来的就是精确轮廓（见
    `app/pipeline.py::run_sprite`）。

    `protect` 就是给照片路线兜底的：它**只用于清零**，不用来算横向范围。
    传一个"宁大勿小"的掩码进来（照片路线传 `change_mask`，是"她 + 她被画出的
    影子"的连通域），暗带最多画不出来，绝不会画到她身上。
    范围仍由精确的 `mask` 决定，所以暗带的位置不会因为兜底而变糊。
    """
    import numpy as np

    if mask is None:
        return None
    W, H = size
    cy = float(contact_y)
    t = max(2.0, float(height_px) * thickness_ratio)
    mask = mask.astype(bool)

    # --- 1. 找她在接触线附近的**占地轮廓** ---
    # 搜索带**偏向上**：照片路线的掩码底边常比真实接触线高出一截
    # （她白色鞋底不是蓝色主导，被 character_mask 漏掉，实测差 20px）。
    # 上下对称的带子会一个像素都搜不到，暗核直接消失（静默失败）。
    y0 = max(0, int(cy - 3.0 * t))
    y1 = min(H, int(cy + 1.5 * t) + 1)
    if y1 <= y0:
        return None
    band = mask[y0:y1]
    if band.size == 0 or not band.any():
        return None
    cols = band.any(axis=0)
    if not cols.any():
        return None

    # --- 2. 横向：占地轮廓做一维高斯，得到向两侧**渐隐**的覆盖 ---
    #
    # 第一版这里是 `dark[:, c0:c1] = prof[:, None]`：只在 y 上做了高斯，
    # x 上直接一刀截断 → 暗带其实是一个**硬边矩形**。
    # 座面路线看不出来（大半被她自己挡住，只露出十几像素的边），
    # 落到地面姿态就现形：人行道上横着一条硬边暗条，比原来的 bug 还显眼。
    #
    # 正确做法是把"她占了哪些列"当成一条一维信号做高斯模糊：
    # 正下方 ≈1，两侧平滑衰减到 0，宽度由 `spread_ratio` 控制。
    # 左右不再有硬边，也不用做二维距离变换。
    pad = max(2.0, t * spread_ratio)
    r = max(1, int(round(3.0 * pad)))
    xs = np.arange(-r, r + 1, dtype=np.float32)
    kern = np.exp(-0.5 * (xs / pad) ** 2).astype(np.float32)
    kern /= kern.sum()
    hprof = np.convolve(np.pad(cols.astype(np.float32), r, mode="constant"),
                        kern, mode="valid")
    hprof = np.clip(hprof, 0.0, 1.0)

    # --- 3. 纵向：接触线上方极窄，下方渐隐 ---
    ys = np.arange(H, dtype=np.float32)
    dy = ys - cy
    # 上方 sigma=0.35t（只为消除硬接缝），下方 sigma=1.0t（受光面上真实存在一段暗部）
    sigma = np.where(dy < 0.0, 0.35 * t, 1.0 * t).astype(np.float32)
    prof = np.exp(-0.5 * (dy / np.maximum(sigma, 1.0)) ** 2).astype(np.float32)
    prof[dy < -1.0 * t] = 0.0        # 再往上就不是接缝了，压暗空气会变成一条横线
    prof[dy > 3.5 * t] = 0.0         # 远端截断，免得留一层看不见但确实存在的灰
    prof *= float(strength)

    ylo = max(0, int(cy - 1.5 * t))
    yhi = min(H, int(cy + 3.5 * t) + 1)
    if yhi <= ylo:
        return None
    dark = np.zeros((H, W), dtype=np.float32)
    dark[ylo:yhi] = prof[ylo:yhi, None] * hprof[None, :]

    # --- 4. 她身上不留暗带 ---
    keep_out = mask if protect is None else (mask | protect.astype(bool))
    dark[_dilate1(keep_out)] = 0.0

    if float(dark.max()) <= 0.01:
        return None
    return dark


def shadow_under(result: Image.Image, base: Image.Image,
                 alpha: "np.ndarray", dark: "np.ndarray") -> Image.Image:
    """把暗化图 `dark` 施加在**角色图层之下**，返回新图。

    ## 为什么需要这个函数

    `render_shadow_matte` 的契约写得很清楚："角色应贴在该层之上（角色遮挡自己的影子）"。
    但立绘路线的实际顺序是**先贴她、再画阴影**，阴影直接盖在她身上——
    代码违反了它自己的契约。

    实测（"开/关阴影"两次成品差分，只看她的 alpha 内部）：

    | 场景 | 她身上被压暗（接触带内均值） | 受影响像素占比 |
    |---|---|---|
    | 蹲姿·地面 | 14.6 灰阶（最大 158） | 16.3% |
    | 坐长椅 | 13.4 灰阶（最大 72） | 31.8% |
    | 坐桌沿 | 27.1 灰阶（最大 143） | **48.7%** |

    桌沿那张接近一半的身体被 AO 椭圆压暗了。物理上说不通：
    被遮住的是**地面/座面**，不是她自己。

    ## 怎么做到"在下面"而不需要重排整条链路

    设 `B` = 干净底图，`a` = 她的覆盖度，`H` = 她（已调过色/打过光）的颜色：

        result  = B·(1-a) + H·a          ← 现在手上这张
        目标    = B·(1-dark)·(1-a) + H·a  ← 阴影只落在背景那一半

    两式相减：

        目标 = result − B·(1-a)·dark

    也就是**从她没盖住的那部分里减掉阴影**，她盖住的部分一点不动。
    逐像素看都对得上：
      a=0（纯背景）→ B − B·dark = B·(1−dark) ✓
      a=1（她本人）→ result 不变 ✓
      a=0.5（边缘）→ B·0.5·(1−dark) + H·0.5 ✓

    这样不用把"贴人 → 调色 → 打光 → 画影"重排成"画影 → 贴人 → 调色 → 打光"，
    也就不会碰到 `match_ambient` 拿"她外圈"当背景基准这件事
    （那个圈如果先被阴影压暗，会把她也一起拉黑）。
    """
    import numpy as np

    if dark is None or alpha is None:
        return result
    nz = dark > 0.002
    if not nz.any():
        return result
    ys, xs = np.nonzero(nz)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    arr = np.asarray(result.convert("RGB"), dtype=np.float32)
    b = np.asarray(base.convert("RGB"), dtype=np.float32)
    occ = (1.0 - alpha[y0:y1, x0:x1])[..., None] * dark[y0:y1, x0:x1, None]
    arr[y0:y1, x0:x1] = np.clip(arr[y0:y1, x0:x1] - b[y0:y1, x0:x1] * occ, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def occlude(result: Image.Image, base: Image.Image, fg: "np.ndarray") -> Image.Image:
    """把前景（人物）重新盖回到成品上 —— 也就是「她被前景挡住」。

    ## 为什么这一层能改变产品形态

    立绘路线原来只有一种位置：**人旁边的空地**，而且她必须完整可见。
    这条约束推出一个错误结论——"等身角色在竖构自拍里放不下"
    （实测 5/5 场景要缩到 21%~72%）。

    但只要允许被挡住，她需要的就不是 700px 空地，而是**一条能露出来的缝**。
    "从身后探出半个身子"、"勾肩搭背" 都是这个原理。

    ## 实现

    `result` 是已经贴过她、调过色、打过光的成品，`base` 是原照片，
    `fg` 是人物前景的软 alpha。

        out = result·(1-fg) + base·fg

    前景`fg=1`的地方原样还原照片（她被挡住），`fg=0`的地方保留成品，
    边界由软 alpha 羽化。放在**最后一步**做，因为：

    - 她的接触阴影不该压到前面的人身上（物理上影子落在她自己的地面）；
    - 重打光只作用在她自己的 alpha 内，被挡住的部分白算，但代价可忽略。
    """
    import numpy as np

    if fg is None:
        return result
    a = fg[..., None]
    r = np.asarray(result.convert("RGB"), dtype=np.float32)
    b = np.asarray(base.convert("RGB"), dtype=np.float32)
    out = r * (1.0 - a) + b * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def silhouette_edge(fg: "np.ndarray", y0: int, y1: int,
                    thr: float = 0.5) -> tuple[int, int] | None:
    """前景在 [y0, y1) 这段纵向带里的左右轮廓 x（取逐行中位数）。

    为什么不直接用人物框：框是矩形的，而她要在**肩线/腰线那一带的真实轮廓**上收边。
    两者差得很远——框的左沿通常在她手臂外侧，而那一段轮廓其实凹进去。

    取中位数而不是最小/最大值：一段带子里可能有手臂、发梢单独伸出去，
    用极值会被那一小撮拉走。
    """
    import numpy as np

    h = fg.shape[0]
    y0 = max(0, min(int(y0), h - 1))
    y1 = max(y0 + 1, min(int(y1), h))
    band = fg[y0:y1] > thr
    lefts, rights = [], []
    for row in band:
        idx = np.flatnonzero(row)
        if idx.size:
            lefts.append(idx[0])
            rights.append(idx[-1])
    if len(lefts) < 5:
        return None
    return int(np.median(lefts)), int(np.median(rights))


def render_shadow_matte(size: tuple[int, int], box: tuple[float, ...],
                        light: LightingState,
                        fov_compress: float = 1.0,
                        matte_scale: float = 0.3,
                        contact_y: float | None = None,
                        reach: float = 1.0) -> np.ndarray:
    """返回 (H, W) 的暗化系数图 dark，取值 [0,1]。

    合成方式：scene = scene * (1 - dark)。角色应贴在该层之上（角色遮挡自己的影子）。
    **这一条现在由 `shadow_under` 强制执行**——在此之前，立绘路线是"先贴她再画阴影"，
    阴影直接盖在她身上（桌沿那张实测 48.7% 的身体被 AO 染黑）。

    性能说明（实测修正 06 号文档的 "<5ms" 估计）：
    直接在全画幅上做 5 层高斯（半径最大 0.25w ≈ 47px）实测 118ms，达不到目标。
    阴影本身是低频信号，因此改为**降采样渲染 + 双线性上采样**（matte_scale=0.3），
    几何全部按比例缩放，实测降到 20ms 量级。若还要更快，可再叠加 ROI 裁剪
    （只渲染影子包围盒），但当前量级已够原型用。
    """
    W, H = size
    s = max(0.1, min(1.0, matte_scale))
    ws, hs = max(1, int(W * s)), max(1, int(H * s))

    w = max((box[2] - box[0]) * s, 1.0)     # 角色底边宽（缩放后）
    cx = (box[0] + box[2]) / 2.0 * s
    # 接触点：站姿就是框底边；坐/趴姿的接触面（坐垫、椅面、桌面）在框底边之上，
    # 由调用方用 contact_y 传入。用框底边当接触点是错的——影子会落到地上，
    # 而人其实坐在垫子上。这是首轮真实照片验证暴露的问题。
    cy_orig = (box[3] if contact_y is None else contact_y)
    cy = cy_orig * s
    # 接触面以下还剩多少"她的身体"：站姿≈0，坐姿≈0.38。
    # 这部分会把 AO 椭圆挡住，所以要按它放宽 AO（见 AO_SPILL_K 的注释）。
    hidden = max(0.0, min(1.0, (box[3] - cy_orig) / max(box[3] - box[1], 1.0)))

    tier = light.shadow_tier()
    layers: list[tuple[Image.Image, float]] = []

    # ---- AO 层（永远渲染，不依赖光向，置信度再低也安全） ----
    a = (AO_A + AO_SPILL_K * min(hidden / AO_SPILL_REF, 1.0)) * w
    b = AO_B_RATIO * fov_compress * a
    ao = Image.new("L", (ws, hs), 0)
    d = ImageDraw.Draw(ao)
    d.ellipse([cx - a, cy - b * 0.9, cx + a, cy + b * 1.1], fill=255)
    ao = ao.filter(ImageFilter.GaussianBlur(radius=float(max(AO_SIGMA * w, 0.8))))
    layers.append((ao, AO_OPACITY))

    if tier != "ao_only":
        segs = _SEG_HARD if light.quality_hardness > 0.85 else _SEGMENTS
        if tier == "ao_near":
            segs = segs[:1]  # 只保留近段
        ang = light.shadow_angle_deg()
        for i, (end_len, sig, op) in enumerate(segs):
            # reach 放大投射距离：晴天低角度阳光的影子远长于室内散射光
            L = end_len * w * reach
            t = _SEG_THICK[min(i, len(_SEG_THICK) - 1)] * w
            r = t / 2.0
            # 轴向右的胶囊：两端圆 + 中间矩形，随后绕接触点旋转到目标方向
            seg = Image.new("L", (ws, hs), 0)
            ds = ImageDraw.Draw(seg)
            ds.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            ds.ellipse([cx + L - r, cy - r, cx + L + r, cy + r], fill=255)
            ds.rectangle([cx, cy - r, cx + L, cy + r], fill=255)
            # 顺时针 ang 度（画面坐标 y 向下）：PIL 逆时针为正，故取 -ang
            seg = seg.rotate(-ang, center=(cx, cy), resample=Image.BILINEAR)
            seg = seg.filter(ImageFilter.GaussianBlur(radius=float(max(sig * w, 0.8))))
            layers.append((seg, op))

    dark = np.zeros((hs, ws), dtype=np.float32)
    for matte, op in layers:
        m = np.asarray(matte, dtype=np.float32) / 255.0
        dark = 1.0 - (1.0 - dark) * (1.0 - op * light.intensity * m)
    dark = np.clip(dark, 0.0, 0.92)  # 封顶，避免纯黑

    if (ws, hs) != (W, H):
        up = Image.fromarray((dark * 255).astype(np.uint8), "L").resize((W, H), Image.BILINEAR)
        dark = np.asarray(up, dtype=np.float32) / 255.0
    return dark


# ---------------------------------------------------------------------------
# S3.1 重力对齐（06 §4）
# ---------------------------------------------------------------------------

def apply_gravity(char_layer: Image.Image, box: tuple[float, ...],
                  gravity_deg: float) -> Image.Image:
    """把整幅角色层绕锚点（框底边中点）旋转 -gravity_deg。

    角色的竖直轴应对齐照片重力方向而非图像竖直轴（三视图是正交正面图）。
    一次矩阵旋转，零推理成本。符号约定待 U-09 校准——若方向反了改一处符号即可。

    性能：只在包围盒 ROI 内旋转（外扩量按旋转后外接框膨胀计算），再贴回整幅层。
    全画幅 RGBA 旋转实测约 56ms，ROI 化后降到个位数毫秒。
    """
    if abs(gravity_deg) < 0.05:
        return char_layer
    w_img, h_img = char_layer.size
    x1, y1, x2, y2 = box
    # 旋转后外接框膨胀：|w·sinθ| + |h·cosθ| 方向上的增量，取保守界
    th = abs(gravity_deg) * 3.14159265 / 180.0
    bw, bh = x2 - x1, y2 - y1
    margin = int(abs(bw * np.sin(th)) + abs(bh * (1 - np.cos(th)))) + 4
    rx1 = max(0, int(x1) - margin)
    ry1 = max(0, int(y1) - margin)
    rx2 = min(w_img, int(x2) + margin)
    ry2 = min(h_img, int(y2) + margin)

    # 锚点换算到 ROI 局部坐标；旋转在 ROI 内以局部坐标为中心进行
    anchor_local = ((x1 + x2) / 2.0 - rx1, y2 - ry1)
    crop = char_layer.crop((rx1, ry1, rx2, ry2))
    rot = crop.rotate(-gravity_deg, center=anchor_local,
                      resample=Image.BICUBIC, expand=False)

    layer = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
    layer.paste(rot, (rx1, ry1))
    return layer


# ---------------------------------------------------------------------------
# S3.2 深度匹配散焦（06 §3）
# ---------------------------------------------------------------------------

def apply_defocus(char_layer: Image.Image, box: tuple[float, ...],
                  focus_y: float, strength: float = 10.0,
                  bands: int = 4) -> Image.Image:
    """按深度做空间变化的散焦（CoC 单调近似）。

    深度代理：地面场景里 y 越大越近。角色所在深度离焦点（focus_y，画面像素）
    越远 blur 越大，满足单调性即可，不必精确算光学公式（06 §3）。
    实现为横向条带分层 blur + 线性羽化过渡。

    性能：只在角色包围盒 ROI 内做 blur（外扩 3 倍最大 sigma），再贴回整幅透明层。
    全画幅做 4 次 RGBA 高斯实测 290ms 量级，ROI 化后降一个数量级。
    """
    w_img, h_img = char_layer.size
    y1, y2 = int(box[1]), int(box[3])
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(y1 + 1, min(h_img, y2))

    # 每条带的中心 y 与 sigma（必须转 Python float：PIL 不接受 numpy 标量）
    edges = np.linspace(y1, y2, bands + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    sigmas = [float(max(0.0, strength * abs(float(cy_) - focus_y) / h_img))
              for cy_ in centers]
    max_sig = max(sigmas) if sigmas else 0.0
    if max_sig < 0.25:
        # 角色基本在焦平面上，散焦不可见，直接返回（省一次全图拷贝）
        return char_layer

    pad = int(max_sig * 3) + 2
    rx1 = max(0, int(box[0]) - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w_img, int(box[2]) + pad)
    ry2 = min(h_img, y2 + pad)
    crop = char_layer.crop((rx1, ry1, rx2, ry2))
    cw, chh = crop.size
    yy = np.arange(chh, dtype=np.float32) + ry1

    out = np.zeros((chh, cw, 4), dtype=np.float32)
    weight = np.zeros((chh, cw), dtype=np.float32)
    for i, sig in enumerate(sigmas):
        blurred = crop if sig < 0.25 else crop.filter(ImageFilter.GaussianBlur(radius=sig))
        top, bot = edges[i], edges[i + 1]
        ramp = np.clip(np.minimum(yy - top, bot - yy) / max((bot - top) / 2, 1), 0, 1)
        ramp = 0.5 + 0.5 * ramp
        m = ((yy >= top) & (yy <= bot)).astype(np.float32) * ramp
        arr = np.asarray(blurred, dtype=np.float32)
        out += arr * m[:, None, None]
        weight += m[:, None]
    weight = np.maximum(weight, 1e-6)
    out /= weight[..., None]

    layer = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
    layer.paste(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA"), (rx1, ry1))
    return layer
