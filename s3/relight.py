"""S3.4 重打光：把中性棚拍光的立绘，按场景主光方向补一层明暗。

## 为什么需要这一步

立绘是在纯白背景、中性棚拍光下生成的。它自带一套"从正前方打光"的明暗，
贴到一张强侧光的照片上就会**穿帮成贴纸**——不是因为她画得不好，
而是因为她的明暗和照片的明暗不是同一个光源。

`match_ambient` 解决不了这个：它只做**直流分量平移**（把她的整体色偏往背景拉），
不改变任何空间分布。实测位移只有 [-1.7, -8.3, -8.3]，也就是说
"调完之后她还是平的"。要补的是**梯度**，不是**偏移**。

## 模型

只做一件事：沿主光方向叠一层线性明暗。

    gain(p) = 1 + amp * t(p)          t ∈ [-1, 1] 沿光方向归一化的位置

三个必须做对的地方：

1. **保能量**。直接乘会让她整体变暗或变亮（一侧 +amp 另一侧 -amp 平均是 1，
   但她的轮廓在 t 上不对称，实际均值会偏）。所以按她自己的像素把 gain 的
   加权均值归一到 1 —— 只改分布，不改总亮度。

2. **按 alpha 羽化**。她边缘的抗锯齿像素不该和内部一样被乘。
   直接 `arr * gain` 会在轮廓上留下一道亮度台阶。

3. **色温也要跟着方向走**。真实光源有颜色：受光面偏光源色、背光面偏环境色。
   这里不拍常数，而是**从照片自己身上量**——把背景像素按亮度分位数切成
   "亮的一撮"和"暗的一撮"，两组的色差就是这张照片光/影的色温差。
   量出来是暖的就往暖走，是冷的就往冷走。

## 什么时候不做

`light_azimuth_deg` 为 None（auto 档没判出方向）时**直接放弃**。
一个方向错的梯度比没有梯度更糟——这和人脸打光反了会立刻被察觉是一个道理。
置信度低于 0.45 也不做，理由相同。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# 色温偏移的封顶。超过 ±0.06 就会把深蓝女仆裙染成紫的。
_TINT_MAX = 0.06

# 量色温时空采样步长：2M 像素全量排序要 180ms，跳着取 1/9 结果不变但快一个量级
_TINT_STRIDE = 3


def light_vector(azimuth_deg: float) -> tuple[float, float]:
    """主光方向单位向量（画面坐标，x 向右、y 向下）。

    约定来自 `LightingState`：0=右 90=下 180=左 270=上，语义是"光**来自**何方"。
    所以 0 → (1,0)：光来自右边；270 → (0,-1)：光来自上方（y 轴向下，故为负）。
    """
    th = np.radians(float(azimuth_deg) % 360.0)
    return float(np.cos(th)), float(np.sin(th))


def side_light(photo: Image.Image, work: int = 128,
               min_contrast: float = 0.03) -> dict | None:
    """量这张照片"右边比左边亮多少"，据此判断侧光来向。

    ## 为什么不信 VLM 的 left/right

    实测在 `s04_full`（秋季公园全身照）上，VLM 回答 `light=left`，
    但画面里树影和她的投影都朝左倒、牛仔裤和球鞋也都是右侧更亮——
    光是从**右边**来的。方向判反了。
    而同一张图的横向亮度差是 +7.3%（向右变亮），和实拍一致。

    方向错了比不打光更糟（这一点在模块开头已经说过），所以方向改由画面决定。
    VLM 仍然有用：它给的 `quality_hardness` 决定梯度幅度，那是个不会"反号"的量。

    ## 为什么只用横向，而且用整幅画面

    **只用横向**：纵向亮度差里内容占比压倒性。实测整幅平面拟合的纵向斜率
    在长椅那张是 +63.4%（上方草地暗、下方水泥地亮），公园那张是 −78.5%
    （上方天空亮、下方深色衣服暗）——那不是光，是"画面里什么东西在哪儿"。

    **用整幅而不是她周围一圈**：她身边那圈常常站着人、摆着家具，
    局部窗口的均值会被单个物体带跑。整幅取列中位数更稳。

    ## 抗噪

    1. 降采样 + 高斯模糊，抹掉纹理；
    2. 每列先取**中位数**再求均值（不是逐像素平均），
       这样天上那块高光斑、地上那块死黑都压不动结果；
    3. 只比较中间 86% 的列宽，避开镜头暗角；
    4. 对比度低于 `min_contrast` 判"没有侧光"——阴天、顶光、室内散射都走这条。
    """
    small = photo.convert("RGB").resize(
        (work, max(8, int(round(photo.height * work / photo.width)))),
        Image.BILINEAR).filter(ImageFilter.GaussianBlur(2))
    lum = np.asarray(small, dtype=np.float32) @ np.array([0.299, 0.587, 0.114],
                                                        dtype=np.float32)
    col = np.median(lum, axis=0)                      # 每列中位数，压掉孤立高光/死黑
    n = col.size
    a, b = int(n * 0.07), int(n * 0.93)               # 去掉两端各 7%，避暗角
    col = col[a:b]
    k = max(1, col.size // 3)
    left, right = float(col[:k].mean()), float(col[-k:].mean())
    denom = max(left + right, 1.0)
    contrast = (right - left) / denom
    if abs(contrast) < min_contrast:
        return None
    return {
        # 右边更亮 → 光从右边来 → azimuth 0；反之 180
        "azimuth": 0.0 if contrast > 0 else 180.0,
        "contrast": round(float(contrast), 4),
        "left": round(left, 1), "right": round(right, 1),
    }


def amp_for(hardness: float, scene_contrast: float = 0.12) -> float:
    """该补多强的梯度。

    两个因子相乘，各自有物理含义：

    - `hardness`（光质）：硬光的方向性强，明暗过渡陡。用 `quality_hardness`
      而不是 `intensity` 当自变量——intensity 是"影子多黑"，hardness 才是
      "方向性多强"，重打光要的正是方向性。基准 0.10 + 0.16×hardness。

    - `scene_contrast`（证据强度）：这张照片本身左右亮度差多少。
      实测四个场景差了近十倍（长椅 7%、餐桌 34%），只按 hardness 给会让
      弱侧光的场景被打上一道不该有的斜坡。这里把它当成**置信度**用：
      证据越弱，打得越轻（下限 0.35，不是归零——太弱就别打光，那由
      `side_light` 的 `min_contrast` 阈值决定）。

    注意不能直接把 `scene_contrast` 当幅度：那是**整幅画面**的左右亮度差，
    而她的宽度只占画面一成多，按比例缩下去等于不打光。真实侧光在她身上的
    明暗来自**曲面法线**随光向变化，不是一层横跨画面的线性斜坡。
    """
    base = float(np.clip(0.10 + 0.16 * float(hardness), 0.08, 0.24))
    evidence = float(np.clip(abs(scene_contrast) / 0.12, 0.35, 1.0))
    return base * evidence


def scene_tint(photo: Image.Image, alpha: np.ndarray,
               q: float = 0.12) -> np.ndarray:
    """从照片背景量出"光 vs 影"的色温差，返回逐通道 tint（已封顶）。

    做法：背景像素按亮度排序，取最亮的 q 和最暗的 q，两组平均色相减。
    这不是在猜光源颜色，而是在读这张照片**已经呈现出来**的结果——
    阴天量出来接近 0（没有色温差），夕阳量出来明显偏暖，都是照片自己说的。
    """
    arr = np.asarray(photo.convert("RGB"), dtype=np.float32)
    bg = alpha < 0.08
    if int(bg.sum()) < 500:
        return np.zeros(3, dtype=np.float32)
    # 跳采样：这里要的是分位数，不是逐像素统计，1/9 的点足够
    sub = arr[::_TINT_STRIDE, ::_TINT_STRIDE]
    bgs = bg[::_TINT_STRIDE, ::_TINT_STRIDE]
    px = sub[bgs]
    if px.shape[0] < 200:
        return np.zeros(3, dtype=np.float32)
    lum = px @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lo_th, hi_th = np.quantile(lum, q), np.quantile(lum, 1.0 - q)
    dark_m, bright_m = lum <= lo_th, lum >= hi_th
    if dark_m.sum() < 20 or bright_m.sum() < 20:
        return np.zeros(3, dtype=np.float32)
    dark = px[dark_m].mean(axis=0)
    bright = px[bright_m].mean(axis=0)
    # 除以亮度差，得到"每单位亮度对应的色偏"，与整体曝光无关
    dl = max(float(lum[bright_m].mean() - lum[dark_m].mean()), 8.0)
    tint = (bright - dark) / dl * 40.0
    # 去掉共同的灰度分量，只保留"相对色偏"
    tint = tint - tint.mean()
    return np.clip(tint, -_TINT_MAX * 255, _TINT_MAX * 255).astype(np.float32) / 255.0


def relight(step: Image.Image, alpha: np.ndarray,
            azimuth_deg: float,
            tint: np.ndarray | None = None,
            amp: float | None = None) -> tuple[Image.Image, dict]:
    """按主光方向给角色补一层明暗。`alpha` 是 (H,W) 的 [0,1] 浮点覆盖度。

    只改角色像素；alpha 之外的像素一个都不动（乘的是 alpha 本身）。
    """
    if alpha is None or alpha.max() <= 0.01:
        return step, {"applied": False, "reason": "no_alpha"}

    H, W = alpha.shape
    ys, xs = np.nonzero(alpha > 0.05)
    if xs.size < 100:
        return step, {"applied": False, "reason": "too_small"}

    if amp is None:
        amp = amp_for(0.55, 0.12)

    # 她自己的包围盒（不是画面），t 在这个盒子里归一化到 [-1,1]
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    hw = max((xs.max() - xs.min()) / 2.0, 1.0)
    hh = max((ys.max() - ys.min()) / 2.0, 1.0)

    # 只算她那一块。alpha 之外本来就乘 0，全画幅跑一遍纯属浪费——
    # 实测 1280×1704 全画幅 345ms，裁到 ROI 后降到几十毫秒。
    pad = 4
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad + 1)

    ux, uy = light_vector(azimuth_deg)
    ax = (np.arange(x0, x1, dtype=np.float32) - cx) / hw
    ay = (np.arange(y0, y1, dtype=np.float32) - cy) / hh
    t = np.clip(ux * ax[None, :] + uy * ay[:, None], -1.0, 1.0).astype(np.float32)

    # 逐通道增益：亮度项 + 色温项
    if tint is not None:
        tc = (amp * t)[..., None] + t[..., None] * tint[None, None, :]
    else:
        tc = np.repeat((amp * t)[..., None], 3, axis=2).astype(np.float32)

    # 保能量：按她自己的像素（alpha 加权）把平均增益拉回 1
    #
    # 只改分布、不改总亮度。少了这一步，长椅那张实测她整体亮了 6%——
    # 因为她的轮廓在 t 上不对称（尾巴甩在背光侧，把"暗"的那半边撑大了）。
    sub_a = alpha[y0:y1, x0:x1]
    num = np.einsum("hwc,hw->c", 1.0 + tc, sub_a)
    mean_gain = num / max(float(sub_a.sum()), 1e-6)
    gain = 1.0 + tc - (mean_gain - 1.0)[None, None, :]

    arr = np.asarray(step.convert("RGB"), dtype=np.float32).copy()
    sub = arr[y0:y1, x0:x1]
    # 只作用在 alpha 之内，且边缘按 alpha 羽化 → 不留亮度台阶
    blended = 1.0 + (gain - 1.0) * sub_a[..., None]
    arr[y0:y1, x0:x1] = np.clip(sub * blended, 0, 255)

    info = {
        "applied": True,
        "azimuth": round(float(azimuth_deg), 1),
        "amp": round(amp, 3),
        "tint": [round(float(v), 4) for v in (tint if tint is not None else np.zeros(3))],
        # 受光侧 / 背光侧的增益，验证时直接看这两个数
        "gain_light_side": round(float(gain[t > 0.5].mean()), 3) if (t > 0.5).any() else 1.0,
        "gain_dark_side": round(float(gain[t < -0.5].mean()), 3) if (t < -0.5).any() else 1.0,
    }
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB"), info
