"""角色立绘（sprite）：脱离场景单独生成角色，程序化合成。

## 为什么这条路比"在照片上抠图"对

原来是在**真实照片**上判断"哪些像素是她"。这个判断在两个颜色相近的地方必然出错：

    - 深藏青袖子压在蓝色牛仔裤上（两边都暗、都偏蓝）
    - 白围裙压在亮地面上（两边都亮）
    - 蓝色头发压在阴影里

出错的表现就是**透明**（背景透出来）和**缺块**（连边界一起切掉）。
实测在真实照片上抠，内部空隙最好也有 5~7%。

改成"先在纯白背景上生成角色立绘，再抠图"，判断退化成：

    **近白 且 与画面边界连通** 的像素才是背景

她的白围裙、白袜、白荷叶边都被深色描边围住，**不与边界连通**，
所以不会被误删。实测内部空隙 **1.8%**（且主要来自描边内的暗部，不是洞）。

## 顺带解决三件事

1. **尺寸**：直接 resize，无损、可控。不再需要"用人物身高推算目标高度再缩放"
   这种间接控制。
2. **位置**：完全由程序计算，模型不参与。
3. **复用**：立绘**与照片无关**——同一个姿态只需要生成一次，
   之后可以贴到任意照片上。成本从"每张照片一次生成"降到"每个姿态一次生成"。

## 代价

- 立绘的**光照是通用的**，不来自照片。需要靠环境色匹配 + 程序化阴影来贴合。
- 没有**遮挡关系**：她不能被人物的手臂挡住一半。
  对"站在/坐在旁边"这类动作无所谓，对"扒着衣摆探头"这类就不行。
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPRITE_DIR = os.path.join(ROOT, "assets", "sprites")


def build_sprite_prompt(action_desc: str, character=None) -> str:
    """立绘提示词。姿态描述直接复用动作库里的 desc，不另写一套。

    三条硬性要求都是为抠图服务的：
    - 纯白背景：否则背景不可预测
    - **轮廓闭合**：外面要有一圈连续深色描边。这是白底抠图成立的前提——
      她的白围裙、白袜靠这圈描边与背景隔开，才不会被一起抠掉
    - 完整入画：任何部位被裁切都无法重建

    `character` 不传时按 Q 版处理（老调用点兼容）；`identity` / `body`
    都从角色注册表来——**体型这条最容易出错**：拿"Q 版 2 头身"去生成
    成人设定图的立绘，模型会把她画成大头娃娃。
    """
    if character is None:
        from .characters import get_character
        character = get_character()
    n_ref = len(character.refs)
    ref_word = ("图1" if n_ref == 1
                else "图1、图" + "、图".join(str(i) for i in range(2, n_ref + 1)))
    return (
        f"{ref_word}是同一个二次元角色的设定图。"
        "请画这个角色的**一张全身立绘**。\n"
        f"角色形象：{character.identity}。"
        f"{character.body}。\n\n"
        f"姿态：{action_desc}。\n\n"
        "硬性要求：\n"
        "1. **纯白背景**（#FFFFFF），背景上不要有任何东西——不要地面、不要投影、"
        "不要渐变、不要光晕、不要边框、不要文字、不要参考图的痕迹；\n"
        "2. 角色**完整入画**：头顶和脚底都留出一小圈白边，任何部位都不要被画面裁切；\n"
        "3. 角色**轮廓必须闭合**：外面有一圈连续完整的深色描边，"
        "白围裙和白袜都被描边围在角色内部；\n"
        # 这一条是被 `rail_back` 打出来的：姿态描述里写"双手抓住栏杆"，
        # 模型就把栏杆一起画进去了——贴到照片里会多出一截**假的**灰色栏杆，
        # 而且抠图时那截栏杆会被当成角色的一部分保留下来。
        "4. **画面里只画角色本人**：不要画她扶着、坐着、靠着、抓着的任何东西"
        "（栏杆、椅子、桌面、台阶、枕头都不要出现）。"
        "身体保持相应的姿势，但那些物体本身**不画**；\n"
        "5. 画面里恰好一个角色，居中构图。\n"
    )


def _flood_from_border(free: np.ndarray, max_iter: int = 600) -> np.ndarray:
    """在 free 区域内从画面四边向内泛洪。

    不用 `PIL.ImageDraw.floodfill`：它在 1024x1536 这种尺寸上会**静默失效**
    ——同样的调用在 10x10 小图上正常，大图上返回时一个像素都没改，
    而且不抛异常。用 numpy 迭代膨胀代替，一遍 4 次数组运算，收敛即停。
    """
    h, w = free.shape
    seed = np.zeros_like(free)
    seed[0, :] = free[0, :]
    seed[-1, :] = free[-1, :]
    seed[:, 0] |= free[:, 0]
    seed[:, -1] |= free[:, -1]
    out = seed.copy()
    for _ in range(max_iter):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        nxt &= free
        if nxt.sum() == out.sum():
            return nxt
        out = nxt
    return out


def key_white(img: Image.Image, thr: int = 232, feather: float = 1.2,
              work: int = 420) -> Image.Image:
    """从纯白背景里抠出角色，返回 RGBA。

    `thr` 是"近白"的亮度下限。白背景实测 252~255，取 232 留足余量
    （JPEG 压缩会把边缘的白压到 240 左右）。

    `feather` 只做 1.2 像素的高斯，给边缘一点抗锯齿。不要加大——
    加大等于把描边也吃软，边缘会发虚。
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    near_white = arr.min(axis=2) > thr

    sw = work
    sh = max(1, int(h * sw / w))
    small = np.asarray(
        Image.fromarray((near_white * 255).astype("uint8"), "L")
        .resize((sw, sh), Image.BILINEAR)) > 127
    bg_small = _flood_from_border(small)
    bg = np.asarray(
        Image.fromarray((bg_small * 255).astype("uint8"), "L")
        .resize((w, h), Image.BILINEAR)) > 64
    # 只认"近白"范围内的背景：角色身上的浅色（围裙、袜、皮肤高光）即使
    # 与背景连通，只要不够白就不会被算成背景
    bg &= near_white

    alpha = np.where(bg, 0, 255).astype("uint8")
    a_img = Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(feather))
    out = img.convert("RGBA")
    out.putalpha(a_img)
    return out


def sprite_path(action_id: str, model: str, character=None) -> str:
    """立绘缓存路径。立绘与照片无关，同一个姿态可以跨照片复用。

    路径按角色分目录（`assets/characters/<id>/sprites/`），否则两个角色的
    同名姿态会**互相覆盖**——Q 版的 `leg_hug_knee` 和成人的 `leg_hug_knee`
    文件名一模一样。`chibi` 的目录指回老的 `assets/sprites/`，不搬文件。
    """
    if character is None:
        from .characters import get_character
        character = get_character()
    tag = model.replace("doubao-", "").replace("seedream-", "")
    return os.path.join(character.sprite_dir_or_default(),
                        f"{action_id}__{tag}.png")


def load_sprite(action_id: str, model: str, character=None) -> Image.Image | None:
    p = sprite_path(action_id, model, character)
    if os.path.exists(p):
        return Image.open(p).convert("RGBA")
    return None


def save_sprite(action_id: str, model: str, rgba: Image.Image, character=None) -> str:
    p = sprite_path(action_id, model, character)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    rgba.save(p)
    return p


def trim(rgba: Image.Image, thr: int = 40) -> Image.Image:
    """按 alpha 裁掉四周空白，返回**只含角色**的图。

    这一步很关键：裁掉之后，图的宽高比就是她真实的宽高比，
    后面按目标高度缩放时不会带上无用的留白，宽度也能算准。
    """
    a = np.asarray(rgba)[..., 3]
    ys = np.argwhere(a.max(axis=1) > thr)
    xs = np.argwhere(a.max(axis=0) > thr)
    if ys.size == 0 or xs.size == 0:
        return rgba
    y0, y1 = int(ys[0][0]), int(ys[-1][0]) + 1
    x0, x1 = int(xs[0][0]), int(xs[-1][0]) + 1
    return rgba.crop((x0, y0, x1, y1))


def alpha_hole_ratio(rgba: Image.Image) -> float:
    """诊断用：行内空隙占不透明像素的比例。用来横向比较抠图质量。"""
    solid = np.asarray(rgba)[..., 3] >= 250
    gap = 0
    for y in range(solid.shape[0]):
        xs = np.flatnonzero(solid[y])
        if xs.size >= 2:
            gap += (xs[-1] - xs[0] + 1) - xs.size
    return gap / max(int(solid.sum()), 1)
