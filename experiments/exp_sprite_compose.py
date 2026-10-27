"""白底立绘 -> 抠图 -> 按目标尺寸/位置合成到真实照片。

抠图判据：**近白 且 与画面边界连通** 的像素才是背景。
她的白围裙、白袜、白荷叶边都被深色描边围住，不与边界连通，所以不会被误删。
这比"在真实照片上判断哪些像素是她"可靠一个量级。
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from api.ark import _post  # noqa: F401  仅为保持导入一致性
from app import pipeline as P
from s3.beautify import character_mask
from s3.rescale import _bbox


def _flood_from_border(free, max_iter=600):
    """在 free 区域内，从画面四边向内泛洪。

    不用 PIL 的 ImageDraw.floodfill：它在 1024x1536 这种尺寸上会**静默失效**
    （同样的调用在 10x10 小图上正常，大图上返回时一个像素都没改）。
    这里用 numpy 迭代膨胀，一遍 4 次数组运算，收敛即停。
    """
    h, w = free.shape
    seed = np.zeros_like(free)
    seed[0, :] = free[0, :]
    seed[-1, :] = free[-1, :]
    seed[:, 0] |= free[:, 0]
    seed[:, -1] |= free[:, -1]
    out = seed.copy()
    for _ in range(max_iter):
        prev = out
        nxt = out.copy()
        nxt[1:, :] |= prev[:-1, :]
        nxt[:-1, :] |= prev[1:, :]
        nxt[:, 1:] |= prev[:, :-1]
        nxt[:, :-1] |= prev[:, 1:]
        nxt &= free
        if nxt.sum() == prev.sum():
            return nxt
        out = nxt
    return out


def key_white(img, thr=232, feather=1.2, work=420):
    """从纯白背景里抠出角色，返回 RGBA。

    判据：**近白 且 与画面边界连通** 的像素才是背景。
    她的白围裙、白袜、白荷叶边都被深色描边围住，不与边界连通，所以不会被误删。
    这比"在真实照片上判断哪些像素是她"可靠一个量级——
    实测内部空隙 0.00%（对比：在真实照片上抠，最好也有 5~7%）。
    """
    rgb = np.asarray(img.convert('RGB'), dtype=np.float32)
    h, w = rgb.shape[:2]
    near_white = rgb.min(axis=2) > thr

    # 降采样做泛洪（快），再把结果放大回原尺寸
    sw = work
    sh = max(1, int(h * sw / w))
    small = np.asarray(Image.fromarray((near_white * 255).astype('uint8'), 'L')
                       .resize((sw, sh), Image.BILINEAR)) > 127
    bg_small = _flood_from_border(small)
    bg = np.asarray(Image.fromarray((bg_small * 255).astype('uint8'), 'L')
                    .resize((w, h), Image.BILINEAR)) > 64
    # 只认"近白"范围内的背景，避免把角色边缘也啃掉
    bg &= near_white

    alpha = np.where(bg, 0, 255).astype('uint8')
    a_img = Image.fromarray(alpha, 'L').filter(ImageFilter.GaussianBlur(feather))
    out = img.convert('RGBA')
    out.putalpha(a_img)
    return out


def main():
    sprite = Image.open('output/step8/sprite_crouch.png').convert('RGB')
    rgba = key_white(sprite)
    A = np.asarray(rgba)[..., 3]
    solid = A >= 250
    holes = 0
    for y in range(solid.shape[0]):
        xs = np.flatnonzero(solid[y])
        if xs.size >= 2:
            holes += (xs[-1] - xs[0] + 1) - xs.size
    print(f'alpha: 不透明 {int(solid.sum())}  内部空隙 {holes / max(solid.sum(),1) * 100:.2f}%')

    # 她的包围盒（按 alpha 裁）
    bb = _bbox(np.asarray(rgba)[..., 3] > 40)
    print('sprite bbox', bb, '高', bb[3] - bb[1])
    cut = rgba.crop(bb)

    # 目标位置
    photo_path = 'output/runs/16afa9bcc5a6/scene.jpg'
    prof = P.vlm_profile(photo_path)
    photo = Image.open(photo_path).convert('RGB')
    W, H = photo.size
    person_h = prof.person_h_px(H)
    foot_y = prof.foot_y_px(H)
    lb = prof.legs
    leg_cx = (lb[0] + lb[2]) / 2 / 999 * W
    leg_w = (lb[2] - lb[0]) / 999 * W
    target_h = person_h * 0.26
    target_w = cut.width * target_h / cut.height
    cx = leg_cx - (leg_w * 0.5 + target_w * 0.5 + 6)
    print(f'target h={target_h:.0f} w={target_w:.0f} cx={cx:.0f} bottom={foot_y:.0f}')

    s = cut.resize((int(target_w), int(target_h)), Image.LANCZOS)
    canvas = photo.copy()
    px, py = int(cx - target_w / 2), int(foot_y - target_h)
    canvas.paste(s, (px, py), s)

    # 接触阴影（复用现有渲染器）
    from s3.lighting import LightingState
    from s3.s3ops import contact_core, render_shadow_matte
    ls = LightingState(light_azimuth_deg=180.0, quality_hardness=0.78,
                       intensity=0.82, confidence=0.88)
    box = (px, py, px + int(target_w), py + int(target_h))
    dark = render_shadow_matte(canvas.size, box, ls, contact_y=foot_y, reach=1.9)
    arr = np.asarray(canvas, dtype=np.float32) * (1 - dark[..., None])
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype('uint8'), 'RGB')
    core = contact_core(canvas.size, np.asarray(s.resize((canvas.width, canvas.height), Image.LANCZOS).split()[3]) > 128
                        if False else None, foot_y, target_h) if False else None
    canvas.save('output/step8/sprite_composite.jpg', quality=95)
    print('saved output/step8/sprite_composite.jpg')


if __name__ == '__main__':
    main()
