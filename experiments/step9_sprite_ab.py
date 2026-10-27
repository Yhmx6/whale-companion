"""对比：在照片上抠图 vs 白底立绘 + 程序化合成。"""
import sys
sys.path.insert(0, '.')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from s3 import sprite as S
from s3.beautify import character_mask
from s3.rescale import _bbox

F = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)

# --- 白底立绘路径 ---
raw = Image.open('output/step8/sprite_crouch.png').convert('RGB')
rgba = S.trim(S.key_white(raw))
print(f'立绘抠图: 内部空隙 {S.alpha_hole_ratio(rgba) * 100:.2f}%  尺寸 {rgba.size}')

# --- 照片抠图路径（同一姿态）---
d = '17df8775497e'
pr = Image.open(f'output/runs/{d}/raw_0.png').convert('RGB')
ps = Image.open(f'output/runs/{d}/scene.jpg').convert('RGB').resize(pr.size)
cf = character_mask(ps, pr)
bb = _bbox(cf)
from s3.rescale import _alpha_in_box
box = (max(0, bb[0] - 2), max(0, bb[1] - 2), min(pr.width, bb[2] + 2), min(pr.height, bb[3] + 2))
al = _alpha_in_box(ps, pr, box, core=cf[box[1]:box[3], box[0]:box[2]])
photo_matte = pr.crop(box).convert('RGBA')
photo_matte.putalpha(al)
photo_matte = S.trim(photo_matte)
print(f'照片抠图: 内部空隙 {S.alpha_hole_ratio(photo_matte) * 100:.2f}%  尺寸 {photo_matte.size}')

# 并排（同高，透明处画棋盘格）
H = 620
tiles = []
for lab, im in [('照片上抠图（现行）', photo_matte), ('白底立绘抠图（新）', rgba)]:
    c = im.resize((int(im.width * H / im.height), H), Image.LANCZOS)
    chk = ((np.indices((c.height, c.width)).sum(0) // 12) % 2)
    bg = np.where(chk[..., None] == 0, 70, 100).astype(np.uint8)
    bg = np.repeat(bg, 3, axis=2) if bg.shape[2] == 1 else bg
    ca = np.asarray(c).astype(np.float32)
    a = (ca[..., 3:4] / 255.0)
    comp = (ca[..., :3] * a + bg[..., :3] * (1 - a)).astype(np.uint8)
    tiles.append((lab, Image.fromarray(comp)))

w = sum(t[1].width + 10 for t in tiles)
s = Image.new('RGB', (w, H + 30), (18, 18, 18))
dr = ImageDraw.Draw(s)
x = 0
for t, im in tiles:
    s.paste(im, (x, 26))
    dr.text((x + 4, 6), t, fill=(255, 225, 80), font=F)
    x += im.width + 10
s.save('output/step8/sprite_vs_matte.jpg', quality=95)
