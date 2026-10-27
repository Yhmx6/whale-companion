"""实验：用"灰色剪影 + 绿边"当占位符，看能否在生成阶段就定住尺寸和位置。

如果成立，S2.5 的"抠图 + 搬移"就能从默认路径里去掉——
透明、边界切口、碎片这一整类 bug 都会消失。

用法：python experiments/exp_placeholder.py <源raw目录> <预缩倍数>
"""
import sys
sys.path.insert(0, '.')
import io, base64
import numpy as np
from PIL import Image

from api.ark import download, generate
from s3.beautify import character_mask
from s3.rescale import _bbox, _alpha_in_box
from app.pipeline import _encode_refs, _pick_size, vlm_profile

SRC_RAW = sys.argv[1] if len(sys.argv) > 1 else '6e19c391960b'
PRESHRINK = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
TAG = f'sil_{int(PRESHRINK * 100)}'

# 1) 从一个已有 raw 里取出她的真实轮廓
rr = Image.open(f'output/runs/{SRC_RAW}/raw_0.png').convert('RGB')
ss = Image.open(f'output/runs/{SRC_RAW}/scene.jpg').convert('RGB').resize(rr.size)
cf = character_mask(ss, rr)
bb = _bbox(cf)
box = (max(0, bb[0] - 2), max(0, bb[1] - 2), min(rr.width, bb[2] + 2), min(rr.height, bb[3] + 2))
sil = _alpha_in_box(ss, rr, box, core=cf[box[1]:box[3], box[0]:box[2]])

# 2) 目标位置/尺寸
photo_path = 'output/runs/16afa9bcc5a6/scene.jpg'
prof = vlm_profile(photo_path)
photo = Image.open(photo_path).convert('RGB')
W, H = photo.size
person_h = prof.person_h_px(H)
foot_y = prof.foot_y_px(H)
lb = prof.legs
leg_cx = (lb[0] + lb[2]) / 2 / 999 * W
leg_w = (lb[2] - lb[0]) / 999 * W
target_h = person_h * 0.26 / PRESHRINK
target_w = sil.width * target_h / sil.height
cx = leg_cx - (leg_w * 0.5 + target_w * 0.5 + 6)
bottom = foot_y
print(f'person_h={person_h:.0f} 目标 h={target_h:.0f} w={target_w:.0f} cx={cx:.0f} bottom={bottom:.0f}')

# 3) 画占位符
sil2 = sil.resize((int(target_w), int(target_h)), Image.LANCZOS)
mask = Image.new('L', (W, H), 0)
mask.paste(sil2, (int(cx - target_w / 2), int(bottom - target_h)))
marked = photo.copy()
marked.paste(Image.new('RGB', (W, H), (120, 120, 120)), (0, 0), mask)
from PIL import ImageFilter
edge = mask.filter(ImageFilter.MaxFilter(9))
edge = Image.fromarray(((np.asarray(edge) > 127) & (np.asarray(mask) <= 127)).astype('uint8') * 255)
marked.paste(Image.new('RGB', (W, H), (0, 220, 0)), (0, 0), edge)
marked.save(f'output/step8/{TAG}_input.jpg', quality=95)

# 4) 生成
prompt = (
    "图1是一张真实的人物照片，画面里有一个**灰色的角色剪影占位符**（带绿色描边）。"
    "图2、图3是同一个二次元角色的设定图。\n"
    "角色形象：蓝色渐变长发、深蓝色鲸鱼耳、藏青女仆裙、白色荷叶边围裙、"
    "身后深蓝色鲸尾、深蓝色小皮鞋。Q 版 2 到 2.5 头身，头大身子小，四肢短圆。\n\n"
    "任务：把这个灰色剪影**填成这个角色**。\n"
    "1. 角色必须**严丝合缝地填满这个剪影**：剪影的头顶就是她的头顶，"
    "剪影的下沿就是她的脚底，剪影的左右边界就是她的头发和尾巴的外缘。"
    "不要比剪影大，也不要比剪影小；\n"
    "2. 填完后把灰色和绿色完全去掉，不要留下任何占位痕迹；\n"
    "3. 姿态：盘腿坐在人物脚边的地面上，仰头看着人物；\n"
    "4. 人物、背景、地面、光线保持原样不变；\n"
    "5. 画面里**恰好一个**角色。\n"
)


def uri(im):
    b = io.BytesIO()
    im.resize((min(W, 1536), int(H * min(W, 1536) / W))).save(b, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()


st, resp = generate(prompt, images=[uri(marked)] + _encode_refs(), size=_pick_size(photo),
                    model='doubao-seedream-5-0-flash-260915', seed=2000)
print('HTTP', st)
if st != 200:
    print(str(resp)[:300])
    sys.exit(1)
out = f'output/step8/{TAG}_raw.png'
download((resp.get('data') or [{}])[0]['url'], out)

# 5) 量误差
raw = Image.open(out).convert('RGB')
sc = Image.open(photo_path).convert('RGB').resize(raw.size)
bb2 = _bbox(character_mask(sc, raw))
x0, y0, x1, y1 = bb2
print(f'实际 bbox={bb2} 高={y1 - y0} 宽={x1 - x0}')
print(f'高度误差 {(y1 - y0 - target_h) / target_h * 100:+.1f}%  '
      f'中心x {(x0 + x1) / 2 - cx:+.0f}px  底边 {y1 - bottom:+.0f}px')
