"""实验：不依附场景，单独生成"纯白底角色立绘"，再程序化抠图与合成。

## 为什么这个方向更对

现状是"在真实照片上判断哪些像素是她"，这个判断在两个颜色相近处必然出错
（深藏青袖压牛仔裤、白围裙压亮地面），于是出透明和缺块。

如果角色本来就在**纯白背景**上生成，判断退化成"哪些像素是白的、且与画面边界连通"——
几乎不可能出错：她的白围裙和白袜是被深色描边围住的，不与边界连通，不会被误删。

顺带解决三件事：
  1. 尺寸：直接 resize，无损、可控
  2. 位置：程序计算，不依赖模型
  3. 复用：同一姿态的立绘可以跨照片复用，成本骤降
"""
import sys
sys.path.insert(0, '.')
import io, base64
import numpy as np
from PIL import Image
from api.ark import download, generate
from app.pipeline import _encode_refs

POSE = sys.argv[1] if len(sys.argv) > 1 else 'crouch'

POSES = {
    'crouch': '蹲在平地上的姿态，双脚平踩地面，两个膝盖并拢曲起来顶到胸前，'
              '两只小手抱着自己的膝盖，仰起头向上看，像一只蹲在门口的小动物',
    'jump': '整个人跳在半空中的姿态，两只手握成小拳头向上举过头顶，'
            '一条腿蜷起来、另一条腿向后伸，裙摆和长发向上飘起，笑得非常开心',
}

prompt = (
    "图1、图2是同一个二次元角色的设定图（正面、侧面）。"
    "请画这个角色的**一张全身立绘**。\n"
    "角色形象：蓝色渐变长发、深蓝色鲸鱼耳、藏青女仆裙、白色荷叶边围裙及围裙上的鲸鱼图案、"
    "身后深蓝色鲸尾、深蓝色小皮鞋。Q 版 2 到 2.5 头身，头大身子小，四肢短圆。\n\n"
    f"姿态：{POSES[POSE]}。\n\n"
    "硬性要求：\n"
    "1. **纯白背景**（#FFFFFF），背景上不要有任何东西——不要地面、不要投影、"
    "不要渐变、不要光晕、不要边框、不要文字、不要参考图的痕迹；\n"
    "2. 角色**完整入画**：头顶和脚底都要留出一小圈白边，任何部位都不要被画面裁切；\n"
    "3. 角色**轮廓必须闭合**：外面有一圈连续完整的深色描边，"
    "白围裙和白袜都被描边围在角色内部；\n"
    "4. 画面里恰好一个角色，居中构图。\n"
)

st, resp = generate(prompt, images=_encode_refs(), size='1024x1536',
                    model='doubao-seedream-5-0-flash-260915', seed=3001)
print('HTTP', st)
if st != 200:
    print(str(resp)[:300])
    sys.exit(1)
out = f'output/step8/sprite_{POSE}.png'
download((resp.get('data') or [{}])[0]['url'], out)
im = Image.open(out).convert('RGB')
print('saved', out, im.size)
W, H = im.size
print('四角像素:', [im.getpixel(p) for p in [(2, 2), (W - 3, 2), (2, H - 3), (W - 3, H - 3)]])
