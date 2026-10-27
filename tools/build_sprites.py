"""批量构建角色立绘并缓存。

立绘**与照片无关**——同一个姿态只需要生成一次，之后贴到任意照片上都是零成本。
所以这是一个"一次性基建"脚本，不是每次请求都跑的东西。

用法：
    python tools/build_sprites.py            # 缺什么补什么
    python tools/build_sprites.py --force    # 全部重生成
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from api.ark import download, generate  # noqa: E402
from app.pipeline import _encode_refs  # noqa: E402
from s3.actions import enabled_actions  # noqa: E402
from s3.sprite import (alpha_hole_ratio, build_sprite_prompt, key_white,  # noqa: E402
                       load_sprite, save_sprite)


def build(action, model: str, force: bool = False) -> None:
    if not force and load_sprite(action.id, model) is not None:
        print(f'  {action.id:18s} 已有缓存，跳过')
        return
    t0 = time.time()
    prompt = build_sprite_prompt(action.desc)
    st, resp = generate(prompt, images=_encode_refs(), size='1024x1536',
                        model=model, seed=3001)
    if st != 200:
        print(f'  {action.id:18s} 生成失败 HTTP {st}: {str(resp)[:120]}')
        return
    tmp = os.path.join(ROOT, 'output', 'sprites_raw', f'{action.id}.png')
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    download((resp.get('data') or [{}])[0]['url'], tmp)

    rgba = key_white(Image.open(tmp).convert('RGB'))
    from s3.sprite import trim
    rgba = trim(rgba)
    save_sprite(action.id, model, rgba)
    print(f'  {action.id:18s} OK  {rgba.size}  内部空隙 {alpha_hole_ratio(rgba) * 100:.1f}%  '
          f'{time.time() - t0:.0f}s')


def main() -> None:
    force = '--force' in sys.argv
    model = 'doubao-seedream-5-0-flash-260915'
    acts = [a for a in enabled_actions() if a.render == 'sprite']
    print(f'需要立绘的动作 {len(acts)} 个（model={model}）')
    for a in acts:
        build(a, model, force)
    print('完成，缓存目录 assets/sprites/')


if __name__ == '__main__':
    main()
