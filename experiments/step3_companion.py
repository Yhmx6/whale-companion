"""第三批：贴贴模式（companion）+ Pro/Flash 引擎对比。

## 为什么底图是 AI 生成的

用户希望"网上找带人物的自拍"来测。这条不走：
1. 拿网上抓的真人脸做合成，正是 Meta Muse Image 3 天下线的死法（未经同意使用肖像）；
2. 《互联网信息服务深度合成管理规定》第 17 条把"人脸生成/操控"列为必须显著标识的情形，
   测试阶段同样适用；
3. 测试要控制变量——AI 生成的底图可以精确指定机位、光线、留白，比随机抓图更有用。

所以用 Seedream 生成写实自拍做底图：零肖像权风险，且构图可控。

## 引擎对比

Pro  0.30 元/张（≤261万像素）
Flash 0.12 元/张
同一底图、同一提示词、同 seed，只换模型。
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from api.ark import download, generate  # noqa: E402
from app import pipeline  # noqa: E402

SCENES = os.path.join(ROOT, "assets", "scenes", "selfie")
OUT = os.path.join(ROOT, "output", "step3")

PRO = "doubao-seedream-5-0-pro-260628"
FLASH = "doubao-seedream-5-0-flash-260915"

BASE_PROMPTS = {
    "s01_mirror": (
        "手机自拍，一位二十多岁的年轻女生在卧室对着穿衣镜自拍半身照，穿浅灰色连帽卫衣，"
        "自然微笑看着镜头，一只手举着手机遮住半张脸，背景是卧室白墙和木质书架，"
        "下午的窗光从画面左侧照进来，真实照片质感，轻微手机镜头畸变，没有文字没有水印"
    ),
    "s02_park": (
        "手机自拍，两个年轻人并肩坐在秋天公园的长椅上自拍合影，两人靠得很近，笑得很开心，"
        "背景是金黄色的树和步道，阳光从画面右侧斜照过来，真实照片质感，没有文字没有水印"
    ),
    "s03_sofa": (
        "手机自拍，一位年轻男生坐在客厅沙发上自拍半身照，穿深灰色短袖，对着镜头微笑，"
        "背景是客厅的白色墙面和浅色窗帘，室内自然光从画面左侧照入，真实照片质感，"
        "没有文字没有水印"
    ),
}

# 每张底图配一个贴贴动作
COMPANION = {
    "s01_mirror": dict(action="cheek", desc="贴在人物脸侧，双手捧着自己圆鼓鼓的脸，和人物头靠头，眯眼笑"),
    "s02_park": dict(action="hug_arm", desc="双手抱住画面左侧那个人物的手臂，整个身体挂在手臂上，脸贴在手臂外侧，做出撒娇的表情"),
    "s03_sofa": dict(action="shoulder", desc="坐在人物的右肩上，一只手扶着人物的头发，两条短腿垂在肩膀前面"),
}


def ensure_bases() -> dict:
    os.makedirs(SCENES, exist_ok=True)
    paths = {}
    for key, prompt in BASE_PROMPTS.items():
        p = os.path.join(SCENES, f"{key}.jpg")
        if os.path.exists(p):
            paths[key] = p
            continue
        print(f"[底图] 生成 {key} ...", flush=True)
        st, resp = generate(prompt, size="1280x1704", model=PRO)
        if st != 200:
            print(f"  失败 {st}: {str(resp)[:200]}")
            continue
        download(resp["data"][0]["url"], p)
        paths[key] = p          # 初版漏了这行：首次生成时 paths 为空，后面循环 0 次，
        print(f"  -> {os.path.basename(p)} {Image.open(p).size}", flush=True)
    return paths


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    bases = ensure_bases()
    records = []

    for key, path in bases.items():
        spec = COMPANION[key]
        photo_bytes = open(path, "rb").read()
        for tag, model, price in (("pro", PRO, 0.30), ("flash", FLASH, 0.12)):
            t0 = time.perf_counter()
            try:
                r = pipeline.run(
                    photo_bytes=photo_bytes,
                    box_norm=(0.4, 0.4, 0.2, 0.3),   # 贴贴模式不用框定位置，占位而已
                    surface="ground", light_key="auto", candidates=1,
                    mode="companion", action=spec["action"], model=model,
                )
            except Exception as e:
                print(f"[ERR] {key}/{tag}: {type(e).__name__}: {e}", flush=True)
                records.append(dict(key=key, engine=tag, error=str(e)))
                continue
            dt = time.perf_counter() - t0
            c = (r.get("candidates") or [{}])[0]
            rec = dict(key=key, engine=tag, price=price, elapsed=round(dt, 1),
                       run=r["run_id"], file=c.get("file"), scores=c.get("scores"))
            records.append(rec)
            print(f"[OK] {key}/{tag} {dt:.1f}s  {c.get('file')}  {c.get('scores')}", flush=True)

    with open(os.path.join(OUT, "companion_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    ok = len([r for r in records if r.get("file")])
    cost = sum(r.get("price", 0) for r in records if r.get("file"))
    print(f"\n完成 {ok}/{len(records)}  贴贴调用花费约 {cost:.2f} 元")


if __name__ == "__main__":
    main()
