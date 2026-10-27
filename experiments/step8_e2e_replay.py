"""Step 8：把已有 raw 回灌进完整管线，验证 S2.5 尺度归一 + S3 后处理全链。

## 为什么这样做

尺度归一接连修了三个 bug（掩码贴边被静默拒绝、alpha 抠掉白色部分、
投影被裁切框切断），每一次都只在"离线单独跑 rescale"里验证过。
但管线里还有 defringe / match_ambient / 接触阴影三步，它们依赖角色位置，
而位置现在会因为缩放而改变。必须让**整条链路**跑一遍才算数。

用 `generate_fn` 桩函数返回已有的 raw 图，跳过付费的图像生成；
唯一会真的调 API 的是 VLM 定位（约 1.5 秒，可忽略）。

用法：
    python experiments/step8_e2e_replay.py 16afa9bcc5a6 leg_hug_knee
"""

from __future__ import annotations

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from app import pipeline  # noqa: E402


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "16afa9bcc5a6"
    action = sys.argv[2] if len(sys.argv) > 2 else "leg_hug_knee"
    run_dir = os.path.join(ROOT, "output", "runs", src)
    scene_path = os.path.join(run_dir, "scene.jpg")
    raw_path = os.path.join(run_dir, "raw_0.png")
    out_dir = os.path.join(ROOT, "output", "step8")
    os.makedirs(out_dir, exist_ok=True)

    with open(scene_path, "rb") as f:
        photo_bytes = f.read()

    def stub_generate(_i: int) -> str:
        return raw_path

    logs: list[str] = []
    res = pipeline.run(
        photo_bytes, (0.0, 0.0, 1.0, 1.0), surface="ground",
        candidates=1, mode="companion", action=action,
        generate_fn=stub_generate,
        progress=lambda s, p: logs.append(f"{p:>3}% {s}"),
    )

    for line in logs:
        print(line)
    item = res["candidates"][0]
    print("\n--- 诊断 ---")
    for k in ("scale", "defringe_replaced_ratio", "ambient", "support", "shadow", "scores"):
        print(f"{k}: {item.get(k)}")

    out_path = os.path.join(ROOT, "output", "runs", res["run_id"], item["file"])
    if os.path.exists(out_path):
        dest = os.path.join(out_dir, f"{action}_e2e.jpg")
        Image.open(out_path).save(dest, quality=95)
        print(f"\n输出: {dest}")


if __name__ == "__main__":
    main()
