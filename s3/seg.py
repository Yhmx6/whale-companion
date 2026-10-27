"""人物前景分割：为了"她被人挡住"这件事。

## 为什么需要它

立绘路线一直有个固有限制：**她永远画在最前面**。
所以位置只能选"完全在人旁边的空地里"——一旦要"从人身后探出半个身子"或"勾肩搭背"，
就必须知道**人物的轮廓**，才能把她压在下面。

这个限制还直接决定了尺度：等身角色宽约 700px，竖构自拍两侧各只剩 400px，
按"完整可见"算是放不下的（实测 5/5 场景要缩到 21%~72%）。
**但只要允许被挡住，"放不下"就不成立了**——她不需要 700px 空地，
只需要一条能露出来的缝。

## 为什么不用 GrabCut

先试过 `cv2.grabCut`（带人物框先验）。在两人重叠、又都穿深色外套的照片上直接失效：
她深色外套整块被判成背景（于是鲸鱼娘会画到外套上面），同时糊到了旁边男生的额头。
它不是"参数没调好"，是没有语义——分不清"两块深色的布"和"两个人"。

改用 MODNet（人像抠图，ONNX，25.9MB）。下载走 hf-mirror：
GitHub releases 在这个环境里不通（curl 返回 HTTP 000），hf-mirror 通。

## 失败要软着陆

模型缺失、onnxruntime 没装、推理报错 —— 一律返回 None，
调用方退回"不遮挡"的老行为。分割是增强，不是主链路，不能因为它挂了出不了图。
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, "assets", "models", "modnet.onnx")

_SESSION = None
_FAILED = False


def _session():
    """懒加载 ONNX 会话。加载失败只试一次，之后直接返回 None。"""
    global _SESSION, _FAILED
    if _SESSION is not None or _FAILED:
        return _SESSION
    if not os.path.exists(MODEL_PATH):
        _FAILED = True
        return None
    try:
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    except Exception:
        _FAILED = True
        return None
    return _SESSION


def available() -> bool:
    return _session() is not None


def _keep_person_components(a: np.ndarray, keep_box, thr: float = 0.5) -> np.ndarray:
    """只保留"穿过人物框中心区"的连通域。

    ## 为什么必须做这一步

    MODNet 是**显著目标**模型，不是人像专用的。实测 `s02_park`（两人坐在长椅上）：
    画面下半部的长椅和背景也被判成前景，y>700 之后掩码的左右边界直接是 0 和 1279
    ——整幅宽都算"人"。

    后果不是"轮廓不准"，是**灾难性的**：遮挡阶段会拿这张掩码把成品盖回去，
    于是她被整片擦掉，成品里只剩下几条蓝边。

    过滤办法：取人物框**内缩 40% 的中心区**作为种子，只保留与种子有交集的连通域。
    长椅在框的左右下角，不与中心区相交，会被丢掉；两个人都在中心区里，都保留。
    """
    if keep_box is None:
        return a
    try:
        import cv2
        W = a.shape[1]
        x1, y1, x2, y2 = [v / 999.0 for v in keep_box]
        mx, my = (x2 - x1) * 0.3, (y2 - y1) * 0.3
        sx1, sx2 = int((x1 + mx) * W), int((x2 - mx) * W)
        sy1 = int((y1 + my) * a.shape[0])
        sy2 = int((y2 - my) * a.shape[0])
        m = (a > thr).astype(np.uint8)
        n, lab = cv2.connectedComponents(m, 8)
        if n <= 2:
            return a
        seed = lab[sy1:sy2, sx1:sx2]
        keep_ids = set(np.unique(seed).tolist()) - {0}
        if not keep_ids:
            return a
        out = np.where(np.isin(lab, list(keep_ids)), a, 0.0).astype(np.float32)
        # 一个都没留下时退回原掩码，宁可粗也不要空
        return out if out.max() > 0.1 else a
    except Exception:
        return a


def person_alpha(photo: Image.Image, work: int = 512,
                 keep_box=None) -> np.ndarray | None:
    """返回人物（前景）的软 alpha，(H, W) float32 ∈ [0,1]；不可用返回 None。

    `work` 是推理时的短边基准。MODNet 对 512 左右最稳；再大只是变慢，
    因为遮挡只要求轮廓准到几个像素，最后会统一双线性放大回原图。

    注意输入是**整张照片**，不是人物框裁片：MODNet 靠"整幅画面里谁最像人"来判断，
    裁成小图反而会丢掉构图信息（实测裁片模式下她旁边的男生容易被当成背景）。
    """
    sess = _session()
    if sess is None:
        return None
    try:
        W, H = photo.size
        s = work / float(max(W, H))
        ws, hs = max(32, int(round(W * s / 32)) * 32), max(32, int(round(H * s / 32)) * 32)
        small = photo.convert("RGB").resize((ws, hs), Image.BILINEAR)
        x = np.asarray(small, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5                       # MODNet: [0,1] -> [-1,1]
        x = np.transpose(x, (2, 0, 1))[None]      # (1,3,H,W)
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        a = np.squeeze(out)
        a = np.clip(a, 0.0, 1.0).astype(np.float32)

        a = _keep_person_components(a, keep_box)
        img = Image.fromarray((a * 255).astype(np.uint8), "L").resize((W, H), Image.BILINEAR)
        # 轻微羽化：硬边会把她的轮廓切出锯齿
        img = img.filter(ImageFilter.GaussianBlur(1.2))
        return np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return None
