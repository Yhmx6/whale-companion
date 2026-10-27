"""火山引擎方舟（Ark）最小客户端：图像生成/编辑 + 视觉语言模型。

只用标准库（urllib），不引第三方依赖，便于原型阶段快速验证。
密钥从项目根目录的 `.env` 读（见 `.env.example`），绝不硬编码、绝不打印完整值。
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request


def load_env(path: str | None = None) -> dict:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = path or os.path.join(here, ".env")
    env: dict = {}
    if not os.path.exists(path):
        return env
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


ENV = load_env()
API_KEY = ENV.get("ARK_API_KEY", "")
BASE_URL = ENV.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
IMAGE_MODEL = ENV.get("ARK_IMAGE_MODEL", "doubao-seedream-5-0-flash-260915")
VLM_MODEL = ENV.get("ARK_VLM_MODEL", "doubao-seed-2-1-lite-260915")


def masked_key() -> str:
    if len(API_KEY) < 12:
        return "(未配置或过短)"
    return f"{API_KEY[:8]}...{API_KEY[-4:]}"


def _post(path: str, payload: dict, timeout: int = 180) -> tuple[int, dict]:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"raw": body[:2000]}
    except Exception as e:  # 网络层错误
        return -1, {"error": f"{type(e).__name__}: {e}"}


def image_path_to_data_uri(path: str, max_side: int = 1536,
                           keep_alpha: bool = False,
                           bg: tuple[int, int, int] | None = None) -> str:
    """本地图转 data URI。顺手压到长边 max_side，控制请求体体积。

    keep_alpha=True 时输出 PNG 并保留透明通道（用来测"透明底是否消除白边"）。
    bg 不为空时把图合成到指定纯色底上（用来测"中性灰底"）。
    """
    from PIL import Image
    import io

    im = Image.open(path)
    if keep_alpha and im.mode in ("RGBA", "LA"):
        im = im.convert("RGBA")
        if bg is not None:
            canvas = Image.new("RGB", im.size, bg)
            canvas.paste(im, (0, 0), im)
            im = canvas
        fmt, quality = "PNG", None
    else:
        im = im.convert("RGB")
        if bg is not None:
            canvas = Image.new("RGB", im.size, bg)
            canvas.paste(im, (0, 0))
            im = canvas
        fmt, quality = "JPEG", 92

    if max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "PNG":
        im.save(buf, format="PNG")
        mime = "image/png"
    else:
        im.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def generate(prompt: str, images: list[str] | None = None,
             size: str = "1024x1024", model: str | None = None,
             watermark: bool = False, seed: int | None = None) -> tuple[int, dict]:
    """图像生成/编辑。images 传 data URI 或公网 URL 列表（Seedream 4.0 支持多图）。

    注意：Seedream 的图生图用 image 字段；单图传字符串、多图传数组（见方舟文档）。
    """
    payload: dict = {
        "model": model or IMAGE_MODEL,
        "prompt": prompt,
        "size": size,
        "response_format": "url",
        "watermark": watermark,
    }
    if images:
        payload["image"] = images if len(images) > 1 else images[0]
    if seed is not None:
        payload["seed"] = seed
    return _post("/images/generations", payload)


def download(url: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp, open(out_path, "wb") as f:
        f.write(resp.read())
    return out_path
