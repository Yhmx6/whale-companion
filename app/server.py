"""一键启动的本地服务（纯标准库，无 web 框架依赖）。

为什么不用 FastAPI：一键部署的第一原则是"少一个装不上的依赖就少一个失败点"。
标准库的 ThreadingHTTPServer 足够撑原型，Docker 镜像也小。

    python -m app.server            # 默认 http://127.0.0.1:8848

接口：
    GET  /                      前端页面
    GET  /static/*              静态资源
    POST /api/generate          发起生成（JSON，图片走 base64，不用 multipart）
    GET  /api/job/<id>          轮询进度与结果
    GET  /files/<run>/<name>    取回产物
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import pipeline  # noqa: E402
from s3.actions import as_dicts, enabled_actions, get_action  # noqa: E402
from s3.profile import pick_action_v2  # noqa: E402

STATIC_DIR = os.path.join(HERE, "static")
FILES_DIR = os.path.join(ROOT, "output", "runs")
# 首页展示角色形象用。只读这一个目录，见 do_GET 里 /assets/ 的处理。
SPRITE_DIR = os.path.join(ROOT, "assets", "sprites")

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _set_job(jid: str, **kw) -> None:
    with LOCK:
        JOBS.setdefault(jid, {}).update(kw)


def _run_job(jid: str, args: dict) -> None:
    def progress(stage: str, pct: int) -> None:
        _set_job(jid, stage=stage, pct=pct)

    try:
        _set_job(jid, status="running", stage="准备", pct=3)
        result = pipeline.run(progress=progress, **args)
        _set_job(jid, status="done", pct=100, stage="完成", result=result)
    except Exception as e:  # 任何异常都要回到前端，不能静默卡住
        _set_job(jid, status="error", stage="失败",
                 error=f"{type(e).__name__}: {e}",
                 trace=traceback.format_exc()[-1500:])


class Handler(BaseHTTPRequestHandler):
    server_version = "BlueWhalePhoto/0.2"

    def log_message(self, fmt, *a):  # 静默默认访问日志，只打错误
        pass

    # ---- 工具 ----
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: str):
        if not os.path.isfile(path):
            self._json({"error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ---- 路由 ----
    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        if p in ("/", "/index.html"):
            self._file(os.path.join(STATIC_DIR, "index.html"))
        elif p.startswith("/static/"):
            self._file(os.path.join(STATIC_DIR, os.path.basename(p)))
        elif p.startswith("/files/"):
            parts = [x for x in p.split("/") if x]
            if len(parts) >= 3:
                self._file(os.path.join(FILES_DIR, parts[1], os.path.basename(parts[2])))
            else:
                self._json({"error": "bad path"}, 400)
        elif p.startswith("/assets/"):
            # 只给前端展示角色形象用（首页那块"她会变成这样"）。
            # **限定在 assets/sprites 一个目录**，别放开整个 assets——
            # 那里面有角色设定图和测试场景，没必要也不该从 HTTP 出去。
            # os.path.basename 挡掉 ../ 穿越。
            name = os.path.basename(p)
            if name.lower().endswith((".png", ".jpg", ".webp")):
                self._file(os.path.join(SPRITE_DIR, name))
            else:
                self._json({"error": "bad path"}, 400)
        elif p.startswith("/api/job/"):
            jid = p.rsplit("/", 1)[-1]
            with LOCK:
                job = dict(JOBS.get(jid, {}))
            if not job:
                self._json({"error": "unknown job"}, 404)
            else:
                self._json(job)
        elif p == "/api/actions":
            self._json({
                "enabled": as_dicts(enabled_actions()),
                "all": as_dicts(),
            })
        elif p == "/api/moods":
            # 表情清单从 s3/moods.py 读，不在前端硬编码——
            # 那边加一个表情、改一句描述，前端自动跟着变。
            # 只给前端展示需要的字段：`face`/`body` 是写给生图模型的，
            # 前端展示它们没有意义，还会让卡片变长。
            from s3.moods import as_dicts as mood_dicts
            self._json({"moods": [
                {"id": m["id"], "name": m["name"], "when": m["when"]}
                for m in mood_dicts()
            ]})
        elif p == "/api/characters":
            from s3.characters import list_characters
            self._json({"characters": list_characters()})
        elif p == "/api/health":
            self._json({"ok": True, "model": os.environ.get("ARK_IMAGE_MODEL", "")})
        else:
            self._json({"error": "not found"}, 404)

    def _handle_match(self):
        """接收一张图，返回 VLM 推荐的动作。约 1.5 秒（关掉思考模式）。"""
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            img = payload["image"]
            b64 = img.split(",", 1)[1] if "," in img else img
            tmp_dir = os.path.join(FILES_DIR, "_match")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp = os.path.join(tmp_dir, uuid.uuid4().hex[:8] + ".jpg")
            with open(tmp, "wb") as f:
                f.write(base64.b64decode(b64))
        except Exception as e:
            self._json({"error": f"请求解析失败: {e}"}, 400)
            return
        try:
            result = pick_action_v2(tmp)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        self._json(result)

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/api/match":
            self._handle_match()
            return
        if p != "/api/generate":
            self._json({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            img = payload["image"]                      # data URI
            b64 = img.split(",", 1)[1] if "," in img else img
            photo_bytes = base64.b64decode(b64)
            box = payload.get("box") or [0.3, 0.4, 0.25, 0.4]
            args = dict(
                photo_bytes=photo_bytes,
                box_norm=tuple(float(v) for v in box),
                surface=payload.get("surface", "ground"),
                light_key=payload.get("light", "auto"),
                candidates=max(1, min(int(payload.get("candidates", 2)), 4)),
                anchor_override=(payload.get("anchor") or "").strip() or None,
                # mode 是**交互模式**（solo 单人 / companion 贴贴），用户在前端选，
                # 默认值和原来保持 solo 不动——它决定了要不要拖框，改它等于改 UI 流程。
                mode=payload.get("mode", "solo"),
                action=payload.get("action") or None,
                model=payload.get("engine") or None,
                character=payload.get("character") or None,
                # route 是渲染路线。**默认 auto = 一键出图**：
                # 位置/动作/表情/大小全部由 AI 决定，调用方一个参数都不用传。
                # 想精确控制时传 sprite（要 action）或 guided（等价 auto）。
                route=(payload.get("route") or "auto"),
                mood=(payload.get("mood") or "").strip() or None,
            )
        except Exception as e:
            self._json({"error": f"请求解析失败: {e}"}, 400)
            return

        jid = uuid.uuid4().hex[:12]
        _set_job(jid, status="queued", stage="排队", pct=0)
        threading.Thread(target=_run_job, args=(jid, args), daemon=True).start()
        self._json({"job_id": jid})


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8848"))
    os.makedirs(FILES_DIR, exist_ok=True)
    # 把 .env 灌进环境变量，供 pipeline / ark 读取
    try:
        from api.ark import ENV
        for k, v in ENV.items():
            os.environ.setdefault(k, v)
    except Exception:
        pass
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"blue-whale-photo 已启动 -> http://{host}:{port}")
    print(f"产物目录 -> {FILES_DIR}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
