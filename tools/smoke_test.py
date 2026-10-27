"""冒烟测试：走一遍真实 HTTP 链路（上传 → 生成 → 轮询 → 落盘）。

用法：
    python -m tools.smoke_test [照片路径] [--surface chair] [--n 2]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("BASE", "http://127.0.0.1:8848")


def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def get_json(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", nargs="?", default="assets/scenes/user/u03_living.jpg")
    ap.add_argument("--surface", default="chair")
    ap.add_argument("--light", default="auto")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--box", default="0.56,0.50,0.22,0.32")
    args = ap.parse_args()

    photo = os.path.join(ROOT, args.photo)
    if not os.path.exists(photo):
        raise SystemExit(f"缺照片: {photo}")
    b64 = base64.b64encode(open(photo, "rb").read()).decode()

    box = [float(v) for v in args.box.split(",")]
    print(f"照片 {args.photo}  框 {box}  接触面={args.surface}  光照={args.light}  候选={args.n}")
    t0 = time.perf_counter()
    resp = post_json("/api/generate", {
        "image": "data:image/jpeg;base64," + b64,
        "box": box, "surface": args.surface, "light": args.light, "candidates": args.n,
    })
    jid = resp["job_id"]
    print("job:", jid)

    last = ""
    for _ in range(200):
        time.sleep(2)
        j = get_json(f"/api/job/{jid}")
        line = f"  [{j.get('pct', 0):3d}%] {j.get('stage', '')}"
        if line != last:
            print(line, flush=True)
            last = line
        if j.get("status") == "done":
            r = j["result"]
            print(f"\n完成，用时 {time.perf_counter() - t0:.1f}s  run={r['run_id']}  输出 {r['size']}")
            for c in r["candidates"]:
                print(f"  #{c['index']} {c['file']}  白边修正 {c['defringe_replaced_ratio']:.3f}  "
                      f"阴影 {c.get('shadow')}  质检 {c['scores']}")
                print(f"       {BASE}/files/{r['run_id']}/{c['file']}")
            if r.get("errors"):
                print("  失败候选:", r["errors"])
            return
        if j.get("status") == "error":
            print("错误:", j.get("error"))
            print(j.get("trace", ""))
            return
    print("超时未完成")


if __name__ == "__main__":
    main()
