"""第二批：5 张户外照片批量生成（强光 / 长影 / 明确的尺度参照物）。

与室内那批的差别：
1. 户外是**方向性硬光**，`auto` 档（只出 AO）会显得没影子，必须给左右方向；
2. 影子更长，用 `sun_*` 预设（reach=1.9）；
3. 户外尺度参照物非常明确（车、雕像、凳子、路灯、石头）→ 锚点能写得很硬，
   但反过来，尺度错了也一眼就看出来。

走真实 HTTP 链路（不打桩），1 候选/案例，串行跑。
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("BASE", "http://127.0.0.1:8851")
SCENES = os.path.join(ROOT, "assets", "scenes", "user")

CASES = [
    dict(key="u06_street", file="u06_street.jpg", surface="ground", light="sun_right",
         box=(0.10, 0.72, 0.06, 0.10),
         title="站在斑马线旁的柏油路上",
         anchor="她站在柏油路面上，整个人只有半个抱枕那么大，只到旁边那辆白色轿车轮胎的一半高"),
    dict(key="u07_plaza", file="u07_plaza.jpg", surface="ground", light="sun_right",
         box=(0.60, 0.66, 0.07, 0.12),
         title="站在玩偶雕像右侧的广场上",
         anchor="她站在广场地面上，整个人只有半个抱枕那么大，比旁边那个橙色雕像的鞋还小很多"),
    dict(key="u08_cats", file="u08_cats.jpg", surface="chair", light="right",
         box=(0.28, 0.38, 0.07, 0.13),
         title="坐在左边那张小折叠凳上",
         anchor="她坐在小折叠凳上，整个人只有半个抱枕那么大，跟旁边那只猫差不多高"),
    dict(key="u09_church", file="u09_church.jpg", surface="ground", light="sun_left",
         box=(0.28, 0.71, 0.06, 0.11),
         title="站在教堂前的石板广场上",
         anchor="她站在广场石板地上，整个人只有半个抱枕那么大，只有那盏绿色路灯底座的一半高"),
    dict(key="u10_river", file="u10_river.jpg", surface="ground", light="sun_right",
         box=(0.10, 0.64, 0.06, 0.11),
         title="站在河岸的枯草坡上",
         anchor="她站在枯草坡上，整个人只有半个抱枕那么大，跟脚边那块白石头差不多大"),
]


def post_json(path: str, payload: dict) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def get_json(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode())


def run_case(c: dict) -> dict:
    photo = os.path.join(SCENES, c["file"])
    b64 = base64.b64encode(open(photo, "rb").read()).decode()
    t0 = time.perf_counter()
    resp = post_json("/api/generate", {
        "image": "data:image/jpeg;base64," + b64,
        "box": list(c["box"]), "surface": c["surface"], "light": c["light"],
        "candidates": 1, "anchor": c["anchor"],
    })
    jid = resp["job_id"]
    for _ in range(200):
        time.sleep(2)
        j = get_json(f"/api/job/{jid}")
        if j.get("status") == "done":
            r = j["result"]
            out = dict(c, run=r["run_id"], elapsed=round(time.perf_counter() - t0, 1),
                       file=(r["candidates"][0]["file"] if r["candidates"] else None),
                       scores=(r["candidates"][0]["scores"] if r["candidates"] else {}),
                       defringe=(r["candidates"][0].get("defringe_replaced_ratio")
                                 if r["candidates"] else None))
            print(f"[OK] {c['key']} {out['elapsed']}s -> {out['file']}  {out['scores']}", flush=True)
            return out
        if j.get("status") == "error":
            print(f"[ERR] {c['key']}: {j.get('error')}", flush=True)
            return dict(c, error=j.get("error"))
    print(f"[TIMEOUT] {c['key']}", flush=True)
    return dict(c, error="timeout")


def main() -> None:
    results = []
    for c in CASES:
        print(f"\n=== {c['key']} | {c['title']} | {c['light']} ===", flush=True)
        results.append(run_case(c))
    out = os.path.join(ROOT, "output", "step2", "outdoor_manifest.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n完成 {len([r for r in results if r.get('file')])} / {len(results)}")


if __name__ == "__main__":
    main()
