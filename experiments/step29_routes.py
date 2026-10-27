"""step29：路线分派的回归测试（0 成本，不调任何 API）。

## 为什么要有

2026-10-27 起引导生成成了**默认路线**（`route="guided"`）。三条路线的分派
分散在 `run()` 开头，一旦改错，用户会看到"点了生成却出来立绘路线的东西"，
而且**不会报错**——静默走错路线是最难发现的那类 bug。

所以用 stub 把三个 `run_*` 函数换掉，只验分派本身：
不调 API、不出图、0.2 秒跑完。

## 覆盖的分派

| 输入 | 应该走 |
|---|---|
| **什么都不传**（一键出图） | guided |
| `mode="companion"`（不传 route） | guided |
| `route="auto"` | guided（和 guided 等价） |
| `route="guided"` 显式 | guided |
| `route="sprite"` + 合法 sprite 动作 | sprite |
| `route="sprite"` 不带 action | 报错（立绘按动作缓存，没动作无从取） |
| `route="sprite"` + scene 类动作（如 shoulder） | 报错（那个动作只能和场景一起生成） |
| `route="planned"` | planned |
| `route="xxx"` 未知 | 报错并列出可选项 |
| `route="solo"` | 走老的"画进场景"路径（不分派给 run_*） |
| `mode="solo"` 但 route 没指定 | **仍然 guided**——mode 已不参与分派 |

跑法：
    python experiments/step29_routes.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import app.pipeline as P  # noqa: E402

# 1×1 的白图。用 PIL 现生成——手写 PNG 字节试过，CRC 对不上会报
# "broken data stream"，虽然不影响分派判据（只看有没有调 run_*），
# 但输出里挂个假错误容易让人误判成回归失败。
def _one_px_png() -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


PNG_1PX = _one_px_png()

BOX = (0.3, 0.4, 0.25, 0.4)


def main() -> int:
    calls: list[tuple[str, dict]] = []
    orig = (P.run_guided, P.run_sprite, P.run_planned)

    def stub(name):
        def _f(*a, **kw):
            calls.append((name, kw))
            return {"mode": name, "stub": True}
        return _f

    P.run_guided, P.run_sprite, P.run_planned = (
        stub("guided"), stub("sprite"), stub("planned"))
    bad = 0
    cases_seen = []

    def case(label, expect_mode, **kw):
        nonlocal bad
        cases_seen.append(label)
        calls.clear()
        try:
            res = P.run(PNG_1PX, BOX, "ground", **kw)
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        got = res.get("mode") or "(none)"
        err = res.get("error") or ""
        ok = False
        if expect_mode == "error":
            ok = bool(err)
        elif expect_mode == "no-dispatch":
            ok = not calls          # 不该碰任何 run_*
        else:
            ok = (got == expect_mode)
        bad += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
        print(f"      -> {got}{'  err: ' + err[:56] if err else ''}")
        return ok

    case("一键出图：什么参数都不传", "guided")
    case("companion 不传 route", "guided", mode="companion")
    case("route=auto（和 guided 等价）", "guided", route="auto")
    case("route=guided 显式", "guided", mode="companion", route="guided")
    case("route=sprite + 合法立绘动作", "sprite", mode="companion",
         route="sprite", action="seat_edge_sit")
    case("route=sprite 不给 action", "error", mode="companion", route="sprite")
    case("route=sprite + scene 类动作", "error", mode="companion",
         route="sprite", action="shoulder")
    case("route=planned", "planned", mode="companion", route="planned")
    case("route=xxx 未知", "error", mode="companion", route="xxx")
    case("route=solo 走老路径", "no-dispatch", route="solo")
    case("mode=solo 但 route 未指定（仍 guided）", "guided", mode="solo")

    P.run_guided, P.run_sprite, P.run_planned = orig
    print(f"\n{len(cases_seen) - bad}/{len(cases_seen)} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
