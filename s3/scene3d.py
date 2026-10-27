"""简易 2.5D 场景：让她站在"能站的地方"，而不是站在咖啡杯里。

## 为什么需要这一层

改造前她落点只用一个数：`center_x`（从规划器的框中心算出来）。
程序不知道那个位置有什么，所以 p04 那张照片里她被放到了**咖啡杯上面**——
规划器说"坐在桌面边缘"，框中心正好压着那杯咖啡。

真实世界当然不是这样：桌面是平的，杯子上没地方坐。要修这个问题，
程序至少得知道"画面里有哪些东西、它们各占哪块地方、谁在谁前面"。

单目照片做不出真三维，但做个**够用的 2.5D** 就足够解决这个问题：

| 维 | 从哪来 | 够用来做什么 |
|---|---|---|
| 横向 x | 物体的外接框 | 判断她会不会横向撞上 |
| 纵向 y | 外接框 | 判断她会不会竖向压上 |
| 深度 z | **物体的接地线 `base_y`** | 谁在前谁在后 |

`base_y` 就是"这件东西和它下面的支撑面接触的那条线"。它是单目照片里
**唯一可靠的深度线索**：脚在地上、离镜头越近，那条线在画面里越靠下。
桌上的杯子也一样——它在桌面上越靠近镜头，杯底那条线越靠下。

## 为什么不用单目深度估计

试过的路不走：跑一个 MiDaS/DepthAnything 只能拿到一张相对深度图，
它是**稠密但不带语义**的，既分不开"杯子"和"杯子后面的墙"，
也没法回答"这个面能不能坐"。而且模型体积、推理耗时都会翻倍。
VLM 给一个框 + 一条接地线，信息量小得多，但每一条都直接可用。

## 三条判定规则

1. **谁更近**：`base_y` 大的离镜头近。
2. **能不能重叠**：她的脚线比她近的物体低 → 她站在它前面，重叠是正常的
   （她挡住它）；她的脚线比她远的物体高 → 她站在它后面，物体该挡住她。
   第二条在**生成路线**里做不到（模型不会主动挡她），所以直接判为冲突。
3. **坐在谁身上**：她的落点落在某个可坐物的 `sit_y` 附近，且横向在它的范围内，
   就是"坐在它上面"，此时**只允许和它自己重叠**。

## 这份实现的边界

- 只做**矩形占位**，不处理物体的实际轮廓（一只杯子会占掉它整个外接框）。
  结果是偏保守：宁可多避一点，也不会撞上。
- 深度只有"接地线"这一维，**同一行的两件东西分不出先后**（桌上并排的两个杯子）。
  这种时候按外接框面积排，大的算近——近似，但够用。
- 不处理旋转、不处理镜面、不处理镂空（椅子的靠背缝隙会被当成实心）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 这些 kind 的物体天然是"能坐的面"，即使 VLM 没给 sit_y 也可以兜一个
SITTABLE = {"sofa", "bench", "chair", "bed", "table", "counter", "step"}

# 小物件：它们不该被撞上，但也不可能坐上去
SMALL_KINDS = {"cup", "plate", "book", "phone", "laptop", "vase",
               "bag", "lamp", "other"}

KIND_CN = {
    "sofa": "沙发", "bench": "长椅", "chair": "椅子", "bed": "床",
    "table": "桌子", "counter": "台面", "step": "台阶",
    "cup": "杯子", "plate": "盘子", "book": "书", "phone": "手机",
    "laptop": "电脑", "vase": "花瓶", "plant": "花盆", "bag": "包",
    "lamp": "台灯", "person": "人物", "other": "物件",
}


@dataclass
class SceneObject:
    kind: str = "other"
    name: str = ""
    box: list[int] = field(default_factory=list)   # 0-999 归一化
    base_y: int = 0                                # 0-999，接地线
    support: str = "unknown"                       # floor / table / wall / unknown
    solid: bool = True
    sit: bool = False
    sit_y: int = 0                                 # 0-999，可坐面
    # 由 area 和 base_y 一起定的相对深度序号（0 = 最近）。见 `Scene2D.build`
    depth: int = 0

    @property
    def name_cn(self) -> str:
        return self.name or KIND_CN.get(self.kind, "物件")

    @property
    def area(self) -> int:
        if len(self.box) != 4:
            return 0
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])

    def x_span(self, w_px: int) -> tuple[float, float]:
        """它在画面上的横向占位（像素）。"""
        if len(self.box) != 4:
            return (0.0, 0.0)
        return (self.box[0] / 999.0 * w_px, self.box[2] / 999.0 * w_px)

    def y_span(self, h_px: int) -> tuple[float, float]:
        if len(self.box) != 4:
            return (0.0, 0.0)
        return (self.box[1] / 999.0 * h_px, self.box[3] / 999.0 * h_px)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name_cn, "box": self.box,
                "base_y": self.base_y, "support": self.support,
                "solid": self.solid, "sit": self.sit, "sit_y": self.sit_y,
                "depth": self.depth}


def parse_objects(raw) -> list[SceneObject]:
    """把 VLM 给的 `objects` 数组解析成 `SceneObject`。

    宽松解析：任一字段坏了就整条丢掉，绝不因为一个杯子解析失败让整张图走不下去。
    但 `box` 是必需的——没有框的物体在这套逻辑里没有任何用处。
    """
    out: list[SceneObject] = []
    if not isinstance(raw, list):
        return out
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        box = item.get("box")
        if not isinstance(box, list) or len(box) < 4:
            continue
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((max(0, min(999, x1)), max(0, min(999, x2))))
        y1, y2 = sorted((max(0, min(999, y1)), max(0, min(999, y2))))
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue
        kind = str(item.get("kind") or "other").strip().lower()
        if kind not in KIND_CN:
            kind = "other"
        try:
            base_y = int(round(float(item.get("base_y") or y2)))
        except (TypeError, ValueError):
            base_y = y2
        base_y = max(0, min(999, base_y))
        support = str(item.get("support") or "unknown").strip().lower()
        if support not in ("floor", "table", "wall", "unknown"):
            support = "unknown"
        solid = item.get("solid")
        solid = True if solid is None else bool(solid)
        sit = bool(item.get("sit"))
        if not sit and kind in SITTABLE:
            sit = True                     # VLM 漏标时按类别兜一个
        sit_y = 0
        try:
            if item.get("sit_y") is not None:
                sit_y = max(0, min(999, int(round(float(item["sit_y"])))))
        except (TypeError, ValueError):
            sit_y = 0
        if sit and not sit_y:
            # VLM 漏给 sit_y 时的兜底。**不能用框的上沿**——那是靠背顶。
            #
            # 实测 p05 的长椅：box=[272,375,999,690]，用 y1 得到 sit_y=375，
            # 而照片里座面在 y≈560。差 185 个归一化单位（约 19% 画面高），
            # 后果是"她坐在靠背上"，比不坐还离谱。
            #
            # 家具的座面高度有个稳定的比例：从地面到座面约是整件家具高度的
            # 55~65%（沙发偏矮、餐椅偏高、长椅居中）。取 0.60 是这三类的中间。
            # 这个数是**近似的**，所以 `sit_target` 的匹配容差要按她的身高给足
            # （见 `avoid_collision` 的 `tol_ratio`），不能当成精确值用。
            sit_y = int(round(y1 + 0.60 * (y2 - y1)))
        out.append(SceneObject(kind=kind, name=str(item.get("name") or ""),
                               box=[x1, y1, x2, y2], base_y=base_y,
                               support=support, solid=solid, sit=sit,
                               sit_y=sit_y))
    return out


@dataclass
class Scene2D:
    """一张照片的物品布局。坐标全部是 0-999 归一化，用时再乘画幅。"""
    objects: list[SceneObject] = field(default_factory=list)

    @classmethod
    def build(cls, objs: list[SceneObject]) -> "Scene2D":
        """排深度。按 `base_y` 越大越近，同 base_y 的按面积大的算近。

        为什么要排这个：判断"她和某个物体谁在前"只靠 base_y 就够了，
        但相遇时要给出**人话解释**（"她把杯子挡住了" / "杯子挡在她前面"），
        有个序号比分小数好读。
        """
        objs = [o for o in objs if len(o.box) == 4]
        # `base_y` 越大越近。排序后倒着编号，让 **0 = 最近**——
        # 编号和 `_DEPTH_COLORS` 的顺序是同向的（0 是红、最远是紫），
        # 这样图例上"数字=深度序号"读起来才顺：红的是身边的东西，
        # 紫的是远处的背景。第一版没倒，结果最远的背景桌拿了红色。
        order = sorted(objs, key=lambda o: (o.base_y, o.area), reverse=True)
        for i, o in enumerate(order):
            o.depth = i
        obj2 = cls(objects=objs)
        return obj2

    # ---- 查询 ----

    def solid_objects(self) -> list[SceneObject]:
        return [o for o in self.objects if o.solid]

    def sit_target(self, center_x: float, bottom_y: float,
                   w_px: int, h_px: int, tol_px: float) -> SceneObject | None:
        """判断她这一下落点是不是"坐在某件东西上"。

        判据三条：横向在她的范围内、可坐、落点离它的可坐面不超过 `tol_px`。
        `tol_px` 传她身高的一部分（见 `avoid_collision` 的调用点）。
        """
        best, best_d = None, 1e9
        for o in self.objects:
            if not o.sit:
                continue
            x1, x2 = o.x_span(w_px)
            if not (x1 - 8 <= center_x <= x2 + 8):
                continue
            sy = o.sit_y / 999.0 * h_px
            d = abs(bottom_y - sy)
            if d <= tol_px and d < best_d:
                best, best_d = o, d
        return best

    def footprint_conflicts(self, x1: float, y1: float, x2: float, y2: float,
                            w_px: int, h_px: int, tol_px: float,
                            sits_on: SceneObject | None) -> list[SceneObject]:
        """她这个框会和哪些实体撞上。

        ## 四道筛子，每一道都是被实测打出来的

        1. **`solid` 且不是 `person`**。人不是障碍——她挡住人物下半身是正常的，
           把她从人身边赶开会毁掉构图，而这条流水线的目的就是让她和人互动。

        2. **横向跨度不超过画面 85%**。被 c05 逼出来的：那张图的"花盆"
           `box=[0,275,999,785]` 其实是背景的整片灌木。把它当障碍，怎么挪都撞。

        3. **她坐在某个面上时，家具本身不算障碍，面上的东西才算**。
           被 c09 逼出来的：台面 `box=[0,605,999,999]` 横跨整个画面，
           而她**就坐在台面上**。台面不是障碍，台上那两个杯子才是。

        4. **不是家具的东西一律躲，不看大小**。这条是被 p04 和 c01 连着打回来的：
           咖啡杯 `base_y=947` 比她的脚线 855 更低（更靠镜头），按"更近就合法"
           的规则该放过，于是她坐在杯子前面、把整杯拿铁挡住。物理上没错
           （她确实挡得住），**但产品上不行**——桌上那杯咖啡是照片的视觉焦点。

           c01 又补了一刀：那张是特写，杯子和托盘各占画面 44% / 54% 宽，
           我一开始给"小物件"设了 35% 的尺寸门槛，于是它们被判成"大家具"、
           走深度规则、被放过。可它们就在她坐着的那个桌面上。
           所以门槛取消：**只要 `sit == False` 且横向没占满大半个画面（筛子 2），
           就躲**。占满的那是背景/地面本身，躲不开也没意义。

        ## 深度判据（这里我改错过一次，把推导写全，别再想反）

        约定：y 越小越靠画面顶部 = **越远**；y 越大越靠底部 = **越近**。
        `base_y` 是物体底部那条线的 y，`foot` 是她的脚线。

        - `base > foot`：物体底在**她脚下方** → 物体离镜头更近 → **物体挡住她**。
          而生成路线画不出"物体挡她"，只会画成她穿过去 → **冲突**。
        - `base < foot`：物体底在**她脚上方** → 物体更远 → **她挡住它**。
          这是自然的前后关系，**合法**。

        第一版写的是 `base < foot - tol → continue`，**这个是对的**。
        后来我在诊断"批量测试 10 张挪开 0 张"时，把 `base < foot` 读成了
        "物体更近"，于是"修正"成了相反的方向——结果 c10 里背包明明在她前面，
        却被判成"她挡住了背包"，她直接穿了过去。
        真正让那 10 张挪不动的根因是 `sit_y` 兜底取了框的上沿（靠背顶），
        跟方向无关。**改之前先用离线 case 验，别靠脑内推理。**
        """
        hits: list[SceneObject] = []
        foot = y2
        for o in self.solid_objects():
            if o.kind == "person":
                continue
            if o is sits_on or (sits_on is not None and o.kind == sits_on.kind
                                and o.box == sits_on.box):
                continue
            ox1, ox2 = o.x_span(w_px)
            oy1, oy2 = o.y_span(h_px)
            span_x = ox2 - ox1
            if span_x > w_px * 0.85:            # 筛子 2：一整片面，躲不开也没意义
                continue
            if sits_on is not None and o.sit:   # 筛子 3：家具本身不是障碍
                continue
            if x2 <= ox1 or x1 >= ox2:          # 横向不挨着
                continue
            if y2 <= oy1 or y1 >= oy2:          # 竖向不挨着
                continue
            is_small = not o.sit
            if is_small:                        # 筛子 4：面的东西无条件躲
                hits.append(o)
                continue
            base = o.base_y / 999.0 * h_px
            if base < foot - tol_px:
                continue                        # 它在更远处，她挡住它，合法
            hits.append(o)                      # 它在近处，该挡住她，生成画不出
        return hits

    def as_dict(self) -> dict:
        return {"objects": [o.as_dict() for o in self.objects]}


def snap_to_surface(scene: Scene2D, center_x: float, bottom_y: float,
                    w_px: int, h_px: int, target_h: float,
                    max_ratio: float = 0.75) -> dict:
    """落点悬空时，把她吸到最近的可坐面上。

    ## 为什么需要这一步

    规划器给的 `region` 底边（落点）经常和真实座面对不上。实测 c05：
    它把她放在 y=548，而长椅座面在 654——差 106 像素，约画面高的 10%。
    差得不多时人眼看着就是"坐在座面上方一点点"，但**程序里差 40 像素就匹配不上
    `sit_target`**，于是长椅从"我坐着的东西"变成"挡住我的障碍"，
    整个避让逻辑跟着跑偏（实测 c05 报了 2 个冲突、一张无碍的图被判成挪不开）。

    c08 床沿更极端：落点 625、床面 796，差 171 像素。

    ## 吸附的边界

    只在**竖向差不超过她身高的 60%** 时才吸。再多就说明规划器不是在指这个面
    （比如她站在地上，旁边三米外有张桌子），硬吸过去会把她从地面拎到桌上。

    横向也要落在物体的范围内——否则"桌子在左边、她在右边地上"也会被吸上去。

    返回 `{y, snapped, from, to, target}`。`snapped=False` 时 `y` 原样返回。
    """
    max_drop = target_h * max_ratio
    best, best_d = None, 1e9
    for o in scene.objects:
        if not o.sit:
            continue
        ox1, ox2 = o.x_span(w_px)
        if not (ox1 - 8 <= center_x <= ox2 + 8):
            continue
        sy = o.sit_y / 999.0 * h_px
        d = abs(bottom_y - sy)
        if d <= max_drop and d < best_d:
            best, best_d = o, d
    if best is None:
        return {"y": bottom_y, "snapped": False}
    return {"y": best.sit_y / 999.0 * h_px, "snapped": True,
            "from": round(bottom_y, 1),
            "to": round(best.sit_y / 999.0 * h_px, 1),
            "object": best.as_dict(), "delta": round(best_d, 1)}


def avoid_collision(scene: Scene2D, center_x: float, bottom_y: float,
                    target_w: float, target_h: float,
                    w_px: int, h_px: int, margin: float = 6.0,
                    tol_ratio: float = 0.15,
                    step_ratio: float = 0.008) -> dict:
    """从规划器给的落点出发，找最近的一个"不会撞到东西"的横向位置。

    为什么只挪 x 不挪 y：`bottom_y` 是落点（她坐在/站在什么高度），
    那是规划器判断的核心，动它等于换了个动作。横向平移则几乎不改变
    "她在哪、在做什么"，只是从杯子左边挪到杯子右边。

    搜索方式：以原位置为中心左右交替试探，取第一个无冲突的。
    这样"没冲突就不动"，有冲突时也是"挪得最少"。

    ## 找不到空位时退到"压得最少"

    有些构图是真的放不下——实测 c01：咖啡杯 + 托盘占掉画面 20%~78%，
    而她本身有 28% 画面宽，左右两侧**确实一寸空地都没有**。

    第一版遇到这种情况就原样返回 + 报 `conflicts`，等于让她压住整杯咖啡。
    但"完全避不开"和"压住整杯"是两件事：往左挪 100 像素就能只压到托盘边，
    那比压住杯子好得多。

    所以搜完整条横向还没空位时，退回去取**重叠面积最小**的那个位置，
    并把这个事实写进 `forced`。仍然如实告警，不假装问题解决了。

    返回 `{x, moved_px, sits_on, conflicts, tried, forced}`。
    """
    tol = target_h * tol_ratio
    half = target_w / 2.0
    lo = margin + half
    hi = w_px - margin - half
    if lo >= hi:                                # 她比画面还宽，无从避让
        return {"x": w_px / 2.0, "moved_px": 0, "sits_on": None,
                "conflicts": [], "tried": 0, "impossible": True}

    sits_on = scene.sit_target(center_x, bottom_y, w_px, h_px, tol)

    def hits_at(cx: float) -> list[SceneObject]:
        return scene.footprint_conflicts(cx - half, bottom_y - target_h,
                                         cx + half, bottom_y,
                                         w_px, h_px, tol, sits_on)

    def overlap_area(cx: float, hits: list[SceneObject]) -> float:
        """她和这些物体重叠的总面积。用来挑"压得最少"的位置。"""
        total = 0.0
        for o in hits:
            ox1, ox2 = o.x_span(w_px)
            oy1, oy2 = o.y_span(h_px)
            ix = max(0.0, min(cx + half, ox2) - max(cx - half, ox1))
            iy = max(0.0, min(bottom_y, oy2) - max(bottom_y - target_h, oy1))
            total += ix * iy
        return total

    now = hits_at(center_x)
    if not now:
        return {"x": center_x, "moved_px": 0,
                "sits_on": sits_on.as_dict() if sits_on else None,
                "conflicts": [], "tried": 1}

    step = max(1.0, w_px * step_ratio)
    max_k = int((hi - lo) / step) + 1
    tried = 1
    # 兜底候选：记住"压得最少"的那个位置，最后可能用得上
    best_x, best_area, best_hits = center_x, overlap_area(center_x, now), now
    for k in range(1, max_k + 1):
        for sign in (1, -1):
            cx = center_x + sign * k * step
            if not (lo <= cx <= hi):
                continue
            tried += 1
            h = hits_at(cx)
            if not h:
                return {"x": cx, "moved_px": round(abs(cx - center_x), 1),
                        "sits_on": sits_on.as_dict() if sits_on else None,
                        "conflicts": [], "tried": tried}
            a = overlap_area(cx, h)
            if a < best_area:
                best_x, best_area, best_hits = cx, a, h

    now_area = overlap_area(center_x, now)
    # 只有"压得明显更少"才值得挪；差别可以忽略就别动了，
    # 免得为了 2% 的面积去挪一大截，反而显得位置莫名其妙。
    if best_area < now_area * 0.75:
        return {"x": best_x, "moved_px": round(abs(best_x - center_x), 1),
                "sits_on": sits_on.as_dict() if sits_on else None,
                "conflicts": [], "tried": tried, "forced": True,
                "overlap_cut": round(1 - best_area / max(now_area, 1e-6), 2),
                "still_hit": [o.as_dict() for o in best_hits]}
    return {"x": center_x, "moved_px": 0,
            "sits_on": sits_on.as_dict() if sits_on else None,
            "conflicts": [o.as_dict() for o in now], "tried": tried,
            "impossible": True}


# ---------------------------------------------------------------------------
# 可视化：把场景图叠回照片上
# ---------------------------------------------------------------------------

# 按深度给颜色：越近越暖（红），越远越冷（青）。人眼对"红近青远"有直觉，
# 比给每个物体标一个数字好读得多。
_DEPTH_COLORS = [
    (232, 92, 58), (240, 148, 46), (232, 196, 60),
    (120, 196, 96), (74, 168, 186), (108, 128, 216), (150, 120, 200),
]


def _depth_color(depth: int) -> tuple[int, int, int]:
    return _DEPTH_COLORS[min(depth, len(_DEPTH_COLORS) - 1)]


def render_scene_overlay(scene: "Scene2D", photo) -> "object":
    """把场景图叠在照片上：每个物体一个框 + 一条接地线 + 名字。

    这张图的用途是**让人一眼验证解析对不对**：
    框画歪了、接地线画在了天花板高度、杯子漏了——都直接看得出来。
    没有这张图，"简易 3D 场景"只是个说了不算的说法。
    """
    from PIL import Image, ImageDraw, ImageFont

    img = photo.convert("RGB").copy()
    W, H = img.size
    # 放大到至少 1000 宽，否则标签挤成一团看不清
    if W < 1000:
        k = 1000 / W
        img = img.resize((int(W * k), int(H * k)), Image.LANCZOS)
        W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
        small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    except OSError:
        font = small = ImageFont.load_default()

    for o in sorted(scene.objects, key=lambda x: -x.depth):
        col = _depth_color(o.depth)
        x1, y1, x2, y2 = (o.box[0] / 999 * W, o.box[1] / 999 * H,
                          o.box[2] / 999 * W, o.box[3] / 999 * H)
        # 外框
        dr.rectangle([x1, y1, x2, y2], outline=col + (220,), width=2)
        # 接地线：实线，这是深度的来源，最重要的一条
        by = o.base_y / 999 * H
        dr.line([x1 - 6, by, x2 + 6, by], fill=col + (255,), width=3)
        # 可坐面：虚线
        if o.sit:
            sy = o.sit_y / 999 * H
            x = x1
            while x < x2:
                dr.line([x, sy, min(x + 10, x2), sy],
                        fill=(255, 255, 255, 240), width=2)
                x += 18
        label = f"{o.depth}·{o.name_cn}"
        if not o.solid:
            label += "（非实体）"
        tw = dr.textlength(label, font=font)
        ly = max(0, y1 - 22)
        dr.rectangle([x1, ly, x1 + tw + 8, ly + 21], fill=col + (235,))
        dr.text((x1 + 4, ly + 1), label, font=font, fill=(255, 255, 255, 255))

    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    # 图例
    dr2 = ImageDraw.Draw(out)
    legend = "细框=外接框  粗横线=接地线（越靠下越近）  白虚线=可坐面  数字=深度序号"
    dr2.rectangle([8, H - 34, 8 + 20 + len(legend) * 13, H - 6],
                  fill=(0, 0, 0))
    dr2.text((14, H - 30), legend, font=small, fill=(255, 255, 255))
    return out
