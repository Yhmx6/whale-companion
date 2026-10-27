"""贴贴动作库 v2：按「支撑方式 × 手部呈现 × 贴附部位」三个维度设计。

## 为什么要重构成三维度

v1 只有"难度档位"，结果出了两个问题：
1. 把所有贴贴当成"挂在人身上"，导致站在地上的动作全部浮空（用户实测发现）；
2. 光按"手的任务"排难度，救不回"手必须暴露在空处还要抓握"的动作。

三个正交维度，每个都对应一个具体实现分支：

| 维度 | 取值 | 决定什么 |
|------|------|----------|
| support | person / ground | 承重在哪 —— 决定要不要程序化画地面阴影 |
| hand_mode | hidden / self / occluded / flat / grip | 手怎么呈现 —— 决定成功率 |
| anchor_region | legs / feet / shoulder / head / back / torso / beside | 贴在哪 —— 决定位置约束 |

## 手部呈现方式的成功率排序（实测得出）

    hidden（完全看不见，如背在身后）
      > self（抱自己膝盖，接触结构明确、不涉及他人）
      > occluded（绕到腿后侧被挡住）
      > flat（整只手掌平贴）
      > grip（精确抓握，几乎必败）

## 用户明确要求

全身照时把角色约束在人物腿部附近。对应 anchor_region in (legs, feet)，
由 vlm_locate() 定位人物腿部区域，把归一化坐标写进提示词做位置约束。
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@dataclass
class Action:
    id: str
    name: str
    desc: str
    tier: str                       # A 可用 / B 待验证 / C 暂不开放
    support: str = "person"         # person 挂在人身上 / ground 站在地上 / object 依附物件
    hand_mode: str = "flat"         # hidden / self / occluded / flat / grip
    anchor_region: str = "beside"   # legs / feet / shoulder / head / back / torso / beside
    # 锚点原型：动作贴在什么几何上。这是路由的依据，和 support 正交——
    # 比如"站在人物腿边"是 anchor_kind=person_part + support=ground。
    #   person_part    人物身上的部位（肩、腿、衣摆）
    #   seat_plane     水平坐面（沙发、长椅、椅面、桌沿、床沿、台阶）
    #   vertical_prop  竖直柱状物（树干、电线杆、栏杆立柱）
    #   horizontal_edge 细长水平边（栏杆横杆、窗台）
    #   ground_plane   开阔地面
    anchor_kind: str = "person_part"
    needs: list[str] = field(default_factory=list)
    framing: list[str] = field(default_factory=list)
    note: str = ""
    # **她画出来的高度占人物身高的比例**。companion 模式下没有框可依，
    # 提示词里的"约40厘米"是软约束（实测偏大 1.5 到 2 倍），所以尺度由
    # "人物全身高当尺子 + 程序化缩放"来定。
    #
    # 标定依据：40cm / 175cm ≈ 0.23，这是"小玩偶"的锚点；实测按 0.20 渲出来
    # 在整张照片里像一个指甲盖大的小手办，存在感太弱。站姿取 0.28（约 46cm，
    # 大号抱枕玩偶）、坐/蜷类取 0.24 到 0.26（坐着的视觉高度天然更矮）。
    # 曾经出现的"角色过大"是 41%，那个值是模型自己画的，不是这里的参数定的。
    # 曾经用过"占腿长比例"（0.4 到 0.5），但腿长取决于 VLM 把胯部标在哪，
    # 胯部一偏整体尺度就跟着漂，已弃用。
    scale_ratio: float = 0.24

    # 重定位时，用锚点框的哪条边当她 bbox 的底边。
    #   top    = 锚点上沿 → 她"坐在"这个面上（坐肩、坐椅面、站地面）
    #   bottom = 锚点下沿 → 她"扒在"这个边缘的上方（扒衣摆探头）
    # 两种物理关系不同，不能混：坐肩要坐在肩线上沿；扒衣摆是扒在下摆上沿，
    # 但表达上用下沿把她的身体压在布边上，视觉上才是"从布后面探出来"。
    anchor_edge: str = "top"

    # 悬空高度：她的 bbox 底边离锚点线还有多远，单位是**她自身高度的比例**。
    # 绝大多数姿态是 0（脚或臀紧贴锚点线），只有"跳起来"这类需要离开地面。
    #
    # 这个字段解决的是"接触点不在她身上"的情况：站着/坐着时，她的 bbox 底边
    # 就是接触点，阴影画在 bb[3] 天经地义；但跳在半空时，她的 bbox 底边离地
    # 还有一段，阴影必须落在地面线（foot_y）上，否则会画成"悬空的人带着
    # 一只脚边的影子"，比不画还假。
    lift_ratio: float = 0.0

    # 渲染路线。两条路的差别是"她需不需要被场景里的东西挡住"：
    #   "sprite" —— 她是一个完整、独立的人形，没有任何东西挡她。
    #               可以用白底立绘 + 程序化合成：抠图近乎无损、尺寸位置完全可控、
    #               而且立绘与照片无关，**一个姿态只生成一次，之后零成本复用**。
    #   "scene"  —— 她必须被场景里的东西挡住一部分（下半身被衣摆盖住、
    #               身体被头肩挡住、坐在肩上时人物衣服压着她）。
    #               只能在她和场景一起生成的图里抠，代价是抠图在低反差处会出错。
    render: str = "sprite"

    # ---- 「被人挡住」----
    #
    # `render="scene"` 的注释里写着"她必须被场景里的东西挡住一部分"，
    # 但那时的答案只有"和场景一起生成"。现在立绘路线也能做**被人物挡住**：
    # 先把前景（人）的轮廓切成 alpha，成品阶段再把它重新盖回来（`s3ops.occlude`）。
    #
    # 这一条改变的不只是构图好看——它推翻了"等身角色放不下"这个结论。
    # 完整可见时她需要约 700px 横向空地，竖构自拍两侧各只剩约 400px；
    # 但被挡住一半时，她只需要一条能露出来的缝。
    #
    # `behind`：站在人物身后，被挡。
    # `visible_ratio`：她的宽度里露出多少。0.55 ≈ "探出半个身子"；
    #                  0.85 ≈ "勾肩搭背"（只被挡掉一条边）。
    behind: bool = False
    visible_ratio: float = 0.55

    # 接触点落在她**自身高度**的哪个位置（0=头顶，1=bbox 底边）。
    # 决定 bbox 底边要比锚点线低多少：
    #   站立/坐地做动作 —— 脚或臀就在 bbox 底边上，取 1.0，bbox 底边对齐锚点线；
    #   坐在高处（肩上/椅面）—— 臀在约 62% 处，双腿垂在下面，取 0.62，
    #                          bbox 底边要往下沉 38% 的身高，臀才落到座位上。
    # 不设这个参数的话，"坐肩"会变成"站在肩上"——实测就是这样：她的鞋底
    # 正好压在肩线上，看起来是站着，和动作名对不上。
    contact_ratio: float = 1.0

    # 这个姿态能不能看到她的脸。
    #
    # 影响的是**表情系统**：`s3/moods.py` 那 9 个表情全是描述五官的
    # （眼睛怎么睁、嘴角朝哪、脸红不红）。背影 / 后脑勺的姿态里这些
    # 一个都画不出来，硬塞进提示词只会让模型在看不见的地方瞎猜，
    # 甚至把她转过来——那就和"她在看远方"的意图直接冲突。
    #
    # 所以背视角动作标 False，`run_guided` 会跳过表情那一段，
    # 提示词里改写成"看不到脸，靠肩膀和头部的角度表达情绪"。
    face_visible: bool = True


ACTION_LIBRARY: list[Action] = [
    # ============ A 档：已验证可用 ============
    Action(
        id="shoulder", name="坐在肩膀上", tier="A",
        desc="坐在人物的右肩上，一只手扶着人物的头发，两条短腿垂在肩膀前面，得意地笑",
        support="person", hand_mode="flat", anchor_region="shoulder",
        render="scene",
        needs=["肩膀", "头"], framing=["half", "full"],
        contact_ratio=0.62,
        note="[可用] 真的坐在肩上、腿垂下、衣服上有接触暗部。Pro 版更好",
    ),
    Action(
        id="peek", name="扒着肩膀探头", tier="A",
        desc="从人物肩膀后面探出半个身子，两只小手扒在肩膀上，只露出头和一点肩膀，好奇地看镜头",
        support="person", hand_mode="flat", anchor_region="shoulder",
        render="scene",
        needs=["肩膀"], framing=["closeup", "half"],
        note="[可用] 手只扒在肩线，遮挡简单，全部动作里最稳",
    ),

    # ============ B 档：接近可用 / 新设计待验证 ============
    Action(
        id="shoulder_back", name="趴在肩后探身", tier="B",
        desc="趴在人物右肩后方，双手从后面环住人物的肩膀，下巴搁在肩上，只露出上半身",
        support="person", hand_mode="occluded", anchor_region="back",
        needs=["肩膀", "后背"], framing=["half", "full"],
        note="[接近可用] 在其右肩后方探身、手扒在肩附近，遮挡合理。建议 Pro 复跑",
    ),
    Action(
        id="leg_hands_behind", name="站在腿边仰头", tier="A",
        desc="双脚踩在地面上，站在人物左小腿的外侧，双手背在身后完全看不见，"
             "仰起头看着人物的脸，身体微微向前倾，一副等着被摸头的样子",
        support="ground", hand_mode="hidden", anchor_region="legs",
        needs=["小腿", "脚"], framing=["full"],
        note="[可用] 双手背在身后完全看不见，手部零风险；位置略偏脚外侧，但落地与阴影正常",
        scale_ratio=0.28,
    ),
    Action(
        id="behind_peek", name="在人物身后探出半个身子", tier="A",
        desc="站在人物身后，只有半边身体从人物的肩膀后面露出来，另一半被人物的身体挡住；"
             "身体微微侧转朝着镜头，双手背在身后完全看不见，"
             "下巴略抬、笑着看向镜头，像突然从背后探出头来打招呼",
        support="ground", hand_mode="hidden", anchor_region="legs",
        needs=["小腿", "脚"], framing=["full"],
        note="[可用] 唯一依赖遮挡的动作：她的右半身被人物的轮廓切掉。"
             "遮挡靠 `s3ops.occlude`（前景 alpha 重新盖回），不是靠生成",
        scale_ratio=0.26, behind=True, visible_ratio=0.55,
    ),
    Action(
        id="leg_hug_knee", name="坐在脚边抱膝盖", tier="A",
        # ## 这条 desc 改过一版，因为第一版画出来的腿是伸直的
        #
        # 旧文案："双腿紧紧蜷起来收成一团、小腿完全折在大腿下面看不见"。
        # 出来的立绘是**膝盖在中间、小腿从膝盖往下延伸到底、脚离屁股很远**——
        # 读作"坐着伸腿"而不是"抱膝"。用户直接指出"腿过长"。
        #
        # 复盘：旧文案三句都在描述**姿态关系**（蜷起来、折在下面），
        # 没有一句描述**画面上应该看到什么**。这个模型对"关系"的遵从度低，
        # 对"看得见/看不见"这种视觉结果反而更听话（同 `simplify_hands` 那次的教训）。
        #
        # 所以新文案改成三段结构：
        #   1. 膝盖到哪（顶到下巴下面）—— 定住最高点，膝盖不能低于胸口；
        #   2. 小腿往哪折（向后折叠压在身体和地面之间）+ **正面看不到小腿和脚**——
        #      用"看不见"这个结果堵住"伸出去"这条歧义路径；
        #   3. 整体轮廓（圆滚滚、接近一个球）—— 给一个可校验的形状目标。
        desc="坐在人物脚边的地面上，把双腿收到胸前、两个膝盖并拢顶到下巴下面，"
             "大腿贴着肚子，小腿整个向后折叠、压在身体和地面之间，"
             "所以**从正面几乎看不到小腿、也看不到脚**，只在身侧露出一点点鞋尖；"
             "两只手环抱着膝盖，下巴搁在膝盖上，仰头看着人物，"
             "整个身体缩成一个圆滚滚的团子，外轮廓接近一个球",
        support="ground", hand_mode="self", anchor_region="feet",
        needs=["脚", "小腿"], framing=["full"],
        note="[可用] 手抱自己结构清楚无畸形，地面阴影正常。"
             "**姿态文案改过一版**：第一版出来腿是伸直的（用户报「腿过长」），"
             "改成描述「正面看不到小腿和脚、轮廓接近一个球」之后才画对",
        scale_ratio=0.26,
    ),
    Action(
        id="leg_lean", name="侧靠在小腿上", tier="B",
        desc="侧身靠在人物左小腿的外侧，脸和肩膀贴着裤腿，一只手臂绕到小腿后方被挡住，"
             "另一只手自然垂在身侧，抬头看人物",
        support="ground", hand_mode="occluded", anchor_region="legs",
        needs=["小腿"], framing=["full"],
        note="[接近可用] 位置对齐小腿、脚落地有阴影，但与腿之间留了缝，「靠」得不实",
        scale_ratio=0.28,
    ),
    Action(
        id="foot_sit", name="坐在鞋面上", tier="C",
        desc="坐在人物一只鞋的鞋面上，双手平撑在鞋的边缘，两条短腿垂在鞋的外侧，仰头笑",
        support="ground", hand_mode="flat", anchor_region="feet",
        needs=["脚", "鞋"], framing=["full"],
        note="[未达成] 结果是站在鞋旁边而非坐在鞋上；鞋面小且底图脚部侧向，坐不上去",
        scale_ratio=0.19,
    ),
    Action(
        id="hem_peek", name="扒着衣摆探头", tier="A",
        desc="从人物上衣下摆的边缘探出上半身，两只小手平扒在下摆的边缘上，"
             "布料被按出一点褶皱挡住部分手，只露出头和两只圆手",
        support="person", hand_mode="flat", anchor_region="torso",
        render="scene",
        needs=["衣摆", "躯干"], framing=["half", "full"],
        anchor_edge="bottom",
        note="[可用] 全部动作里最自然：扒在卫衣下摆边缘探出上半身，衣摆是长边、定位准",
    ),

    # ============ 姿态族：蹲 / 跳 / 盘腿坐 ============
    #
    # 这三条和前面的动作不是"又一个点子"，而是补上几何模型缺的两类关系：
    #
    #   蹲   接触点 = 双脚（bbox 底边），但身体被压缩，所以 scale_ratio 要比
    #        站立小得多——她是同一个人，只是蹲下来变矮了。
    #   跳   接触点**不在她身上**。bbox 底边离地还有一段，靠 lift_ratio 表达，
    #        阴影必须落到地面线而不是 bb[3]（否则像"悬空的人带着脚边的影子"）。
    #   盘腿坐 接触点 = 臀底且贴地（contact_ratio=1.0），和"坐高处的 0.62"
    #        是两种坐法，不能混。
    Action(
        id="crouch_by_leg", name="蹲在脚边", tier="A",
        desc="蹲在人物脚边的地面上，双脚平踩在地面上，两个膝盖并拢曲起来顶到胸前，"
             "两只小手抱着自己的膝盖，仰起头看着人物的脸，像一只蹲在门口的小动物",
        support="ground", hand_mode="self", anchor_region="legs",
        needs=["小腿", "脚", "地面"], framing=["full"],
        # 统一 0.26，和 leg_hug_knee 同档。**不是**按"蹲下去更矮"给小系数：
        # 实测这个模型画蹲姿时，raw 里的包围盒和站姿几乎一样高（691 vs 727，
        # 差 5%），它并不会因为蹲下就把人画矮。给小系数只会把同一个角色
        # 渲染成明显更小的一只，三个姿态并排时头部尺寸对不上。
        scale_ratio=0.26,
        note="[可用] 蹲姿成立：膝盖曲起顶到胸前、脚接地，读得出蹲。"
             "尺度系数与站姿同档——实测模型不会因蹲下而压缩包围盒",
    ),
    Action(
        id="jump_air", name="跳起来", tier="A",
        desc="整个人跳在半空中，脚下离地面有一小段空隙，两只手握成小拳头向上举过头顶，"
             "一条腿蜷起来、另一条腿向后伸，裙摆和长发因为跳跃向上飘起，笑得非常开心",
        support="ground", hand_mode="self", anchor_region="legs",
        needs=["小腿", "脚", "地面"], framing=["full"],
        scale_ratio=0.26,
        lift_ratio=0.30,
        note="[可用] 悬空姿态成立：脚下有明显空隙、头发裙摆上飘，"
             "影子落在地面线而不是她脚下。lift=0.30×身高",
    ),
    Action(
        id="sit_cross_legged", name="盘腿坐", tier="A",
        desc="盘腿坐在人物脚边的地面上，两条腿交叉盘起来藏在裙子下面，"
             "两只手分别搭在自己两侧的膝盖上，仰起头看着人物笑，尾巴在身后摊开",
        support="ground", hand_mode="self", anchor_region="feet",
        needs=["脚", "地面"], framing=["full"],
        scale_ratio=0.26,
        note="[可用] 盘腿坐成立：臀底贴地、双腿盘在裙下，contact_ratio=1.0"
             "（和坐高处的 0.62 是两种坐法）",
    ),

    # ============ 物件路线：照片里没有人物时的兜底 ============
    #
    # 只做一个动作，因为"坐沙发前沿""坐长椅前沿""坐桌沿""坐床沿"在几何上是
    # **同一件事**：一个水平面的前边缘 + 双腿垂在外面。差异只在文案里的名词，
    # 用 {anchor} 占位符在运行时替换成实际物件名（沙发/长椅/桌子…）。
    #
    # 这样 8 个场景 × 各自的动作收敛成 1 条代码路径，新增一种家具只需要
    # 在 seat_plane 的 kind 白名单里加个词，不需要新动作、不需要重新实测。
    # ============ 背视角：她背对镜头，和人一起看风景 ============
    #
    # ## 为什么需要这一族
    #
    # 改造前 23 个动作**全部是正面朝向**，而且没有一个是 `horizontal_edge`
    # （栏杆、窗台、栏杆横杆这类水平细长边）。
    #
    # 用户拿一张阳台照指出来：照片里的人是**背对镜头扶着栏杆看夕阳**的。
    # 这时如果她还是正面朝镜头，会出现两个问题：
    #   1. 视线方向打架——人在看远方，她在看观众；
    #   2. 姿态关系不对——两个人没有在"一起看"某样东西。
    #
    # 所以补两条：`face_visible=False` 的背视角，锚点是栏杆横杆。
    Action(
        id="back_look_up", name="背对站着仰头看", tier="C",
        # ## 这一条是背视角的"安全款"
        #
        # `rail_back` 试了两次都没洗干净：desc 里写"手扶着栏杆"，模型就会
        # **把栏杆画进立绘**——贴到照片里多出一截假横杆，抠图时还会被当成
        # 角色的一部分保留。第二次改进提示词（"画面里只画角色本人"）后
        # 仍然剩一根横杆，而且手臂画成完全伸直、像在拉单杠。
        #
        # 根因是：只要姿态依赖某个外部物体，立绘阶段就必然要"画点什么给她抓"。
        # 所以补这一条**不需要抓任何东西**的：站着、仰头、手自然垂着。
        # 它同样满足"背对镜头、和人一起看远处"，而且构图干净、零依赖。
        desc="**完全背对镜头**站着，只能看到她的后脑勺、头发和背影，看不到脸；"
             "双手自然垂在身体两侧，双脚并拢站在地面上，"
             "**头微微仰起来看着前方的远处**，蓝色长发从背后一路垂下去，"
             "鲸鱼尾巴在身后自然垂着、尾尖轻轻翘起，站得笔直",
        support="ground", hand_mode="hidden", anchor_region="legs",
        anchor_kind="ground_plane",
        needs=["地面"], framing=["full"],
        scale_ratio=0.28,
        face_visible=False,
        note="[有风险] 立绘本身很干净（无道具、无畸形），但**实测在"
             "「照片里的人也是背对镜头」时会出事**：h08 阳台照上，"
             "成品里原照片的女性整个消失了——模型把'小背影'和'大背影'搞混，"
             "连同尺寸一起放大到 47%，直接把她当人物本体重画。"
             "**只在照片里的人正面或侧身时用**；人背对时请用 rail_back",
    ),
    Action(
        id="rail_back", name="扶着栏杆一起看远处", tier="B",
        desc="**背对镜头**站在地面上，两只手向前上方伸出、手肘微微弯着，"
             "十指虚握成扶着某物的姿势（**那个物体本身不要画出来，手是空的**），"
             "**仰起头看着前方的远处，只能看到她的后脑勺、背影和一点点侧脸轮廓**，"
             "蓝色长发从背后垂下来，鲸鱼尾巴在身后轻轻翘起，站得笔直",
        support="ground", hand_mode="grip", anchor_region="legs",
        anchor_kind="horizontal_edge",
        needs=["栏杆", "地面"], framing=["full"],
        # 降为 C 档：试了两次，"扶着栏杆"这个语义会**必然把栏杆画进立绘**
        # （见 `back_look_up` 的注释）。等找到"手抓空物"的写法再升档。
        scale_ratio=0.28,
        face_visible=False,
        note="[可用] 实测 h08 阳台照：她背对镜头站在人腿边、双手举着，"
             "视线方向和人一致，原照片的人物完整保留。"
             "**已知瑕疵**：立绘里原本画着一根横杆（试了两次没洗干净），"
             "贴上去后手是空抓的，看起来像虚握着空气。"
             "动态姿势反而避免了'和人物姿态太像被合并'的问题",
    ),
    Action(
        id="rail_lean_back", name="趴在栏杆上看远处", tier="C",
        desc="**背对镜头**趴着，上半身向前靠在某个齐腰高的横杆上，两条短腿在横杆外侧悬着晃，"
             "两只小手平搭在横杆上沿（**横杆本身不要画出来**），"
             "**头微微抬起来看着前方的远处，只能看到她的背影**，"
             "蓝色长发垂在身前，鲸鱼尾巴在身后翘起来",
        support="ground", hand_mode="flat", anchor_region="legs",
        anchor_kind="horizontal_edge",
        needs=["栏杆"], framing=["full"],
        scale_ratio=0.26,
        anchor_edge="bottom",
        face_visible=False,
        note="[未验证] 同样依赖一个外部横杆，先标 C 档。等 `back_look_up` "
             "跑通后再回来看这条值不值得做",
    ),
    Action(
        id="seat_edge_sit", name="坐在边缘垂腿", tier="A",
        desc="坐在{anchor}的前边缘上，两条短腿自然垂在{anchor}外侧，"
             "两只小手撑在身侧的边缘上保持平衡，仰起头看镜头，笑得很开心",
        support="object", hand_mode="flat", anchor_region="seat",
        anchor_kind="seat_plane",
        needs=["座面前边缘"], framing=["full", "half"],
        anchor_edge="bottom",
        # 坐在座面上时臀在自身高度约 62% 处，bbox 底边（脚）要沉到座面线以下
        contact_ratio=0.62,
        # 物件路线**不用**这个值：照片里没有人物，就没有"人物身高"这把尺子，
        # 实际尺度由 pipeline 里的 SEAT_H_FRAC / SEAT_W_RATIO 按家具种类算。
        # 留 0.33 只是让字段有合理默认，避免别的代码路径拿到 0。
        scale_ratio=0.33,
        note="[可用] 软硬座面都成立。三张实测：软沙发（家居照）、木板条公园长椅（中景）、"
             "原木桌沿（近景特写）——臀底都落在座面前沿线上、双腿垂在座面外，读得出「坐」。"
             "限定：座面必须有清晰的横向前沿线；近景（座面占画面宽 >60%）靠第二把尺子补偿，"
             "再近到只剩桌角时会偏大",
    ),

    # ============ C 档：暂不开放 ============
    Action(
        id="on_head", name="坐在头顶", tier="C",
        desc="盘腿正坐在人物头顶的正中央，双手撑在头发上保持平衡，低头看着镜头做鬼脸",
        support="person", hand_mode="flat", anchor_region="head",
        needs=["头顶"], framing=["closeup", "half"],
        note="[待优化] 偏到头发右侧有悬浮感，缺「正中央」这个词；头顶面积小、坐姿不稳",
    ),
    Action(
        id="hug_calf", name="抱住小腿", tier="C",
        desc="蹲坐在人物左脚边，双手环抱住人物的小腿，脸贴在小腿外侧，仰头看着人物撒娇",
        support="ground", hand_mode="grip", anchor_region="legs",
        needs=["小腿", "脚"], framing=["full"],
        note="[偏弱] 蹲在脚边位置对，但看不出手在环抱；grip 类，优先改用 leg_hug_knee",
        scale_ratio=0.26,
    ),
    Action(
        id="hug_thigh", name="抱大腿", tier="C",
        desc="坐在人物脚边，双臂环抱住人物的大腿，整张脸埋进裤腿里，只露出一点侧脸",
        support="ground", hand_mode="grip", anchor_region="legs",
        needs=["大腿"], framing=["full"],
        note="[待优化] 身体贴在大腿侧面尚可，手部不清；同属 grip 类",
        scale_ratio=0.26,
    ),
    Action(
        id="hug_arm", name="抱住手臂", tier="C",
        desc="双手抱住人物的手臂，整个身体挂在手臂上，脸贴在手臂外侧",
        support="person", hand_mode="grip", anchor_region="beside",
        needs=["手臂"], framing=["half", "full"],
        note="[待优化] 偏「贴在手臂旁」而非抱住，grip 类的天花板",
    ),
    Action(
        id="pull_hem", name="扯衣角", tier="C",
        desc="双脚站在地面上，站在人物身侧，身体微微向后倾，一只手向前伸直手臂拉住上衣下摆",
        support="ground", hand_mode="grip", anchor_region="legs",
        needs=["衣摆"], framing=["full"],
        note="[维持C] 补地面阴影后不浮空，但后倾加伸直手臂在 2.5 头身下没表达空间",
        scale_ratio=0.28,
    ),
    Action(
        id="pull_trouser", name="扯裤脚", tier="C",
        desc="蹲在人物脚边，一只手拉住人物的裤脚往下拽，仰头嘟嘴看着人物",
        support="ground", hand_mode="grip", anchor_region="feet",
        needs=["裤脚", "腿"], framing=["full"],
        note="[维持C] 位置飘到腿外侧，手没接触裤脚",
        scale_ratio=0.24,
    ),
    Action(
        id="pocket", name="从口袋里探出", tier="C",
        desc="从人物的上衣口袋里探出上半身，两只小手搭在口袋边缘",
        support="person", hand_mode="flat", anchor_region="torso",
        needs=["口袋"], framing=["half", "full"],
        note="[未达成] 口袋这种小区域模型定位不住；已由 hem_peek 替代",
    ),
    Action(
        id="hand_hold", name="牵手", tier="C",
        desc="站在人物身侧，一只手牵着人物的一根手指",
        support="ground", hand_mode="grip", anchor_region="beside",
        needs=["手"], framing=["half", "full"],
        note="[失败] 手指对位是模型最弱项",
        scale_ratio=0.28,
    ),
    Action(
        id="lap", name="坐在腿上", tier="C",
        desc="坐在人物的腿上，仰头看着人物，双手拉着人物的衣角",
        support="person", hand_mode="grip", anchor_region="torso",
        needs=["腿", "躯干"], framing=["half", "full"],
        note="[失败] 需配坐姿照片；本轮底图为站姿，属配对错误",
    ),
    Action(
        id="cheek", name="贴脸", tier="C",
        desc="贴在人物脸侧，双手捧着自己的脸，和人物头靠头",
        support="person", hand_mode="flat", anchor_region="head",
        needs=["脸"], framing=["closeup"],
        note="[失败] 浮在脸旁，做不到头靠头",
    ),
]

# 默认开放白名单：显式列举，不按档位筛（实测 A/B 档互有胜负）
DEFAULT_ENABLED_IDS = ("shoulder", "peek", "leg_hug_knee", "hem_peek",
                       "leg_hands_behind", "seat_edge_sit",
                       "crouch_by_leg", "jump_air", "sit_cross_legged",
                       "behind_peek",
                       # 背视角：只放实测可用的那一条。
                       # `back_look_up` / `rail_lean_back` 都是 C 档——
                       # 前者会把照片里背对的人擦掉，后者没验证过、也依赖外部横杆。
                       # 它们留在动作库里但不在白名单，需要时用 `action=` 显式指定。
                       "rail_back")

# 用户明确要求：全身照时把角色约束在人物腿部附近
LEG_ACTIONS = ("leg_hands_behind", "leg_hug_knee", "leg_lean", "foot_sit")


def get_action(action_id: str) -> Action | None:
    for a in ACTION_LIBRARY:
        if a.id == action_id:
            return a
    return None


def enabled_actions() -> list[Action]:
    return [a for a in ACTION_LIBRARY if a.id in DEFAULT_ENABLED_IDS]


def experimental_actions() -> list[Action]:
    return [a for a in ACTION_LIBRARY if a.id not in DEFAULT_ENABLED_IDS]


def as_dicts(actions: list[Action] | None = None) -> list[dict]:
    return [
        {"id": a.id, "name": a.name, "tier": a.tier, "support": a.support,
         "hand_mode": a.hand_mode, "anchor_region": a.anchor_region,
         "anchor_kind": a.anchor_kind,
         "framing": a.framing, "needs": a.needs, "note": a.note}
        for a in (actions if actions is not None else ACTION_LIBRARY)
    ]


# ---------------------------------------------------------------------------
# VLM：定位人物部位（供"约束在腿部附近"用）
# ---------------------------------------------------------------------------

LOCATE_SYS = (
    "你是图像标注助手。给你一张人物照片，请给出指定部位的归一化外接框。\n"
    "坐标归一化到 0 到 999：左上角是 (0,0)，右下角是 (999,999)。\n"
    "只输出 JSON，不要解释、不要代码块标记。格式必须是：\n"
    '{"bbox": [x1, y1, x2, y2], "confidence": 0.9}\n'
    "bbox 是四个整数按顺序排列：左、上、右、下。\n"
    '如果该部位在画面里看不见，输出 {"error": "not_visible"}。'
)


def vlm_locate(photo_path: str, part: str = "人物的双腿（从小腿到鞋）",
               timeout: int = 60) -> dict:
    """让 VLM 给出某个部位的归一化外接框。失败返回带 error 的 dict。"""
    from api.ark import ENV, _post, image_path_to_data_uri

    payload = {
        "model": ENV.get("ARK_VLM_MODEL", "doubao-seed-2-1-lite-260915"),
        "messages": [
            {"role": "system", "content": LOCATE_SYS},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": image_path_to_data_uri(photo_path, max_side=640)}},
                {"type": "text", "text": f"请给出{part}的外接框。"},
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": 160,
        # 关思考：默认模式 39 秒，关掉约 1.5 秒
        "thinking": {"type": "disabled"},
    }
    st, resp = _post("/chat/completions", payload, timeout=timeout)
    if st != 200:
        return {"error": f"HTTP {st}: {str(resp)[:160]}"}
    try:
        txt = resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return {"error": "bad_response"}
    txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    # 主路径：标准 JSON
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and not data.get("error"):
        bb = data.get("bbox")
        if isinstance(bb, list) and len(bb) >= 4:
            return _checked([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
                            data.get("confidence", 0.0))
        if all(isinstance(data.get(k), (int, float)) for k in ("x1", "y1", "x2", "y2")):
            return _checked([int(data["x1"]), int(data["y1"]),
                             int(data["x2"]), int(data["y2"])],
                            data.get("confidence", 0.0))
    if isinstance(data, dict) and data.get("error"):
        return data

    # 兜底：模型常把 JSON 拼坏（实测吐过 {"x1": 386, 573, "x2": 613, 986}），
    # 但数值本身是对的。按出现顺序取前 4 个数当 x1,y1,x2,y2。
    import re as _re
    nums = [int(v) for v in _re.findall(r"-?\d+", txt)][:4]
    if len(nums) == 4:
        return _checked(nums, 0.0, "regex_fallback")
    return {"error": "not_json", "raw": txt[:200]}


def _checked(bbox: list, conf: float, parsed: str = "json") -> dict:
    """合理性校验：退化框一律判为不可用。

    半身照里腿本来就看不见，VLM 会返回贴着底边的整幅横条（实测 [0,917,999,999]），
    这种框拿去做位置约束会把角色按到画面最底下，必须拦掉。
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 40 or h < 60:
        return {"error": "bbox_too_small", "bbox": bbox}
    if w > 850:
        return {"error": "bbox_too_wide", "bbox": bbox}
    if y1 > 900:
        return {"error": "bbox_at_bottom", "bbox": bbox}
    return {"bbox": bbox, "confidence": conf, "parsed": parsed}


PAIR_SYS = (
    "你是图像标注助手。给你一张人物照片，请给出两个归一化外接框，并判断光线方向。\n"
    "坐标归一化到 0 到 999：左上角是 (0,0)，右下角是 (999,999)。\n"
    "person 是整个人的外接框：上边取头发最高点，下边取鞋底最低点。\n"
    "legs 是双腿的外接框：上边取胯部，下边取鞋底最低点。\n"
    "light 是主光来向，只能取 \"left\"、\"right\"、\"top\"、\"flat\" 之一：\n"
    "  left  影子朝画面右侧 / 人物右脸更亮（光从左边来）\n"
    "  right 影子朝画面左侧 / 人物左脸更亮（光从右边来）\n"
    "  top   正午顶光，影子在脚下缩成一圈\n"
    "  flat  阴天或室内散射光，几乎看不出影子方向\n"
    "只输出 JSON，不要解释、不要代码块标记。格式必须是：\n"
    '{"person": {"bbox": [x1, y1, x2, y2]}, "legs": {"bbox": [x1, y1, x2, y2]}, '
    '"light": "left"}\n'
    "bbox 是四个整数按顺序排列：左、上、右、下。\n"
    '如果这个人不在画面里，输出 {"error": "not_visible"}。'
)


def vlm_locate_pair(photo_path: str, timeout: int = 60) -> dict:
    """一次调用同时拿到「整个人」和「双腿」两个框。

    ## 为什么必须用整人身高当尺子

    贴贴模式下角色该多大，不能用提示词里的"约 40 厘米"来定（软约束，实测偏大
    1.5 到 2 倍），也不能用"腿长 × 某个比例"——腿长取决于 VLM 把胯部画在哪，
    胯部一偏，目标高度跟着整体放大或缩小，误差不可控。

    整人身高是唯一稳定的基准：40cm / 175cm ≈ 0.23，这是纯人体测量学比例，
    不依赖任何估算。所以 scale_ratio 的语义定义为**占人物身高的比例**。

    两次分别调用也能做，但会多花 1.5 秒和一次配额；实测一次问两个框，
    VLM 输出稳定（本机实测返回 {"person": {...}, "legs": {...}}）。
    """
    from api.ark import ENV, _post, image_path_to_data_uri

    payload = {
        "model": ENV.get("ARK_VLM_MODEL", "doubao-seed-2-1-lite-260915"),
        "messages": [
            {"role": "system", "content": PAIR_SYS},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": image_path_to_data_uri(photo_path, max_side=640)}},
                {"type": "text", "text": "请给出这两个外接框。"},
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": 220,
        "thinking": {"type": "disabled"},      # 关思考：39 秒 -> 约 1.5 秒
    }
    st, resp = _post("/chat/completions", payload, timeout=timeout)
    if st != 200:
        return {"error": f"HTTP {st}: {str(resp)[:160]}"}
    try:
        txt = resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return {"error": "bad_response"}
    txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    out: dict = {}
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # 兜底：VLM 偶尔把 JSON 拼坏，按 key 逐个用正则挖方括号里的四个整数
        data = {}
        for key in ("person", "legs"):
            m = re.search(key + r'"?\s*:\s*\{?\s*"?bbox"?\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
                          txt)
            if m:
                data[key] = {"bbox": [int(g) for g in m.groups()]}

    if not isinstance(data, dict) or data.get("error"):
        return data if isinstance(data, dict) else {"error": "bad_json"}

    for key in ("person", "legs"):
        node = data.get(key)
        bb = node.get("bbox") if isinstance(node, dict) else None
        if isinstance(bb, list) and len(bb) >= 4:
            chk = _checked([int(v) for v in bb[:4]], 0.9)
            if "bbox" in chk:
                out[key] = chk["bbox"]
    light = data.get("light")
    if isinstance(light, str) and light.strip().lower() in ("left", "right", "top", "flat"):
        out["light"] = light.strip().lower()
    if "person" not in out and "legs" not in out:
        return {"error": "no_valid_bbox", "raw": txt[:160]}
    return out


MATCH_SYS = (
    "你是贴图姿势顾问。给你一张人物照片，从给定的动作清单里选出**最能成立的一个**。\n"
    "判断依据：照片里哪些身体部位清晰可见、人物的姿态、取景景别（特写/半身/全身）。\n"
    "硬性规则：\n"
    "1. 只从清单里选，不要发明新动作；\n"
    "2. 动作需要的部位在照片里看不见的，直接排除（例如半身照看不到腿，就不能选腿部动作）；\n"
    "3. 全身照优先选腿部动作，半身照优先选上半身或衣摆动作；\n"
    "4. 人物贴身物件（肩膀、衣摆、头发）比小区域（口袋、手指）可靠，同等条件下优先。\n"
    "只输出 JSON，不要解释、不要代码块标记。格式必须是：\n"
    '{"action_id": "清单里的某个 id", "confidence": 0.0 到 1.0 的小数, '
    '"reason": "一句中文说明为什么选它", "visible_parts": ["可见部位", "..."]}\n'
)


def vlm_match_action(photo_path: str, actions: list["Action"] | None = None,
                     timeout: int = 60) -> dict:
    """看照片，从候选动作里挑一个最可能成功的，返回给界面做「自动匹配」。

    返回结构要和 `app/static/app.js` 对齐：
        {action_id, confidence, reason, visible_parts[]}
    失败时返回 {"error": ...}。

    `actions` 由调用方传入**已经按景别过滤过**的候选集（见 `s3/profile.py`
    的 `filter_by_framing`）。不在这里自己过滤，是为了避免 actions -> profile
    的反向依赖：profile 需要 actions 里的 Action 类型和 _checked。

    ## 为什么让 VLM 选而不是用规则

    规则版要写「如果能看到脚就选腿部动作」，但「能看到脚」本身就不好判定：
    半身照可能露出一点鞋尖，规则会误判成全身照。VLM 看整张图直接判断景别，
    一次调用 1.5 秒，比堆规则准。

    ## 清单里为什么要带 desc 和 needs

    只给 id 和名字，模型不知道每个动作到底要什么。把「需要哪些部位」和
    「适配哪种景别」一起写进提示词，它才能做排除法。
    """
    acts = actions if actions is not None else enabled_actions()
    if not acts:
        return {"error": "no_candidate_action"}
    catalog = "\n".join(
        f'- id={a.id}｜{a.name}｜{a.desc}｜需要部位: {"、".join(a.needs) or "无"}'
        f'｜适配景别: {"、".join(a.framing) or "不限"}'
        for a in acts
    )

    from api.ark import ENV, _post, image_path_to_data_uri

    payload = {
        "model": ENV.get("ARK_VLM_MODEL", "doubao-seed-2-1-lite-260915"),
        "messages": [
            {"role": "system", "content": MATCH_SYS},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": image_path_to_data_uri(photo_path, max_side=640)}},
                {"type": "text", "text": f"可选动作清单：\n{catalog}\n\n请选出最合适的一个。"},
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "thinking": {"type": "disabled"},
    }
    st, resp = _post("/chat/completions", payload, timeout=timeout)
    if st != 200:
        return {"error": f"HTTP {st}: {str(resp)[:160]}"}
    try:
        txt = resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return {"error": "bad_response"}
    txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data = None
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # 兜底：JSON 拼坏时按 key 单个挖
        m = re.search(r'"?action_id"?\s*:\s*"([a-z_]+)"', txt)
        if m:
            cm = re.search(r'"?confidence"?\s*:\s*([0-9.]+)', txt)
            rm = re.search(r'"?reason"?\s*:\s*"([^"]*)"', txt)
            data = {"action_id": m.group(1),
                    "confidence": float(cm.group(1)) if cm else 0.5,
                    "reason": rm.group(1) if rm else ""}
    if not isinstance(data, dict) or data.get("error"):
        return data if isinstance(data, dict) else {"error": "bad_json"}

    ids = [a.id for a in acts]
    aid = data.get("action_id")
    # "none" 是合法信号：模型认为清单里没有一个能成立。不要当成解析错误，
    # 上层要靠它做"换一条路线再试"的决策（实测空沙发照被误判成有人物时，
    # 模型就是靠这个信号告诉我们走错路线了）。
    if str(aid).lower() in ("none", "null", ""):
        return {"error": "no_fit", "reason": str(data.get("reason", ""))[:200]}
    if aid not in ids:
        return {"error": f"模型给了清单外的动作: {aid}", "raw": txt[:160]}

    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    parts = data.get("visible_parts")
    if not isinstance(parts, list):
        parts = []
    return {
        "action_id": aid,
        "confidence": max(0.0, min(1.0, conf)),
        "reason": str(data.get("reason", ""))[:200],
        "visible_parts": [str(p) for p in parts][:8],
    }


if __name__ == "__main__":
    print("默认开放:", [a.id for a in enabled_actions()])
    print("总共:", len(ACTION_LIBRARY), "个动作")
    scene = os.path.join(ROOT, "assets", "scenes", "selfie", "s04_full.jpg")
    print("定位双腿:", vlm_locate(scene))
    print("自动匹配:", vlm_match_action(
        os.path.join(ROOT, "output", "runs", "16afa9bcc5a6", "scene.jpg")))
