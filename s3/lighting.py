"""LightingState — 贯穿全链的光源数据契约。

依据：《06-管线设计与失败形态映射》§1 与《07-待实测登记表》。
设计要点：光源是结构化对象而非提示词字符串。S1 估算，S2 序列化成提示词，
S3 直接读字段渲染阴影。同一个数值，两个消费者，物理上不可能打架。

字段注释里的"待实测"均对应 07 号文档的 U 编号。
"""

from dataclasses import dataclass


@dataclass
class LightingState:
    # 主光方向（世界坐标近似到画面坐标的投影）
    light_azimuth_deg: float = 180.0   # 光来自画面何方：0=右 90=下 180=左 270=上
    elevation_deg: float = 40.0        # 仰角，影响影子长度（越高越短）

    quality_hardness: float = 0.55     # 0=柔光 1=硬光；>0.85 时投射阴影改 4 段
    intensity: float = 0.85            # 整体强度，缩放所有阴影不透明度

    confidence: float = 0.78           # 估光置信度；<0.5 降级 AO-only（U-09 联动）

    # 重力方向（06 §4）：照片重力相对图像竖直轴的偏角，仰拍时非零。
    gravity_deg: float = 0.0
    gravity_source: str = "plane_normal"  # plane_normal / vanishing_point / zero

    # ---- 派生量 -----------------------------------------------------------

    def shadow_angle_deg(self) -> float:
        """投射阴影在画面上的延伸方向（度，顺时针，0=向右，90=向下）。

        约定：light_azimuth_deg 是"光来自画面何方"，用同一套角度语义
        （0=右 90=下 180=左 270=上）。影子向光的反方向延伸，所以直接
        +180 取反即可 —— 注意只能取反一次。

        历史 bug（2026-10-26 修复）：初版在 +180 之后又做了一次
        (180 - x) 变换，导致光从左来时影子指向左（方向完全反了）。
        这正是"错的阴影比没阴影更糟"要防的情形，已在合成测试图上肉眼验出。
        """
        # 影子延伸方向 = 光来向的反向，然后叠加地面透视偏置（影子随地面向灭点收）
        return (self.light_azimuth_deg + 180.0 + 12.0) % 360.0

    def shadow_tier(self) -> str:
        """三级降级（06 §2.4 / 07 §3 第 9 条）。

        full    三段投射 + AO
        ao_near 仅 AO + 近段（方向大概对但不敢赌长距离）
        ao_only 仅 AO（无方向，置信度不足时永远安全）
        """
        if self.confidence >= 0.6:
            return "full"
        if self.confidence >= 0.4:
            return "ao_near"
        return "ao_only"

    def serialize_prompt(self) -> str:
        """S2 消费用：序列化成提示词片段。与 S3 读的是同一份字段。"""
        side = {0: "from the right", 90: "from below", 180: "from the left",
                270: "from above"}.get(int(self.light_azimuth_deg) % 360 // 90 * 90,
                                       "from the left")
        hard = "hard" if self.quality_hardness > 0.7 else "soft"
        return (f"main light {side}, {hard} quality, "
                f"elevation {self.elevation_deg:.0f} degrees; "
                "do NOT generate ground cast shadow")

    def as_dict(self) -> dict:
        return {
            "light_azimuth_deg": self.light_azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "quality_hardness": self.quality_hardness,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "gravity_deg": self.gravity_deg,
            "gravity_source": self.gravity_source,
            "shadow_tier": self.shadow_tier(),
        }
