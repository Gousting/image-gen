"""风格锚定注册表。

一套图锁一个风格：调用时传风格名（如 ``ink_frenzy``），provider 拿到对应的
Z-Image style 组合值 + 锚定提示词 + 默认约束，保证跨图一致。

每个风格 = ``Style``：
- ``key``           : 注册表键名（对外风格名）
- ``zimage_style``  : StylePromptEncoder2 节点的 style 组合值（带引号），"none" 表示不套风格
- ``anchor``        : 风格锚定提示词（正向描述，拼接到画面 prompt 上）
- ``constraints``   : 默认约束（正向描述）

Z-Image 的坑：CFG=0、无 negative，所以「文字最少化/无文字」这类约束必须写成正向
描述（"text-free, no watermark"），而不是"不要文字"。见 README「Z-Image 的坑」。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Style:
    key: str
    zimage_style: str                       # StylePromptEncoder2 的 style 值（带引号）或 "none"
    anchor: str = ""                        # 风格锚定正向提示词
    constraints: tuple[str, ...] = field(default_factory=tuple)  # 默认正向约束

    def build_prompt(self, prompt: str) -> str:
        """把画面描述 + 风格锚定 + 默认约束拼成最终正向 prompt。"""
        parts = [prompt.strip()]
        if self.anchor:
            parts.append(self.anchor.strip())
        parts.extend(c.strip() for c in self.constraints if c.strip())
        return ", ".join(parts)


# 通用约束：Z-Image 画面内文字易乱码，默认要求文字最少化（正向描述）。
_NO_TEXT = "text-free, no visible text, no watermark, no lettering, no captions"


STYLES: dict[str, Style] = {
    # 塔罗牌牌面 / TRPG 场景卡素材：手绘水墨狂草
    "ink_frenzy": Style(
        key="ink_frenzy",
        zimage_style='"Ink Frenzy"',
        anchor=(
            "hand-drawn ink frenzy illustration, bold dynamic ink strokes, "
            "high-contrast black ink on textured paper, tarot card and tabletop RPG art style"
        ),
        constraints=(_NO_TEXT,),
    ),
    # 胶片写实：Classic Film Photo
    "classic_film": Style(
        key="classic_film",
        zimage_style='"Classic Film Photo"',
        anchor=(
            "classic film photograph, analog film grain, natural light, "
            "realistic documentary look, muted tones"
        ),
        constraints=(_NO_TEXT,),
    ),
    # 新海诚动画雨景（Z-Image 无「Shinkai」样式，用 Anime 组合值 + 锚定提示词逼近）
    "shinkai": Style(
        key="shinkai",
        zimage_style='"Anime"',
        anchor=(
            "Makoto Shinkai anime style, cinematic rain, luminous volumetric clouds, "
            "detailed sky, vivid color grading, nostalgic atmospheric lighting"
        ),
        constraints=("no subtitles, text-free, no watermark",),
    ),
    # 通用无风格占位：不套任何 Z-Image 预设样式，只约束文字
    "generic": Style(
        key="generic",
        zimage_style="none",
        anchor="",
        constraints=(_NO_TEXT,),
    ),
}


def resolve_style(name: str) -> Style:
    """按名取风格，未知风格抛 KeyError 并列出可用名。"""
    style = STYLES.get(name)
    if style is None:
        raise KeyError(f"未知风格 '{name}'，可用风格: {', '.join(sorted(STYLES))}")
    return style


def list_styles() -> dict[str, Style]:
    """返回全部已注册风格（副本）。"""
    return dict(STYLES)
