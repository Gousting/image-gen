"""Provider 抽象层 —— 本项目的「灵魂」。

调用方只面向 ``ImageProvider`` 接口，不写死任何具体模型：
换模型 = 换 provider，``generate(req) -> GenResult`` 的契约不变。

- ``GenRequest``  : 一次生成请求（画面描述 + 风格 + 尺寸 + seed + negative）
- ``GenResult``   : 一次生成结果（落盘路径 + 来源 provider + 元信息）
- ``ImageProvider``: 所有模型统一实现的抽象接口
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenRequest:
    """一次生图请求。

    Attributes:
        prompt:   正向画面描述。注意 Z-Image 走 CFG=0、不依赖 negative，
                  因此 prompt 里不能出现「不要 XX」这类否定式约束，
                  必须转成正向描述（例：「不要亮色」→「strictly monochrome dark sepia」）。
        style:    风格锚定名，如 "ink_frenzy" / "classic_film" / "shinkai"。
                  一套图传同一个 style 名即可保证跨图风格一致。
        width:    目标宽度（Z-Image 会就近映射到其 ratio 档位）。
        height:   目标高度。
        size:     Z-Image 尺寸档位（small / medium (recommended) / large，取值来自
                  object_info 的 EmptyZImageLatentImage.size 档位）；None = 用 config
                  默认（zimage.default_size）。其它 provider 忽略。
        seed:     随机种子；None 表示由 provider 自动生成。
        negative: 负向提示词，仅供支持 negative 的模型使用；Z-Image 会直接忽略。
        provider: 可选，指定由哪个 provider 出图（分派模式用）；None 走默认 provider。
    """
    prompt: str
    style: str = "generic"
    width: int = 1344
    height: int = 768
    size: str | None = None
    seed: int | None = None
    negative: str | None = None
    provider: str | None = None


@dataclass
class GenResult:
    """一次生图结果。

    Attributes:
        image_path: 落盘后的图片绝对路径。
        provider:   是哪个模型出的图（对应 ``ImageProvider.name``）。
        meta:       附加元信息（seed / 耗时 / 原始响应等），供上层与审查循环使用。
    """
    image_path: str
    provider: str
    meta: dict[str, Any] = field(default_factory=dict)


class ImageProvider(ABC):
    """所有生图模型统一实现的抽象接口。"""

    name: str = "base"
    is_remote: bool = False  # False=本地，True=外部 API

    @abstractmethod
    def generate(self, req: GenRequest) -> GenResult:
        """执行一次生成，返回落盘结果。"""
        raise NotImplementedError

    @abstractmethod
    def available(self) -> bool:
        """健康检查：就绪返回 True，未就绪（离线/未配置）返回 False。"""
        raise NotImplementedError
