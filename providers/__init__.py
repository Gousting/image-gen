"""providers 子包：provider 注册表（按 name 取 provider）。

用法：
    from providers import get_provider
    p = get_provider("zimage", config)   # -> ComfyUIZImageProvider 实例
    p2 = get_provider("remote")          # -> RemoteOpenAICompatProvider 实例（懒加载默认配置）
"""
from __future__ import annotations

from providers.base import GenRequest, GenResult, ImageProvider
from providers.comfyui_qwen_image import ComfyUIQwenImageProvider
from providers.comfyui_zimage import ComfyUIZImageProvider
from providers.remote_openai_compat import RemoteOpenAICompatProvider

__all__ = [
    "GenRequest",
    "GenResult",
    "ImageProvider",
    "ComfyUIZImageProvider",
    "ComfyUIQwenImageProvider",
    "RemoteOpenAICompatProvider",
    "PROVIDER_CLASSES",
    "get_provider",
    "list_providers",
    "DEFAULT_PROVIDER",
]

# 注册表：name -> provider 类。新增 provider 只需在此登记 + 实现 ImageProvider 接口。
PROVIDER_CLASSES: dict[str, type[ImageProvider]] = {
    "zimage": ComfyUIZImageProvider,
    "qwen_image": ComfyUIQwenImageProvider,
    "remote": RemoteOpenAICompatProvider,
}

DEFAULT_PROVIDER = "zimage"


def get_provider(name: str, config: dict | None = None) -> ImageProvider:
    """按 name 取 provider 实例。未知 name 抛 KeyError。"""
    cls = PROVIDER_CLASSES.get(name)
    if cls is None:
        raise KeyError(f"未知 provider '{name}'，可用: {', '.join(sorted(PROVIDER_CLASSES))}")
    return cls(config)


def list_providers() -> dict[str, type[ImageProvider]]:
    """返回全部已注册 provider 类（副本）。"""
    return dict(PROVIDER_CLASSES)
