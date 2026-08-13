"""批量 / 混合出图编排（BatchOrchestrator）。

三个模式（见方案书 §4）：

1. 分派模式（本版实现）：``generate(reqs, provider)`` ——
   一次 batch 里显式指定哪些图走本地、哪些走外部。例：78 张塔罗牌走本地 Z-Image，
   封面主图走外部高质量模型。每张图的 style/seed 由 ``GenRequest`` 各自指定。

2. 对比模式（二期）：同一 prompt 本地出 N 张 + 外部出 M 张，全过 VLM 审查，
   取评分最高那张。

3. 升级模式（二期）：本地先出底图（快、构图/风格锁定），外部对底图做 img2img 精修。

升级 / 对比模式本版只留方法签名 + TODO，二期实现（见用户拍板决策）。
"""
from __future__ import annotations

from providers import DEFAULT_PROVIDER, get_provider
from providers.base import GenRequest, GenResult, ImageProvider


class BatchOrchestrator:
    """批量出图编排器。"""

    def __init__(self, config: dict | None = None):
        self._config = config
        self._provider_cache: dict[str, ImageProvider] = {}

    def _get(self, name: str) -> ImageProvider:
        """按名取 provider 实例（带缓存）。"""
        if name not in self._provider_cache:
            self._provider_cache[name] = get_provider(name, self._config)
        return self._provider_cache[name]

    # ------------------------------------------------------------------ #
    # 分派模式（本版实现）
    # ------------------------------------------------------------------ #
    def generate(
        self,
        reqs: list[GenRequest],
        provider: str | None = None,
    ) -> list[GenResult]:
        """分派模式：逐个出图，每张可按 ``GenRequest.provider`` 指定来源。

        ``provider`` 为默认来源（缺省 DEFAULT_PROVIDER，即本地 Z-Image）；
        单个 req 通过 ``req.provider`` 覆盖。
        """
        default_name = provider or DEFAULT_PROVIDER
        results: list[GenResult] = []
        for req in reqs:
            p = self._get(req.provider or default_name)
            results.append(p.generate(req))
        return results

    # ------------------------------------------------------------------ #
    # 对比模式（二期，只留签名）
    # ------------------------------------------------------------------ #
    def compare(
        self,
        reqs: list[GenRequest],
        local: int = 3,
        remote: int = 2,
    ) -> list[GenResult]:
        """二期：同一 prompt 本地 + 外部各出图，全过 VLM 审查取评分最高。

        TODO(二期):
        - 对每个 req 分别用本地 provider 出 local 张、外部 provider 出 remote 张
        - 全量走 VLMReviewer.score，取评分最高者返回
        - 需要外部 provider 已配置（available() == True）
        """
        raise NotImplementedError(
            "对比模式（双来源择优）留二期实现，本版只留方法签名。见 batch.py docstring。"
        )

    # ------------------------------------------------------------------ #
    # 升级模式（二期，只留签名）
    # ------------------------------------------------------------------ #
    def upgrade(
        self,
        reqs: list[GenRequest],
        provider: str | None = None,
    ) -> list[GenResult]:
        """二期：本地底图 + 外部 img2img 精修提升质感。

        TODO(二期):
        - 本地 provider 先出底图（快，构图/风格锁定）
        - 外部 provider 对底图做 img2img 精修
        - 需要外部 provider 已配置（available() == True）
        """
        raise NotImplementedError(
            "升级模式（本地底图 + 外部 img2img 精修）留二期实现，本版只留方法签名。见 batch.py docstring。"
        )
