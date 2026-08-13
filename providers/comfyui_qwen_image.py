"""本地 Qwen-Image provider —— 空骨架（未来接入位，本次不实现）。

这是「模型无关」设计的验证点：未来要换 Qwen-Image，只需把这个类填上，
调用方与 VLM 审查循环零改动。当前 ``available()`` 恒为 False。

TODO（二期）：
- 调 ComfyUI 的 Qwen-Image 节点（本机已有 qwen_image_edit_2509_fp8 相关模型文件，
  但节点/workflow 未验证，暂不接入）。
- 实现 ``generate`` 时复用 ``ComfyUIZImageProvider`` 的提交/轮询/下载逻辑。
"""
from __future__ import annotations

from providers.base import GenRequest, GenResult, ImageProvider


class ComfyUIQwenImageProvider(ImageProvider):
    name = "qwen_image"
    is_remote = False

    def generate(self, req: GenRequest) -> GenResult:
        raise NotImplementedError(
            "Qwen-Image 本地接入留二期（本类为骨架占位，available() 恒为 False）"
        )

    def available(self) -> bool:
        # 骨架：未接入，永远不可用
        return False
