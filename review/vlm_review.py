"""VLM 视觉审查循环：出图 → 评分 → 低分换 seed 重生成。

视觉模型：qwen3.8-max（OpenAI-compatible，走 opencode.ai/zen/go/v1 网关，key 从配置读）。
送审前把图片做 JPEG 压缩（默认目标 ~80KB，最长边缩放），转 base64 data URL。

评分维度：
- 构图（composition）
- 光影（lighting）
- 风格一致（style consistency）
- 文字不乱码（text legibility）
- AI 味（AI-flavor：对称构图、聚光灯、套路隐喻 → 扣分）

阈值默认 75 分，低分换 seed 重生成，最多 N 次（可配）。
"""
from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import requests

from config import is_placeholder, load_config
from providers.base import GenRequest, GenResult, ImageProvider

_SCORE_PROMPT = """You are a strict art director reviewing an AI-generated image.

Score the image on 0-100 across these dimensions:
- composition (构图): balanced, intentional, no lazy centered symmetry
- lighting (光影): natural, motivated light; no flat "spotlight" look
- style consistency (风格一致): matches the requested style
- text legibility (文字): any text/letters in the image must be clean, NOT garbled/mojibake
- AI-flavor (AI 味): penalize obvious AI tells — perfectly symmetric composition,
  dramatic spotlight, cliché metaphors, over-saturated glow, plastic skin

Reply with ONLY a JSON object, no markdown fences, in this exact shape:
{"score": <int 0-100>, "opinion": "<one concise Chinese sentence explaining the main issue or why it passes>"}
"""


@dataclass
class ReviewResult:
    """一次审查结果（含重生成次数）。"""
    score: float
    opinion: str
    image_path: str
    attempts: int = 1
    meta: dict[str, Any] = field(default_factory=dict)


class VLMReviewer:
    """视觉审查器 + 重生成循环。"""

    def __init__(self, config: dict | None = None):
        cfg = config if config is not None else load_config()
        vcfg = cfg.get("vlm", {}) or {}
        self.base_url = (vcfg.get("base_url") or "").rstrip("/")
        self.api_key = vcfg.get("api_key") or ""
        self.model = vcfg.get("model", "qwen3.8-max")
        self.threshold = float(vcfg.get("score_threshold", 75))
        self.max_retries = int(vcfg.get("max_retries", 3))

        ocfg = cfg.get("output", {}) or {}
        self.jpeg_max_side = int(ocfg.get("jpeg_max_side", 1024))
        self.jpeg_quality = int(ocfg.get("jpeg_quality", 80))

    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        """是否可用：已配置真实 api_key。"""
        return bool(self.api_key and not is_placeholder(self.api_key))

    # ------------------------------------------------------------------ #
    # 图片预处理
    # ------------------------------------------------------------------ #
    def _prepare_image(self, image_path: str) -> str:
        """把图片缩放 + JPEG 压缩到 ~80KB，返回 base64 data URL。"""
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        # 缩放最长边
        w, h = img.size
        longest = max(w, h)
        if longest > self.jpeg_max_side:
            scale = self.jpeg_max_side / longest
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

        # 质量逐级下调，逼近 ~80KB（bytes），避免超大 payload
        quality = self.jpeg_quality
        buf = io.BytesIO()
        while quality >= 40:
            buf.seek(0)
            buf.truncate(0)
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= 80 * 1024:
                break
            quality -= 10
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # ------------------------------------------------------------------ #
    # 单次评分
    # ------------------------------------------------------------------ #
    def score(self, image_path: str) -> ReviewResult:
        """对单张图评分，返回 ReviewResult（attempts 恒为 1）。"""
        data_url = self._prepare_image(image_path)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _SCORE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = requests.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=120
        )
        if r.status_code != 200:
            raise RuntimeError(f"VLM 审查返回 {r.status_code}: {r.text[:500]}")
        content = r.json()["choices"][0]["message"]["content"]
        score, opinion = self._parse(content)
        return ReviewResult(score=score, opinion=opinion, image_path=image_path)

    @staticmethod
    def _parse(content: str) -> tuple[float, str]:
        """从 VLM 返回文本里解析 {score, opinion}（容忍 markdown 代码块等噪声）。"""
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            raise RuntimeError(f"VLM 返回无法解析: {content[:300]}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"VLM 返回 JSON 解析失败: {content[:300]}") from e
        score = float(data.get("score", 0))
        opinion = str(data.get("opinion", "")).strip()
        return score, opinion

    # ------------------------------------------------------------------ #
    # 审查循环：出图 → 评分 → 低分换 seed 重生成
    # ------------------------------------------------------------------ #
    def review_with_retry(self, provider: ImageProvider, req: GenRequest) -> ReviewResult:
        """生成并审查，低分自动换 seed 重生成，直到达标或超过 max_retries。"""
        import random

        current = req
        last = None
        for attempt in range(1, self.max_retries + 2):  # 首张 + 最多 max_retries 次重生成
            result: GenResult = provider.generate(current)
            review = self.score(result.image_path)
            review.attempts = attempt
            review.image_path = result.image_path
            review.meta = {
                "seed": result.meta.get("seed"),
                "provider": result.provider,
                "threshold": self.threshold,
            }
            last = review
            if review.score >= self.threshold:
                return review
            # 低分：换 seed 重生成（绕开 ComfyUI 执行缓存）
            current = replace(current, seed=random.randint(1, 2**63 - 1))
        return last  # 达到最大次数仍不达标，返回最后一次（得分最高者由调用方决定）
