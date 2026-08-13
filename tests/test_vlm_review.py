"""VLM 审查循环单元测试。

不依赖网络 / PIL：通过 monkeypatch 掉 ``VLMReviewer.score``，
用 fake provider 驱动 ``review_with_retry``；``_parse`` 直接喂文本测解析。
"""
from __future__ import annotations

from providers.base import GenRequest, GenResult, ImageProvider
from review.vlm_review import ReviewResult, VLMReviewer


class _FakeProvider(ImageProvider):
    name = "fake"
    is_remote = False

    def generate(self, req):
        return GenResult(image_path=f"img-{req.seed}.png", provider=self.name, meta={"seed": req.seed})

    def available(self):
        return True


# --------------------------------------------------------------------------- #
# 审查循环：低分换 seed，不达标返回最高分（而非最后一次）
# --------------------------------------------------------------------------- #
def test_review_with_retry_returns_best(monkeypatch):
    provider = _FakeProvider()
    # max_retries=1 → 首张 + 1 次重生成 = 2 次尝试
    reviewer = VLMReviewer({"vlm": {"max_retries": 1, "score_threshold": 75}})

    scores = iter([74.0, 55.0])

    def fake_score(self, image_path):
        s = next(scores)
        return ReviewResult(score=s, opinion=f"opinion-{s}", image_path=image_path)

    monkeypatch.setattr(VLMReviewer, "score", fake_score)

    res = reviewer.review_with_retry(provider, GenRequest(prompt="x"))

    # 两次都低于阈值：应返回最高分的 74 那次，而非最后一次 55
    assert res.score == 74.0
    assert res.opinion == "opinion-74.0"
    assert res.meta["attempt_scores"] == [74.0, 55.0]
    assert res.attempts == 2


def test_review_with_retry_returns_early_on_pass(monkeypatch):
    provider = _FakeProvider()
    reviewer = VLMReviewer({"vlm": {"max_retries": 3, "score_threshold": 75}})

    scores = iter([60.0, 88.0])  # 第二次达标

    def fake_score(self, image_path):
        s = next(scores)
        return ReviewResult(score=s, opinion=f"opinion-{s}", image_path=image_path)

    monkeypatch.setattr(VLMReviewer, "score", fake_score)

    res = reviewer.review_with_retry(provider, GenRequest(prompt="x"))

    assert res.score == 88.0
    assert res.attempts == 2
    assert res.meta["attempt_scores"] == [60.0, 88.0]


# --------------------------------------------------------------------------- #
# _parse：分维度评分解析 + 向后兼容（dimensions 缺失降级空 dict）
# --------------------------------------------------------------------------- #
def test_parse_dimensions():
    content = (
        '{"score": 85, "dimensions": {"composition": 88, "lighting": 82, '
        '"style_consistency": 90, "text_legibility": 75, "ai_flavor": 70}, '
        '"opinion": "ok"}'
    )
    score, opinion, dimensions = VLMReviewer._parse(content)
    assert score == 85.0
    assert opinion == "ok"
    assert dimensions == {
        "composition": 88.0,
        "lighting": 82.0,
        "style_consistency": 90.0,
        "text_legibility": 75.0,
        "ai_flavor": 70.0,
    }


def test_parse_missing_dimensions_falls_back_to_empty():
    content = '{"score": 72, "opinion": "no dims"}'
    score, opinion, dimensions = VLMReviewer._parse(content)
    assert score == 72.0
    assert opinion == "no dims"
    assert dimensions == {}


def test_parse_tolerates_markdown_fence():
    content = '```json\n{"score": 90, "dimensions": {"composition": 91}, "opinion": "good"}\n```'
    score, opinion, dimensions = VLMReviewer._parse(content)
    assert score == 90.0
    assert dimensions == {"composition": 91.0}
