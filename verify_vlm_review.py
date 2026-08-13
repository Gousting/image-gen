"""VLM 审查闭环真实验证脚本：出图 → 评分 → 低分换 seed 重生成，并逐项打印结果。

相比 cli.py（只打印 score/attempts/seed/opinion），本脚本额外把 VLM 返回的
5 维度分（composition/lighting/style_consistency/text_legibility/ai_flavor）
与每次尝试的 attempt_scores 逐项打印，用于验证「VLM 5 维度分」修复真实生效。

用法：
    python verify_vlm_review.py --prompt "..." --style ink_frenzy
"""
from __future__ import annotations

import argparse

from config import load_config
from providers.base import GenRequest
from providers.comfyui_zimage import ComfyUIZImageProvider
from review.vlm_review import VLMReviewer

_DIM_ORDER = (
    "composition",
    "lighting",
    "style_consistency",
    "text_legibility",
    "ai_flavor",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--style", default="ink_frenzy")
    args = parser.parse_args()

    cfg = load_config()
    cfg["output"]["dir"] = "output/test_verify"
    provider = ComfyUIZImageProvider(cfg)
    reviewer = VLMReviewer(cfg)

    if not provider.available():
        print("[error] ComfyUI 不在线", flush=True)
        return 1
    if not reviewer.available():
        print("[error] VLM 未配置（vlm.api_key 为空或占位符）", flush=True)
        return 1

    print(f"[review] 阈值 {reviewer.threshold}，最多重生成 {reviewer.max_retries} 次",
          flush=True)
    print(f"[review] VLM model={reviewer.model} base_url={reviewer.base_url}", flush=True)

    req = GenRequest(prompt=args.prompt, style=args.style)
    res = reviewer.review_with_retry(provider, req)

    print("=" * 60, flush=True)
    print(f"总分 score     : {res.score:.0f}", flush=True)
    print(f"opinion        : {res.opinion}", flush=True)
    print(f"attempts       : {res.attempts}", flush=True)
    print(f"attempt_scores : {res.meta.get('attempt_scores')}", flush=True)
    print(f"seed           : {res.meta.get('seed')}", flush=True)
    print(f"image_path     : {res.image_path}", flush=True)
    print("-" * 60, flush=True)
    print("5 维度分：", flush=True)
    for k in _DIM_ORDER:
        v = res.dimensions.get(k)
        print(f"  {k:18s}: {v}", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
