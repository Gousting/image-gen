"""image-gen 命令行入口。

用法：
    python cli.py generate --prompt "..." --style ink_frenzy --count 4
    python cli.py generate --prompt "..." --style shinkai --review        # 带 VLM 审查循环
    python cli.py styles                                                  # 列出可用风格
    python cli.py providers                                               # 列出 provider 及可用性
"""
from __future__ import annotations

import argparse
import sys

from config import load_config
from providers import DEFAULT_PROVIDER, PROVIDER_CLASSES, get_provider
from providers.base import GenRequest
from styles.style_registry import STYLES


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = load_config()
    provider = get_provider(args.provider, cfg)

    if not provider.available():
        print(f"[error] provider '{args.provider}' 不可用（未就绪/未配置）。", file=sys.stderr)
        return 1

    reqs = [
        GenRequest(
            prompt=args.prompt,
            style=args.style,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )
        for _ in range(args.count)
    ]

    if args.review:
        from review.vlm_review import VLMReviewer

        reviewer = VLMReviewer(cfg)
        if not reviewer.available():
            print("[error] VLM 未配置（vlm.api_key 为空或占位符），无法审查。", file=sys.stderr)
            return 1
        print(f"[review] 阈值 {reviewer.threshold}，最多重生成 {reviewer.max_retries} 次")
        for i, req in enumerate(reqs, 1):
            res = reviewer.review_with_retry(provider, req)
            print(
                f"[{i}/{len(reqs)}] score={res.score:.0f} attempts={res.attempts} "
                f"seed={res.meta.get('seed')} -> {res.image_path}"
            )
            print(f"        {res.opinion}")
    else:
        for i, req in enumerate(reqs, 1):
            res = provider.generate(req)
            print(f"[{i}/{len(reqs)}] seed={res.meta.get('seed')} -> {res.image_path}")
    return 0


def _cmd_styles(_args: argparse.Namespace) -> int:
    for key, s in STYLES.items():
        print(f"- {key:14s} zimage_style={s.zimage_style!r:22s} anchor={s.anchor[:50]}")
    return 0


def _cmd_providers(_args: argparse.Namespace) -> int:
    cfg = load_config()
    for name, cls in PROVIDER_CLASSES.items():
        try:
            inst = get_provider(name, cfg)
            status = "available" if inst.available() else "unavailable"
        except Exception as e:  # noqa: BLE001
            status = f"error: {e}"
        flag = "remote" if cls.is_remote else "local "
        print(f"- {name:12s} [{flag}] {status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image-gen", description="统一生图服务")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="本地出图（可指定 style/seed/count）")
    g.add_argument("--prompt", required=True, help="正向画面描述")
    g.add_argument("--style", default="generic", help="风格锚定名（见 `styles`）")
    g.add_argument("--count", type=int, default=1, help="出图张数")
    g.add_argument("--seed", type=int, default=None, help="随机种子（缺省随机，绕开缓存）")
    g.add_argument("--width", type=int, default=1344)
    g.add_argument("--height", type=int, default=768)
    g.add_argument("--provider", default=DEFAULT_PROVIDER, help="provider 名")
    g.add_argument("--review", action="store_true", help="启用 VLM 审查循环（低分换 seed 重生成）")

    sub.add_parser("styles", help="列出可用风格")
    sub.add_parser("providers", help="列出 provider 及可用性")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "styles":
        return _cmd_styles(args)
    if args.command == "providers":
        return _cmd_providers(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
