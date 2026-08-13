"""外部 OpenAI-compatible 生图 API provider —— 骨架插槽（不接具体云端模型）。

走 OpenAI-compatible 的 ``POST {base_url}/images/generations`` 协议：国内多数云端
生图（通义万相、即梦、硅基流动 FLUX 等）都兼容该协议，一个 provider 覆盖多个云端模型。

配置驱动：``base_url`` / ``api_key`` / ``model`` 全部从 config.yaml 读，
换云端模型只改配置不写码。api_key 用占位符入库，真实值放 .env（不入库）。

当前状态：``available()`` 返回 False —— 因为尚未配置真实云端模型。
二期填上 base_url/api_key/model 后，本类的 generate 即可直接调用，无需改代码。
未来遇到非 OpenAI 协议的模型（Ideogram、Gemini 等），再新增 adapter 类，
同样实现 ``ImageProvider`` 接口即可。
"""
from __future__ import annotations

import base64
import time

import requests

from config import is_placeholder, load_config, resolve_output_dir
from providers.base import GenRequest, GenResult, ImageProvider


class RemoteOpenAICompatProvider(ImageProvider):
    name = "remote"
    is_remote = True

    def __init__(self, config: dict | None = None):
        cfg = config if config is not None else load_config()
        self._cfg = cfg
        rcfg = cfg.get("remote", {}) or {}
        self.base_url = (rcfg.get("base_url") or "").rstrip("/")
        self.api_key = rcfg.get("api_key") or ""
        self.model = rcfg.get("model") or ""
        self.output_dir = resolve_output_dir(cfg)

    def _configured(self) -> bool:
        """是否已填真实配置（base_url / api_key / model 三者齐备）。"""
        return bool(
            self.model
            and not is_placeholder(self.base_url)
            and not is_placeholder(self.api_key)
        )

    def available(self) -> bool:
        # 骨架插槽：未配置真实云端模型，恒为 False。
        # 二期填好 base_url/api_key/model 后，这里可改为 GET /models 探测。
        return False

    def generate(self, req: GenRequest) -> GenResult:
        if not self._configured():
            raise RuntimeError(
                "外部 API 未配置（base_url/api_key/model 为空或仍为占位符）。"
                "二期填好 config.yaml 的 remote 段后再调用。"
            )

        # OpenAI-compatible images/generations 协议
        payload = {
            "model": self.model,
            "prompt": req.prompt,
            "n": 1,
            "size": f"{req.width}x{req.height}",
            "response_format": "b64_json",
        }
        # 注意：OpenAI images 接口不支持 negative prompt，req.negative 忽略。
        headers = {"Authorization": f"Bearer {self.api_key}"}
        started = time.time()
        r = requests.post(
            f"{self.base_url}/images/generations",
            json=payload,
            headers=headers,
            timeout=300,
        )
        if r.status_code != 200:
            raise RuntimeError(f"外部 API 返回 {r.status_code}: {r.text[:500]}")
        data = r.json()
        elapsed = time.time() - started

        items = data.get("data") or []
        if not items:
            raise RuntimeError(f"外部 API 未返回图片: {data}")

        item = items[0]
        seed = req.seed if req.seed is not None else item.get("seed")
        out_path = self.output_dir / f"remote_{self.model}_{int(time.time())}.png"

        if item.get("b64_json"):
            out_path.write_bytes(base64.b64decode(item["b64_json"]))
        elif item.get("url"):
            img = requests.get(item["url"], timeout=120)
            if img.status_code != 200:
                raise RuntimeError(f"下载外部图片失败 {img.status_code}")
            out_path.write_bytes(img.content)
        else:
            raise RuntimeError(f"外部 API 返回格式不支持: {item}")

        return GenResult(
            image_path=str(out_path),
            provider=self.name,
            meta={
                "model": self.model,
                "seed": seed,
                "elapsed_seconds": round(elapsed, 1),
                "raw": item,
            },
        )
