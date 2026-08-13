"""本地 Z-Image 生图 provider（完整实现）。

调宿主机 ComfyUI 的 Z-Image Turbo 节点，走 ``POST /prompt`` + 轮询 ``/history`` 拿图。

================================================================================
Z-Image 的坑（全部沉淀于此，README 同步说明）：
--------------------------------------------------------------------------------
1. 节点名带 `` //ZImagePowerNodes`` 后缀（只有 ``TextEncodeZImageOmni`` 不带）。
   提交前建议 GET /object_info 确认全名，本实现已按本机节点名写死：
   - ``EmptyZImageLatentImage //ZImagePowerNodes``
   - ``ZSamplerTurbo2 //ZImagePowerNodes``
   - ``StylePromptEncoder2 //ZImagePowerNodes``
   - ``SaveImage //ZImagePowerNodes``
   - ``VAEDecode`` / ``UNETLoader`` / ``CLIPLoader`` / ``VAELoader``（comfy 原生节点）

2. CFG=0、完全忽略 negative prompt：
   ``ZSamplerTurbo2`` 根本没有 cfg / negative 输入，``GenRequest.negative`` 直接忽略。
   因此 prompt 里「不要 XX」这类否定约束无效，必须转成正向描述：
   「不要亮色」→「strictly monochrome dark sepia」。风格约束里文字最少化也写成
   「text-free, no watermark」等正向描述。

3. 画面内文字易乱码：风格锚定里默认约束「文字最少化或无文字」（见 style_registry）。

4. 换 seed 才能绕开 ComfyUI 执行缓存：同 seed + 同参数会被缓存直接复用旧图，
   所以生成多样图时必须换 seed（本实现默认随机 seed）。

5. Z-Image 不支持自由宽高，只有「横竖 + 比例档位 + 尺寸档位」：
   ``EmptyZImageLatentImage`` 用 landscape / ratio / size 三参数，本实现把
   width/height 就近映射到 ratio 档位（见 ``_ratio_and_orientation``）。
================================================================================
"""
from __future__ import annotations

import random
import time
import uuid
from typing import Any

import requests

from config import PROJECT_ROOT, is_placeholder, load_config, resolve_output_dir
from providers.base import GenRequest, GenResult, ImageProvider
from styles.style_registry import resolve_style

# Z-Image 比例档位（均为横版比例；竖版由 landscape=False 翻转，值不变）
_RATIOS: tuple[tuple[float, str], ...] = (
    (1.0, "1:1  (square)"),
    (1.333, "4:3  (retro tv)"),
    (1.5, "3:2  (photo)"),
    (1.6, "16:10  (monitor)"),
    (1.778, "16:9  (widescreen)"),
    (2.0, "2:1  (univisium)"),
    (2.333, "21:9  (ultrawide)"),
    (2.4, "12:5  (anamorphic)"),
    (2.593, "70:27  (cinerama)"),
    (3.556, "32:9  (super wide)"),
)

# 节点全名（带 //ZImagePowerNodes 后缀）
_N_EMPTY = "EmptyZImageLatentImage //ZImagePowerNodes"
_N_SAMPLER = "ZSamplerTurbo2 //ZImagePowerNodes"
_N_STYLE = "StylePromptEncoder2 //ZImagePowerNodes"
_N_SAVE = "SaveImage //ZImagePowerNodes"


def _ratio_and_orientation(width: int, height: int) -> tuple[bool, str]:
    """把 width/height 映射为 (landscape, ratio 档位)。

    Z-Image 只接受横版比例档位 + landscape 布尔值；竖版用 landscape=False 翻转。
    """
    landscape = width >= height
    aspect = (width / height) if landscape else (height / width)
    best = min(_RATIOS, key=lambda r: abs(r[0] - aspect))
    return landscape, best[1]


class ComfyUIZImageProvider(ImageProvider):
    """本地 Z-Image Turbo（FP8）provider。"""

    name = "zimage"
    is_remote = False

    def __init__(self, config: dict | None = None):
        cfg = config if config is not None else load_config()
        self._cfg = cfg
        self.comfyui_url = (cfg.get("comfyui", {}) or {}).get("url", "http://127.0.0.1:8188").rstrip("/")
        self.timeout = int((cfg.get("comfyui", {}) or {}).get("timeout", 600))

        zcfg = cfg.get("zimage", {}) or {}
        self.unet_name = zcfg.get("unet_name", "z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors")
        self.clip_name = zcfg.get("clip_name", "qwen3_4b_fp8_scaled.safetensors")
        self.clip_type = zcfg.get("clip_type", "lumina2")
        self.vae_name = zcfg.get("vae_name", "ae.safetensors")
        self.steps = int(zcfg.get("steps", 8))
        self.default_size = zcfg.get("default_size", "medium (recommended)")

        self.output_dir = resolve_output_dir(cfg)

    # ------------------------------------------------------------------ #
    # 健康检查
    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        """检测 ComfyUI 是否在线（GET /system_stats）。"""
        try:
            r = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------ #
    # 生成入口
    # ------------------------------------------------------------------ #
    def generate(self, req: GenRequest) -> GenResult:
        if not self.available():
            raise RuntimeError(f"ComfyUI 不在线: {self.comfyui_url}（Z-Image provider 不可用）")

        seed = req.seed if req.seed is not None else random.randint(1, 2**63 - 1)
        # 注意：negative 在 Z-Image 下被完全忽略（CFG=0），这里记录但不使用。
        if req.negative:
            # 只提示，不改行为 —— 见模块 docstring 的坑 #2
            pass

        workflow = self._build_workflow(req, seed)
        started = time.time()
        filename, subfolder = self._submit_and_wait(workflow)
        elapsed = time.time() - started

        png = self._download_image(filename, subfolder)
        out_path = self.output_dir / f"{req.style or 'zimage'}_{seed}_{int(time.time())}.png"
        out_path.write_bytes(png)

        return GenResult(
            image_path=str(out_path),
            provider=self.name,
            meta={
                "seed": seed,
                "elapsed_seconds": round(elapsed, 1),
                "style": req.style,
                "prompt": req.prompt,
                "comfyui_filename": filename,
            },
        )

    # ------------------------------------------------------------------ #
    # 工作流构建（纯函数，便于单测）
    # ------------------------------------------------------------------ #
    def _build_workflow(self, req: GenRequest, seed: int) -> dict[str, dict]:
        """构建 ComfyUI API 工作流（POST /prompt 的 prompt 字段）。"""
        style = resolve_style(req.style)
        prompt = style.build_prompt(req.prompt)
        landscape, ratio = _ratio_and_orientation(req.width, req.height)

        return {
            # 模型 / 文本编码器 / VAE（FP8 版，与本机 ComfyUI 已装模型一致）
            "53": {
                "inputs": {"unet_name": self.unet_name, "weight_dtype": "default"},
                "class_type": "UNETLoader",
            },
            "54": {
                "inputs": {"clip_name": self.clip_name, "type": self.clip_type, "device": "default"},
                "class_type": "CLIPLoader",
            },
            "55": {
                "inputs": {"vae_name": self.vae_name},
                "class_type": "VAELoader",
            },
            # 空 latent（横竖 + 比例 + 尺寸档位）
            "32": {
                "inputs": {
                    "landscape": landscape,
                    "ratio": ratio,
                    "size": self.default_size,
                    "batch_size": 1,
                },
                "class_type": _N_EMPTY,
            },
            # 风格 + prompt 编码（style 用带引号的组合值；gallery/spacer 是装饰输入，传 None）
            "45": {
                "inputs": {
                    "style": style.zimage_style,
                    "gallery": None,
                    "spacer": None,
                    "text": prompt,
                    "clip": ["54", 0],
                },
                "class_type": _N_STYLE,
            },
            # Z-Sampler Turbo ^G2：无 cfg / negative 输入（CFG=0）；divider/divider2 是装饰输入，传 None
            "48": {
                "inputs": {
                    "latent_input": ["32", 0],
                    "model": ["53", 0],
                    "positive": ["45", 0],
                    "seed": seed,
                    "steps": self.steps,
                    "denoise": 1.0,
                    "divider": None,
                    "initial_sample_size": "full_size",
                    "divider2": None,
                    "intensity": 0.0,
                    "intensity_bias": 0.0,
                    "turbo_creativity": "off",
                },
                "class_type": _N_SAMPLER,
            },
            # VAE 解码
            "8": {
                "inputs": {"samples": ["48", 0], "vae": ["55", 0]},
                "class_type": "VAEDecode",
            },
            # 保存
            "31": {
                "inputs": {
                    "images": ["8", 0],
                    "filename_prefix": "image_gen/ZI",
                    "civitai_compatible_metadata": True,
                },
                "class_type": _N_SAVE,
            },
        }

    # ------------------------------------------------------------------ #
    # 提交 + 轮询
    # ------------------------------------------------------------------ #
    def _submit_and_wait(self, workflow: dict) -> tuple[str, str]:
        """提交工作流并轮询 /history 直到完成，返回 (filename, subfolder)。"""
        client_id = str(uuid.uuid4())
        try:
            r = requests.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"提交 ComfyUI 工作流失败: {e}") from e
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI /prompt 返回 {r.status_code}: {r.text[:500]}")
        data = r.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"ComfyUI /prompt 无 prompt_id: {data}")
        prompt_id = data["prompt_id"]

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            time.sleep(3)
            try:
                h = requests.get(f"{self.comfyui_url}/history/{prompt_id}", timeout=30).json()
            except requests.RequestException:
                continue
            if prompt_id not in h:
                continue
            hist = h[prompt_id]
            status = hist.get("status", {})
            if status.get("status_str") == "error":
                msgs = hist.get("messages", [])
                detail = " | ".join(str(m) for m in msgs)[:2000]
                raise RuntimeError(f"ComfyUI 生成失败: {detail}")
            if status.get("completed"):
                for out in hist.get("outputs", {}).values():
                    if "images" in out and out["images"]:
                        img = out["images"][0]
                        return img["filename"], img.get("subfolder", "")
                raise RuntimeError("ComfyUI 完成但未返回图片")
        raise TimeoutError(f"ComfyUI 生成超时（>{self.timeout}s）")

    def _download_image(self, filename: str, subfolder: str) -> bytes:
        """从 ComfyUI /view 下载图片原始字节。"""
        import urllib.parse

        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": "output"}
        )
        r = requests.get(f"{self.comfyui_url}/view?{params}", timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"下载图片失败 {r.status_code}: {r.text[:200]}")
        return r.content
