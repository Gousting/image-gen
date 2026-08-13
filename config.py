"""配置加载：config.yaml + .env + 环境变量覆盖。

加载优先级（低 → 高）：
  1. config.yaml（入库，api_key 只放占位符）
  2. 项目根目录 .env（不入库，放真实密钥）
  3. 进程环境变量（最高）

.env 与真实环境变量统一走 ``IMAGE_GEN_*`` 前缀的覆盖映射（见 ``_ENV_MAP``）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

# 环境变量名 -> (config 路径元组)。优先级最高，方便 CI / 容器注入。
_ENV_MAP: dict[str, tuple[str, ...]] = {
    "IMAGE_GEN_COMFYUI_URL": ("comfyui", "url"),
    "IMAGE_GEN_ZIMAGE_UNET": ("zimage", "unet_name"),
    "IMAGE_GEN_ZIMAGE_CLIP": ("zimage", "clip_name"),
    "IMAGE_GEN_ZIMAGE_CLIP_TYPE": ("zimage", "clip_type"),
    "IMAGE_GEN_ZIMAGE_VAE": ("zimage", "vae_name"),
    "IMAGE_GEN_VLM_BASE_URL": ("vlm", "base_url"),
    "IMAGE_GEN_VLM_API_KEY": ("vlm", "api_key"),
    "IMAGE_GEN_VLM_MODEL": ("vlm", "model"),
    "IMAGE_GEN_REMOTE_BASE_URL": ("remote", "base_url"),
    "IMAGE_GEN_REMOTE_API_KEY": ("remote", "api_key"),
    "IMAGE_GEN_REMOTE_MODEL": ("remote", "model"),
}


def _parse_env_file(path: Path) -> dict[str, str]:
    """解析极简 .env（KEY=VALUE，支持 # 注释），不依赖 python-dotenv。"""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def _deep_set(cfg: dict, path_tuple: tuple[str, ...], value: Any) -> None:
    node = cfg
    for key in path_tuple[:-1]:
        node = node.setdefault(key, {})
    node[path_tuple[-1]] = value


def load_config(path: str | Path | None = None) -> dict:
    """加载配置并合并 .env / 环境变量覆盖。

    返回嵌套 dict；不含密钥的真实值（真实值经 .env/环境变量注入）。
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    # .env 覆盖（不覆盖真实进程环境变量，进程环境变量优先级最高）
    file_env = _parse_env_file(cfg_path.parent / ".env")
    merged_env = {**file_env, **os.environ}
    for var, target in _ENV_MAP.items():
        if merged_env.get(var):
            _deep_set(cfg, target, merged_env[var])

    return cfg


def is_placeholder(value: Any) -> bool:
    """判断某个配置值是否仍是占位符（未填真实值）。"""
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.startswith("YOUR_")


def resolve_output_dir(cfg: dict) -> Path:
    """解析出图落盘目录（相对路径基于项目根），并确保存在。"""
    rel = (cfg.get("output", {}) or {}).get("dir", "output")
    out = (PROJECT_ROOT / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    out.mkdir(parents=True, exist_ok=True)
    return out
