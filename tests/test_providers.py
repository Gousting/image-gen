"""provider 抽象 + 分派模式 + 风格锚定的单元测试。

不依赖网络：Z-Image 只测工作流构建（纯函数）；分派模式用 fake provider。
"""
from __future__ import annotations

import pytest

from batch import BatchOrchestrator
from providers import (
    PROVIDER_CLASSES,
    ComfyUIQwenImageProvider,
    ComfyUIZImageProvider,
    RemoteOpenAICompatProvider,
    get_provider,
)
from providers.base import GenRequest, GenResult, ImageProvider
from providers.comfyui_zimage import _ratio_and_orientation
from styles.style_registry import STYLES, resolve_style


def _fake_config(tmp_path):
    return {
        "comfyui": {"url": "http://127.0.0.1:8188", "timeout": 60},
        "zimage": {
            "unet_name": "z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors",
            "clip_name": "qwen3_4b_fp8_scaled.safetensors",
            "clip_type": "lumina2",
            "vae_name": "ae.safetensors",
            "steps": 8,
            "default_size": "medium (recommended)",
        },
        "remote": {
            "base_url": "https://example.com/v1",
            "api_key": "YOUR_REMOTE_API_KEY",
            "model": "",
        },
        "output": {"dir": str(tmp_path / "out")},
    }


# --------------------------------------------------------------------------- #
# GenRequest / GenResult 数据契约
# --------------------------------------------------------------------------- #
def test_genrequest_defaults():
    req = GenRequest(prompt="a cat")
    assert req.style == "generic"
    assert req.width == 1344
    assert req.height == 768
    assert req.seed is None
    assert req.negative is None
    assert req.provider is None


def test_genresult_meta_defaults():
    res = GenResult(image_path="/tmp/x.png", provider="zimage")
    assert res.meta == {}


# --------------------------------------------------------------------------- #
# ImageProvider 抽象接口
# --------------------------------------------------------------------------- #
def test_imageprovider_is_abstract():
    with pytest.raises(TypeError):
        ImageProvider()  # 抽象类不可实例化


# --------------------------------------------------------------------------- #
# provider 注册表
# --------------------------------------------------------------------------- #
def test_registry_contains_all_providers():
    assert set(PROVIDER_CLASSES) == {"zimage", "qwen_image", "remote"}


def test_get_provider_unknown_raises():
    with pytest.raises(KeyError):
        get_provider("does_not_exist")


def test_get_provider_returns_instance(tmp_path):
    cfg = _fake_config(tmp_path)
    p = get_provider("zimage", cfg)
    assert isinstance(p, ComfyUIZImageProvider)
    assert p.is_remote is False


# --------------------------------------------------------------------------- #
# Qwen 骨架 / 外部 API 骨架
# --------------------------------------------------------------------------- #
def test_qwen_skeleton_unavailable(tmp_path):
    p = get_provider("qwen_image", _fake_config(tmp_path))
    assert p.available() is False
    with pytest.raises(NotImplementedError):
        p.generate(GenRequest(prompt="x"))


def test_remote_skeleton_unavailable(tmp_path):
    p = get_provider("remote", _fake_config(tmp_path))
    assert isinstance(p, RemoteOpenAICompatProvider)
    assert p.is_remote is True
    assert p.available() is False  # 未配置真实云端模型


def test_remote_available_when_configured(tmp_path):
    cfg = _fake_config(tmp_path)
    cfg["remote"] = {
        "base_url": "https://real.example.com/v1",
        "api_key": "sk-real-key-123",
        "model": "flux-pro",
    }
    p = get_provider("remote", cfg)
    assert p.available() is True  # base_url/api_key/model 三者齐备且非占位符


# --------------------------------------------------------------------------- #
# 风格锚定
# --------------------------------------------------------------------------- #
def test_styles_have_required_entries():
    # 至少 3 个已验证风格 + generic 占位
    assert {"ink_frenzy", "classic_film", "shinkai", "generic"} <= set(STYLES)


def test_resolve_style_unknown_raises():
    with pytest.raises(KeyError):
        resolve_style("nope")


def test_style_build_prompt_applies_anchor_and_constraint():
    s = resolve_style("ink_frenzy")
    prompt = s.build_prompt("a tower card")
    assert prompt.startswith("a tower card")
    assert "ink frenzy" in prompt
    assert "no visible text" in prompt  # 默认文字约束（正向描述）


# --------------------------------------------------------------------------- #
# Z-Image 工作流构建（纯函数，不联网）
# --------------------------------------------------------------------------- #
def _build(tmp_path, seed=42, **kw):
    p = get_provider("zimage", _fake_config(tmp_path))
    req = GenRequest(prompt="a castle in rain", **kw)
    return p._build_workflow(req, seed=seed)


def test_zimage_node_names_have_suffix(tmp_path):
    wf = _build(tmp_path)
    assert wf["32"]["class_type"] == "EmptyZImageLatentImage //ZImagePowerNodes"
    assert wf["45"]["class_type"] == "StylePromptEncoder2 //ZImagePowerNodes"
    assert wf["48"]["class_type"] == "ZSamplerTurbo2 //ZImagePowerNodes"
    assert wf["31"]["class_type"] == "SaveImage //ZImagePowerNodes"


def test_zimage_sampler_has_no_cfg_and_no_negative(tmp_path):
    # 坑 #2：CFG=0，sampler 无 cfg / negative 输入
    inputs = _build(tmp_path)["48"]["inputs"]
    assert "cfg" not in inputs
    assert "negative" not in inputs
    assert inputs["denoise"] == 1.0
    assert inputs["turbo_creativity"] == "off"


def test_zimage_seed_enters_workflow(tmp_path):
    assert _build(tmp_path, seed=123)["48"]["inputs"]["seed"] == 123


def test_zimage_style_mapped_to_combo(tmp_path):
    wf = _build(tmp_path, style="ink_frenzy")
    assert wf["45"]["inputs"]["style"] == '"Ink Frenzy"'
    assert "no visible text" in wf["45"]["inputs"]["text"]


def test_ratio_mapping():
    assert _ratio_and_orientation(1344, 768) == (True, "16:9  (widescreen)")
    assert _ratio_and_orientation(768, 1344) == (False, "16:9  (widescreen)")
    assert _ratio_and_orientation(1024, 1024) == (True, "1:1  (square)")


# --------------------------------------------------------------------------- #
# 分派模式（fake provider，不联网）
# --------------------------------------------------------------------------- #
class _FakeLocal(ImageProvider):
    name = "fake_local"
    is_remote = False

    def generate(self, req):
        return GenResult(image_path=f"local:{req.prompt}", provider=self.name, meta={"seed": req.seed})

    def available(self):
        return True


class _FakeRemote(ImageProvider):
    name = "fake_remote"
    is_remote = True

    def generate(self, req):
        return GenResult(image_path=f"remote:{req.prompt}", provider=self.name, meta={"seed": req.seed})

    def available(self):
        return True


def test_dispatch_routes_to_requested_provider(monkeypatch):
    fake_map = {"fake_local": _FakeLocal(), "fake_remote": _FakeRemote()}

    def _fake_get(name, config=None):
        return fake_map[name]

    monkeypatch.setattr("batch.get_provider", _fake_get)
    orch = BatchOrchestrator(config=None)

    reqs = [
        GenRequest(prompt="a", provider="fake_local"),
        GenRequest(prompt="b", provider="fake_remote"),
        GenRequest(prompt="c"),  # 走默认
    ]
    results = orch.generate(reqs, provider="fake_local")
    assert [r.provider for r in results] == ["fake_local", "fake_remote", "fake_local"]
    assert results[1].image_path == "remote:b"


def test_phase2_stubs_raise_not_implemented():
    orch = BatchOrchestrator(config=None)
    with pytest.raises(NotImplementedError):
        orch.compare([GenRequest(prompt="x")])
    with pytest.raises(NotImplementedError):
        orch.upgrade([GenRequest(prompt="x")])
