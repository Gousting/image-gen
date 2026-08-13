"""验证 remote provider 的 ``available()`` 配置翻转修复。

修复前 ``available()`` 恒为 ``False``；修复后改为按配置判定
（base_url / api_key / model 三者齐备且非占位符即返回 True）。

本测试不依赖网络：只构造 ``RemoteOpenAICompatProvider`` 实例并断言
``available()`` 的返回值，用「假但非占位」的配置验证翻转行为。
"""
from __future__ import annotations

from providers.remote_openai_compat import RemoteOpenAICompatProvider


def _flip_config(tmp_path, *, base_url, api_key, model):
    """构造 remote 段配置（含合法输出目录，避免 resolve_output_dir 落盘失败）。"""
    return {
        "remote": {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        },
        "output": {"dir": str(tmp_path / "out")},
    }


def test_remote_available_flips_true_with_fake_non_placeholder(tmp_path):
    # 「假但非占位」的配置：三者齐备且均非占位符 → 修复后 available() 应为 True
    cfg = _flip_config(
        tmp_path,
        base_url="https://fake.example.com/v1",
        api_key="sk-fake123",
        model="dummy",
    )
    p = RemoteOpenAICompatProvider(cfg)
    assert p.available() is True  # 修复前恒 False，修复后应 True


def test_remote_available_false_with_placeholder(tmp_path):
    # 占位符 / 空值 → 仍应 False（占位符不视为已配置）
    cfg = _flip_config(
        tmp_path,
        base_url="https://example.com/v1",
        api_key="YOUR_REMOTE_API_KEY",  # 占位符
        model="",
    )
    p = RemoteOpenAICompatProvider(cfg)
    assert p.available() is False


def test_remote_available_false_with_missing_model(tmp_path):
    # base_url / api_key 齐备但 model 为空 → 仍应 False（三者缺一不可）
    cfg = _flip_config(
        tmp_path,
        base_url="https://fake.example.com/v1",
        api_key="sk-fake123",
        model="",
    )
    p = RemoteOpenAICompatProvider(cfg)
    assert p.available() is False
