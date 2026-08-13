"""size 档位暴露验证（真实端到端）：medium vs large 分辨率对比。

直接构造 ``GenRequest``，分别用 ``size="medium (recommended)"`` 与 ``size="large"``
真实调 ComfyUI 出图，再用 PIL 读实际尺寸，断言 large 分辨率 > medium，
证明 size 档位真实生效（参数未被忽略）。

注：任务书示例写 ``size="medium"``，但 ComfyUI ``EmptyZImageLatentImage`` 的 size
COMBO 实际可选值为 ``small / medium (recommended) / large``（见 fix_report），
裸 ``"medium"`` 会被 ComfyUI 拒绝，故 medium 档位用合法值 ``"medium (recommended)"``。
"""
from __future__ import annotations

from PIL import Image

from config import load_config
from providers.base import GenRequest
from providers.comfyui_zimage import ComfyUIZImageProvider

PROMPT = "a lone lighthouse on a stormy cliff, bold ink strokes, high contrast"


def main() -> int:
    cfg = load_config()
    # 测试产物落盘 output/test_verify/（红线 #4）
    cfg["output"]["dir"] = "output/test_verify"
    provider = ComfyUIZImageProvider(cfg)

    req_medium = GenRequest(
        prompt=PROMPT, style="ink_frenzy", size="medium (recommended)"
    )
    req_large = GenRequest(prompt=PROMPT, style="ink_frenzy", size="large")

    res_medium = provider.generate(req_medium)
    res_large = provider.generate(req_large)

    w_m, h_m = Image.open(res_medium.image_path).size
    w_l, h_l = Image.open(res_large.image_path).size

    print(f"medium (recommended): {w_m} x {h_m}  -> {res_medium.image_path}")
    print(f"large              : {w_l} x {h_l}  -> {res_large.image_path}")
    print(f"medium area={w_m * h_m}, large area={w_l * h_l}")

    assert (w_l * h_l) > (w_m * h_m), (
        f"size 档位未生效：large ({w_l}x{h_l}) 未大于 medium ({w_m}x{h_m})"
    )
    assert w_m > 0 and w_l > 0
    print("OK: large 分辨率 > medium，size 档位真实生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
