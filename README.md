# image-gen —— 统一生图服务

把散在各处的生图能力（TRPG 场景卡、塔罗牌牌面、小红书封面、AI 视频关键帧）收拢成一个统一服务。
核心三条：**模型无关**、**质量保障（VLM 审查闭环）**、**本地 + 外部混合**。

## 快速开始

```bash
# 本地 Z-Image 出一张图
python cli.py generate --prompt "a lone samurai in the rain" --style ink_frenzy

# 出 4 张（每张自动随机 seed，绕开缓存）
python cli.py generate --prompt "..." --style classic_film --count 4

# 带 VLM 审查循环：低分自动换 seed 重生成
python cli.py generate --prompt "..." --style shinkai --review

# 查看可用风格 / provider
python cli.py styles
python cli.py providers
```

### 环境要求

- Python 3.11+，依赖 `requests`、`PyYAML`、`Pillow`（`pip install requests pyyaml pillow`）
- 本地出图需宿主机 ComfyUI 运行在 `127.0.0.1:8188`（默认，`config.yaml` 可改）

### 配置与密钥

`config.yaml` 入库，但 **api_key 一律用占位符**；真实密钥放项目根目录 `.env`（已被 `.gitignore` 排除，永不提交）：

```bash
# .env（不入库）
IMAGE_GEN_VLM_API_KEY=sk-xxxx     # VLM 审查模型密钥
IMAGE_GEN_REMOTE_API_KEY=sk-yyyy  # 外部云端生图密钥（二期）
```

加载优先级：`config.yaml` → `.env` → 进程环境变量（最高）。

## 架构：Provider 抽象（模型无关，本项目的灵魂）

调用方只面向 `ImageProvider` 接口，不写死任何模型。换模型 = 换 provider，契约不变：

```python
from providers import get_provider
from providers.base import GenRequest

p = get_provider("zimage")               # 本地 Z-Image
result = p.generate(GenRequest(prompt="...", style="ink_frenzy"))
print(result.image_path)                 # 落盘 PNG 路径
```

| provider | name | 状态 | 说明 |
| --- | --- | --- | --- |
| 本地 Z-Image | `zimage` | ✅ 完整实现 | 调 ComfyUI Z-Image Turbo（FP8） |
| 本地 Qwen-Image | `qwen_image` | 🦴 骨架 | `available()` 恒 False，二期填 |
| 外部 OpenAI-compat | `remote` | 🧩 插槽 | 协议已实现，未接具体云端模型 |

分派模式（`batch.py`）批量出图，每张可指定来源：

```python
from batch import BatchOrchestrator
from providers.base import GenRequest

orch = BatchOrchestrator()
reqs = [GenRequest(prompt="卡面", style="ink_frenzy", provider="zimage"),
        GenRequest(prompt="封面", style="classic_film", provider="remote")]
results = orch.generate(reqs)
```

> 升级模式（本地底图 + 外部 img2img 精修）、对比模式（双来源择优）留二期，`batch.py` 只留方法签名。

## VLM 审查循环（`review/vlm_review.py`）

出图 → 评分 → 低分换 seed 重生成。评分维度：构图 / 光影 / 风格一致 / 文字不乱码 / AI 味
（对称构图、聚光灯、套路隐喻扣分）。阈值默认 75，最多重生成 3 次（可配）。

```python
from review.vlm_review import VLMReviewer
from providers import get_provider

reviewer = VLMReviewer()                      # 读 config.yaml + .env
res = reviewer.review_with_retry(get_provider("zimage"),
                                 GenRequest(prompt="...", style="ink_frenzy"))
print(res.score, res.opinion, res.image_path)
```

## 风格锚定（`styles/style_registry.py`）

一套图锁一个风格，传同一个 style 名即可保证跨图一致。已注册：

| style | Z-Image 样式 | 用途 |
| --- | --- | --- |
| `ink_frenzy` | `"Ink Frenzy"` | 塔罗牌 / 跑团素材（水墨狂草） |
| `classic_film` | `"Classic Film Photo"` | 胶片写实 |
| `shinkai` | `"Anime"` + 锚定词 | 新海诚动画雨景 |
| `generic` | `none` | 通用无风格占位 |

新增风格 = 在 `style_registry.py` 的 `STYLES` 加一条。

---

## Z-Image 的坑（必读，全部已沉淀进 provider 注释）

1. **节点名带 ` //ZImagePowerNodes` 后缀**：`EmptyZImageLatentImage //ZImagePowerNodes`、
   `ZSamplerTurbo2 //ZImagePowerNodes`、`StylePromptEncoder2 //ZImagePowerNodes`、
   `SaveImage //ZImagePowerNodes`；`TextEncodeZImageOmni` 不带后缀。提交前建议
   `GET /object_info` 确认全名（本实现已按本机节点名写死）。

2. **CFG=0、完全忽略 negative prompt**：`ZSamplerTurbo2` 根本没有 cfg / negative 输入，
   `GenRequest.negative` 直接忽略。prompt 里「不要 XX」无效，必须转正向描述：
   「不要亮色」→「strictly monochrome dark sepia」。

3. **画面内文字易乱码**：风格锚定默认约束「文字最少化或无文字」（正向描述
   `text-free, no visible text, no watermark`）。

4. **换 seed 才能绕开 ComfyUI 执行缓存**：同 seed + 同参数直接复用旧图；批量出图务必随机 seed。

5. **Z-Image 不支持自由宽高**：只有「横竖 + 比例档位 + 尺寸档位」，`width/height` 会被
   就近映射到比例档位（如 1344×768 → 16:9 widescreen）。见 `_ratio_and_orientation`。

## 目录结构

```
├── cli.py                        # 命令行入口
├── config.py                     # 配置加载（config.yaml + .env + 环境变量）
├── config.yaml                   # 本地 ComfyUI 地址 + VLM + 外部 API 占位
├── batch.py                      # 分派模式；升级/对比留二期签名
├── providers/
│   ├── base.py                   # ImageProvider 抽象 + GenRequest/GenResult
│   ├── __init__.py               # provider 注册表
│   ├── comfyui_zimage.py         # 本地 Z-Image（完整实现，含全部坑）
│   ├── comfyui_qwen_image.py     # 本地 Qwen-Image（空骨架）
│   └── remote_openai_compat.py   # 外部 OpenAI-compat（插槽）
├── review/vlm_review.py          # VLM 审查循环
├── styles/style_registry.py      # 风格锚定模板
└── tests/test_providers.py       # 单测（provider 抽象 + 分派模式 + 风格）
```

## 测试

```bash
python -m pytest tests/ -q
```
