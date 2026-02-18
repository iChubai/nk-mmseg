# 推理与可视化

本文档说明 nk-mmseg 的推理入口、输入输出格式及可视化方式。

## mmseg 风格推理

主入口为 `tools/mmseg_infer.py`：

```bash
python tools/mmseg_infer.py \
  --config local_configs.NYUDepthv2.DFormer_Large \
  --checkpoint checkpoints/trained/NYUv2_DFormer_Large.pth \
  --img /path/to/rgb.png \
  --modal-x /path/to/depth.png \
  --out-file output/vis.png
```

### 参数

| 参数 | 说明 |
|------|------|
| `--config` | 配置模块路径 |
| `--checkpoint` | 模型权重路径 |
| `--img` | RGB 图像路径 |
| `--modal-x` | 额外模态（如 depth）路径 |
| `--out-file` | 输出可视化图像路径 |

## 编程接口

### 使用 mmseg API

```python
from mmseg.apis import init_model, inference_model

model = init_model(config_path, checkpoint_path)
result = inference_model(model, img_path, modal_x_path=depth_path)
```

### 使用 utils 直接推理

```python
from utils.jt_utils import load_model
from models import build_model

model = build_model(config)
model = load_model(model, checkpoint_path)
model.eval()

# 准备 RGB + depth 输入
# 前向得到 logits
logits = model(rgb, depth)
pred = logits.argmax(dim=1)
```

## 输入格式

- **RGB**：`[N, 3, H, W]`，CHW，已归一化
- **Depth**：单通道 `[N, 1, H, W]`（DFormerv2）或 3 通道复制（DFormer v1）
- 尺寸需与训练时的 crop_size 或 eval_crop_size 一致，或由预处理 resize

## 输出格式

- **logits**：`[N, num_classes, H, W]`
- **pred**：`[N, H, W]`，类别 ID，`255` 为 ignore

## 可视化

结果可映射到颜色表生成彩色分割图，与 mmseg 的 `LocalVisualizer` 或自定义 palette 一致。
