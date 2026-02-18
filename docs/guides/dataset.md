# 数据集准备

本文档详细说明 nk-mmseg 支持的数据集格式、目录结构与配置方式。

## 支持的数据集

| 数据集 | 类别数 | 训练样本 | 测试样本 | 分辨率 |
|--------|--------|----------|----------|--------|
| NYUDepthv2 | 40 | 795 | 654 | 480×640 |
| SUNRGBD | 37 | 5,285 | 5,050 |  varies |

## NYUDepthv2

### 目录结构

```
NYUDepthv2/
├── RGB/              # RGB 图像，格式 .jpg
├── Depth/            # 深度图，格式 .png（单通道灰度）
├── Label/            # 语义标签，格式 .png
├── train.txt         # 训练集文件列表（每行一个文件名，不含路径）
└── test.txt          # 测试集文件列表
```

### 标签语义

- 原始标签范围：`1..40`（有效类），`0` 为无效
- 配置中 `C.gt_transform = True` 时，会做 `gt = gt - 1`，原 `0` 映射为 `ignore_index=255`
- 若标签变换错误，mIoU 会明显偏离，务必与 PyTorch 基线一致

### 关键配置（local_configs/_base_/datasets/NYUDepthv2.py）

```python
C.dataset_name = "NYUDepthv2"
C.dataset_path = osp.join(C.root_dir, "NYUDepthv2")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_is_single_channel = True   # Depth 为单通道
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.num_classes = 40
C.background = 255
```

### 归一化

- RGB：ImageNet mean/std `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`
- Depth：`mean=[0.48], std=[0.28]`（固定，不要改为 RGB 参数）

## SUNRGBD

结构与 NYUDepthv2 类似，需修改 `local_configs/_base_/datasets/SUNRGBD.py` 中的 `dataset_name`、`num_classes`、`class_names` 等。

## Depth 通道处理

- **DFormerv2**：单通道 depth `[H,W,1]`，backbone 预期单通道
- **DFormer v1**：depth 复制为 3 通道
- 配置 `C.x_is_single_channel` 需与 backbone 预期一致

## 软链接

若 nk-mmseg 与 DFormer-Jittor 分离部署，可软链数据与 checkpoints：

```bash
ln -s /path/to/DFormer-Jittor/datasets ./datasets
ln -s /path/to/DFormer-Jittor/checkpoints ./checkpoints
```

## 常见问题

1. **软链或路径错误**：检查 `train.txt` / `test.txt` 中的文件名与 `RGB/`、`Depth/`、`Label/` 是否一一对应
2. **depth 通道不匹配**：DFormer v1 用 3 通道，DFormerv2 用单通道，不要混用
3. **GT 标签偏移**：NYU 必须做 `gt - 1` 并将原 0 映射为 255，否则 mIoU 异常
