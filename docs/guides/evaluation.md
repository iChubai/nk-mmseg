# 评测指南

本文档说明 nk-mmseg 的评测入口、评测策略（单尺度、多尺度、滑窗）及与论文复现的对应关系。

## 评测入口

主入口为 `utils/eval.py`：

```bash
python utils/eval.py \
  --config local_configs.NYUDepthv2.DFormer_Large \
  --continue_fpath /path/to/best.pth
```

## 命令行参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--config` | str | 必填 | 配置模块路径 |
| `--continue_fpath` | str | 必填 | checkpoint 路径 |
| `--strict-load` | flag | False | 严格加载，key/shape 不匹配则报错 |
| `--gpus` | int | 1 | GPU 数量 |
| `--batch-size` | int | 1 | 验证 batch size |
| `--num-workers` | int | 4 | DataLoader workers |
| `--multi_scale` | flag | - | 多尺度评测 |
| `--scales` | float+ | 0.5 0.75 1.0 1.25 1.5 | 多尺度比例（与 `--multi_scale` 配合） |
| `--flip` | flag | - | 水平翻转增强 |
| `--sliding` | flag | - | 滑窗推理 |
| `--window-size` | int int | 512 512 | 滑窗尺寸 |
| `--stride` | int int | 256 256 | 滑窗步长 |
| `--save-pred` | flag | - | 保存预测结果 |
| `--pred-dir` | str | ./predictions | 预测保存目录 |
| `--verbose` | flag | - | 详细输出 |

## 评测策略

### 1. 单尺度（Standard）

- 输入 resize 到固定尺寸（如 480×640）
- 前向一次，argmax 得到预测
- 用于快速验证与调试

### 2. 多尺度 + Flip（MSF）

- 对多个 scale（如 0.5, 0.75, 1.0, 1.25, 1.5）分别推理
- 每个 scale 的结果 resize 回原图尺寸
- 累加 softmax 概率（不是 logits），可选 flip 分支再累加
- 最终 argmax 得到预测
- 用于论文复现

### 3. 滑窗（Sliding Window）

- 按 `crop_size` 和 `stride_rate` 划窗
- 每窗预测后回填到全图，重叠区域做平均
- 适用于大图或显存有限场景

## 论文复现脚本

```bash
bash tools/reproduce_dformer_scores.sh
```

该脚本会：

1. 链接 DFormer-Jittor 的 checkpoints
2. 对 DFormer-Large 和 DFormerv2-Large 执行 MS + flip + sliding 评测
3. 输出 mIoU 等指标

环境变量：

```bash
DFORMER_ROOT=/path/to/DFormer-Jittor
PYTHON_BIN=python
GPUS=1
STRICT_LOAD=1
MS_SCALES="0.5 0.75 1.0 1.25 1.5"
```

## 评测协议一致性

与 PyTorch 基线对比时，需确保以下一致：

- scale 列表
- flip 开关
- sliding 参数（crop_size、stride_rate）
- `align_corners`
- `ignore_index`（通常 255）

## 指标说明

`SegmentationMetric.get_results()` 返回：

- `IoU_per_class`
- `mIoU`
- `Acc_per_class`
- `mAcc`
- `Overall_Acc`
- `FWIoU`

注意：`compute_iou` 等返回百分比样式；`get_results` 返回 0~1 浮点，对比时注意单位。
