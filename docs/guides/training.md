# 训练指南

本文档说明 nk-mmseg 的训练流程、命令行参数、checkpoint 与 resume 行为。

## 训练入口

主入口为 `utils/train.py`：

```bash
python utils/train.py --config local_configs.NYUDepthv2.DFormer_Base
```

## 命令行参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--config` | str | 必填 | 配置模块路径，如 `local_configs.NYUDepthv2.DFormer_Large` |
| `--gpus` | int | 1 | GPU 数量 |
| `--checkpoint_dir` | str | 配置内 |  checkpoint 保存目录 |
| `--continue_fpath` | str | - | 从 checkpoint 继续训练（resume） |
| `--epochs` | int | 配置内 | 覆盖 `C.nepochs` |
| `--max-iters` | int | 0 | 每 epoch 最大迭代（0=不限制，用于 smoke） |
| `--max-val-iters` | int | 0 | 验证最大迭代（0=不限制） |
| `--batch-size-override` | int | 0 | 覆盖 batch_size（>0 时生效） |
| `--num-workers-override` | int | 0 | 覆盖 num_workers |
| `--no-mst` | flag | - | 禁用多尺度训练 |
| `--no-amp` | flag | - | 禁用混合精度 |
| `--no-val_amp` | flag | - | 验证时禁用混合精度 |
| `--no-sliding` | flag | - | 验证时禁用滑窗 |
| `--no-syncbn` | flag | - | 禁用 SyncBN |

## 训练流程概要

1. 加载配置、覆盖命令行参数
2. 设置 Jittor flags、随机种子
3. 构建 train/val DataLoader
4. 构建模型、优化器、学习率调度器
5. 若指定 `--continue_fpath`，加载 checkpoint 并 resume
6. 主循环：每个 epoch 前向、反向、优化、评测（按策略）
7. 保存 best mIoU 的 checkpoint

## 评测触发策略

`is_eval(epoch, config)` 控制何时进行验证：

- epoch 1–5：不评测
- epoch 6–20：每 5 epoch
- epoch 21–50：每 10 epoch
- 之后：按 `checkpoint_start_epoch` 或每 25 epoch

## Checkpoint 与 Resume

### 支持的 checkpoint 类型

1. **Jittor 训练态**：含 model + optimizer + epoch/iter，可完整 resume
2. **Jittor model-only**：仅模型参数
3. **PyTorch model-only**（`.pth`/`.pth.tar`）：自动解析并加载，optimizer/epoch 不恢复
4. **PyTorch 训练态**：若格式可识别，亦可尝试完整恢复

### Resume 行为

- 指定 `--continue_fpath` 时，优先尝试完整 resume
- 若 checkpoint 为 model-only 或格式不兼容，自动 fallback 为仅加载模型参数
- 日志中会提示：`Loaded model weights only ... optimizer/epoch not restored`

### 权重加载与 key 映射

`utils/jt_utils.py` 负责加载，会自动处理：

- `module.` 前缀
- `backbone.` / `model.` 前缀
- `gamma_1` → `gamma1` 等命名差异
- `.bn.` → `.norm.`
- decode head：`conv_seg` → `cls_seg`

## 优化器与学习率

- 默认优化器：AdamW（可配置）
- 参数分组：`group_weight`，`len(shape)>=2` 的参数做 weight decay
- 调度器：WarmUpPolyLR
  - poly: `lr = base_lr * (1 - t/T)^power`，`power` 常为 0.9

## 快速 Smoke 示例

```bash
python utils/train.py \
  --config local_configs.NYUDepthv2.DFormer_Large \
  --epochs 10 --max-iters 1 --max-val-iters 1 \
  --batch-size-override 1 --num-workers-override 0 \
  --gpus 1 --no-mst --no-amp --no-val_amp --no-sliding --no-syncbn
```
