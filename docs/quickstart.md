# 快速开始

本指南帮助你在约 5 分钟内完成数据准备、训练与评测。

## 1. 克隆与安装

```bash
git clone https://github.com/VCIP-RGBD/DFormer-Jittor.git nk-mmseg
cd nk-mmseg
# 按 installation.md 完成依赖安装
```

## 2. 数据集准备

### NYUDepthv2

下载链接：[Google Drive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl) | [百度网盘](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q)

解压后目录结构：

```
NYUDepthv2/
├── RGB/          # RGB 图像 *.jpg
├── Depth/        # 深度图 *.png
├── Label/        # 语义标签 *.png
├── train.txt     # 训练集列表
└── test.txt      # 测试集列表
```

配置数据路径（修改 `local_configs/_base_/datasets/NYUDepthv2.py` 或通过 `C.root_dir` 指定）：

```python
C.root_dir = "/path/to/your/datasets"
C.dataset_path = osp.join(C.root_dir, "NYUDepthv2")
```

### SUNRGBD

结构类似，需包含 `RGB/`、`Depth/`、`Label/` 及 `train.txt`、`test.txt`。

## 3. 预训练权重（可选）

从 [模型库](model_zoo.md) 下载预训练 backbone 或完整模型，放入 `checkpoints/pretrained/` 或 `checkpoints/trained/`。

项目支持直接加载 PyTorch `.pth` / `.pth.tar`，无需转换。

## 4. 快速 smoke 训练

单步训练 + 有限评测，用于验证环境：

```bash
export CUDA_HOME=/usr/local/cuda-11.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python utils/train.py \
  --config local_configs.NYUDepthv2.DFormer_Large \
  --epochs 10 \
  --max-iters 1 \
  --max-val-iters 1 \
  --batch-size-override 1 \
  --num-workers-override 0 \
  --gpus 1 --no-mst --no-amp --no-val_amp --no-sliding --no-syncbn
```

## 5. 完整训练

```bash
python utils/train.py --config local_configs.NYUDepthv2.DFormer_Base
```

或使用脚本：

```bash
bash train.sh
```

## 6. 评测

单尺度评测：

```bash
python utils/eval.py \
  --config local_configs.NYUDepthv2.DFormer_Base \
  --continue_fpath checkpoints/trained/your_model.pth
```

论文复现（多尺度 + flip + 滑窗）：

```bash
bash tools/reproduce_dformer_scores.sh
```

可设置环境变量：

```bash
DFORMER_ROOT=/path/to/DFormer-Jittor
PYTHON_BIN=python
GPUS=1
bash tools/reproduce_dformer_scores.sh
```

## 7. 推理与可视化

```bash
python tools/mmseg_infer.py \
  --config local_configs.NYUDepthv2.DFormer_Large \
  --checkpoint checkpoints/trained/NYUv2_DFormer_Large.pth \
  --img /path/to/rgb.png \
  --modal-x /path/to/depth.png \
  --out-file output/vis.png
```

## 8. 迁移与兼容性检查

```bash
python tools/migration_audit.py --eval-samples 5
python tools/mmengine_runner_smoke.py
python tools/mmseg_api_smoke.py
```

## 下一步

- [训练指南](guides/training.md)：训练参数、checkpoint、resume
- [评测指南](guides/evaluation.md)：评测策略与协议
- [配置参考](configuration.md)：配置字段说明
