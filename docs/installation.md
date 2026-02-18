# 安装指南

本文档详细说明 nk-mmseg 的安装步骤、依赖与常见问题。

## 系统要求

- **操作系统**：Linux（推荐 Ubuntu 20.04+）
- **Python**：3.8 / 3.9 / 3.10
- **GPU**：NVIDIA GPU，支持 CUDA 11.x
- **显存**：建议 8GB+（DFormer-Large 训练 batch_size=8 约需 24GB）

## 1. 创建 conda 环境

```bash
conda create -n nk_mmseg python=3.10 -y
conda activate nk_mmseg
```

建议使用 Python 3.10，与 Jittor 1.3.x 兼容性良好。

## 2. 安装 Jittor

```bash
pip install jittor
```

**重要**：Jittor 首次运行会编译自定义算子，需要安装：

- **libGL**（OpenCV 依赖）：若出现 `libGL.so.1: cannot open shared object file`，执行：

  ```bash
  sudo apt-get update
  sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
  ```

- **CUDA**：确保 `nvcc` 可用，环境变量正确：

  ```bash
  export CUDA_HOME=/usr/local/cuda-11.4
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

## 3. 安装 numpy（兼容性要求）

使用 Jittor 1.3.10.x 时，推荐 `numpy<2`：

```bash
pip install "numpy>=1.24,<2"
# 推荐固定版本以复现
pip install numpy==1.26.4
```

`numpy>=2` 可能与 Jittor 的 tensor 转换和 checkpoint 加载存在兼容问题。

## 4. 安装项目依赖

```bash
cd /path/to/nk-mmseg
pip install opencv-python pillow numpy scipy tqdm tensorboardX tabulate easydict
```

## 5. 安装可选依赖（mmseg 风格 API）

若使用 `mmseg.apis`、`tools/mmseg_infer.py` 等：

```bash
pip install PyYAML addict
```

## 6. 验证安装

```bash
python -c "import jittor as jt; print(jt.__version__)"
python -c "import cv2; print('cv2 OK')"
```

运行快速 smoke 测试：

```bash
python tools/compat_smoke.py
```

## CUTLASS 相关说明

Jittor 会尝试使用 CUTLASS 加速部分算子。若遇到 CUTLASS 下载失败或崩溃：

| 方案 | 操作 | 适用场景 |
|------|------|----------|
| 跳过 CUTLASS | `export use_cutlass=0` | 仅跑通训练/评测 |
| 手动安装 CUTLASS | 见下方命令 | 保留 CUTLASS 算子 |
| 升级 Jittor | `pip install -U jittor` | 使用已修复下载链接的版本 |

手动安装 CUTLASS：

```bash
rm -rf ~/.cache/jittor/cutlass
mkdir -p ~/.cache/jittor/cutlass
cd ~/.cache/jittor/cutlass
git clone --depth 1 https://github.com/NVIDIA/cutlass.git cutlass
```

## 使用 Docker（可选）

若在无显示环境中运行，确保已安装 `libgl1-mesa-glx` 等依赖，或在 Dockerfile 中：

```dockerfile
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0
```

## 下一步

完成安装后，请参考 [快速开始](quickstart.md) 准备数据并运行首次训练。
