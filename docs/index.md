# nk-mmseg 文档

nk-mmseg 是 **DFormer** 与 **DFormerv2** 的 Jittor 实现，面向 RGB-D 语义分割。本仓库基于 [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) 深度学习框架，提供训练、评测与推理的完整流程。

## 关于论文

- **DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation** (ICLR 2024)  
  [论文](https://arxiv.org/abs/2309.09668) | [主页](https://yinbow.github.io/Projects/DFormer/index.html)

- **DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation** (CVPR 2025)  
  [论文](https://arxiv.org/abs/2504.04701)

## 核心特性

- **纯 Jittor 实现**：无 PyTorch 运行时依赖
- **PyTorch 权重兼容**：可直接加载 `.pth` / `.pth.tar` 预训练权重
- **mmseg/mmengine 兼容**：支持 mmseg 风格 API 与配置
- **多模态输入**：支持 RGB + Depth 联合输入

## 快速链接

| 文档 | 说明 |
|------|------|
| [安装指南](installation.md) | 环境、依赖与安装步骤 |
| [快速开始](quickstart.md) | 5 分钟跑通训练与评测 |
| [训练指南](guides/training.md) | 训练流程与参数说明 |
| [评测指南](guides/evaluation.md) | 单尺度 / 多尺度 / 滑窗评测 |
| [配置参考](configuration.md) | 配置字段详解 |
| [模型库](model_zoo.md) | 预训练权重与复现指标 |
| [故障排除](troubleshooting.md) | 常见问题与解决方案 |

## 目录结构概览

```
nk-mmseg/
├── configs/               # mmseg 风格配置
├── local_configs/         # DFormer 训练配置
├── models/                # DFormer 模型定义（encoders/decoders）
├── mmseg/                 # mmseg 兼容层
├── mmengine/              # mmengine Runner/Hook 兼容
├── utils/                 # 训练/评测/数据加载
├── tools/                 # 脚本与工具
├── tests/                 # 回归测试
└── docs/                  # 本文档
```

## 环境要求

- Python 3.8+
- Jittor 1.3.9+
- CUDA 11.x（GPU 训练）
- numpy<2（推荐 1.26.4，与 Jittor 1.3.10.x 兼容）

## 支持的数据集

- **NYUDepthv2**：室内 RGB-D 语义分割
- **SUNRGBD**：室内场景理解

## 下一步

1. 阅读 [安装指南](installation.md) 完成环境配置
2. 按照 [快速开始](quickstart.md) 运行训练与评测
3. 查阅 [配置参考](configuration.md) 自定义训练与数据路径
