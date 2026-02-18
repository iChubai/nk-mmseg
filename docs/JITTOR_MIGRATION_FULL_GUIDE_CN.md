# nk-mmseg Jittor 迁移全景手册（高层原理到低层细节）

## 0. 阅读说明

这份文档是面向“完整迁移与长期维护”的工程手册，不是简单的使用教程。  
如果你的目标是：

1. 明确为什么 `DFormer` 与 `mmseg` 代码看起来“割裂”；
2. 精确理解当前 Jittor 迁移链路如何工作；
3. 在不破坏复现点数的前提下完成单栈收敛；
4. 让后续新模型迁移有统一方法论；

那么这份文档就是为这个目标写的。

---

## 1. 迁移目标的严格定义

“迁移完成”在工程上至少要满足四个等价：

1. 语义等价（Semantic Equivalence）
   - 模型结构、前向路径、loss 计算、后处理语义一致。
2. 协议等价（Protocol Equivalence）
   - 训练协议与评测协议一致：数据预处理、尺度策略、flip、sliding、ignore_index、保存恢复策略等。
3. 运行时等价（Runtime Equivalence）
   - 不依赖 PyTorch 运行，纯 Jittor 框架可训练、可评测、可推理。
4. 工程等价（Engineering Equivalence）
   - 配置、注册器、Runner、API、checkpoint、测试系统可持续维护。

只做到第 1 条通常“能跑”；做到 1+2 才可能“点数接近”；做到 1+2+3+4 才是“完整迁移”。

---

## 2. 为什么迁移不是“把 torch 改成 jittor”

很多项目迁移失败，不是模型错，而是系统错。典型原因：

1. 只改了算子 API，没有校验训练/评测协议。
2. 只改了模型代码，没改 checkpoint 读写语义。
3. 只跑了前向 smoke test，没跑 full train + resume + full eval。
4. 忽略了“工程结构迁移”（配置体系、Runner、Hook、Inferencer）。

在 `nk-mmseg` 中，正确路线是“分层迁移 + 最终收敛”：

1. 模型层先可用；
2. 训练评测先可复现；
3. 框架兼容层补齐；
4. 最后把多入口收敛为单入口。

---

## 3. 当前仓库结构与双栈现象

### 3.1 顶层目录角色

以 `/defaultShare/archive/yinbowen/Houjd/nk-mmseg` 为核心：

1. `models/`
   - DFormer 最早迁移实现（encoder/decoder/build）。
2. `utils/`
   - DFormer 训练/评测主链路（`train.py`, `eval.py`, `infer.py`, `jt_utils.py`）。
3. `local_configs/`
   - DFormer 原链路配置体系。
4. `mmseg/`
   - mmseg 风格组件（datasets/models/apis/engine 等）。
5. `mmengine/`
   - Runner/loop/hook/checkpoint/optim/scheduler 兼容实现。
6. `mmcv/`
   - 常用模块/算子兼容实现。
7. `torch/`
   - 运行时兼容层（Jittor 上模拟常用 PyTorch 行为）。
8. `tests/`
   - 回归测试集合。
9. `tools/`
   - 兼容性与迁移验证脚本。

### 3.2 为什么你会感知“割裂”

当前确实存在两条历史路径：

1. DFormer 路径
   - 入口：`utils/train.py`、`utils/eval.py`
   - 模型：`models/builder.py` + `models/encoders/*`
   - 配置：`local_configs/*`

2. mmseg/mmengine 路径
   - 入口：`mmengine.runner.Runner`、`mmseg.apis.*`
   - 模型：`mmseg/models/*` 注册体系
   - 配置：`configs/*`（通用 mmseg 风格）

双路径并存的优点是“过渡期稳定”，缺点是“长期维护成本高，认知负担大”。

---

## 4. 当前代码状态（结论先行）

结合已有实现与验证，当前仓库状态可概括为：

1. 纯 Jittor 训练/评测可跑通（DFormer 主路径）。
2. torch 权重文件可以被纯 Jittor 路径读取（不依赖 torch 运行时）。
3. checkpoint resume 路径已补强（含 model-only fallback）。
4. mmseg/mmengine/mmcv 兼容层基础已具备。
5. 主要剩余问题不是“能不能跑”，而是“如何收敛为单入口单配置单实现”。

---

## 5. 术语与概念统一

为了避免讨论时混淆，统一术语如下：

1. model-only checkpoint
   - 只含参数字典，不含优化器和 epoch/iter。
2. training-state checkpoint
   - 含模型参数 + 优化器状态 + 训练进度状态。
3. strict load
   - key 和 shape 必须完全匹配，否则报错。
4. non-strict load
   - 允许部分 key 不匹配，以日志统计差异。
5. protocol
   - 不只是模型，而是完整数据和评测行为约定。
6. single-stack
   - 一个入口、一套配置、一套注册体系。

---

## 6. 从高层到低层的迁移分层模型

建议把整个迁移拆为 6 层，每层有输入输出：

1. L0 数据层
   - 输入：原始数据目录和标注
   - 输出：统一数据读取语义（RGB + modal_x + gt）
2. L1 变换层
   - 输入：样本
   - 输出：训练/验证增强与归一化结果
3. L2 模型层
   - 输入：张量
   - 输出：logits 或 loss 结构
4. L3 训练评测层
   - 输入：model + dataloader
   - 输出：checkpoint 与 metric
5. L4 框架层
   - 输入：配置与注册对象
   - 输出：Runner/Hook/API 可用
6. L5 工程层
   - 输入：多路径代码
   - 输出：单入口可维护系统

只要某层协议没锁死，上层结论都不稳定。

---

## 7. 数据层迁移细节（L0）

核心文件：

1. `utils/dataloader/RGBXDataset.py`
2. `local_configs/_base_/datasets/NYUDepthv2.py`
3. `local_configs/_base_/datasets/SUNRGBD.py`

### 7.1 NYUDepthv2 配置关键字段

关键约定：

1. `dataset_name = NYUDepthv2`
2. RGB：`RGB/*.jpg`
3. GT：`Label/*.png`
4. Depth：`Depth/*.png`
5. `num_classes = 40`
6. `background = 255`

### 7.2 标签变换语义（非常关键）

`RGBXDataset._gt_transform` 对 NYU 的语义：

1. 原标签通常为 `1..40`，`0` 为无效。
2. 迁移后做 `gt = gt - 1`。
3. 原 `0` 变成 `-1`，再映射为 `255`（ignore_index）。

这一步如果错，mIoU 会明显偏离。

### 7.3 多模态通道语义

`RGBXDataset.__getitem__` 的逻辑：

1. 对 modal `d`（depth）先按灰度读入。
2. 若 backbone 是 `DFormerv2*`，保留单通道（扩到 `[H,W,1]`）。
3. 若 backbone 是 DFormer v1，复制为 3 通道。

这也是 DFormer v1/v2 差异点之一，不能混用。

### 7.4 常见数据层错误

1. 软链错误导致读错目录。
2. train.txt/test.txt 对不上数据目录。
3. depth 单通道和三通道处理不匹配 backbone。
4. GT 未执行 NYU 标签偏移。

---

## 8. 变换层迁移细节（L1）

核心文件：

1. `utils/dataloader/dataloader.py`
2. `utils/transforms.py`

### 8.1 训练预处理 `TrainPre`

顺序（不可随意调整）：

1. `random_mirror`（RGB/GT/modal 同步翻转）
2. `random_scale`
3. RGB normalize（ImageNet mean/std）
4. depth normalize（固定参数）
5. 随机 crop + pad 到固定尺寸
6. 转 CHW

### 8.2 depth 归一化参数

在 `TrainPre`/`ValPre` 中：

1. 单通道 depth：`mean=[0.48], std=[0.28]`
2. 三通道 depth：每通道同样的 `[0.48]/[0.28]`

这套参数是为了对齐原 PyTorch 训练分布。  
如果误改为 RGB 的 mean/std，会产生稳定精度损失。

### 8.3 验证预处理 `ValPre`

不做 random augment，只做：

1. normalize
2. HWC->CHW

验证必须和训练解耦，否则 metric 抖动大。

### 8.4 worker 与随机性

`seed_worker` 使用全局 seed 派生 worker seed。  
建议与 `set_seed` 配合，保证可复现实验。

---

## 9. 模型层迁移细节（L2）

核心文件：

1. `models/builder.py`
2. `models/encoders/DFormer.py`
3. `models/encoders/DFormerv2.py`
4. `models/decoders/ham_head.py`

### 9.1 DFormer v1 与 v2 的核心差异

主要差异包括：

1. backbone 结构和通道配置不同。
2. depth 输入通道预期不同（v2 常单通道）。
3. Drop path 等超参默认不同。

迁移时必须确保：

1. stage 输出通道和顺序一致；
2. decode head `in_channels/in_index` 对应正确；
3. 上采样分辨率恢复逻辑一致。

### 9.2 execute / forward 双入口

在 Jittor 模型里常保留：

1. `execute` 真正执行路径。
2. `forward` 调 `execute` 提供外部兼容。

这可同时满足：

1. Jittor 原生调用；
2. mmseg/兼容层调用习惯；
3. 测试脚本和 API 的统一行为。

### 9.3 训练返回结构兼容

当前模型实现需兼容多种调用上下文：

1. 推理路径：返回 logits。
2. 训练路径：返回可被训练循环解析的 loss 结构。
3. mm 风格路径：可按 `mode='loss'/'predict'` 或类似参数分发。

如果训练循环和模型返回协议没对齐，会出现“能前向、不能训练”的断裂。

---

## 10. 指标与度量细节（L3-Metric）

核心文件：

1. `utils/metric.py`

### 10.1 Confusion Matrix 更新

核心逻辑：

1. flatten pred/gt
2. 过滤 ignore_index=255
3. `np.bincount` 构建 hist
4. 累积 hist

### 10.2 指标计算

`get_results` 返回：

1. `IoU_per_class`
2. `mIoU`
3. `Acc_per_class`
4. `mAcc`
5. `Overall_Acc`
6. `FWIoU`

注意：

1. `compute_iou/compute_f1/compute_pixel_acc` 返回百分比样式；
2. `get_results` 返回 0~1 浮点比例；
3. 日志和对比时要注意单位，不要混淆。

---

## 11. 权重迁移总流程（L3-Checkpoint）

核心文件：

1. `utils/jt_utils.py`

### 11.1 支持的权重来源

当前加载器支持：

1. 原生 Jittor checkpoint（`jt.save`）
2. torch zip checkpoint（`.pth`, `.pt`, `.pth.tar`, `.bin`）

### 11.2 torch zip 读取机制

关键步骤：

1. 用 `zipfile` 找到 `data.pkl`
2. 自定义 unpickler 解析 storage/tensor rebuild
3. 从 zip 内 `data/<key>` 读 storage
4. 组装出 numpy 张量
5. 提取 `state_dict` 或 `model` 字段

这一机制避免了运行时强依赖 PyTorch。

### 11.3 参数名映射策略

`_convert_param_name` 处理以下场景：

1. `module.` 前缀
2. `backbone.` 前缀
3. `model.` 前缀
4. `gamma_1/gamma_2` 到 `gamma1/gamma2`
5. `.bn.` 到 `.norm.`
6. decode head 显式映射（`conv_seg` -> `cls_seg`）

### 11.4 shape 与 dtype 对齐

加载时做：

1. shape 检查，不匹配直接跳过并统计；
2. dtype 按目标参数 dtype 对齐；
3. 更新后 `sync`。

### 11.5 strict 模式

`strict=True` 会在以下情况报错：

1. missing_after_mapping 非空
2. shape mismatch > 0
3. failed assign > 0

建议只在“最终复现验收”开启 strict。

---

## 12. checkpoint 恢复（resume）语义与修复

核心文件：

1. `utils/engine/engine.py`
2. `utils/jt_utils.py`

### 12.1 关键语义

恢复分两种：

1. 完整恢复
   - 恢复模型参数 + optimizer 状态 + epoch/iteration。
2. model-only 恢复
   - 仅加载模型参数，不改 optimizer 和 epoch/iteration。

### 12.2 已修复问题

1. `is_restore=True` 时避免把已带 `module.` 的 key 再次加前缀，防止 `module.module.*`。
2. `--continue_fpath` 指向 torch model-only 权重时不再强行加载 optimizer 状态。
3. 通过优化器状态特征判定是否可完整恢复；否则自动 fallback model-only。

### 12.3 为什么这很重要

过去典型崩溃点：

1. `_pickle.UnpicklingError`
2. `optimizer.load_state_dict` 格式不匹配
3. 恢复后 epoch/iter 错位

当前修复后：

1. torch model-only 文件能作为继续训练的初始权重
2. jittor training-state 文件能完整恢复训练进度

---

## 13. 训练链路（DFormer 主路径）逐步拆解

核心文件：

1. `utils/train.py`

### 13.1 参数层

常用参数：

1. `--config`
2. `--checkpoint_dir`
3. `--continue_fpath`
4. `--epochs`
5. `--max-iters` / `--max-val-iters`（smoke/debug）
6. `--no-mst` / `--sliding` 等评测策略开关

### 13.2 初始化层

关键步骤：

1. 设置 Jittor flags（`use_cuda`、`lazy_execution` 等）
2. 读取配置 `local_configs.*`
3. 根据命令行覆盖训练参数
4. 初始化 Engine 与 logger
5. 若给定 `continue_fpath`，接入 restore 流程

### 13.3 数据层构建

1. `get_train_loader`
2. `get_val_loader`
3. 训练与验证 worker 配置分开（验证常设为 0 防死锁）

### 13.4 优化器与调度

1. 参数分组来自 `group_weight`
2. 优化器：`AdamW` 或 `SGDM`
3. 调度器：`WarmUpPolyLR`

### 13.5 主循环

每个 epoch：

1. 取 batch
2. 前向计算 loss
3. `optimizer.step(loss)`
4. `lr_policy.step(global_iter)`
5. 记录日志
6. 在设定 epoch 执行评测
7. best mIoU 时保存 checkpoint

### 13.6 评测触发策略

`is_eval(epoch, config)` 当前策略是分阶段：

1. 前 5 个 epoch 不评测
2. 6~20：每 5 epoch
3. 21~50：每 10 epoch
4. 后续按 checkpoint_start_epoch 或固定间隔

这决定了你何时会看到 checkpoint 产出。

---

## 14. 学习率与优化器细节

核心文件：

1. `utils/lr_policy.py`
2. `utils/init_func.py`

### 14.1 参数分组 `group_weight`

规则：

1. `len(shape) >= 2` 参数进入 decay 组（卷积/线性权重）
2. 其余（bias/bn 等）进入 no-decay 组

优点：

1. 符合常见视觉模型实践
2. 更接近原始训练协议

### 14.2 WarmUpPolyLR 公式

阶段 1：warmup（linear 或 constant）  
阶段 2：poly decay

poly 核心：

`lr = base_lr * (1 - t / T)^power`

其中：

1. `t` 为当前迭代
2. `T` 为最大迭代
3. `power` 常为 `0.9`

---

## 15. 评测链路逐步拆解

核心文件：

1. `utils/eval.py`
2. `utils/val_mm.py`
3. `utils/metric.py`

### 15.1 标准评测 `evaluate`

流程：

1. `model.eval()`
2. 遍历 val loader
3. 前向得到 logits
4. `argmax` 得到预测标签
5. 更新 confusion matrix
6. 汇总指标

### 15.2 多尺度 + flip 评测 `evaluate_msf`

关键点：

1. 对每个 scale 推理后 resize 回原尺寸
2. 累加 softmax 概率，而不是直接累加 logits
3. 可选 flip 分支再累加
4. 最终 `argmax` 生成预测

### 15.3 滑窗推理 `slide_inference`

关键点：

1. 根据 crop_size / stride_rate 划窗
2. 每窗预测后回填到全图张量
3. 用 count matrix 做重叠区域平均

### 15.4 评测一致性要求

和原 PyTorch 对齐时，必须确保以下全部一致：

1. scale 列表
2. flip 开关
3. sliding 参数（crop/stride）
4. align_corners
5. ignore_index

---

## 16. mmseg API 兼容层（当前桥接）

核心文件：

1. `mmseg/apis/inference.py`
2. `mmseg/apis/mmseg_inferencer.py`

### 16.1 当前实现方式

`mmseg.apis.init_model` 目前会调用：

1. `load_local_config`
2. `models.build_model`
3. `utils.jt_utils.load_model`

这说明 mmseg API 能用，但底层仍桥接到 DFormer 原链路。

### 16.2 意义

它是“兼容入口”，不是“最终收敛状态”。  
最终目标是 API 入口和底层实现都统一走 mmseg registry + mmengine runner。

---

## 17. mmengine Runner 兼容层解读

核心文件：

1. `mmengine/runner/runner.py`
2. `mmengine/runner/loops.py`
3. `mmengine/runner/checkpoint.py`

### 17.1 Runner 能力

支持：

1. `from_cfg` 构建
2. train/val/test loop
3. hook 调度
4. load/save checkpoint
5. optimizer wrapper 与 scheduler

### 17.2 checkpoint 适配

`mmengine/runner/checkpoint.py` 做了：

1. `_load_checkpoint_any` 对接项目统一加载器
2. `_extract_state_dict` 提取多种字段名
3. 支持去掉 `module.` 前缀的 fallback
4. 支持部分加载（按 key+shape）

这层是把“加载兼容”从业务脚本抽成基础能力的关键。

---

## 18. mmseg 模型注册现状与缺口

从当前注册内容看：

1. `mmseg/models` 已有大量通用 backbone/head/segmentor。
2. DFormer 主体实现仍主要在 `models/`。
3. `mmseg` 路径虽有 `MultimodalEncoderDecoder`，但 DFormer 主模型未完全注册到 mmseg 主族群里。

这就是“代码功能可用但架构未收敛”的根因之一。

---

## 19. “割裂”问题的本质诊断

不是一个 bug，而是工程阶段问题：

1. 阶段目标 1：先复现点数（达成）
2. 阶段目标 2：再统一架构（进行中）

常见误区：

1. 看到双路径就想立刻删旧代码
2. 在统一前就改评测协议
3. 没有迁移映射表直接大重构

正确做法是“可回滚的阶段性收敛”。

---

## 20. 单栈收敛路线图（可直接执行）

下面给的是可以直接排期执行的方案。

### 20.1 阶段 A：入口收敛

目标：

1. 对外只暴露一个训练入口和一个评测入口。

实施：

1. 新增 `tools/train.py`、`tools/test.py`（统一 CLI）
2. `utils/train.py` 和 `utils/eval.py` 变成兼容包装器，内部转发到新入口
3. README 只写新入口命令

验收：

1. 老命令可用
2. 新命令可用
3. 两者结果一致

### 20.2 阶段 B：配置收敛

目标：

1. DFormer 训练也走 `configs/` 风格，而不是只依赖 `local_configs/`。

实施：

1. 新建 `configs/dformer/*`
2. 把 local config 字段映射为 mmseg config schema
3. 保留 local->configs 映射适配器（短期）

验收：

1. 同一模型通过两套配置跑出的关键结果一致
2. CI 默认只跑 `configs/` 路径

### 20.3 阶段 C：实现收敛

目标：

1. DFormer backbone/decode head 注册进 `mmseg.registry.MODELS`。

实施：

1. 把 `models/encoders/*` 迁入 `mmseg/models/backbones/` 或做注册包装
2. 把 `models/decoders/*` 迁入 `mmseg/models/decode_heads/` 或做注册包装
3. `models/builder.py` 退化为 thin adapter

验收：

1. mmengine runner 可直接构建 DFormer 模型
2. `mmseg.apis.init_model` 不再依赖 `models.build_model`

### 20.4 阶段 D：权重与恢复统一

目标：

1. checkpoint 加载统一从 mmengine checkpoint adapter 进入。

实施：

1. 把 `utils/jt_utils.py` 的加载能力封装成 mmengine 标准接口
2. `utils/engine/engine.py` 调用统一加载 API
3. 减少多处重复 fallback 逻辑

验收：

1. torch 权重、jittor 权重、训练态恢复都能从统一接口完成

---

## 21. 配置字段映射表（local_configs -> mmseg configs）

建议定义映射（示例）：

1. `C.backbone` -> `model.backbone.type`
2. `C.decoder` -> `model.decode_head.type`
3. `C.decoder_embed_dim` -> `model.decode_head.channels`
4. `C.num_classes` -> `model.decode_head.num_classes`
5. `C.pretrained_model` -> `load_from` 或 `model.backbone.pretrained`
6. `C.batch_size` -> `train_dataloader.batch_size`
7. `C.num_workers` -> `train_dataloader.num_workers`
8. `C.lr` -> `optim_wrapper.optimizer.lr`
9. `C.weight_decay` -> `optim_wrapper.optimizer.weight_decay`
10. `C.nepochs` / `C.niters_per_epoch` -> `train_cfg.max_iters`
11. `C.eval_iter` -> `default_hooks.checkpoint.interval` 或 `val_interval`

建议把映射固化为脚本，避免手工迁移造成不一致。

---

## 22. checkpoint 兼容矩阵

建议在文档和测试中明确 4 类输入：

1. torch model-only（zip）
2. torch training-state（若存在）
3. jittor model-only
4. jittor training-state

对每类定义行为：

1. 是否可 strict
2. 是否可 resume optimizer
3. 是否可恢复 epoch/iter
4. 失败时 fallback 策略

---

## 23. 训练复现标准作业流程（SOP）

以下流程用于“不是 smoke，而是复现验收”。

### 23.1 环境固定

1. 固定 python：`/defaultShare/archive/yinbowen/Houjd/envs/jittordet/bin/python`
2. 固定 CUDA 环境变量
3. 固定随机种子
4. 固定数据与权重软链

### 23.2 训练阶段

1. DFormer-Large 全训练
2. DFormerv2-L 全训练
3. 每个实验都保存：
   - 配置快照
   - 训练日志
   - 验证日志
   - best checkpoint

### 23.3 评测阶段

1. 单尺度评测
2. MS+Flip
3. Sliding（如原协议需要）

### 23.4 对比阶段

1. 与 PyTorch 基线点数对比
2. 与 DFormer-Jittor 基线对比
3. 与上次 nk-mmseg 版本对比

---

## 24. 数值偏差分析方法

如果最终点数差异大，按下面顺序排查：

1. 数据是否一致
   - 样本总数、split 文件、标签变换
2. 评测协议是否一致
   - scale、flip、sliding、align_corners
3. 权重加载是否一致
   - loaded/skipped/mismatch 日志
4. 优化器和 lr 是否一致
   - warmup/poly 参数
5. 模型结构是否一致
   - 特征维度、head 输入索引、上采样尺寸

建议每次只改一个维度，避免多变量同时变化导致难以定位。

---

## 25. 典型日志解释（你最常见会看到的）

### 25.1 `Loaded X params, skipped Y`

解释：

1. `X` 越接近模型参数总数越好；
2. `Y` 不一定错误，可能是预期过滤项（如 `num_batches_tracked`）；
3. 但 `Y` 突然增大通常意味着映射回归。

### 25.2 `Missing key(s) before mapping`

解释：

1. 在带前缀/跨框架文件里常见；
2. 看的是“原始 key 空间”，不等价于最终加载失败；
3. 要看 `missing_after_mapping` 才能判定实质问题。

### 25.3 `Loaded model weights only ... optimizer/epoch not restored`

解释：

1. 当前 checkpoint 被判定为 model-only；
2. 是合理 fallback，不是错误；
3. 适合做“从预训练继续训练”的场景。

---

## 26. 低层实现伪代码（便于快速理解）

### 26.1 统一 checkpoint 加载

```text
load_checkpoint_any(path):
    if suffix in {pth, pt, pth.tar, bin}:
        try torch_zip_parse
        except -> try jt.load
    else:
        try jt.load
```

### 26.2 key 映射加载

```text
for (src_key, src_tensor) in checkpoint_state:
    dst_key = convert_param_name(src_key)
    if dst_key is None: skip
    if shape mismatch: skip
    cast to target dtype
    update target param
```

### 26.3 resume 判断

```text
if checkpoint has model and optimizer looks like jittor optimizer-state:
    full resume (model + optimizer + epoch/iter)
else:
    model-only load
```

---

## 27. 测试体系建议（必须工程化）

当前测试已经覆盖很多兼容能力，但建议长期固定三层测试：

1. 单元测试
   - key 映射、shape 对齐、loss 数值、dataset transform。
2. 集成测试
   - train -> save -> resume。
3. 端到端测试
   - full eval 点数验收。

推荐关键测试项：

1. `tests/test_checkpoint_resume.py`
2. `tests/test_models.py`
3. `tests/test_data.py`
4. `tests/test_losses.py`
5. `tests/test_mmcv_ops_compat.py`

---

## 28. 针对“DFormer 与 mmseg 割裂”的落地建议（具体到文件）

### 28.1 第一批改造（低风险）

1. 新增统一 CLI：
   - `tools/train.py`
   - `tools/test.py`
2. 旧脚本改为转发：
   - `utils/train.py`
   - `utils/eval.py`
3. 文档入口统一：
   - `README.md`
   - `README_CN.md`

### 28.2 第二批改造（中风险）

1. DFormer 注册到 mmseg registry：
   - `mmseg/models/backbones/`
   - `mmseg/models/decode_heads/`
2. `models/builder.py` 改 thin adapter。
3. local config 到 mmseg config 的转换脚本。

### 28.3 第三批改造（高风险，需要基线回归）

1. 删除重复实现（在确认结果一致后）
2. checkpoint 和 runner 逻辑收口
3. 删除不再使用的兼容胶水代码

每批都必须跑完整回归，不建议跨批大跳跃。

---

## 29. 迁移中的性能与稳定性权衡

### 29.1 稳定优先策略

为了先保证不崩，当前存在一些偏保守设置：

1. val loader worker 设为 0（避免死锁）
2. 部分循环中主动 `jt.clean()/jt.gc()`
3. 早期 epoch 减少评测频率

### 29.2 性能优化顺序建议

先保证复现，再优化吞吐：

1. 锁定精度协议
2. 固定 checkpoint 与 resume 行为
3. 再尝试提高 dataloader worker
4. 再优化显存策略和 mixed precision

避免“提速后点数飘了但不知道哪步导致”。

---

## 30. 版本管理与可追溯性建议

为了让后续每次改动可追溯，建议固定产物结构：

1. 每个实验目录包含：
   - config 副本
   - 训练日志
   - 验证日志
   - checkpoint
   - commit hash
2. 每次权重映射改动都记录：
   - 映射规则变更
   - loaded/skipped 对比
   - 复现实验结果变化

这样后续出现偏差能快速回溯。

---

## 31. 新模型迁移模板（可复制到任何项目）

### 31.1 Stage A：结构迁移

1. 搭 backbone
2. 搭 decode head
3. 跑 shape smoke

### 31.2 Stage B：训练闭环

1. 跑 loss backward
2. 跑短训
3. 跑 save/resume

### 31.3 Stage C：评测闭环

1. 单尺度
2. MS+Flip
3. Sliding

### 31.4 Stage D：工程闭环

1. 注册器接入
2. Runner 接入
3. API 接入
4. 测试接入

---

## 32. 常见反模式（尽量避免）

1. 为了“看起来整洁”提前删掉旧路径。
2. 没做 full eval 就宣布复现完成。
3. 只对齐 mIoU，不对齐评测协议。
4. 把 model-only 权重误当成 resume checkpoint。
5. 在未固化测试前进行大规模重构。

---

## 33. 未来去兼容层方向（长期）

`torch/` 兼容层是阶段性资产。  
长期建议：

1. 新增功能优先用原生 Jittor 接口实现。
2. 逐步减少 `torch` 兼容调用面。
3. 最终把兼容层限制在少量第三方依赖桥接。

这样可降低维护成本与隐性行为差异。

---

## 34. 迁移验收总清单（最终版）

请将以下作为“迁移完成”判据：

1. 纯 Jittor train/eval/infer 全可运行。
2. torch 权重可加载，且关键模型点数达到目标区间。
3. jittor training-state checkpoint 可完整 resume。
4. 单尺度与多尺度评测协议都有稳定结果。
5. 全量测试通过（包含 checkpoint resume 回归）。
6. 对外只有一套推荐入口与一套推荐配置。
7. 文档、脚本、CI 与实际代码路径一致。

---

## 35. 快速问答（FAQ）

### Q1：为什么我用 `jt.load` 直接读 `.pth` 会炸？

因为很多 `.pth` 是 torch zip 格式，不是原生 Jittor pickle。  
应使用项目统一加载器。

### Q2：为什么 `continue_fpath` 指向 torch 权重时不能恢复 epoch？

因为那通常是 model-only，不含可用训练状态。  
系统会自动 fallback 到模型参数加载。

### Q3：为什么点数一直比 torch 低一截？

优先检查：

1. 数据软链和 split
2. depth 通道处理
3. 评测协议（MS/flip/sliding）
4. 权重映射统计
5. `align_corners` 与 resize 逻辑

---

## 36. 结论

`nk-mmseg` 的迁移已经跨过“能跑”阶段，核心挑战转为“架构收敛”。  
最优策略不是推倒重写，而是：

1. 先锁协议与结果；
2. 再做分阶段收敛；
3. 每阶段都用回归测试和点数对比守住质量。

当 DFormer 主实现、配置、入口都收敛到 mmseg/mmengine 主栈，且点数与原基线稳定一致时，这个迁移才真正工程闭环。

---

## 附录 A：建议执行命令模板

以下命令示例用于“全流程复现”。

### A.1 环境

```bash
cd /defaultShare/archive/yinbowen/Houjd/nk-mmseg
export CUDA_HOME=/usr/local/cuda-11.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
PY=/defaultShare/archive/yinbowen/Houjd/envs/jittordet/bin/python
```

### A.2 训练（DFormer-Large）

```bash
$PY utils/train.py \
  --config=local_configs.NYUDepthv2.DFormer_Large \
  --gpus=1 \
  --checkpoint_dir=/tmp/repro_dformer_large
```

### A.3 训练（DFormerv2-L）

```bash
$PY utils/train.py \
  --config=local_configs.NYUDepthv2.DFormerv2_L \
  --gpus=1 \
  --checkpoint_dir=/tmp/repro_dformerv2_l
```

### A.4 评测（单尺度）

```bash
$PY utils/eval.py \
  --config=local_configs.NYUDepthv2.DFormer_Large \
  --continue_fpath=/path/to/best.pth
```

### A.5 评测（MS + Flip + Sliding）

```bash
$PY utils/eval.py \
  --config=local_configs.NYUDepthv2.DFormer_Large \
  --continue_fpath=/path/to/best.pth \
  --multi_scale --scales 0.5 0.75 1.0 1.25 1.5 \
  --flip --sliding
```

### A.6 测试

```bash
$PY tests/run_tests.py
```

---

## 附录 B：建议新增的长期自动化任务

1. 每日回归：
   - `tests/run_tests.py`
   - 2 个 quick eval 样本
2. 每周回归：
   - train->save->resume smoke
   - 单尺度 + MS 评测 smoke
3. 发布前回归：
   - full training + full eval（关键模型）

---

## 附录 C：迁移状态看板模板

建议维护一个简洁看板（可写进 issue 或 docs）：

1. 模型迁移状态
2. 配置迁移状态
3. 入口收敛状态
4. 权重兼容状态
5. 复现点数状态
6. 测试覆盖状态

每项给：

1. Owner
2. 当前状态
3. 风险
4. 下一步

这会显著降低多人协作时的认知损耗。

