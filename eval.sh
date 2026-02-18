#!/bin/bash

# DFormer Jittor Evaluation Script
# Adapted from PyTorch version for Jittor framework

# Set environment variables
export CUDA_VISIBLE_DEVICES="0"

# Evaluation configuration
GPUS=1
CONFIG="configs/dformer/dformer_large_8xb8-500e_nyudepthv2-480x640.py"
CHECKPOINT="checkpoints/trained/NYUv2_DFormer_Large.pth"

# Run evaluation using advanced settings to match PyTorch baseline
python tools/test.py \
    --config=$CONFIG \
    --gpus=$GPUS \
    --checkpoint=$CHECKPOINT \
    --mode=val \
    --multi_scale \
    --flip \
    --sliding \
    --verbose

# Available configurations and checkpoints:

# NYUv2 DFormers
# --config=configs/dformer/dformer_large_8xb8-500e_nyudepthv2-480x640.py
# --checkpoint=checkpoints/trained/NYUv2_DFormer_Large.pth
# --config=configs/dformer/dformer_base_8xb8-500e_nyudepthv2-480x640.py
# --checkpoint=checkpoints/trained/NYUv2_DFormer_Base.pth
# --config=configs/dformer/dformer_small_8xb8-500e_nyudepthv2-480x640.py
# --checkpoint=checkpoints/trained/NYUv2_DFormer_Small.pth
# --config=configs/dformer/dformer_tiny_8xb8-500e_nyudepthv2-480x640.py
# --checkpoint=checkpoints/trained/NYUv2_DFormer_Tiny.pth

# NYUv2 DFormerv2
# --config=configs/dformer/dformerv2_l_8xb16-500e_nyudepthv2-480x640.py
# --checkpoint=checkpoints/trained/DFormerv2_Large_NYU.pth
# --config=configs/dformer/dformerv2_b_8xb16-500e_nyudepthv2-480x640.py
# --checkpoint=checkpoints/trained/DFormerv2_Base_NYU.pth
# --config=configs/dformer/dformerv2_s_8xb4-500e_nyudepthv2-480x640.py
# --checkpoint=checkpoints/trained/DFormerv2_Small_NYU.pth

# SUNRGBD DFormers
# --config=configs/dformer/dformer_large_8xb16-300e_sunrgbd-480x480.py
# --checkpoint=checkpoints/trained/SUNRGBD_DFormer_Large.pth
# --config=configs/dformer/dformer_base_8xb16-300e_sunrgbd-480x480.py
# --checkpoint=checkpoints/trained/SUNRGBD_DFormer_Base.pth
# --config=configs/dformer/dformer_small_8xb16-300e_sunrgbd-480x480.py
# --checkpoint=checkpoints/trained/SUNRGBD_DFormer_Small.pth
# --config=configs/dformer/dformer_tiny_8xb16-300e_sunrgbd-480x480.py
# --checkpoint=checkpoints/trained/SUNRGBD_DFormer_Tiny.pth

# SUNRGBD DFormerv2
# --config=configs/dformer/dformerv2_l_8xb16-300e_sunrgbd-480x480.py
# --checkpoint=checkpoints/trained/DFormerv2_Large_SUNRGBD.pth
# --config=configs/dformer/dformerv2_b_8xb16-300e_sunrgbd-480x480.py
# --checkpoint=checkpoints/trained/DFormerv2_Base_SUNRGBD.pth
# --config=configs/dformer/dformerv2_s_8xb16-300e_sunrgbd-480x480.py
# --checkpoint=checkpoints/trained/DFormerv2_Small_SUNRGBD.pth
