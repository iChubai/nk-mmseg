# DFormer / DFormerv2 (Pure Jittor, mmengine Configs)

These configs migrate the legacy `local_configs/*` DFormer setup into
`configs/dformer/*` so they can run through `tools/train.py` in mmengine mode.

## Mapping from legacy configs

### NYUDepthv2

- `local_configs.NYUDepthv2.DFormer_Tiny` -> `configs/dformer/dformer_tiny_8xb8-500e_nyudepthv2-480x640.py`
- `local_configs.NYUDepthv2.DFormer_Small` -> `configs/dformer/dformer_small_8xb8-500e_nyudepthv2-480x640.py`
- `local_configs.NYUDepthv2.DFormer_Base` -> `configs/dformer/dformer_base_8xb8-500e_nyudepthv2-480x640.py`
- `local_configs.NYUDepthv2.DFormer_Large` -> `configs/dformer/dformer_large_8xb8-500e_nyudepthv2-480x640.py`
- `local_configs.NYUDepthv2.DFormerv2_S` -> `configs/dformer/dformerv2_s_8xb4-500e_nyudepthv2-480x640.py`
- `local_configs.NYUDepthv2.DFormerv2_B` -> `configs/dformer/dformerv2_b_8xb16-500e_nyudepthv2-480x640.py`
- `local_configs.NYUDepthv2.DFormerv2_L` -> `configs/dformer/dformerv2_l_8xb16-500e_nyudepthv2-480x640.py`

### SUNRGBD

- `local_configs.SUNRGBD.DFormer_Tiny` -> `configs/dformer/dformer_tiny_8xb16-300e_sunrgbd-480x480.py`
- `local_configs.SUNRGBD.DFormer_Small` -> `configs/dformer/dformer_small_8xb16-300e_sunrgbd-480x480.py`
- `local_configs.SUNRGBD.DFormer_Base` -> `configs/dformer/dformer_base_8xb16-300e_sunrgbd-480x480.py`
- `local_configs.SUNRGBD.DFormer_Large` -> `configs/dformer/dformer_large_8xb16-300e_sunrgbd-480x480.py`
- `local_configs.SUNRGBD.DFormerv2_S` -> `configs/dformer/dformerv2_s_8xb16-300e_sunrgbd-480x480.py`
- `local_configs.SUNRGBD.DFormerv2_B` -> `configs/dformer/dformerv2_b_8xb16-300e_sunrgbd-480x480.py`
- `local_configs.SUNRGBD.DFormerv2_L` -> `configs/dformer/dformerv2_l_8xb16-300e_sunrgbd-480x480.py`

## Run

```bash
python tools/train.py \
  --config configs/dformer/dformer_large_8xb8-500e_nyudepthv2-480x640.py
```

For smoke tests, use:

```bash
python tools/train.py \
  --config configs/dformer/dformer_large_8xb8-500e_nyudepthv2-480x640.py \
  --cfg-options train_cfg.max_iters=1 train_cfg.val_interval=100000000 \
  --work-dir /tmp/nk_dformer_smoke
```

Advanced evaluation (paper-style multi-scale + flip + sliding):

```bash
python tools/test.py \
  --config configs/dformer/dformer_large_8xb8-500e_nyudepthv2-480x640.py \
  --checkpoint checkpoints/trained/NYUv2_DFormer_Large.pth \
  --mode val \
  --multi_scale --flip --sliding \
  --scales 0.5 0.75 1.0 1.25 1.5
```
