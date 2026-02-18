"""Compatibility tests for mmcv.ops and related mmseg loss paths."""

import os
import sys
import unittest
import math
import tempfile

import jittor as jt
import jittor.nn as nn
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv.ops import (CrissCrossAttention, DeformConv2dPack,
                      ModulatedDeformConv2dPack, DeformConv2d)
from mmcv.cnn import build_conv_layer
from mmcv.cnn.bricks import Conv2dAdaptivePadding
from mmcv.transforms.processing import TestTimeAug
from mmengine.config import Config
from mmengine.model import BaseModule
from mmengine.model.weight_init import (bias_init_with_prob, constant_init,
                                        kaiming_init, normal_init,
                                        trunc_normal_, trunc_normal_init,
                                        xavier_init)
from mmseg.structures import SegDataSample
from mmseg.registry import MODELS, TASK_UTILS
from mmseg.models.losses.focal_loss import FocalLoss
from mmseg.models.segmentors.multimodal_encoder_decoder import (
    MultimodalEncoderDecoder,)


class TestMMCVCompatOps(unittest.TestCase):

    def test_criss_cross_attention_forward(self):
        op = CrissCrossAttention(16)
        x = jt.randn((2, 16, 8, 8))

        # gamma is initialized as 0, output should still be a valid tensor.
        y = op(x)
        self.assertEqual(tuple(y.shape), (2, 16, 8, 8))

        # Ensure non-zero gamma path also runs.
        op.gamma.update(jt.ones((1, )))
        y2 = op(x)
        self.assertEqual(tuple(y2.shape), (2, 16, 8, 8))

    def test_deform_conv_pack_forward(self):
        op = DeformConv2dPack(8, 8, 3, stride=1, padding=1, bias=True)
        x = jt.randn((1, 8, 16, 16))
        y = op(x)
        self.assertEqual(tuple(y.shape), (1, 8, 16, 16))

    def test_modulated_deform_conv_pack_forward(self):
        op = ModulatedDeformConv2dPack(8, 8, 3, stride=1, padding=1, bias=True)
        x = jt.randn((1, 8, 16, 16))
        y = op(x)
        self.assertEqual(tuple(y.shape), (1, 8, 16, 16))

    def test_deform_conv2d_forward_cpu_and_cuda(self):
        op = DeformConv2d(8, 8, 3, stride=1, padding=1, deform_groups=1, bias=True)
        x = jt.randn((1, 8, 16, 16))
        offset = jt.randn((1, 2 * 3 * 3, 16, 16))

        orig_flag = int(getattr(jt.flags, 'use_cuda', 0))
        try:
            jt.flags.use_cuda = 0
            y_cpu = op(x, offset)
            self.assertEqual(tuple(y_cpu.shape), (1, 8, 16, 16))

            if bool(jt.has_cuda):
                jt.flags.use_cuda = 1
                y_cuda = op(x, offset)
                self.assertEqual(tuple(y_cuda.shape), (1, 8, 16, 16))
        finally:
            jt.flags.use_cuda = orig_flag

    def test_mmseg_focal_loss_multiclass_paths(self):
        orig_flag = int(getattr(jt.flags, 'use_cuda', 0))
        try:
            loss_fn = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25)
            pred = jt.randn((2, 3, 8, 8))
            target = jt.randint(0, 3, (2, 8, 8))

            jt.flags.use_cuda = 0
            loss_cpu_path = loss_fn(pred, target)
            self.assertEqual(loss_cpu_path.numel(), 1)

            jt.flags.use_cuda = 1
            loss_cuda_path = loss_fn(pred, target)
            self.assertEqual(loss_cuda_path.numel(), 1)
        finally:
            jt.flags.use_cuda = orig_flag

    def test_test_time_aug_cartesian_product(self):
        img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        data = {
            'img': img,
            'img_shape': img.shape[:2],
            'ori_shape': img.shape[:2],
            'seg_fields': []
        }
        tta = TestTimeAug(
            transforms=[
                [
                    dict(type='Resize', scale_factor=0.5, keep_ratio=True),
                    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
                ],
                [
                    dict(type='RandomFlip', prob=0.0, direction='horizontal'),
                    dict(type='RandomFlip', prob=1.0, direction='horizontal'),
                ],
            ])
        out = tta(data)
        self.assertIn('img', out)
        self.assertIn('flip', out)
        self.assertEqual(len(out['img']), 4)
        self.assertEqual(len(out['flip']), 4)

    def test_conv2d_adaptive_padding_shape(self):
        conv = Conv2dAdaptivePadding(
            3, 8, kernel_size=3, stride=2, padding=0, dilation=1, bias=False)
        x = jt.randn((1, 3, 65, 67))
        y = conv(x)
        self.assertEqual(tuple(y.shape), (1, 8, math.ceil(65 / 2), math.ceil(67 / 2)))

    def test_build_conv_layer_adaptive_padding(self):
        conv = build_conv_layer(
            dict(type='Conv2dAdaptivePadding'),
            3,
            8,
            3,
            stride=2,
            padding=0,
            bias=False,
        )
        self.assertIsInstance(conv, Conv2dAdaptivePadding)

    def test_conv_module_accepts_mmcv_style_kwargs(self):
        from mmcv.cnn import ConvModule
        mod = ConvModule(
            8,
            8,
            3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            inplace=False,
            order=('conv', 'norm', 'act'))
        x = jt.randn((1, 8, 16, 16))
        y = mod(x)
        self.assertEqual(tuple(y.shape), (1, 8, 16, 16))

    def test_conv_module_norm_before_conv_uses_input_channels(self):
        from mmcv.cnn import ConvModule
        mod = ConvModule(
            16,
            8,
            3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            order=('norm', 'act', 'conv'))
        x = jt.randn((1, 16, 16, 16))
        y = mod(x)
        self.assertEqual(tuple(y.shape), (1, 8, 16, 16))

    def test_torch_modulelist_subclass_uses_forward_not_sequential_execute(self):
        class ToyPPM(torch.nn.ModuleList):

            def __init__(self):
                super().__init__([
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.AdaptiveAvgPool2d(3),
                ])

            def forward(self, feats):
                outs = []
                for ppm in self:
                    o = ppm(feats)
                    outs.append(o.view(*feats.shape[:2], -1))
                return torch.cat(outs, dim=2)

        module = ToyPPM()
        x = jt.randn((1, 4, 16, 16))
        y = module(x)
        self.assertEqual(tuple(y.shape), (1, 4, 10))

    def test_torch_modulelist_setitem_replaces_entry(self):
        modules = torch.nn.ModuleList(
            [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()])
        modules[1] = torch.nn.GELU()
        self.assertEqual(len(modules), 3)
        self.assertEqual([type(m).__name__ for m in modules],
                         ['ReLU', 'GELU', 'ReLU'])

    def test_mmengine_weight_init_module_signatures(self):
        conv = nn.Conv2d(3, 8, 3, bias=True)
        xavier_init(conv, gain=0.5, bias=0.2, distribution='uniform')
        self.assertTrue(np.allclose(conv.bias.numpy(), 0.2, atol=1e-6))

        kaiming_init(
            conv,
            a=0.1,
            mode='fan_in',
            nonlinearity='leaky_relu',
            bias=-0.3,
            distribution='normal')
        self.assertTrue(np.allclose(conv.bias.numpy(), -0.3, atol=1e-6))

        constant_init(conv, val=1.0, bias=0.0)
        self.assertTrue(np.allclose(conv.weight.numpy(), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(conv.bias.numpy(), 0.0, atol=1e-6))

    def test_mmengine_weight_init_parameter_support(self):
        token = jt.zeros((512, ))
        normal_init(token, mean=0.0, std=0.02)
        token_np = token.numpy()
        self.assertGreater(float(np.std(token_np)), 0.005)

        trunc_normal_(token, mean=0.0, std=1.0, a=-0.2, b=0.2)
        token_np2 = token.numpy()
        self.assertLessEqual(float(np.max(token_np2)), 0.205)
        self.assertGreaterEqual(float(np.min(token_np2)), -0.205)

        linear = nn.Linear(32, 8, bias=True)
        trunc_normal_init(linear, mean=0.0, std=0.02, a=-0.04, b=0.04, bias=0.7)
        self.assertTrue(np.allclose(linear.bias.numpy(), 0.7, atol=1e-6))

    def test_mmengine_bias_init_with_prob(self):
        val = bias_init_with_prob(0.01)
        expected = -math.log((1 - 0.01) / 0.01)
        self.assertAlmostEqual(val, expected, places=7)

    def test_base_module_init_cfg_layer_override_and_guard(self):
        class ToyNet(BaseModule):

            def __init__(self):
                super().__init__(
                    init_cfg=[
                        dict(type='Constant', val=1.0, bias=0.1, layer='Conv2d'),
                        dict(
                            type='Normal',
                            mean=0.0,
                            std=0.02,
                            layer='Linear',
                            bias=-0.2,
                            override=dict(name='head')),
                    ])
                self.stem = nn.Conv2d(3, 4, 3, bias=True)
                self.head = nn.Conv2d(4, 2, 1, bias=True)

        model = ToyNet()
        model.init_weights()

        self.assertTrue(np.allclose(model.stem.weight.numpy(), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(model.stem.bias.numpy(), 0.1, atol=1e-6))
        self.assertTrue(np.allclose(model.head.bias.numpy(), -0.2, atol=1e-6))
        self.assertGreater(float(np.std(model.head.weight.numpy())), 0.005)

        stem_before = model.stem.weight.numpy().copy()
        head_before = model.head.weight.numpy().copy()
        model.init_weights()
        self.assertTrue(np.allclose(model.stem.weight.numpy(), stem_before, atol=1e-6))
        self.assertTrue(np.allclose(model.head.weight.numpy(), head_before, atol=1e-6))

    def test_base_module_init_cfg_recursive_child_override(self):
        class Child(BaseModule):

            def __init__(self):
                super().__init__(init_cfg=dict(type='Constant', val=0.3, layer='Conv2d'))
                self.conv = nn.Conv2d(4, 4, 3, padding=1, bias=False)

        class Parent(BaseModule):

            def __init__(self):
                super().__init__(init_cfg=dict(type='Constant', val=1.0, layer='Conv2d'))
                self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
                self.child = Child()

        model = Parent()
        model.init_weights()

        self.assertTrue(np.allclose(model.conv.weight.numpy(), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(model.child.conv.weight.numpy(), 0.3, atol=1e-6))

    def test_base_module_pretrained_prefix_load(self):
        class PrefixNet(BaseModule):

            def __init__(self, init_cfg=None):
                super().__init__(init_cfg=init_cfg)
                self.conv = nn.Conv2d(3, 2, 1, bias=True)

        source = PrefixNet()
        constant_init(source.conv, val=0.42, bias=0.7)

        ckpt_data = {
            'state_dict': {
                'toy.conv.weight': jt.array(source.conv.weight.numpy()),
                'toy.conv.bias': jt.array(source.conv.bias.numpy()),
            }
        }
        tmp = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            jt.save(ckpt_data, tmp_path)
            target = PrefixNet(
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint=tmp_path,
                    prefix='toy.',
                    strict=False))
            constant_init(target.conv, val=0.0, bias=0.0)

            target.init_weights()
            self.assertTrue(np.allclose(target.conv.weight.numpy(), 0.42, atol=1e-6))
            self.assertTrue(np.allclose(target.conv.bias.numpy(), 0.7, atol=1e-6))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_mmseg_encoder_decoder_tensor_forward(self):
        cfg = dict(
            type='EncoderDecoder',
            backbone=dict(
                type='ResNetV1c',
                depth=18,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True),
            decode_head=dict(
                type='FCNHead',
                in_channels=512,
                in_index=3,
                channels=64,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.0,
                num_classes=19,
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0)),
            train_cfg=dict(),
            test_cfg=dict(mode='whole'))
        model = MODELS.build(cfg)
        model.init_weights()
        x = jt.randn((1, 3, 64, 64))
        out = model(x, data_samples=None, mode='tensor')
        self.assertEqual(tuple(out.shape), (1, 19, 2, 2))

    def test_registry_build_default_args(self):
        ctx = type('Ctx', (), {'ignore_index': 255, 'loss_decode': nn.ModuleList()})()
        sampler = TASK_UTILS.build(
            dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10),
            default_args=dict(context=ctx))
        self.assertIsNotNone(sampler)

    def test_torch_creation_kwargs_compat(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        self.assertTrue(bool(t.requires_grad))
        z = torch.zeros(size=(2, 3), requires_grad=True)
        self.assertEqual(tuple(z.shape), (2, 3))
        self.assertTrue(bool(z.requires_grad))

        x = jt.randn((1, 3, 4, 5))
        nz = x.new_zeros(size=(1, 3, 4, 5))
        self.assertEqual(tuple(nz.shape), (1, 3, 4, 5))

    def test_torch_tensor_argmax_mean_sum_compat(self):
        x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])
        idx_row = x.argmax(dim=1)
        self.assertEqual(tuple(idx_row.shape), (2, ))
        self.assertTrue(np.array_equal(idx_row.numpy(), np.array([1, 2])))

        m = torch.mean(x, dim=0)
        s = torch.sum(x, dim=0)
        self.assertTrue(np.allclose(m.numpy(), np.array([2.5, 1.5, 3.5])))
        self.assertTrue(np.allclose(s.numpy(), np.array([5.0, 3.0, 7.0])))

        vals, inds = torch.max(x, dim=1, keepdim=True)
        self.assertEqual(tuple(vals.shape), (2, 1))
        self.assertEqual(tuple(inds.shape), (2, 1))
        self.assertTrue(np.array_equal(inds.numpy(), np.array([[1], [2]])))

    def test_torch_var_bmm_method_compat(self):
        a = jt.randn((2, 3, 4))
        b = jt.randn((2, 4, 5))
        y = a.bmm(b)
        y_ref = torch.bmm(a, b)
        self.assertEqual(tuple(y.shape), (2, 3, 5))
        self.assertTrue(np.allclose(y.numpy(), y_ref.numpy(), atol=1e-6))

    def test_torch_upsample_size_mode_compat(self):
        x = jt.randn((1, 3, 4, 3))

        up_size = torch.nn.Upsample(size=(8, 6), mode='nearest')
        y_size = up_size(x)
        self.assertEqual(tuple(y_size.shape), (1, 3, 8, 6))

        up_scale = torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        y_scale = up_scale(x)
        self.assertEqual(tuple(y_scale.shape), (1, 3, 8, 6))

    def test_pointrend_tensor_forward_from_config(self):
        cfg = Config.fromfile(
            'configs/point_rend/pointrend_r50_4xb4-160k_ade20k-512x512.py')
        model = MODELS.build(cfg.model)
        x = jt.randn((1, 3, 64, 64))
        out = model(x, data_samples=None, mode='tensor')
        self.assertEqual(len(out.shape), 4)
        self.assertEqual(int(out.shape[0]), 1)
        self.assertEqual(int(out.shape[1]), 150)

    def test_hrnet_backbone_build_from_base_config(self):
        cfg = Config.fromfile('configs/_base_/models/fcn_hr18.py')
        backbone = MODELS.build(cfg.model.backbone)
        self.assertIsNotNone(backbone)

    def test_hrnet_backbone_forward_from_base_config(self):
        cfg = Config.fromfile('configs/_base_/models/fcn_hr18.py')
        backbone = MODELS.build(cfg.model.backbone)
        feats = backbone(jt.randn((1, 3, 64, 64)))
        self.assertIsInstance(feats, list)
        self.assertEqual(len(feats), 4)

    def test_encoder_decoder_without_decode_head_builds_but_runtime_fails(self):
        cfg = dict(
            type='EncoderDecoder',
            backbone=dict(
                type='ResNetV1c',
                depth=18,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True),
            train_cfg=dict(),
            test_cfg=dict(mode='whole'))
        model = MODELS.build(cfg)
        self.assertFalse(model.with_decode_head)
        with self.assertRaises(RuntimeError):
            _ = model(jt.randn((1, 3, 64, 64)), data_samples=None, mode='tensor')

    def test_clip_text_encoder_lazy_build_and_runtime_guard(self):
        encoder = MODELS.build(
            dict(
                type='CLIPTextEncoder',
                dataset_name=None,
                vocabulary=None,
                cache_feature=False,
                cat_bg=False))
        with self.assertRaises(RuntimeError):
            _ = encoder()

    def test_vpd_base_config_model_build(self):
        cfg = Config.fromfile('configs/_base_/models/vpd_sd.py')
        model = MODELS.build(cfg.model)
        self.assertIsNotNone(model)
        self.assertFalse(model.with_decode_head)

    def test_maskformer_head_fallback_predict(self):
        head = MODELS.build(
            dict(
                type='MaskFormerHead',
                in_channels=[64, 128],
                in_index=[0, 1],
                feat_channels=32,
                out_channels=32,
                num_classes=7,
                num_queries=16))
        feats = [jt.randn((2, 64, 16, 16)), jt.randn((2, 128, 8, 8))]
        metas = [
            dict(img_shape=(32, 32), pad_shape=(32, 32), ori_shape=(32, 32))
            for _ in range(2)
        ]
        seg_logits = head.predict(feats, metas, test_cfg=dict())
        self.assertEqual(tuple(seg_logits.shape), (2, 7, 32, 32))

    def test_mask2former_head_fallback_predict(self):
        head = MODELS.build(
            dict(
                type='Mask2FormerHead',
                in_channels=[64, 128, 256],
                in_index=[0, 1, 2],
                feat_channels=64,
                out_channels=64,
                num_classes=5,
                num_queries=16))
        feats = [
            jt.randn((1, 64, 24, 24)),
            jt.randn((1, 128, 12, 12)),
            jt.randn((1, 256, 6, 6))
        ]
        metas = [dict(img_shape=(48, 48), pad_shape=(48, 48), ori_shape=(48, 48))]
        seg_logits = head.predict(feats, metas, test_cfg=dict())
        self.assertEqual(tuple(seg_logits.shape), (1, 5, 48, 48))

    def test_vpd_backbone_fallback_forward(self):
        backbone = MODELS.build(dict(type='VPD', diffusion_cfg=None))
        feats = backbone(jt.randn((1, 3, 64, 64)))
        self.assertIsInstance(feats, list)
        self.assertEqual(len(feats), 4)
        self.assertEqual(tuple(feats[0].shape[:2]), (1, 320))
        self.assertEqual(tuple(feats[1].shape[:2]), (1, 640))
        self.assertEqual(tuple(feats[2].shape[:2]), (1, 1280))
        self.assertEqual(tuple(feats[3].shape[:2]), (1, 1280))

    def test_vpd_depth_estimator_fallback_tensor_shape(self):
        cfg = Config.fromfile('configs/vpd/vpd_sd_4xb8-25k_nyu-480x480.py')
        model = MODELS.build(cfg.model)
        out = model(jt.randn((1, 3, 64, 64)), data_samples=None, mode='tensor')
        self.assertEqual(tuple(out.shape), (1, 1, 480, 480))

    def test_multimodal_encoder_decoder_tensor_forward_uses_encode_decode(self):
        model = object.__new__(MultimodalEncoderDecoder)
        calls = {}

        def fake_encode_decode(inputs, batch_img_metas):
            calls['metas'] = batch_img_metas
            return jt.ones((inputs.shape[0], 2, inputs.shape[2], inputs.shape[3]))

        model.encode_decode = fake_encode_decode
        x = jt.randn((2, 3, 16, 20))

        out = MultimodalEncoderDecoder._forward(model, x, data_samples=None)
        self.assertEqual(tuple(out.shape), (2, 2, 16, 20))
        self.assertEqual(len(calls['metas']), 2)
        self.assertEqual(tuple(calls['metas'][0]['ori_shape']), (16, 20))

        ds = SegDataSample()
        ds.set_metainfo(dict(ori_shape=(10, 11), img_shape=(8, 9)))
        _ = MultimodalEncoderDecoder._forward(model, x[:1], data_samples=[ds])
        self.assertEqual(calls['metas'][0]['ori_shape'], (10, 11))

    def test_seg_data_sample_accepts_metainfo_ctor(self):
        ds = SegDataSample(metainfo=dict(img_shape=(8, 9), ori_shape=(10, 11)))
        self.assertEqual(ds.metainfo['img_shape'], (8, 9))
        self.assertEqual(ds.metainfo['ori_shape'], (10, 11))


if __name__ == '__main__':
    unittest.main()
