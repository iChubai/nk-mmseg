"""
Checkpoint loading and resume regression tests.
"""

import os
import sys
import tempfile
import unittest

import jittor as jt
import numpy as np
from jittor import nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.engine.engine import Engine
from utils.jt_utils import load_model


class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)

    def execute(self, x):
        return self.conv(x)


def _set_weight(model, value):
    arr = np.full(tuple(model.conv.weight.shape), value, dtype=np.float32)
    model.conv.weight.update(jt.array(arr))
    jt.sync_all()


class TestCheckpointResume(unittest.TestCase):

    def setUp(self):
        jt.flags.use_cuda = 0

    def test_load_model_restore_accepts_prefixed_keys(self):
        model = TinyConv()
        _set_weight(model, 0.0)
        shape = tuple(model.conv.weight.shape)

        ckpt = {"module.conv.weight": np.full(shape, 3.0, dtype=np.float32)}
        model, summary = load_model(
            model, ckpt, is_restore=True, return_stats=True)

        loaded = model.conv.weight.numpy()
        self.assertAlmostEqual(float(loaded.reshape(-1)[0]), 3.0, places=6)
        self.assertEqual(summary["load_stats"]["loaded_count"], 1)

    def test_load_model_restore_accepts_unprefixed_keys(self):
        model = TinyConv()
        _set_weight(model, 0.0)
        shape = tuple(model.conv.weight.shape)

        ckpt = {"conv.weight": np.full(shape, 2.0, dtype=np.float32)}
        model, summary = load_model(
            model, ckpt, is_restore=True, return_stats=True)

        loaded = model.conv.weight.numpy()
        self.assertAlmostEqual(float(loaded.reshape(-1)[0]), 2.0, places=6)
        self.assertEqual(summary["load_stats"]["loaded_count"], 1)

    def test_engine_restore_fallback_model_only_checkpoint(self):
        src_model = TinyConv()
        _set_weight(src_model, 5.0)
        model_only_ckpt = {
            "conv.weight": src_model.conv.weight.numpy().copy(),
        }

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "model_only.pth")
            jt.save(model_only_ckpt, ckpt_path)

            dst_model = TinyConv()
            _set_weight(dst_model, 0.0)
            optimizer = jt.optim.SGD(dst_model.parameters(), lr=0.1)

            engine = Engine()
            engine.register_state(model=dst_model, optimizer=optimizer)
            engine.state.epoch = 7
            engine.state.iteration = 123
            engine.continue_state_object = ckpt_path
            engine.restore_checkpoint()

            loaded = dst_model.conv.weight.numpy()
            self.assertAlmostEqual(float(loaded.reshape(-1)[0]), 5.0, places=6)
            # model-only fallback should not modify training progress state
            self.assertEqual(engine.state.epoch, 7)
            self.assertEqual(engine.state.iteration, 123)

    def test_engine_restore_training_checkpoint_increments_epoch(self):
        src_model = TinyConv()
        optimizer = jt.optim.Adam(src_model.parameters(), lr=1e-3)
        x = jt.ones((1, 1, 1, 1))
        loss = ((src_model(x) - 1.0) ** 2).mean()
        optimizer.step(loss)
        jt.sync_all()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "train_state.pth")

            save_engine = Engine()
            save_engine.register_state(model=src_model, optimizer=optimizer)
            save_engine.state.epoch = 3
            save_engine.state.iteration = 42
            save_engine.save_checkpoint(ckpt_path)

            dst_model = TinyConv()
            dst_optimizer = jt.optim.Adam(dst_model.parameters(), lr=1e-3)
            restore_engine = Engine()
            restore_engine.register_state(model=dst_model, optimizer=dst_optimizer)
            restore_engine.continue_state_object = ckpt_path
            restore_engine.restore_checkpoint()

            np.testing.assert_allclose(
                src_model.conv.weight.numpy(),
                dst_model.conv.weight.numpy(),
                rtol=1e-6,
                atol=1e-6)
            self.assertEqual(restore_engine.state.epoch, 4)
            self.assertEqual(restore_engine.state.iteration, 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
