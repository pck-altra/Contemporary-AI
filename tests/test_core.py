"""Structural correctness checks, independent of real training outcomes."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from data import split, augment, load_data
from models import Classifier
from evaluate import metrics, gradcam
import torch.nn.functional as F


class CoreTests(unittest.TestCase):
    def test_split(self):
        y = np.repeat(np.arange(10), 5000)
        tr, va = split(y)
        self.assertEqual(len(set(tr) & set(va)), 0)
        self.assertEqual(len(set(tr) | set(va)), 50000)
        np.testing.assert_array_equal(np.bincount(y[va]), np.repeat(500, 10))
        np.testing.assert_array_equal(np.bincount(y[tr]), np.repeat(4500, 10))
        np.testing.assert_array_equal(tr, split(y)[0])

    def test_models_and_gradients(self):
        for residual, se in [(False, False), (True, False), (True, True)]:
            model = Classifier(residual, se)
            out = model(torch.randn(2, 3, 32, 32))
            self.assertEqual(tuple(out.shape), (2, 10))
            out.square().mean().backward()
            self.assertTrue(
                all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            )

    def test_crop_is_per_sample_and_seeded(self):
        x = torch.arange(3 * 32 * 32).reshape(1, 3, 32, 32).float().repeat(8, 1, 1, 1)
        a = augment(x, torch.Generator().manual_seed(10))
        b = augment(x, torch.Generator().manual_seed(10))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.shape, x.shape)
        self.assertFalse(torch.equal(a[0], a[1]))

    def test_metrics_on_known_probabilities(self):
        y = np.arange(10)
        probs = np.eye(10) * 0.8 + 0.02
        m = metrics(y, probs)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertAlmostEqual(m["ece15"], 0.18)
        self.assertAlmostEqual(m["nll"], -np.log(0.82))
        np.testing.assert_array_equal(m["confusion"], np.eye(10, dtype=int))

    def test_gradcam_matches_cam_for_gap_linear_head(self):
        torch.manual_seed(1)
        model = Classifier(True, True).eval()
        x = torch.randn(1, 3, 32, 32)
        target = 2
        actual = gradcam(model, x, target)
        with torch.no_grad():
            features = model.features(x)
            # With GAP + linear head, Grad-CAM equals ReLU(weighted feature sum)
            # up to a positive spatial constant removed by normalization.
            expected = torch.relu(
                (features * model.head.weight[target][None, :, None, None] / 16).sum(
                    1, keepdim=True
                )
            )
            expected = F.interpolate(expected, (32, 32), mode="bilinear", align_corners=False)[0, 0]
            expected = (expected - expected.min()) / (expected.max() - expected.min() + 1e-8)
        np.testing.assert_allclose(actual, expected.numpy(), atol=1e-5)

    def test_official_integrity(self):
        x, y = load_data()
        self.assertEqual(x.shape, (50000, 3, 32, 32))
        self.assertEqual(y.shape, (50000,))
        x, y = load_data(test=True)
        self.assertEqual(x.shape, (10000, 3, 32, 32))
        np.testing.assert_array_equal(np.bincount(y), np.repeat(1000, 10))


if __name__ == "__main__":
    unittest.main()
