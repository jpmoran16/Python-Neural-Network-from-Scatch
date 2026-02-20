"""Tests for losses module."""

import unittest
import math
from nn.math_utils import Matrix
from nn.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy


class TestMSE(unittest.TestCase):
    """Test Mean Squared Error loss."""

    def test_mse_perfect_predictions_zero_loss(self):
        """MSE: predictions equal targets -> loss = 0."""
        pred = Matrix([[1.0], [2.0], [3.0]])
        tgt = Matrix([[1.0], [2.0], [3.0]])
        mse = MSE()
        loss = mse.forward(pred, tgt)
        self.assertAlmostEqual(loss, 0.0, places=10)

    def test_mse_known_loss_value(self):
        """MSE with known diff gives expected loss."""
        pred = Matrix([[2.0], [4.0]])  # targets 0, 0 -> diff 2, 4 -> squared 4, 16 -> mean 10
        tgt = Matrix([[0.0], [0.0]])
        mse = MSE()
        loss = mse.forward(pred, tgt)
        self.assertAlmostEqual(loss, 10.0, places=6)

    def test_mse_gradient_shape(self):
        """MSE backward returns gradient with same shape as predictions."""
        pred = Matrix([[1.0, 2.0], [3.0, 4.0]])
        tgt = Matrix([[0.0, 0.0], [0.0, 0.0]])
        mse = MSE()
        grad = mse.backward(pred, tgt)
        self.assertEqual(grad.shape(), pred.shape())


class TestBinaryCrossEntropy(unittest.TestCase):
    """Test Binary Cross-Entropy loss."""

    def test_bce_perfect_predictions_near_zero_loss(self):
        """BinaryCrossEntropy: perfect predictions -> near-zero loss."""
        pred = Matrix([[1.0], [0.0], [1.0], [0.0]])
        tgt = Matrix([[1.0], [0.0], [1.0], [0.0]])
        bce = BinaryCrossEntropy()
        loss = bce.forward(pred, tgt)
        self.assertLess(loss, 0.01)
        self.assertGreaterEqual(loss, 0.0)

    def test_bce_clips_extreme_values_no_inf_nan(self):
        """BinaryCrossEntropy clips extreme values (no inf/nan)."""
        pred = Matrix([[0.0], [1.0]])   # would be log(0) without clipping
        tgt = Matrix([[0.0], [1.0]])
        bce = BinaryCrossEntropy()
        loss = bce.forward(pred, tgt)
        self.assertFalse(math.isinf(loss))
        self.assertFalse(math.isnan(loss))
        grad = bce.backward(pred, tgt)
        for i in range(grad.rows):
            for j in range(grad.cols):
                self.assertFalse(math.isinf(grad.data[i][j]))
                self.assertFalse(math.isnan(grad.data[i][j]))


class TestCategoricalCrossEntropy(unittest.TestCase):
    """Test Categorical Cross-Entropy loss."""

    def test_cce_correct_class_lowest_loss(self):
        """CategoricalCrossEntropy: correct class prediction gives lower loss than wrong."""
        # One sample, 3 classes. Correct = class 1.
        pred_correct = Matrix([[0.1, 0.8, 0.1]])
        pred_wrong = Matrix([[0.8, 0.1, 0.1]])
        tgt = Matrix([[0.0, 1.0, 0.0]])
        cce = CategoricalCrossEntropy()
        loss_correct = cce.forward(pred_correct, tgt)
        loss_wrong = cce.forward(pred_wrong, tgt)
        self.assertLess(loss_correct, loss_wrong)

    def test_cce_gradient_shape(self):
        """CategoricalCrossEntropy backward returns gradient matching input shape."""
        pred = Matrix([[0.3, 0.4, 0.3], [0.5, 0.3, 0.2]])
        tgt = Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cce = CategoricalCrossEntropy()
        grad = cce.backward(pred, tgt)
        self.assertEqual(grad.shape(), pred.shape())


class TestLossGradientShapes(unittest.TestCase):
    """Test that all loss gradients match input shapes."""

    def test_mse_gradient_shapes_match(self):
        """MSE gradient shape matches predictions shape."""
        pred = Matrix([[1.0], [2.0]])
        tgt = Matrix([[0.0], [1.0]])
        self.assertEqual(MSE().backward(pred, tgt).shape(), pred.shape())

    def test_bce_gradient_shapes_match(self):
        """BCE gradient shape matches predictions shape."""
        pred = Matrix([[0.9], [0.1]])
        tgt = Matrix([[1.0], [0.0]])
        self.assertEqual(BinaryCrossEntropy().backward(pred, tgt).shape(), pred.shape())


if __name__ == "__main__":
    unittest.main()
